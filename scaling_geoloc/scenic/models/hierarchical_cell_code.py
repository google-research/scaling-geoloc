# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hierarchical cell code retrieval model."""

from collections.abc import Sequence
from typing import Any

from etils.array_types import Array
from etils.array_types import FloatArray
from etils.array_types import IntArray
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from scenic.train_lib import train_utils

from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.models import cell_code
from scaling_geoloc.utils import misc


def batch_take(x: Array['N D'], idx: IntArray['... K']) -> Array['... K D']:
  return jnp.apply_along_axis(
      lambda idx_: jnp.take_along_axis(x, idx_[:, None], axis=0),
      -1,
      idx,
  )


def concat_tree(
    cell_codes: Sequence[FloatArray['... Ni D']],
    tree_idx: IntArray['... N H'],
    xnp: Any = jnp,
) -> FloatArray['...N D*H']:
  """Gather cell codes at each level and concatenate them in the feature dim."""
  return xnp.concatenate(
      [
          xnp.take_along_axis(cell_codes[i], tree_idx[..., i : i + 1], axis=-2)
          for i in range(len(cell_codes))
      ],
      axis=-1,
  )


class CellTreeCodes(nn.Module):
  """New tree embedding."""

  cell_tree: Sequence[IntArray['N K']]  # Each entry specifies K leaf idx.
  cell_code_dim: int | Sequence[int]
  global_contrastive: bool
  cell_centers_ll: Sequence[FloatArray['N 2']] | None = None
  temperature: float = 0.1
  dtype: jnp.dtype = jnp.float32
  max_num_children_inference: int | None = None
  pooling: str | None = None  # Pool finest cell codes.
  shared_projections: bool = False  # Share embedding projections across levels.
  cumprod: bool = False
  sum_probs: bool = False
  cell_code_dims: Sequence[int] | None = None
  l2_normalize: bool = False
  compose_loss: str = ''
  learned_temperature: bool = False

  def setup(self):
    self._cell_code_dims = self.cell_code_dims or [
        self.cell_code_dim for _ in self.cell_tree
    ]
    self.cell_codes = [
        self.param(
            f'cell_codes_{i}',
            nn.initializers.normal(stddev=1 / np.sqrt(self._cell_code_dims[i])),
            (s2cell_ids_at_level.shape[0], self._cell_code_dims[i]),
            self.dtype,
        )
        for i, s2cell_ids_at_level in enumerate(self.cell_tree)
    ]
    self.total_cell_code_dims = sum([x.shape[1] for x in self.cell_codes])
    self.num_childs = self.cell_tree[0].shape[
        1 if len(self.cell_tree) > 1 else 0
    ]
    self.num_roots = self.cell_tree[0].shape[0]

    self.num_levels = len(self.cell_tree)
    self.idx_to_tree_idx = self.cell_tree[-1]

    self.temperature_per_level = jnp.array([self.temperature] * self.num_levels)
    if self.learned_temperature:
      self.temperature_per_level = self.param(
          'temperature_per_level',
          nn.initializers.constant(self.temperature),
          (self.num_levels,),
          self.dtype,
      )

  def pool_cell_code_tree(
      self, finest_cell_codes: FloatArray['N D'], pooling: str
  ) -> Sequence[FloatArray['Ni D']]:
    cell_codes = [finest_cell_codes]
    for i in range(self.num_levels - 2, -1, -1):
      children_cell_idxs = self.cell_tree[i][:, :4]
      cell_codes_i = batch_take(cell_codes[-1], children_cell_idxs)
      cell_codes_i = {
          'mean': lambda x, valid: jnp.mean(x, axis=-2, where=valid),
          'max': lambda x, valid: jnp.max(
              x, axis=-2, where=valid, initial=-jnp.inf
          ),
      }[pooling](cell_codes_i, children_cell_idxs[..., None] > -1)
      cell_codes.append(cell_codes_i)
    return cell_codes[::-1]

  def get_cell_code_tree(self) -> Sequence[FloatArray['Ni D']]:
    if self.pooling is not None and self.pooling != 'none':
      assert True
      return self.pool_cell_code_tree(self.cell_codes, self.pooling)
    else:
      return self.cell_codes

  def traverse_tree(
      self,
      hierarchical_logits: Sequence[FloatArray['B D']],
      cell_codes: Sequence[FloatArray['Ni D']],
  ) -> types.Predictions:
    selected_cell_idxs = []
    selected_cell_probs = []
    sim = misc.compute_similarity(
        hierarchical_logits[0][:, None],
        cell_codes[0][None],
        normalize=self.l2_normalize,
    )[..., 0]
    probs = nn.softmax(sim / self.temperature_per_level[0], axis=-1)
    best_root_idx = jnp.argmax(probs, axis=-1)
    selected_cell_idxs.append(best_root_idx)  # B
    selected_cell_probs.append(
        jnp.take_along_axis(probs, best_root_idx[:, None], axis=-1)[..., 0]
    )
    # Traverse the tree down to the leaf cell.
    for i in range(1, len(self.cell_tree)):
      leafs = self.cell_tree[i - 1][selected_cell_idxs[-1]]  # B x K
      if self.max_num_children_inference is not None:
        leafs = leafs[:, : self.max_num_children_inference]
      sim = misc.compute_similarity(
          hierarchical_logits[i][:, None],
          batch_take(cell_codes[i], leafs),
          normalize=self.l2_normalize,
      )[..., 0]
      probs = nn.softmax(
          sim / self.temperature_per_level[i], axis=-1, where=leafs > -1
      )
      best_cell_idx_in_k = jnp.argmax(probs, axis=-1)  # B
      selected_cell_idxs.append(
          jnp.take_along_axis(leafs, best_cell_idx_in_k[:, None], axis=-1)[:, 0]
      )
      selected_cell_probs.append(
          jnp.take_along_axis(probs, best_cell_idx_in_k[:, None], axis=-1)[
              ..., 0
          ]
      )
    tree_cell_probs = jnp.stack(selected_cell_probs, axis=-1)
    return {
        'tree_cell_idxs': jnp.stack(selected_cell_idxs, axis=-1),
        'best_cell_idx': selected_cell_idxs[-1],
        'best_cell_prob': jnp.prod(tree_cell_probs, axis=-1),
        'tree_cell_probs': tree_cell_probs,
    }

  def flat_forward(
      self,
      hierarchical_logits: Sequence[FloatArray['B D']],
      cell_codes: Sequence[FloatArray['Ni D']],
  ) -> types.Predictions:
    tree_similarity_per_level = [
        misc.compute_similarity(
            h_logits[:, None],
            cell_codes[i][None],
            normalize=self.l2_normalize,
        )
        for i, h_logits in enumerate(hierarchical_logits)
    ]
    tree_similarity = concat_tree(
        tree_similarity_per_level,
        self.cell_tree[-1][None],
    )
    global_similarity = tree_similarity.sum(-1)
    global_log_probs = nn.log_softmax(
        global_similarity / self.temperature, axis=-1
    )
    global_probs = jnp.exp(global_log_probs)
    best_global_idx = jnp.argmax(global_probs, axis=-1)
    best_global_tree_idx = batch_take(
        self.cell_tree[-1], best_global_idx[:, None]
    )  # B x 1 x H
    best_global_prob = jnp.take_along_axis(
        global_probs, best_global_idx[:, None], axis=-1
    )[..., 0]

    tree_log_probs = nn.log_softmax(tree_similarity / self.temperature, axis=-1)
    tree_probs = jnp.exp(tree_log_probs)
    tree_prob = jnp.take_along_axis(tree_probs, best_global_tree_idx, axis=-2)
    tree_prob = tree_prob[:, 0]  # B x H

    return {
        'global_probs': global_probs,
        'tree_cell_idxs': best_global_tree_idx[:, 0],
        'best_cell_idx': best_global_idx,
        'best_cell_prob': best_global_prob,
        'tree_cell_probs': tree_prob,
        'tree_similarity': tree_similarity,
        'tree_similarity_per_level': [
            x[..., 0] for x in tree_similarity_per_level
        ],
    }

  @nn.compact
  def __call__(
      self, logits: FloatArray['... D'], encode_only: bool = False
  ) -> types.Predictions:
    cell_codes = self.get_cell_code_tree()
    # Project input logits to cell code space.

    if isinstance(logits, jnp.ndarray) and logits.ndim == 2:
      if self.shared_projections:
        # Directly use the input logits as cell code logits.
        hierarchical_logits_f = logits
        if logits.shape[-1] != self.total_cell_code_dims:
          raise ValueError(
              f'logits.shape[-1] ({logits.shape[-1]}) !='
              f' self.total_cell_code_dims ({self.total_cell_code_dims})'
          )
      else:
        # Add a projection layer to the input logits.
        hierarchical_logits_f = nn.Dense(
            self.total_cell_code_dims, name='hierarchical_logits'
        )(logits)
      hierarchical_logits = []
      offset = 0
      for i in range(self.num_levels):
        hierarchical_logits.append(
            hierarchical_logits_f[
                ..., offset : offset + self._cell_code_dims[i]
            ]
        )
        offset += self._cell_code_dims[i]
    else:
      hierarchical_logits = logits
      assert hierarchical_logits.ndim == 3  # Find root cell idx.

    if encode_only:
      return {
          'hierarchical_logits': hierarchical_logits,
      }

    if self.global_contrastive:
      pred = self.flat_forward(hierarchical_logits, cell_codes)
    else:
      pred = self.traverse_tree(hierarchical_logits, cell_codes)
    pred['hierarchical_logits'] = hierarchical_logits
    return pred

  def loss_metrics_function(
      self, batch: types.Batch, pred: types.Predictions
  ) -> types.LossMetricsTuple:
    losses, metrics = {}, {}
    cell_codes = self.get_cell_code_tree()
    hierarchical_logits = pred['hierarchical_logits']  # H * (B x Di)
    gt_cell_idx = self.idx_to_tree_idx[batch['cell_idx']]
    log_prob = jnp.zeros(gt_cell_idx.shape, dtype=jnp.float32)
    selected_cell_idx = jnp.zeros_like(gt_cell_idx)
    all_candidates = []
    for i in range(self.num_levels):
      if self.global_contrastive or i == 0:
        candidates = jnp.arange(cell_codes[i].shape[0])[None].repeat(
            gt_cell_idx.shape[0], axis=0
        )
        if 'adjacent_cell_idx_tree' in batch:
          correct_local_idx = batch['adjacent_cell_idx_tree'][:, i]
          weights = batch['adjacent_weights_tree'][:, i]
        else:
          correct_local_idx = gt_cell_idx[:, i : i + 1]
          weights = jnp.ones_like(correct_local_idx)
        cc = cell_codes[i][None]
      else:
        candidates = self.cell_tree[i - 1][gt_cell_idx[:, i - 1]]
        correct_local_idx = jax.vmap(
            lambda idx, cand: jnp.where(cand == idx, size=1, fill_value=-1)[0]
        )(gt_cell_idx[:, i], candidates)
        weights = jnp.ones_like(correct_local_idx)
        cc = batch_take(cell_codes[i], candidates)
      all_candidates.append(candidates)
      similarity = misc.compute_similarity(
          hierarchical_logits[i][:, None],  # b 1 d
          cc,  # b k d
          normalize=self.l2_normalize,
      )[..., 0]
      log_probs = nn.log_softmax(
          similarity / self.temperature_per_level[i],
          axis=-1,
          where=candidates > -1,
      )  # B x N
      log_probs_pos = jnp.take_along_axis(log_probs, correct_local_idx, axis=-1)
      log_prob_i = misc.masked_mean(
          log_probs_pos, weights * (correct_local_idx > -1), axis=-1
      )
      log_prob = log_prob.at[:, i].set(log_prob_i)

      selected_local_idx = jnp.argmax(log_probs, axis=-1)
      selected_cell_idx = selected_cell_idx.at[:, i].set(
          jnp.take_along_axis(candidates, selected_local_idx[:, None], axis=-1)[
              ..., 0
          ]
      )

    if 'global_probs' in pred:
      # B x N x H
      gt_mask_per_level = misc.tree_mask(
          self.cell_tree[-1], batch['cell_idx'][..., None]
      )
      cell_probs = pred['global_probs']
      gt_prob_tree = (gt_mask_per_level * cell_probs[..., None]).sum(-2)
      gt_log_prob_tree = misc.log_safe(gt_prob_tree)
      # The weights are the probabilies of the parent cell.
      weights = misc.weights_from_tree(gt_prob_tree)
      for i in range(self.num_levels):
        metrics[f'tree_gt_prob_{i:02d}'] = gt_prob_tree[..., i]
        losses[f'tree_nll_{i:02d}'] = -gt_log_prob_tree[..., i]
      losses['hierarchical_nll'] = -gt_log_prob_tree.mean(-1)
      losses['hierarchical_nll_weighted'] = -(
          jax.lax.stop_gradient(weights) * gt_log_prob_tree
      ).mean(-1)
      gt_sim = jnp.take_along_axis(
          pred['tree_similarity'].sum(-1), batch['cell_idx'][:, None], axis=-1
      )[:, 0]
      losses['sig_nll'] = -nn.log_sigmoid(gt_sim)
      losses['sig_prob'] = nn.sigmoid(gt_sim)
      losses['hierarchical_nll_weighted_grad'] = -(
          weights * gt_log_prob_tree
      ).mean(-1)

      frustum_cell_idxs = batch['frustum_cell_idx']
      frustum_probs = jnp.take_along_axis(
          cell_probs, frustum_cell_idxs, axis=-1
      )
      frustum_probs = jnp.where(frustum_cell_idxs > -1, frustum_probs, 0.0)
      losses['frustum_nll'] = -misc.log_safe(frustum_probs.sum(-1))

      if 'multi_similarity' in self.compose_loss:
        ms_loss = misc.MultiSimilarityLoss(
            alpha=2.0,
            beta=100.0,
            base=0.2,
        )
        distance_cell_to_image = misc.haversine_distance(
            self.cell_centers_ll[-1][None], batch['latlng'][:, None]
        )[..., 0]
        ms_loss.add_with_indices(
            pred['tree_similarity'].sum(-1),
            batch['cell_idx'][:, None],
            where=distance_cell_to_image > 200,
        )

        ms_pos, ms_neg = ms_loss.compute()
        losses['global_multi_similarity'] = ms_pos + ms_neg

      if 'flat' in self.compose_loss:
        flat_similarity = jnp.concat(pred['tree_similarity_per_level'], axis=-1)
        flat_probs = nn.softmax(flat_similarity / self.temperature, axis=-1)
        offsets = jnp.array(
            [0] + [x.shape[0] for x in self.cell_tree[:-1]]
        ).cumsum()
        gt_probs = jnp.take_along_axis(
            flat_probs, gt_cell_idx + offsets[None], axis=-1
        )
        losses['flat_nll'] = -misc.log_safe(gt_probs.sum(-1))
      if 'avg' in self.compose_loss:
        tree_gt_probs = jnp.concatenate(
            [
                jnp.take_along_axis(
                    nn.softmax(
                        pred['tree_similarity_per_level'][i] / self.temperature,
                        axis=-1,
                    ),
                    gt_cell_idx[:, i : i + 1],
                    axis=-1,
                )
                for i in range(self.num_levels)
            ],
            axis=-1,
        )
        losses['avg_nll'] = -misc.log_safe(tree_gt_probs.mean(-1))
      if 'cascading' in self.compose_loss:
        tree_similarity = pred['tree_similarity']
        cascading_tree_similarity = jnp.cumsum(tree_similarity, axis=-1)
        cascading_probs = nn.softmax(
            cascading_tree_similarity / self.temperature, axis=-2
        )
        cascading_tree_prob = (gt_mask_per_level * cascading_probs).sum(-2)
        cascading_weights = misc.weights_from_tree(cascading_tree_prob)
        losses['cascading_nll'] = -misc.log_safe(cascading_tree_prob).mean(-1)
        losses['cascading_nll_weighted'] = -(
            jax.lax.stop_gradient(cascading_weights)
            * misc.log_safe(cascading_tree_prob)
        ).mean(-1)

      if 'haversine' in self.compose_loss:
        distance_cell_to_image = misc.haversine_distance(
            self.cell_centers_ll[-1][None], batch['latlng'][:, None]
        )[..., 0]
        losses['haversine_nll_mask_weighted'] = misc.masked_distances_loss(
            (10_000, 1_000, 100, 10),
            pred['global_probs'],
            distance_cell_to_image.clip(min=0.0),
            weighted=True,
        )

      if 'interpolation' in self.compose_loss:
        adjacent_cell_idxs = batch['adjacent_cell_idx_tree']
        adjacent_weights = batch['adjacent_weights_tree']  # B x H x K
        adjacent_similarities = jnp.stack(  # B x H x K
            [
                jnp.take_along_axis(
                    pred['tree_similarity_per_level'][i],  # B x Ni
                    adjacent_cell_idxs[:, i],  # B x K
                    axis=-1,
                )
                for i in range(self.num_levels)
            ],
            axis=-2,
        )
        adjacent_similarity_tree = (  # B x H
            adjacent_similarities * adjacent_weights
        ).sum(-1)
        adjacent_similarity = adjacent_similarity_tree.sum(-1)  # B
        metrics['adjacent_similarity'] = adjacent_similarity

        global_probs = nn.softmax(
            jnp.put_along_axis(
                pred['tree_similarity'].sum(-1),
                batch['cell_idx'][:, None],
                adjacent_similarity[:, None],
                axis=-1,
                inplace=False,
            )
            / self.temperature,
            axis=-1,
        )
        distance_cell_to_image = misc.haversine_distance(
            self.cell_centers_ll[-1][None], batch['latlng'][:, None]
        )[..., 0]
        distance_gt_cell_to_image = misc.haversine_distance(
            self.cell_centers_ll[-1][batch['cell_idx']], batch['latlng']
        )
        delta_distances = distance_cell_to_image - distance_gt_cell_to_image

        losses['interpolation_haversine_nll_mask_weighted'] = (
            misc.masked_distances_loss(
                (10_000, 1_000, 100, 10),
                global_probs,
                delta_distances.clip(min=0.0),
                weighted=True,
            )
        )

        losses['interpolation_nll'] = -misc.log_safe(
            jnp.take_along_axis(
                arr=global_probs, indices=batch['cell_idx'][:, None], axis=-1
            )
        )[..., 0]

    if self.cumprod:
      log_prob = jnp.cumsum(log_prob, axis=-1)
    if self.sum_probs:
      log_prob = misc.log_safe(jnp.exp(log_prob).sum(-1, keepdims=True))
    prob = jnp.exp(log_prob)
    loss_per_level = -log_prob
    losses['cell_nll_tree'] = jnp.mean(loss_per_level, axis=-1)  # B

    weights = jax.lax.stop_gradient(jnp.exp(log_prob))
    weights = jnp.roll(weights, 1, axis=-1)
    weights = weights.at[:, 0].set(1.0)
    losses['cell_nll_tree_weighted'] = jnp.mean(
        loss_per_level * weights, axis=-1
    )  # B
    predicted_cell_idx = pred['tree_cell_idxs']  # B x H
    is_close = jnp.stack(
        [
            (predicted_cell_idx[:, i : i + 1] == all_candidates[i]).any(-1)
            for i in range(self.num_levels)
        ],
        axis=-1,
    )  # B x H
    is_correct = gt_cell_idx == predicted_cell_idx
    is_locally_correct = gt_cell_idx == selected_cell_idx
    metrics['num_levels_correct'] = is_correct.sum(-1)
    metrics['num_levels_close'] = is_close.sum(-1)
    metrics['is_correct'] = is_correct[:, -1]
    metrics['precision'] = is_correct[:, -1]
    metrics['is_close'] = is_close[:, -1]
    metrics['is_neighbor'] = jax.vmap(lambda x, y: jnp.isin(x, y).any())(
        predicted_cell_idx, batch['neighbor_cell_idx']
    )
    metrics['is_frustum'] = jax.vmap(lambda x, y: jnp.isin(x, y).any())(
        predicted_cell_idx, batch['frustum_cell_idx']
    )
    metrics['is_best_frustum'] = (
        batch['frustum_cell_idx'][:, 0] == predicted_cell_idx[:, -1]
    ).astype(jnp.float32)
    for level in range(self.num_levels):
      metrics[f'z_cell_nll_{level:02d}'] = loss_per_level[..., level]
      metrics[f'z_is_correct_{level:02d}'] = is_correct[..., level]
      metrics[f'z_is_close_{level:02d}'] = is_close[..., level]
      metrics[f'z_locally_correct_{level:02d}'] = is_locally_correct[..., level]
      metrics[f'z_gt_prob_{level:02d}'] = prob[..., level]
      metrics[f'z_pred_prob_{level:02d}'] = pred['tree_cell_probs'][..., level]
      metrics[f'z_valid_candidates_{level:02d}'] = (
          all_candidates[level] > -1
      ).sum(-1)
    return losses, metrics


class HierarchicalCellCodeRetrieval(cell_code.CellCodeRetrieval):
  """Hierarchical cell code retrieval model."""

  cell_tree: Sequence[IntArray['N K']]  # Each entry specifies K leaf idx.
  cell_centers_tree: Sequence[FloatArray['N 2']] = ()

  def setup(self):
    super().setup()
    self.cell_tree_codes = CellTreeCodes(
        self.cell_tree,
        cell_centers_ll=self.cell_centers_tree,
        cell_code_dim=self.config.cell_code_dim,
        temperature=self.config.get('temperature', 0.1),
        global_contrastive=self.config.get('global_contrastive', False),
        max_num_children_inference=self.config.get(
            'max_num_children_inference', 16
        ),
        dtype=self.dtype,
        pooling=self.config.get('pooling', None),
        shared_projections=self.config.get('shared_projections', False),
        cumprod=self.config.get('cumprod', False),
        cell_code_dims=self.config.get('cell_code_dims', None),
        compose_loss=self.config.get('compose_loss', None),
        l2_normalize=self.config.get('l2_normalize', False),
        learned_temperature=self.config.get('learned_temperature', False),
    )

  def encode_multiview(
      self,
      batch: types.Batch,
      train: bool,
      sow_attention: bool = False,
      **kwargs,
  ) -> tuple[FloatArray['... D'], FloatArray['... N D']]:
    """Encode multiview image pixels (interface for eval)."""
    logits, tokens = super().encode_multiview(
        batch,
        train=train,
        sow_attention=self.config.get('sow_attention', False) and not train,
        **kwargs,
    )

    hierarchical_logits = self.cell_tree_codes(logits, encode_only=True)[
        'hierarchical_logits'
    ]
    return jnp.concat(hierarchical_logits, axis=-1), tokens

  def __call__(
      self, batch: types.Batch, *, train: bool, debug: bool = False
  ) -> types.Predictions:
    image_pixels = self.get_image_pixels(batch, train)
    batch = cell_code.flatten_batch(batch, stride=None)
    logits, tokens = self.encode_multiview(
        {'image_pixels': image_pixels},
        train=train,
        sow_attention=self.config.get('sow_attention', False) and not train,
    )
    if logits.ndim == 3:
      logits, _ = logits[:, 0], logits[:, 1:]

    cell_pred = self.cell_tree_codes(logits)
    pred = {
        'logits': logits,
        'tokens': tokens,
        'augmented_pixels': image_pixels,
        **cell_pred,
    }

    if 'cell_idx' in batch:
      # GT is available, so compute loss.
      losses, metrics = self.cell_tree_codes.loss_metrics_function(batch, pred)
      pred = {
          'losses': losses,
          'metrics': metrics,
          **pred,
      }
    return pred


class HierarchicalCellCodeRetrievalModel(cell_code.CellCodeRetrievalModel):
  """Trainer-facing wrapper for the cell code retrieval model."""

  def build_flax_model(self) -> nn.Module:
    self.s2cell_ids = self.dataset_meta_data['s2cell_ids']
    self.s2cell_centers = jnp.asarray(
        self.dataset_meta_data['s2cell_centers_ecef'], dtype=self.dtype
    )
    self.s2cell_centers_ll = jnp.asarray(
        self.dataset_meta_data['s2cell_centers_ll'], dtype=self.dtype
    )
    self.cell_tree = self.dataset_meta_data['cell_tree']
    self.s2cell_ids_tree = self.dataset_meta_data['s2cell_ids_tree']

    self.s2cell_centers_ll_tree = []

    return HierarchicalCellCodeRetrieval(
        cell_tree=[jnp.asarray(x) for x in self.cell_tree],
        cell_centers=self.s2cell_centers_ll,
        cell_centers_tree=[self.s2cell_centers_ll],
        num_cells=len(self.s2cell_ids),
        config=self.config,
        dtype=self.dtype,
    )

  def init_state(
      self, model_params: types.ModelParams, model_state: types.ModelState
  ) -> tuple[types.ModelParams, types.ModelState]:
    model_params, model_state = super().init_state(model_params, model_state)
    return model_params, model_state

  def loss_metrics_function(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      model_params: jnp.ndarray | None = None,
      model_state: Any | None = None,
      learning_rates: types.LearningRates | None = None,
  ) -> types.LossMetricsTuple:
    losses, metrics = pred['losses'], pred['metrics']
    losses['total'] = types.compose_loss(losses, self.config.compose_loss)
    metrics = {
        **metrics,
        **self.distance_metrics(pred['best_cell_idx'], batch['cell_idx']),
    }
    return losses, metrics

  def get_cell_codes(
      self,
      train_state: train_utils.TrainState,
      do_normalize: bool | None = None,
  ) -> FloatArray['N D']:
    tree_height = len(self.cell_tree)
    tree_codes = [
        train_state.params['cell_tree_codes'][f'cell_codes_{i}']
        for i in range(tree_height)
    ]
    return concat_tree(tree_codes, self.cell_tree[-1], xnp=np)
