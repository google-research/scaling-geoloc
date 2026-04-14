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

"""SV-aerial alignment model with cell codes."""

from collections.abc import Mapping
from typing import Any

from etils.array_types import FloatArray
from etils.array_types import IntArray
import flax.linen as nn
import jax
import jax.numpy as jnp

from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.models import sv_aerial
from scaling_geoloc.scenic.models import sv_aerial_code
from scaling_geoloc.utils import misc


def sharded_argmax(
    x: FloatArray['ND*B N//ND'],
) -> tuple[FloatArray['B'], IntArray['B']]:
  """Sharded argmax."""

  best_idx_on_device = jnp.argmax(
      jax.lax.stop_gradient(x), axis=-1, keepdims=True
  )
  best_sim_on_device = jnp.take_along_axis(
      jax.lax.stop_gradient(x), best_idx_on_device, axis=-1
  )
  best_sims, best_idxs = [
      jax.lax.all_to_all(
          x,
          axis_name='batch',
          tiled=True,
          split_axis=0,
          concat_axis=len(x.shape) - 1,
      )
      for x in [best_sim_on_device, best_idx_on_device]
  ]
  best_device_idx = jnp.argmax(best_sims, axis=-1, keepdims=True)
  best_idx = jnp.take_along_axis(best_idxs, best_device_idx, axis=-1)
  best_sim = jnp.take_along_axis(best_sims, best_device_idx, axis=-1)
  # Add device offset.
  best_idx = best_idx + best_device_idx * x.shape[-1]
  return best_sim[..., 0], best_idx[..., 0]


def max_and_masked_expsum(
    batch_logits: FloatArray['B D'],  # total K dimensions
    batch_latlng: FloatArray['B 2'],  # total K dimensions
    sharded_logits: FloatArray['N//ND D'],  # total K dimensions
    global_latlng: FloatArray['N 2'],  # total K dimensions
    normalize: bool = False,
    beta: float = 1.0,
    base: float = 0.0,
    distance_threshold: float | None = None,
    required_idx: IntArray['B K'] | None = None,
) -> Mapping[str, Any]:
  """Computes the similarity between batch logits and sharded logits."""
  # Gather batch logits across devices.
  all_logits = jax.lax.all_gather(
      jnp.expand_dims(batch_logits, axis=-2),
      axis_name='batch',
      tiled=True,  # Concat in axis 0.
  )  # ND*B x ... x D

  # Compute similarity to sharded cell codes per device.
  similarity = misc.compute_similarity(
      all_logits,  # ND*B x 1 x D
      sharded_logits[None],  # Add device dim: 1 x N // ND x D
      normalize=normalize,
  )[..., 0]

  best_sim, best_idx = sharded_argmax(similarity)

  pos_sim = None
  if required_idx is not None:
    size_per_shard = similarity.shape[-1]
    device_offset = jax.lax.axis_index('batch') * size_per_shard
    all_required_idx = (
        jax.lax.all_gather(required_idx, axis_name='batch', tiled=True)
        - device_offset
    )  # ND*B x K
    valid = (all_required_idx >= 0) & (all_required_idx < size_per_shard)
    pos_sim = jnp.take_along_axis(
        similarity, all_required_idx, axis=-1, fill_value=0.0
    )
    pos_sim = jnp.where(valid, pos_sim, 0.0)
    pos_sim = jax.lax.psum_scatter(pos_sim, axis_name='batch', tiled=True)

    # Remove positives from sum.
    oob_required_idx = jnp.where(
        all_required_idx < 0, similarity.shape[-1], all_required_idx
    )
    similarity = jnp.put_along_axis(
        similarity,
        oob_required_idx,
        -jnp.inf,
        axis=-1,
        mode='drop',
        inplace=False,
    )

  if distance_threshold is not None:
    all_latlng = jax.lax.all_gather(
        batch_latlng, axis_name='batch', tiled=True  # Concat in axis 0.
    )  # ND*B x ... x 2
    latlng_on_device = global_latlng.reshape((jax.device_count(), -1, 2))
    distances = misc.haversine_distance(  # ND*B x N//ND
        jnp.expand_dims(all_latlng, axis=-2),
        latlng_on_device[jax.lax.axis_index('batch')][None],
    )[..., 0]
    valid_negatives = distances > distance_threshold
    similarity = jnp.where(valid_negatives, similarity, -jnp.inf)

  expsum = jnp.exp((similarity - base) * beta).sum(-1)
  # Similarity is ND*B x ... x N // ND
  expsum = jax.lax.psum_scatter(expsum, axis_name='batch', tiled=True)
  output = {
      'best_sim': best_sim,
      'best_idx': best_idx,
      'expsum': expsum,
  }
  if pos_sim is not None:
    output['pos_sim'] = pos_sim
  return output


class SVAerialCellCodeRetrieval(sv_aerial_code.SVAerialCellCodeRetrieval):
  """SV-aerial alignment model with cell codes."""

  def topk_similarities(
      self,
      logits: FloatArray['B D'],
      cell_codes: FloatArray['N|N//ND D'],
      k: int,
  ) -> tuple[FloatArray['B N'], IntArray['B N']]:
    """Computes similarity to cell codes."""
    if self.config.get('shard_codes', False) and not self.is_initializing():
      neg_sim, neg_idx = misc.distributed_topk(
          logits,
          cell_codes,
          k=k,
          ascending=False,
          recall_target=self.config.get('topk_recall_target', None),
          final_topk=False,
      )
      return neg_sim, neg_idx
    else:
      logits = jnp.expand_dims(logits, axis=-2)
      cell_codes = cell_codes[None]
      return misc.compute_similarity(
          logits,
          cell_codes,
          normalize=False,
      )[
          ..., 0
      ], jnp.arange(self.num_cells)[None].repeat(logits.shape[0], axis=0)

  @nn.compact
  def __call__(
      self,
      batch: types.Batch,
      *,
      train: bool,
      debug: bool = False,
  ) -> types.Predictions:
    pred = self.sv_aerial(batch, train=train, debug=debug)
    batch = sv_aerial.flatten_batch(batch)
    all_logits = {
        'sv': pred['logits'],
        'aerial': pred['aerial_logits_p'],
    }

    # Compute SV-CC similarity.
    pos_idxs = {
        'sv': batch['adjacent_cell_idx'],
        'aerial': batch['aerial']['adjacent_cell_idx'],
    }
    for name, logits in all_logits.items():
      cell_codes = self.get_cell_codes(self.cell_code_dim)
      if self.config.get(f'detach_{name}_cell_codes', False):
        cell_codes = jax.lax.stop_gradient(cell_codes)
      pos_sim = None
      if self.config.get('masked_expsum') and not self.is_initializing():
        assert self.config.get('shard_codes', False)
        res = max_and_masked_expsum(
            logits,
            batch['latlng'],
            cell_codes,
            self.cell_centers,
            normalize=False,
            beta=self.config.get('multi_similarity_beta', 50.0),
            base=self.config.get('multi_similarity_base', 0.0),
            required_idx=pos_idxs[name],
        )
        masked_expsum = res['expsum']
        best_cell_idx = res['best_idx']
        if 'pos_sim' in res:
          pos_sim = res['pos_sim']
      else:
        if self.config.get('topk') and self.config.topk < cell_codes.shape[0]:
          all_sim, all_idx = self.topk_similarities(
              logits, cell_codes, k=self.config.topk
          )
        else:
          all_sim = self.compute_similarity_to_cell_codes(logits, cell_codes)
          all_idx = jnp.arange(self.num_cells)[None].repeat(
              logits.shape[0], axis=0
          )
          pos_sim = jnp.take_along_axis(all_sim, pos_idxs[name], axis=-1)
        pred[f'{name}_sim'] = all_sim

        valid_negatives = jax.vmap(lambda x, y: ~jnp.isin(x, y) & (x > -1))(
            all_idx,
            pos_idxs[name],
        ) & (all_idx > -1)

        if self.config.get('filter_by_distance', False):
          cell_centers = jnp.take_along_axis(
              self.cell_centers[None],
              all_idx[..., None],
              axis=-2,
          )
          sv_cc_dist = misc.haversine_distance(
              cell_centers,
              batch['latlng'][:, None],
          )[..., 0]
          sv_cc_mask = sv_cc_dist > self.config.get(
              'cc_dist_threshold', self.config.get('same_cell_threshold', 200.0)
          )
          valid_negatives = sv_cc_mask
        neg_sim = jnp.where(valid_negatives, all_sim, -jnp.inf)
        masked_expsum = (
            valid_negatives
            * jnp.exp(
                (neg_sim - self.config.get('multi_similarity_base', 0.0))
                * self.config.get('multi_similarity_beta', 50.0)
            )
        ).sum(-1)

        best_local_idx = jnp.argmax(all_sim, axis=-1, keepdims=True)
        best_cell_idx = jnp.take_along_axis(all_idx, best_local_idx, axis=-1)[
            ..., 0
        ]

      if pos_sim is None:
        if cell_codes.shape[-1] < logits.shape[-1]:
          logits = logits[..., : cell_codes.shape[-1]]
          if self.config.get('l2_normalize', False):
            logits = misc.l2_normalize(logits)
        if self.config.get('shard_codes', False) and not self.is_initializing():
          pos_cc = misc.sharded_batched_gather(cell_codes, pos_idxs[name])
        else:
          pos_cc = jnp.take_along_axis(
              cell_codes[None],
              pos_idxs[name][..., None],
              axis=-2,
          )

        pos_sim = misc.compute_similarity(
            logits[..., None, :],
            pos_cc,
            normalize=False,
        )[..., 0]

      pred[f'{name}_masked_expsum'] = masked_expsum

      temperature = getattr(self, f'{name}_temperature')

      pred[f'{name}_cell_idx'] = best_cell_idx
      pred[f'{name}_best_cell_idx'] = pred['sv_cell_idx']
      pred[f'{name}_temperature'] = temperature[None].repeat(
          logits.shape[0], axis=0
      )

      pred[f'{name}_pos_sim'] = pos_sim
      pred[f'{name}_pos_idxs'] = pos_idxs[name]

    if 'cell_idx' in batch:
      pred['gt_cell_idx'] = batch['cell_idx']
    return pred


class SVAerialCellCodeRetrievalModel(
    sv_aerial_code.SVAerialCellCodeRetrievalModel
):
  """SV-aerial alignment model with cell codes."""

  def build_flax_model(self) -> nn.Module:
    self.s2cell_centers_ll = jnp.asarray(
        self.dataset_meta_data['s2cell_centers_ll'], dtype=self.dtype
    )
    return SVAerialCellCodeRetrieval(
        self.config,
        self.dtype,
        num_cells=self.s2cell_centers_ll.shape[0],
        cell_centers=self.s2cell_centers_ll,
    )

  def _loss_metrics_function(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      model_params: jnp.ndarray | None = None,
      model_state: Any | None = None,
      learning_rates: types.LearningRates | None = None,
  ) -> types.LossMetricsTuple:
    losses, metrics = sv_aerial.SVAerialRetrievalModel._loss_metrics_function(  # pylint: disable=protected-access
        self,
        pred,
        batch,
    )
    batch = sv_aerial.flatten_batch(batch)
    gt_cell_idx = batch['cell_idx']
    for name in ['sv', 'aerial']:
      gt_sim_l2 = (
          pred[f'{name}_pos_sim']
          * (pred[f'{name}_pos_idxs'] == gt_cell_idx[:, None])
      ).sum(-1)
      losses[f'{name}_sim_l2'] = gt_sim_l2
      losses[f'{name}_dist_l2'] = 1 - gt_sim_l2
      metrics[f'{name}_precision'] = (
          gt_cell_idx == pred[f'{name}_cell_idx']
      ).astype(jnp.float32)
      metrics[f'{name}_is_neighbor'] = jax.vmap(
          lambda x, y: (jnp.isin(x, y) & (y > -1)).any()
      )(
          pred[f'{name}_cell_idx'],
          batch['neighbor_cell_idx'],
      )
      metrics[f'{name}_temperature'] = pred[f'{name}_temperature'][0]

    sv_pred_cc_dist = misc.haversine_distance(
        self.s2cell_centers_ll[batch['cell_idx']],
        self.s2cell_centers_ll[pred['sv_best_cell_idx']],
    )[..., 0]

    metrics = metrics | misc.distance_metrics(sv_pred_cc_dist)

    for name in ['sv', 'aerial']:
      cosface_loss_name = f'{name}_cc_cosface'
      if cosface_loss_name in self.config.compose_loss:
        similarity = pred[f'{name}_sim']
        losses[cosface_loss_name] = misc.CosFace(
            scale=self.config.get('cosface_scale', 64.0),
            margin=self.config.get('cosface_margin', 0.35),
        )(
            similarity,
            gt_cell_idx,
        )

    if 'sv_aerial_cc_multi_similarity' in self.config.compose_loss:
      ms_loss = misc.MultiSimilarityLoss(
          alpha=self.config.get('multi_similarity_alpha', 1.0),
          beta=self.config.get('multi_similarity_beta', 50.0),
          base=self.config.get('multi_similarity_base', 0.0),
          margin=self.config.get('multi_similarity_margin', 0.0),
          negative_weight=self.config.get(
              'multi_similarity_negative_weight', -1
          ),
          relative_to_pos=self.config.get(
              'multi_similarity_relative_to_pos', False
          ),
          skip_positives=self.config.get(
              'multi_similarity_skip_positives', False
          ),
          split_terms=self.config.get('multi_similarity_split_terms', False),
      )
      # Add SV-Aerial similarities.
      pos_sims = {}
      for name, stochastic_pred in {
          'sv-a': pred['sv_aerial_stochastic_probs'],
          'a-sv': pred['aerial_sv_stochastic_probs'],
      }.items():
        if self.config.get('multi_similarity_factors', 'all') != 'all':
          if name not in self.config.multi_similarity_factors:
            continue
        valid = ~stochastic_pred['duplicates']
        similarity = stochastic_pred['similarity']
        ms_loss.add_with_indices(
            similarity,
            stochastic_pred['gt_index'][:, None],
            where=valid,
            add_positives=False,
        )
        pos_sims[name] = stochastic_pred['gt_similarity']

      # Add SV Proxy similarities.
      for name, prefix in {'sv-cc': 'sv', 'a-cc': 'aerial'}.items():
        if self.config.get('multi_similarity_factors', 'all') != 'all':
          if name not in self.config.multi_similarity_factors:
            continue
        proxy_pos = self.config.get(
            f'{prefix}_multi_similarity_proxy_positives',
        )
        weights = 1.0
        if proxy_pos == 'cell' or proxy_pos is None:
          gt_idx = batch['cell_idx'][:, None]
        elif proxy_pos == 'frustum':
          gt_idx = batch['frustum_cell_idx']
        elif proxy_pos == 'neighbor':
          gt_idx = batch['neighbor_cell_idx']
        elif proxy_pos == 'adjacent':
          if prefix == 'sv':
            gt_idx = batch['adjacent_cell_idx']
            weights = batch['adjacent_weights']
          elif prefix == 'aerial':
            gt_idx = batch['aerial']['adjacent_cell_idx']
            weights = batch['aerial']['adjacent_weights']
          else:
            raise ValueError(f'Unknown prefix for adjacent: {prefix}')
          weights = jnp.where(gt_idx > -1, weights, 0)
          weights = (
              (pred[f'{prefix}_pos_idxs'][..., None] == gt_idx[..., None, :])
              * weights[..., None, :]
          ).sum(-1)
        else:
          raise ValueError(f'Unknown proxy_pos: {proxy_pos}')

        pos_weights = jax.vmap(lambda x, y: jnp.isin(x, y) & (x > -1))(
            pred[f'{prefix}_pos_idxs'], gt_idx
        )

        if (
            self.config.get('multi_similarity_mean_pos_sim', False)
            and proxy_pos != 'adjacent'
        ):
          pos_sim = misc.masked_mean(
              pred[f'{prefix}_pos_sim'], pos_weights * weights, axis=-1
          )
        else:
          pos_sim = (pos_weights * weights * pred[f'{prefix}_pos_sim']).sum(-1)
        metrics[f'{prefix}_area_pos_sim'] = pos_sim
        metrics[f'{prefix}_area_size'] = pos_weights.sum(-1)
        pos_sims[name] = pos_sim
        ms_loss.neg.append(pred[f'{prefix}_masked_expsum'])

      if 'a-cc' in pos_sims and self.config.get(
          'multi_similarity_add_aerial_cc_positives', False
      ):
        ms_loss.pos.append(ms_loss.pos_scale(pos_sims['a-cc']))
        ms_loss.num_positives += 1

      if self.config.get('combined_positives'):
        combined_sim = jnp.max(
            jnp.stack([pos_sims['sv-a'], pos_sims['sv-cc']], axis=0), axis=0
        )
        ms_loss.pos.append(ms_loss.pos_scale(combined_sim))
        ms_loss.num_positives += 1
      else:
        if 'sv-a' in pos_sims:
          ms_loss.pos.append(ms_loss.pos_scale(pos_sims['sv-a']))
        if 'sv-cc' in pos_sims:
          ms_loss.pos.append(ms_loss.pos_scale(pos_sims['sv-cc']))
        ms_loss.num_positives += 2

      ms_pos, ms_neg = ms_loss.compute()
      losses['sv_aerial_cc_multi_similarity'] = ms_pos + ms_neg
      losses['sv_aerial_cc_multi_similarity_pos'] = ms_pos
      losses['sv_aerial_cc_multi_similarity_neg'] = ms_neg

      metrics['sv_aerial_cc_valid_negatives'] = ms_loss.num_negatives
      metrics['sv_aerial_cc_valid_positives'] = ms_loss.num_positives
    return losses, metrics
