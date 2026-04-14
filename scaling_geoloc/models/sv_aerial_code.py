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

import functools
import logging
from typing import Any, Mapping, Sequence

from etils.array_types import FloatArray
from etils.array_types import IntArray
from flax import jax_utils
import flax.linen as nn
import grain.python as grain
import jax
import jax.numpy as jnp
import jaxtyping as jt
import ml_collections
import numpy as np
from scenic.train_lib import train_utils

from scaling_geoloc.scenic import sv_aerial_data_source
from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.models import location_encoder
from scaling_geoloc.scenic.models import sv_aerial
from scaling_geoloc.utils import misc


def filter_and_remap_indices(
    similarity: FloatArray['B K'],
    neighbors: IntArray['B K'],
    index_map: IntArray['N'],
    cell_centers_ll: FloatArray['N//DG 2'],
    min_distance_meters: float = 200.0,
) -> tuple[FloatArray['B K'], IntArray['B K'], FloatArray['B K']]:
  """Filter and remap indices."""
  neighbor_centers_ll = jnp.take_along_axis(
      jax.lax.all_gather(cell_centers_ll, axis_name='batch', tiled=True)[None],
      neighbors[..., None],
      axis=-2,
  )  # B x K x 2
  distance_meters = misc.haversine_distance(
      cell_centers_ll[:, None], neighbor_centers_ll
  )[..., 0]
  neighbors_mapped = jax.vmap(lambda x: jnp.where(x > -1, index_map[x], -1))(
      neighbors
  )
  valid_neighbors = (neighbors_mapped > -1) & (
      distance_meters > min_distance_meters
  )
  similarity_filtered = jnp.where(valid_neighbors, similarity, -jnp.inf)
  # Recover 'self' at position 0.
  similarity_filtered = similarity_filtered.at[:, 0].set(similarity[:, 0])
  local_indices = jnp.argsort(similarity_filtered, axis=-1, descending=True)
  similarity = jnp.take_along_axis(similarity_filtered, local_indices, axis=-1)
  neighbors = jnp.take_along_axis(neighbors_mapped, local_indices, axis=-1)
  distance_meters = jnp.take_along_axis(distance_meters, local_indices, axis=-1)
  return similarity, neighbors, distance_meters


class SVAerialCellCodeRetrieval(sv_aerial.SVAerialRetrieval):
  """SV-aerial alignment model with cell codes."""

  num_cells: int = 0

  def setup(self):
    super().setup()
    self.sv_temperature = jnp.array(
        self.config.get('cc_temperature', self.config.get('temperature', 0.1))
    )
    self.aerial_temperature = jnp.array(
        self.config.get('cc_temperature', self.config.get('temperature', 0.1))
    )
    if self.config.get('learned_temperature', False):
      self.sv_temperature = self.param(
          'sv_temperature',
          nn.initializers.constant(self.sv_temperature),
          (),
          self.dtype,
      )

      self.aerial_temperature = self.param(
          'aerial_temperature',
          nn.initializers.constant(self.aerial_temperature),
          (),
          self.dtype,
      )

    self.cell_code_dim = self.config.get(
        'cell_code_dim', self.backbone.output_dim()
    )

  @nn.compact
  def get_implicit_cell_codes(self, cell_code_dim: int) -> FloatArray['N D']:
    num_cells = self.num_cells
    if self.config.get('shard_codes', False) and not self.is_initializing():
      num_cells = num_cells // jax.device_count()

    init_fn = nn.initializers.normal(stddev=1 / np.sqrt(cell_code_dim))
    cell_codes = self.param(
        'cell_codes_implicit',
        init_fn,
        (num_cells, cell_code_dim),
        self.dtype,
    )
    return cell_codes

  @nn.compact
  def encode_latlngs(
      self,
      latlng: FloatArray['B 2'],
  ) -> FloatArray['B D']:
    ll_encoder = location_encoder.LocationEncoder(
        output_dim=self.cell_code_dim,
        name='location_encoder',
    )
    return ll_encoder(latlng)

  @nn.compact
  def get_cell_codes(self, cell_code_dim: int) -> FloatArray['N D']:
    num_cells = self.num_cells
    if self.config.get('shard_codes', False) and not self.is_initializing():
      num_cells = num_cells // jax.device_count()
    if self.config.get('cell_codes_momentum'):
      cell_codes = self.variable(
          'model_stats',
          'ema_cell_codes',
          jnp.zeros,
          shape=(num_cells, cell_code_dim),
      )
      cell_codes = cell_codes.value
    elif self.config.get('location_encoder'):
      if self.config.get('shard_codes', False) and not self.is_initializing():
        cell_codes = self.encode_latlngs(
            self.cell_centers.reshape(jax.device_count(), -1, 2)[
                jax.lax.axis_index('batch')
            ]
        )
      else:
        cell_codes = self.encode_latlngs(self.cell_centers)
    else:
      init_type = self.config.get('init_cell_codes', 'normal')
      if init_type == 'orthogonal':
        init_fn = nn.initializers.orthogonal(scale=1 / np.sqrt(cell_code_dim))
      elif init_type == 'normal':
        init_fn = nn.initializers.normal(stddev=1 / np.sqrt(cell_code_dim))
      else:
        raise ValueError(f'Unknown init_cell_codes: {init_type}')
      cell_codes = self.param(
          'cell_codes',
          init_fn,
          (num_cells, cell_code_dim),
          self.dtype,
      )

    if self.config.get('l2_normalize_per_head', False):
      cell_codes = misc.l2_normalize_per_head_vit(
          cell_codes, self.backbone.head
      )
    if self.config.get('l2_normalize', False):
      cell_codes = misc.l2_normalize(cell_codes)
    return cell_codes

  def compute_similarity_to_cell_codes(
      self,
      logits: FloatArray['B ... D'],
      cell_codes: FloatArray['... N|N//ND D'],
      normalize: bool = False,
  ) -> FloatArray['B N']:
    """Computes similarity to cell codes."""
    logits = jnp.expand_dims(logits, axis=-2)
    cell_codes = cell_codes[None]
    if self.config.get('shard_codes', False) and not self.is_initializing():
      return misc.sharded_similarity(logits, cell_codes, normalize=normalize)
    else:
      return misc.compute_similarity(
          logits,
          cell_codes,
          normalize=normalize,
      )[..., 0]

  def compute_multihead_similarity_to_cell_codes(
      self,
      logits: FloatArray['B D'],
      cell_codes: FloatArray['N D'],
      num_heads: int,
      offset: int = 0,
      normalize: bool = False,
  ) -> FloatArray['B N H']:
    """Computes multi-head similarity to cell codes."""

    def add_head_dim(x, num_heads):
      x = x.reshape(x.shape[:-1] + (num_heads, x.shape[-1] // num_heads))
      return x  # B x H x N x D

    mh_similarity = self.compute_similarity_to_cell_codes(
        add_head_dim(logits[..., offset:], num_heads),
        jnp.swapaxes(add_head_dim(cell_codes[..., offset:], num_heads), 0, 1),
        normalize=normalize,
    )  # B x H x N
    if offset != 0:
      mh_similarity = jnp.concatenate(
          [
              self.compute_similarity_to_cell_codes(
                  logits[..., None, :offset],
                  jnp.swapaxes(add_head_dim(cell_codes[..., :offset], 1), 0, 1),
                  normalize=normalize,
              ),
              mh_similarity,
          ],
          axis=-2,
      )
    mh_similarity = jnp.swapaxes(mh_similarity, -1, -2)
    return mh_similarity

  def find_spatial_nn(
      self,
      latlng_center: FloatArray['B 2'],
      threshold_meters: float = 200.0,
  ) -> tuple[IntArray['B'], FloatArray['B']]:
    """Finds spatial nearest neighbors."""
    distance_meters = misc.haversine_distance(
        self.cell_centers[None],
        latlng_center[:, None],
    )  # B x 1
    cell_idxs = jnp.argmin(
        distance_meters,
        axis=-2,
    )  # B x 1
    min_distance = jnp.take_along_axis(
        distance_meters[..., 0], cell_idxs, axis=-1
    )  # B x 1
    cell_idxs = jnp.where(
        min_distance < threshold_meters,
        cell_idxs,
        -1,
    )
    return cell_idxs, min_distance

  def encode_aerial(
      self,
      aerial_batch: types.Batch,
      *args,
      **kwargs,
  ) -> tuple[jt.Float[jt.Array, '... N D'], jt.Float[jt.Array, '... N L D']]:
    """Interface to aerial backbone encoder."""
    if (
        'cell_idxs' in aerial_batch or 'latlng_center' in aerial_batch
    ) and self.config.get('implicit_cell_code_dim', 0) > 0:
      if 'latlng_center' in aerial_batch:
        cell_idxs, _ = self.find_spatial_nn(
            aerial_batch['latlng_center'],
            self.config.get('same_cell_threshold', 200.0),
        )
      else:
        cell_idxs = aerial_batch['cell_idxs']
      cell_codes = self.get_implicit_cell_codes(
          self.config.implicit_cell_code_dim
      )
      if self.config.get('shard_codes', False) and not self.is_initializing():
        extra_tokens = misc.sharded_batched_gather(cell_codes, cell_idxs)
      else:
        extra_tokens = jnp.take_along_axis(
            cell_codes[None], cell_idxs[..., None], axis=-2
        )

      # Perform random dropout of cell codes.
      if (
          kwargs.get('train', False)
          and self.config.get('implicit_cell_codes_dropout', 0.0) > 0
      ):
        extra_tokens = misc.dropout_token(
            self.make_rng('implicit_cell_codes_dropout'),
            extra_tokens,
            self.config.implicit_cell_codes_dropout,
            fill_value=0.0,
        )

      if aerial_batch['image_pixels'].ndim == 5:
        extra_tokens = extra_tokens[..., None, :, :].repeat(
            aerial_batch['image_pixels'].shape[-4], axis=-3
        )
    else:
      extra_tokens = None
    return super().encode_aerial(
        aerial_batch, *args, extra_tokens=extra_tokens, **kwargs
    )

  @nn.compact
  def sv_aerial(self, batch: types.Batch, train: bool, debug: bool = False):
    return super().__call__(batch, train=train, debug=debug)

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
    if 'latlng' in batch:
      pred['sv_cc_dist'] = misc.haversine_distance(
          self.cell_centers[None],
          batch['latlng'][:, None],
      )[..., 0]
      pred['sv_cc_mask'] = pred['sv_cc_dist'] > self.config.get(
          'cc_dist_threshold', self.config.get('same_cell_threshold', 200.0)
      )
    all_logits = {
        'sv': pred['logits'],
        'aerial': pred['aerial_logits_p'],
    }
    for name, logits in all_logits.items():
      cell_codes = self.get_cell_codes(self.cell_code_dim)
      if cell_codes.shape[-1] < logits.shape[-1]:
        logits = logits[..., : cell_codes.shape[-1]]
        if self.config.get('l2_normalize', False):
          logits = misc.l2_normalize(logits)
      if self.config.get(f'detach_{name}_cell_codes', False):
        cell_codes = jax.lax.stop_gradient(cell_codes)
      if self.config.get(f'detach_{name}_embeddings', False):
        logits = jax.lax.stop_gradient(logits)
      sim = self.compute_similarity_to_cell_codes(
          logits,
          cell_codes,
          normalize=False,
      )

      temperature = getattr(self, f'{name}_temperature')
      pred[f'{name}_sim'] = sim
      pred[f'{name}_cell_idx'] = jnp.argmax(sim, axis=-1)
      pred[f'{name}_best_cell_idx'] = pred[f'{name}_cell_idx']
      pred[f'{name}_cell_prob'] = jnp.exp(
          jnp.take_along_axis(
              sim,
              pred[f'{name}_cell_idx'][:, None],
              axis=-1,
          )[:, 0]
      )
      pred[f'{name}_best_cell_prob'] = jnp.exp(
          jnp.take_along_axis(
              sim,
              pred[f'{name}_cell_idx'][:, None],
              axis=-1,
          )[:, 0]
      )
      pred[f'{name}_temperature'] = temperature[None].repeat(
          sim.shape[0], axis=0
      )

    if 'cell_idx' in batch:
      pred['gt_cell_idx'] = batch['cell_idx']
    return pred


class SVAerialCellCodeRetrievalModel(sv_aerial.SVAerialRetrievalModel):
  """SV-aerial alignment model with cell codes."""

  required_rngs: tuple[str, ...] = (
      'masking_rng',
      'dropout',
      'augment',
      'augment_aerial',
      'droptok',
      'crop',
      'dropsim',
      'implicit_cell_codes_dropout',
      'rope',
  )

  def __init__(
      self,
      config: ml_collections.ConfigDict,
      dataset_meta_data: dict[str, Any],
      dtype: jnp.dtype[Any] = jnp.float32,
  ):
    super().__init__(config, dataset_meta_data, dtype)
    self.s2cell_centers_ll = jnp.asarray(
        self.dataset_meta_data['s2cell_centers_ll'], dtype=self.dtype
    )
    if 'cell_tree' in self.dataset_meta_data:
      self.cell_tree = [
          jnp.asarray(x) for x in self.dataset_meta_data['cell_tree']
      ]
    else:
      self.cell_tree = None

  def build_flax_model(self) -> nn.Module:
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
    losses, metrics = super()._loss_metrics_function(
        pred,
        batch,
    )
    batch = sv_aerial.flatten_batch(batch)
    gt_cell_idx = batch['cell_idx']
    for name in ['sv', 'aerial']:
      gt_sim_l2 = jnp.take_along_axis(
          pred[f'{name}_sim'], gt_cell_idx[:, None], axis=-1
      )[:, 0]
      if f'{name}_cc_nll' in self.config.compose_loss:
        gt_labels = (
            jnp.arange(pred[f'{name}_sim'].shape[-1])[None]
            == gt_cell_idx[:, None]
        )
        to_supervise = gt_labels | pred['sv_cc_mask']

        if self.config.get('label_smoothing', 0.0) > 0.0:
          gt_labels = jnp.where(
              gt_labels,
              gt_labels - self.config.label_smoothing,
              to_supervise / gt_labels.shape[-1],
          )
        sim = jnp.where(
            to_supervise,
            pred[f'{name}_sim'] / pred[f'{name}_temperature'][0],
            -jnp.inf,
        )
        log_probs = nn.log_softmax(sim, axis=-1, where=to_supervise)

        losses[f'{name}_cc_nll'] = -(gt_labels * log_probs).sum(-1)  # pytype: disable=attribute-error
      losses[f'{name}_sim_l2'] = gt_sim_l2
      losses[f'{name}_dist_l2'] = 1 - gt_sim_l2
      metrics[f'{name}_sim_l2_avg'] = pred[f'{name}_sim'].mean(-1)
      metrics[f'{name}_precision'] = (
          gt_cell_idx == pred[f'{name}_cell_idx']
      ).astype(jnp.float32)
      metrics[f'{name}_is_neighbor'] = jax.vmap(
          lambda x, y: (jnp.isin(x, y) & (y > -1)).any()
      )(
          pred[f'{name}_cell_idx'],
          batch['neighbor_cell_idx'],
      )
      top_k = jax.lax.approx_max_k(pred[f'{name}_sim'], k=100)[1]
      metrics[f'{name}_false_positives'] = jnp.cumprod(
          1 - (gt_cell_idx[:, None] == top_k), axis=-1
      ).sum(-1)
      metrics[f'{name}_temperature'] = pred[f'{name}_temperature'][0]
    if 'sv_cc_mask' in pred:
      sv_cc_mask = pred['sv_cc_mask']
    else:
      sv_cc_dist = misc.haversine_distance(
          self.s2cell_centers_ll[None],
          batch['latlng'][:, None],
      )[..., 0]
      sv_cc_mask = sv_cc_dist > self.config.get(
          'cc_dist_threshold', self.config.get('same_cell_threshold', 200.0)
      )

    sv_pred_cc_dist = misc.haversine_distance(
        self.s2cell_centers_ll[batch['cell_idx']],
        self.s2cell_centers_ll[pred['sv_best_cell_idx']],
    )[..., 0]

    metrics = metrics | misc.distance_metrics(sv_pred_cc_dist)

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
            add_positives=name == 'sv-a',
        )

        # Add SV-CC and Aerial-CC similarities.
      if self.config.get('mask_multi_similarity_by_distance'):
        valid_cc = sv_cc_mask
      else:
        valid_cc = jnp.ones_like(pred['sv_sim'], dtype=jnp.bool_)

      # Add Proxy similarities.
      for name, prefix in {'sv-cc': 'sv', 'a-cc': 'aerial'}.items():
        if self.config.get('multi_similarity_factors', 'all') != 'all':
          if name not in self.config.multi_similarity_factors:
            continue
        similarity = pred[f'{prefix}_sim']
        if name == 'a-cc' and self.config.get('multi_similarity_a_cc_base'):
          # Put more force on A-CC.
          similarity = similarity - self.config.multi_similarity_a_cc_base
        proxy_pos = self.config.get(
            f'{prefix}_multi_similarity_proxy_positives',
        )
        if proxy_pos == 'cell' or proxy_pos is None:
          gt_idx = batch['cell_idx'][:, None]
        elif proxy_pos == 'frustum':
          gt_idx = batch['frustum_cell_idx']
        elif proxy_pos == 'neighbor':
          gt_idx = batch['neighbor_cell_idx']
        elif proxy_pos == 'adjacent':
          if prefix == 'sv':
            gt_idx = batch['adjacent_cell_idx']
            gt_weights = jnp.where(gt_idx > -1, batch['adjacent_weights'], 0)
          elif prefix == 'aerial':
            gt_idx = batch['aerial']['adjacent_cell_idx']
            gt_weights = jnp.where(
                gt_idx > -1, batch['aerial']['adjacent_weights'], 0
            )
          else:
            raise ValueError(f'Unknown prefix: {prefix}')
          gt_sim = jnp.take_along_axis(similarity, gt_idx, axis=-1)
          interpolated_sim = (gt_weights * gt_sim).sum(-1)
          if self.config.get(f'multi_similarity_add_{prefix}_cc_positives'):
            ms_loss.pos.append(ms_loss.pos_scale(interpolated_sim))
            metrics[f'{prefix}_area_pos_sim'] = interpolated_sim
            metrics[f'{prefix}_area_size'] = (gt_idx > -1).sum(-1)
          if self.config.get(
              f'multi_similarity_add_{prefix}_cc_negatives', True
          ):
            ms_loss.add_negatives(similarity, valid_cc)
          continue
        else:
          raise ValueError(f'Unknown proxy_pos: {proxy_pos}')

        gt_sim = jnp.take_along_axis(similarity, gt_idx, axis=-1)
        metrics[f'{prefix}_area_pos_sim'] = misc.masked_mean(
            gt_sim, gt_idx > -1, axis=-1
        )
        metrics[f'{prefix}_area_size'] = (gt_idx > -1).sum(-1)
        ms_loss.add_with_indices(
            similarity,
            gt_idx,
            where=valid_cc,
            add_negatives=self.config.get(
                f'multi_similarity_add_{prefix}_cc_negatives', True
            ),
            add_positives=self.config.get(
                f'multi_similarity_add_{prefix}_cc_positives', True
            ),
        )
      ms_pos, ms_neg = ms_loss.compute()
      losses['sv_aerial_cc_multi_similarity'] = ms_pos + ms_neg
      losses['sv_aerial_cc_multi_similarity_pos'] = ms_pos
      losses['sv_aerial_cc_multi_similarity_neg'] = ms_neg

      metrics['sv_aerial_cc_valid_negatives'] = ms_loss.num_negatives
      metrics['sv_aerial_cc_valid_positives'] = ms_loss.num_positives
    return losses, metrics

  def loss_metrics_function(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      model_params: jnp.ndarray | None = None,
      model_state: Any | None = None,
      learning_rates: Mapping[str, float] | None = None,
  ) -> types.LossMetricsTuple:
    losses, metrics = self._loss_metrics_function(
        pred, batch, model_params, model_state, learning_rates
    )
    losses['total'] = types.compose_loss(losses, self.config.compose_loss)
    return losses, metrics

  def get_cell_codes(
      self,
      train_state: train_utils.TrainState,  # replicated
  ) -> FloatArray['N D']:
    return jax.pmap(
        self.inference_fn('get_cell_codes'),
        axis_name='batch',
        static_broadcasted_argnums=(1,),
    )(train_state, self.config.cell_code_dim)

  def build_s2cell_database(
      self,
      train_state: train_utils.TrainState,  # replicated
  ) -> dict[str, Any]:
    s2cell_database = {
        'tokens': misc.gather_globally_safe(self.get_cell_codes(train_state)),
        'ecef_positions': self.dataset_meta_data['s2cell_centers_ecef'],
        'll_positions': self.dataset_meta_data['s2cell_centers_ll'],
        's2cell': np.array(
            [x.encode('utf-8') for x in self.dataset_meta_data['s2cell_tokens']]
        ),
        'dataset_index': np.arange(
            len(self.dataset_meta_data['s2cell_tokens'])
        ),
    }
    return s2cell_database

  def get_target_shapes(self, is_replicated: bool) -> Sequence[tuple[int, ...]]:
    if is_replicated:
      num_cells = self.s2cell_centers_ll.shape[0] // jax.device_count()
    else:
      num_cells = self.s2cell_centers_ll.shape[0]
    target_dims = [(num_cells, self.config.cell_code_dim)]
    if self.config.get('implicit_cell_code_dim', 0) > 0:
      target_dims.append((num_cells, self.config.implicit_cell_code_dim))
    if is_replicated:
      target_dims = [(jax.local_device_count(), *x) for x in target_dims]
    return target_dims

  def unreplicate_state(
      self, train_state: train_utils.TrainState
  ) -> train_utils.TrainState:
    # pytype: disable=attribute-error
    if self.config.get('shard_codes', False):
      train_state = misc.unshard_or_unreplicate_train_state(
          train_state,
          misc.unshard_or_unreplicate_from_shape_fn(
              self.get_target_shapes(is_replicated=True),
              half_precision=self.config.get('store_half_precision', False),
          ),
      )
      return train_state
    else:
      return super().unreplicate_state(train_state)
    # pytype: enable=attribute-error

  def replicate_state(
      self, train_state: train_utils.TrainState
  ) -> train_utils.TrainState:
    # pytype: disable=attribute-error
    if self.config.get('shard_codes', False):
      # We check for the cell codes by shape here. This is a bit hacky.
      train_state = misc.shard_or_replicate_train_state(
          train_state,
          misc.shard_or_replicate_from_shape_fn(
              self.get_target_shapes(is_replicated=False),
          ),
      )
      return train_state
    else:
      return super().replicate_state(train_state)
    # pytype: enable=attribute-error

  def aggregate_grads(self, grad: types.Grad) -> types.Grad:
    if self.config.get('shard_codes', False) and 'cell_codes' in grad:
      target_shapes = [grad['cell_codes'].shape]
      if self.config.get('implicit_cell_code_dim', 0) > 0:
        target_shapes.append(grad['cell_codes_implicit'].shape)
      return jax.tree.map(
          misc.pmean_or_divide_from_shape_fn(target_shapes), grad
      )
    else:
      return jax.lax.pmean(grad, axis_name='batch')

  def mine_examples(
      self,
      train_state: train_utils.TrainState,  # replicated
      data_source: grain.RandomAccessDataSource,
      batch_size: int = 2,
      batch_size_topk: int = 512,
      min_distance_meters: float = 200.0,
      similarity_fn: str = 'cosine_similarity',
      return_metadata: bool = False,
      **kwargs,
  ) -> types.DataDict:
    assert isinstance(data_source, sv_aerial_data_source.S2CellDataSource)
    cell_centers_ll = self.s2cell_centers_ll
    index_map = np.array([
        data_source.s2cell_idx.get(s2_id.item(), -1)
        for s2_id in self.dataset_meta_data['s2cell_ids']
    ])
    logging.info('Number of valid cells: %s', (index_map > -1).mean().item())
    cell_codes = self.get_cell_codes(train_state)
    if self.config.get('shard_codes', False):
      # Cell codes are already sharded, nothing to do here.
      cell_codes_sharded = cell_codes
    else:
      if cell_codes.shape[1] % jax.device_count() != 0:
        to_shard_indices = jnp.arange(
            cell_codes.shape[1] // jax.device_count() * jax.device_count()
        )
        cell_centers_ll = cell_centers_ll[to_shard_indices]
        index_map = index_map[to_shard_indices]
      else:
        to_shard_indices = jnp.arange(cell_codes.shape[1])
      # Gather and distribute the replicated cell codes.
      cell_codes_sharded = misc.gather_from_replicated(
          cell_codes,
          misc.distribute_globally(to_shard_indices),
      )
    logging.info('Sharded cell codes: %s', cell_codes_sharded.shape)
    similarity, neighbors = misc.pmap_self_topk(
        cell_codes_sharded,
        k=100,
        batch_size=batch_size_topk,
        show_progress=True,
        similarity_fn=similarity_fn,
        # Consider also masking by distance here, but this is expensive.
        where=misc.distribute_globally(index_map > -1),
        **kwargs,
    )
    logging.info('Similarity: %s', similarity.shape)
    logging.info('Neighbors: %s', neighbors.shape)

    similarity, neighbors, haversine_distances = jax.pmap(
        functools.partial(
            filter_and_remap_indices, min_distance_meters=min_distance_meters
        ),
        'batch',
        donate_argnums=(0, 1),
    )(
        similarity,
        neighbors,
        jax_utils.replicate(index_map),
        misc.distribute_globally(cell_centers_ll),
    )

    logging.info('Similarity: %s', similarity.shape)
    logging.info('Neighbors: %s', neighbors.shape)
    logging.info('Haversine distances: %s', haversine_distances.shape)

    index_map_p = misc.get_process_slice(index_map)

    mining_data = {
        'mining_indices': np.asarray(
            misc.gather_locally(neighbors)[index_map_p > -1, :batch_size]
        ),
    }
    logging.info(
        'Final mining indices: %s, %f valid.',
        mining_data['mining_indices'].shape,
        (mining_data['mining_indices'] > -1).mean(),
    )
    if return_metadata:
      mining_data['mining_haversine_distances'] = np.asarray(
          misc.gather_locally(haversine_distances)[
              index_map_p > -1, :batch_size
          ]
      )
      index_map_inverse = np.array([
          self.dataset_meta_data['s2cell_id_to_idx'].get(s2_id, -1)
          for s2_id in data_source.s2cells
      ])
      mining_data['mining_centers_ll'] = np.array(
          cell_centers_ll[index_map_inverse]
      )
      mining_data['mining_similarity'] = np.asarray(
          misc.gather_locally(similarity)[index_map_p > -1, :batch_size]
      )
    return mining_data

  def update_state(
      self,
      pred: types.Predictions,
      model_params: types.ModelParams,
      model_state: types.ModelState,
      learning_rates: types.LearningRates,
  ) -> tuple[types.ModelParams, types.ModelState]:
    if self.config.get('cell_codes_momentum'):
      momentum = learning_rates.get(
          'cell_codes_momentum', self.config.cell_codes_momentum
      )
      all_cell_idx = jax.lax.all_gather(
          pred['gt_cell_idx'], axis_name='batch', tiled=True
      )
      ema_over = self.config.get('cell_codes_ema_over', 'sv')
      if ema_over == 'sv':
        logits = pred['logits']
      elif ema_over == 'aerial':
        logits = pred['aerial_logits_p']
      elif ema_over == 'both':
        logits = pred['logits'] + pred['aerial_logits_p']
      else:
        raise ValueError(f'Unknown cell_codes_ema_over: {ema_over}')
      all_logits = jax.lax.all_gather(logits, axis_name='batch', tiled=True)

      for k, v in model_state['model_stats'].items():
        if k.startswith('ema_cell_codes'):
          num_cells, _ = v.shape
          if self.config.get('shard_codes', False):
            shard_cell_idx = (
                all_cell_idx - jax.lax.axis_index('batch') * num_cells
            )
            shard_cell_idx = jnp.where(
                shard_cell_idx < 0,
                num_cells,
                shard_cell_idx,
            )
          else:
            shard_cell_idx = all_cell_idx

          updates = jnp.zeros_like(v)
          counts = jnp.zeros_like(v[:, 0], dtype=jnp.int32)
          updates = updates.at[shard_cell_idx].add(all_logits, mode='drop')
          counts = counts.at[shard_cell_idx].add(1, mode='drop')
          updates = updates / counts[..., None].clip(min=1.0)
          model_state['model_stats'][k] = jnp.where(
              counts[..., None] > 0,
              (momentum * v + (1 - momentum) * updates),
              v,
          )
    return model_params, model_state

  def sync_model_state_across_replicas(
      self,
      train_state: train_utils.TrainState,
  ) -> types.ModelState:
    """Sync the model_state across replicas."""
    if (
        self.config.get('shard_codes', False)
        and 'model_stats' in train_state.model_state
    ):
      return misc.map_dict(
          lambda k, v: (v if 'cell_codes' in k else train_utils.pmap_mean(v)),
          train_state.model_state,
          sep='.',
      )
    else:
      return super().sync_model_state_across_replicas(train_state)
