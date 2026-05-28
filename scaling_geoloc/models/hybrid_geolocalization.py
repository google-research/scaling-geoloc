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

"""Hybrid geolocalization model."""

from typing import Any, Mapping

from etils.array_types import FloatArray, IntArray
import flax.linen as nn
import jax
import numpy as np
import jax.numpy as jnp
import jaxtyping as jt
import ml_collections

from scaling_geoloc import types
from scaling_geoloc.configs import defaults
from scaling_geoloc.models import base
from scaling_geoloc.models import vit
from scaling_geoloc.utils import augmentations
from scaling_geoloc.utils import misc, distributed


def get_augment_fn(
    do_padding_augment: bool, do_stretching_augment: bool = False
) -> augmentations.AugmentationFn:
  """Returns an augmentation function."""

  list_augmentations = [
      augmentations.random_color_jitter(0.8, 0.1, 0.4, 0.2, 0.1),
      augmentations.random_grayscale(0.2),
      augmentations.random_gaussian_blur(1.0, 0.1, 2.0),
      augmentations.random_solarize(0.1, 0.5),
  ]

  if do_stretching_augment:
    list_augmentations.append(augmentations.random_stretch(0.5, 0.5))

  if do_padding_augment:
    list_augmentations.append(augmentations.random_padding(0.5, 0.5))

  return augmentations.per_image(augmentations.sequential(list_augmentations))


def flatten_batch(batch: types.Batch, stride: int | None = None) -> types.Batch:
  """Simplifies a batch by removing unnecessary dimensions."""
  batch_views = dict(batch['query'])
  if stride is not None:
    batch_views = {
        k: (
            misc.crop_to_multiple(v, stride)
            if (isinstance(v, jnp.ndarray) and v.ndim == 4)
            else v
        )
        for k, v in batch_views.items()
    }
  return {**batch, **batch_views}


class GeoLocalizer(nn.Module):
  """SV-aerial retrieval model."""

  config: ml_collections.ConfigDict
  dtype: jnp.dtype
  cell_centers: FloatArray['N 2'] | None = None

  def setup(self):
    self.class_backbone = vit.ViTBackbone
    self.backbone = self.class_backbone.from_config(self.config.vit, self.dtype)
    self.aerial_backbone = self.class_backbone.from_config(
        self.config.vit_aerial, self.dtype
    )
    self.crop_fn = augmentations.per_image(
        augmentations.get_crop_fn(
            self.config.crop,
            self.config.crop_size,
        )
    )

    temperature = self.config.temperature
    self.temperature = jnp.array(temperature)

    self.query_temperature = self.temperature
    self.aerial_temperature = self.temperature

    self.prototype_dim = self.config.prototype_dim or self.backbone.output_dim()
    self.num_cells = self.cell_centers.shape[0]

  @nn.compact
  def get_prototypes(self, prototype_dim: int) -> FloatArray['N D']:
    num_cells = self.num_cells
    if self.config.shard_codes and not self.is_initializing():
      num_cells = num_cells // jax.device_count()
    init_fn = nn.initializers.normal(stddev=1 / np.sqrt(prototype_dim))
    prototypes = self.param(
        'prototypes',
        init_fn,
        (num_cells, prototype_dim),
        self.dtype,
    )
    if self.config.l2_normalize:
      prototypes = misc.l2_normalize(prototypes)
    return prototypes

  def compute_similarity_to_prototypes(
      self,
      logits: FloatArray['B ... D'],
      prototypes: FloatArray['... N|N//ND D'],
      normalize: bool = False,
  ) -> FloatArray['B N']:
    """Computes similarity to cell codes."""
    logits = jnp.expand_dims(logits, axis=-2)
    prototypes = prototypes[None]
    if self.config.shard_codes and not self.is_initializing():
      # @TODO: Probably only need this one.
      return distributed.sharded_similarity(logits, prototypes, normalize=normalize)
    else:
      return misc.compute_similarity(
          logits,
          prototypes,
          normalize=normalize,
      )[..., 0]

  def encode_query(
      self,
      batch: types.Batch,
      *args,
      **kwargs,
  ) -> tuple[jt.Float[jt.Array, '... D'], jt.Float[jt.Array, '... L D']]:
    """Interface to backbone encoder."""
    kwargs['normalize'] = self.config.l2_normalize
    kwargs['normalize_per_head'] = self.config.l2_normalize_per_head
    logits, tokens = self.backbone.encode_multiview(
        batch['image'], *args, **kwargs
    )
    return logits, tokens

  def topk_similarities(
      self,
      logits: FloatArray['B D'],
      prototypes: FloatArray['N|N//ND D'],
      k: int,
  ) -> tuple[FloatArray['B N'], IntArray['B N']]:
    """Computes similarity to cell codes."""
    if self.config.shard_codes and not self.is_initializing():
      neg_sim, neg_idx = distributed.distributed_topk(
          logits,
          prototypes,
          k=k,
          ascending=False,
          recall_target=self.config.topk_recall_target,
          final_topk=False,
      )
      return neg_sim, neg_idx
    else:
      logits = jnp.expand_dims(logits, axis=-2)
      prototypes = prototypes[None]
      return misc.compute_similarity(
          logits,
          prototypes,
          normalize=False,
      )[
          ..., 0
      ], jnp.arange(self.num_cells)[None].repeat(logits.shape[0], axis=0)

  @nn.compact
  def encode_aerial(
      self,
      aerial_batch: types.Batch,
      *args,
      **kwargs,
  ) -> tuple[jt.Float[jt.Array, '... N D'], jt.Float[jt.Array, '... N L D']]:
    """Interface to aerial backbone encoder."""
    kwargs['normalize'] = self.config.l2_normalize
    kwargs['normalize_per_head'] = self.config.l2_normalize_per_head
    aerial_pixels = aerial_batch['image']
    logits, tokens = self.aerial_backbone.encode_multiview(
        aerial_pixels, *args, **kwargs
    )
    return logits, tokens

  def get_image_pixels(
      self, batch: types.Batch, train: bool
  ) -> tuple[FloatArray['... H W C'], FloatArray['... N G G C']]:
    """Preprocess the image pixels (crop, augment)."""
    batch = flatten_batch(batch, stride=None)
    image_pixels = batch['image']
    aerial_pixels = batch['aerial']['image']

    # Crop to multiple of patch size to avoid boundary effects.
    image_pixels = misc.crop_to_multiple(
        image_pixels,
        self.config.vit.get('patches', {}).get('size', [16])[0],
    )
    aerial_pixels = misc.crop_to_multiple(
        aerial_pixels,
        self.config.vit_aerial.get('patches', {}).get('size', [16])[0],
    )

    if train and self.config.do_augment:
      # Augment images
      rng = self.make_rng('augment')
      do_padding_augment = self.config.do_padding_augment
      do_stretching_augment = self.config.do_stretching_augment
      image_pixels = get_augment_fn(do_padding_augment, do_stretching_augment)(
          image_pixels, rng
      )

      rng_a = self.make_rng('augment_aerial')
      aerial_pixels = get_augment_fn(False)(aerial_pixels, rng_a)

    if train and self.config.crop != 'none':
      image_pixels = self.crop_fn(image_pixels, self.make_rng('crop'))
    return image_pixels, aerial_pixels

  def stochastic_probs(
      self,
      logits: FloatArray['B D'],
      source_latlngs: FloatArray['B 2'],
      target_logits: FloatArray['B D'],
      target_latlngs: FloatArray['B 2'],
      batch: types.Batch,
      axis_name: str | None = 'batch',
      target_name: str = 'aerial',
  ) -> types.Predictions:
    """Compute similarity between SV and aerial logits within batch (training)."""
    batch_size = logits.shape[0]
    distributed = axis_name is not None and not self.is_initializing()
    if distributed:
      target_logits_in_batch = jax.lax.all_gather(
          target_logits, axis_name=axis_name, tiled=True
      )  # NB x D
    else:
      target_logits_in_batch = target_logits

    cell_idx = jnp.asarray(batch['cell_idx'])
    cell_centers_ll = jnp.asarray(target_latlngs)
    if distributed:
      all_cell_idx_in_batch = jax.lax.all_gather(
          cell_idx, axis_name=axis_name, tiled=True
      )  # B x N
      all_cell_centers_ll_in_batch = jax.lax.all_gather(
          cell_centers_ll, axis_name=axis_name, tiled=True
      )  # B x N x 2
    else:
      all_cell_idx_in_batch = cell_idx
      all_cell_centers_ll_in_batch = cell_centers_ll

    # Handle memory:
    to_memorize = {
        'all_cell_idx': all_cell_idx_in_batch,
        'all_cell_centers_ll': all_cell_centers_ll_in_batch,
        f'all_{target_name}_logits': target_logits_in_batch,
    }
    updated_variables = to_memorize
    all_cell_idx = updated_variables['all_cell_idx']
    all_cell_centers_ll = updated_variables['all_cell_centers_ll']
    aerial_logits = updated_variables[f'all_{target_name}_logits']

    gt_index = jnp.arange(batch_size)  # B

    if distributed:
      gt_index = jax.lax.axis_index(axis_name) * batch_size + gt_index

    # B x NB
    # same_cell = cell_idx[:, None] == all_cell_idx[None]
    distance_meters = misc.haversine_distance(
        source_latlngs[:, None], all_cell_centers_ll[None]
    )[..., 0]
    same_cell = distance_meters < self.config.same_cell_threshold
    duplicates = same_cell.at[jnp.arange(batch_size), gt_index].set(False)

    # Sim / probs: B x N*B
    similarity = misc.compute_similarity(
        logits[:, None],
        aerial_logits[None],
        normalize=False,
    )[
        ..., 0
    ]  # B x N*B

    # Currently use 1.
    return {
        'similarity': similarity,
        'duplicates': duplicates,
        'same_cell': same_cell,
        'gt_index': gt_index,
        'all_cell_idx': all_cell_idx,
        f'all_{target_name}_logits': target_logits,
        f'all_{target_name}_in_batch': target_logits_in_batch,
        'gt_similarity': jnp.take_along_axis(
            similarity, gt_index[:, None], axis=-1
        )[..., 0],
        'temperature': self.temperature[None].repeat(batch_size, axis=0),
    }

  @nn.compact
  def __call__(
      self,
      batch: types.Batch,
      *,
      train: bool,
      debug: bool = False,
      axis_name: str | None = 'batch',
  ) -> types.Predictions:
    image_pixels, aerial_pixels = self.get_image_pixels(batch, train)
    batch = flatten_batch(batch, stride=None)
    logits, _ = self.encode_query(
        {'image': image_pixels},
        train=train,
        sow_attention=self.config.sow_attention and not train,
    )

    aerial_batch = {
        'image': aerial_pixels[..., :3],
    }
    if 'cell_idx' in batch:
      aerial_batch['cell_idxs'] = batch['cell_idx'][:, None]
    aerial_logits, _ = self.encode_aerial(
        aerial_batch,
        train=train,
        sow_attention=self.config.sow_attention and not train,
    )

    pred = {
        'logits': logits,
        'aerial_logits': aerial_logits,
        'aerial_logits': aerial_logits,
        'augmented_pixels': image_pixels,
        'augmented_aerial_pixels': aerial_pixels[..., :3],
    }

    # Stochastic probability from SV to all aerial in batch.
    pred['query_aerial_stochastic_probs'] = self.stochastic_probs(
        logits,
        batch['latlng'],
        aerial_logits,
        batch['aerial']['latlng'],
        batch,
        axis_name=axis_name,
    )
    # Inverse probability from aerial to all SV in batch.
    aerial_sv_probs = self.stochastic_probs(
        aerial_logits,
        batch['aerial']['latlng'],
        logits,
        batch['latlng'],
        batch,
        axis_name=axis_name,
        target_name='aerialquery',
    )
    pred['aerial_query_stochastic_probs'] = aerial_sv_probs

    # Compute SV-CC similarity.
    pos_idxs = {
        'query': batch['adjacent_cell_idx'],
        'aerial': batch['aerial']['adjacent_cell_idx'],
    }
    all_logits = {
        'query': pred['logits'],
        'aerial': pred['aerial_logits'],
    }
    for name, logits in all_logits.items():
      prototypes = self.get_prototypes(self.prototype_dim)
      if getattr(self.config, f'detach_{name}_prototypes'):
        prototypes = jax.lax.stop_gradient(prototypes)
      pos_sim = None
      if self.config.masked_expsum and not self.is_initializing():
        assert self.config.shard_codes
        shard_max = distributed.max_and_masked_expsum(
            logits,
            batch['latlng'],
            prototypes,
            self.cell_centers,
            normalize=False,
            beta=self.config.multi_similarity_beta,
            base=self.config.multi_similarity_base,
            required_idx=pos_idxs[name],
        )
        masked_expsum = shard_max['expsum']
        best_cell_idx = shard_max['best_idx']
        if 'pos_sim' in shard_max:
          pos_sim = shard_max['pos_sim']
      else:
        if self.config.topk and self.config.topk < prototypes.shape[0]:
          all_sim, all_idx = self.topk_similarities(
              logits, prototypes, k=self.config.topk
          )
        else:
          all_sim = self.compute_similarity_to_prototypes(logits, prototypes)
          all_idx = jnp.arange(self.num_cells)[None].repeat(
              logits.shape[0], axis=0
          )
          pos_sim = jnp.take_along_axis(all_sim, pos_idxs[name], axis=-1)
        pred[f'{name}_sim'] = all_sim

        valid_negatives = jax.vmap(lambda x, y: ~jnp.isin(x, y) & (x > -1))(
            all_idx,
            pos_idxs[name],
        ) & (all_idx > -1)

        if self.config.filter_by_distance:
          cell_centers = jnp.take_along_axis(
              self.cell_centers[None],
              all_idx[..., None],
              axis=-2,
          )
          query_p_dist = misc.haversine_distance(
              cell_centers,
              batch['latlng'][:, None],
          )[..., 0]
          query_p_mask = query_p_dist > self.config.same_cell_threshold
          valid_negatives = query_p_mask
        neg_sim = jnp.where(valid_negatives, all_sim, -jnp.inf)
        masked_expsum = (
            valid_negatives
            * jnp.exp(
                (neg_sim - self.config.multi_similarity_base)
                * self.config.multi_similarity_beta
            )
        ).sum(-1)

        best_local_idx = jnp.argmax(all_sim, axis=-1, keepdims=True)
        best_cell_idx = jnp.take_along_axis(all_idx, best_local_idx, axis=-1)[
            ..., 0
        ]

      if pos_sim is None:
        if prototypes.shape[-1] < logits.shape[-1]:
          logits = logits[..., : prototypes.shape[-1]]
          if self.config.l2_normalize:
            logits = misc.l2_normalize(logits)
        if self.config.shard_codes and not self.is_initializing():
          pos_p = distributed.sharded_batched_gather(prototypes, pos_idxs[name])
        else:
          pos_p = jnp.take_along_axis(
              prototypes[None],
              pos_idxs[name][..., None],
              axis=-2,
          )

        pos_sim = misc.compute_similarity(
            logits[..., None, :],
            pos_p,
            normalize=False,
        )[..., 0]

      pred[f'{name}_masked_expsum'] = masked_expsum

      temperature = getattr(self, f'{name}_temperature')

      pred[f'{name}_cell_idx'] = best_cell_idx
      pred[f'{name}_best_cell_idx'] = best_cell_idx
      pred[f'{name}_temperature'] = temperature[None].repeat(
          logits.shape[0], axis=0
      )

      pred[f'{name}_pos_sim'] = pos_sim
      pred[f'{name}_pos_idxs'] = pos_idxs[name]

    if 'cell_idx' in batch:
      pred['gt_cell_idx'] = batch['cell_idx']
    return pred


class GeoLocalizerModel(base.BaseModel):
  """Trainer-facing wrapper for the hybrid geolocalization model."""

  required_rngs: tuple[str, ...] = (
      'masking_rng',
      'dropout',
      'augment',
      'augment_aerial',
      'crop',
      'rope',
  )

  def __init__(
      self,
      config: ml_collections.ConfigDict,
      dataset_meta_data: dict[str, Any],
      dtype: jnp.dtype[Any] = jnp.float32,
  ):
    self.s2cell_centers_ll = jnp.asarray(dataset_meta_data['cell_centers_ll'])
    super().__init__(config, dataset_meta_data, dtype)
    base_cfg = defaults.geo_localizer()
    base_cfg.update(self.config)
    config = base_cfg.copy_and_resolve_references().unlock()
    aerial_patch = config.vit_aerial
    config.vit_aerial = config.vit.copy_and_resolve_references()
    config.vit_aerial.update(aerial_patch)
    self.config = config
    self.flax_model = self.build_flax_model()

  def build_flax_model(self) -> nn.Module:
    return GeoLocalizer(
        self.config, self.dtype,
        cell_centers=self.s2cell_centers_ll,
    )

  def init_state(
      self, model_params: types.ModelParams, model_state: types.ModelState
  ) -> tuple[types.ModelParams, types.ModelState]:
    class_backbone = vit.ViTBackbone
    model_params['backbone'] = class_backbone.maybe_load_params(
        model_params['backbone'],
        self.config.vit.get('pretrained_weights', None),
        self.config.vit.get('pretrained_checkpoint', None),
    )
    model_params['aerial_backbone'] = class_backbone.maybe_load_params(
        model_params['aerial_backbone'],
        self.config.vit_aerial.get('pretrained_weights', None),
        self.config.vit_aerial.get('pretrained_checkpoint', None),
    )
    return model_params, model_state

  def _loss_metrics_function(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      model_params: jnp.ndarray | None = None,
      model_state: Any | None = None,
      learning_rates: types.LearningRates | None = None,
  ) -> types.LossMetricsTuple:
    losses, metrics = {}, {}
    batch = flatten_batch(batch)
    gt_cell_idx = batch['cell_idx']
    # Log metrics of gt cell.
    for name in ['query', 'aerial']:
      gt_sim_l2 = (
          pred[f'{name}_pos_sim']
          * (pred[f'{name}_pos_idxs'] == gt_cell_idx[:, None])
      ).sum(-1)
      losses[f'{name}_sim_l2'] = gt_sim_l2
      metrics[f'{name}_precision'] = (
          gt_cell_idx == pred[f'{name}_cell_idx']
      ).astype(jnp.float32)

    # Log distance metrics of best cell.
    query_pred_p_dist = misc.haversine_distance(
        self.s2cell_centers_ll[batch['cell_idx']],
        self.s2cell_centers_ll[pred['query_best_cell_idx']],
    )[..., 0]

    metrics = metrics | misc.distance_metrics(query_pred_p_dist)
    if 'query_aerial_p_multi_similarity' in self.config.compose_loss:
      ms_loss = misc.MultiSimilarityLoss(
          alpha=self.config.multi_similarity_alpha,
          beta=self.config.multi_similarity_beta,
          base=self.config.multi_similarity_base,
          margin=self.config.multi_similarity_margin,
      )
      # Add SV-Aerial similarities.
      pos_sims = {}
      for name, stochastic_pred in {
          'query-a': pred['query_aerial_stochastic_probs'],
          'a-query': pred['aerial_query_stochastic_probs'],
      }.items():
        if self.config.multi_similarity_factors != 'all':
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
      for name, prefix in {'query-p': 'query', 'a-p': 'aerial'}.items():
        if self.config.multi_similarity_factors != 'all':
          if name not in self.config.multi_similarity_factors:
            continue
        proxy_pos = getattr(self.config, f'{prefix}_multi_similarity_proxy_positives')
        weights = 1.0
        if proxy_pos == 'cell' or proxy_pos is None:
          gt_idx = batch['cell_idx'][:, None]
        elif proxy_pos == 'frustum':
          gt_idx = batch['frustum_cell_idx']
        elif proxy_pos == 'adjacent':
          if prefix == 'query':
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
            self.config.multi_similarity_mean_pos_sim
            and proxy_pos != 'adjacent'
        ):
          # We actually do this! (SV)
          pos_sim = misc.masked_mean(
              pred[f'{prefix}_pos_sim'], pos_weights * weights, axis=-1
          )
        else:
          pos_sim = (pos_weights * weights * pred[f'{prefix}_pos_sim']).sum(-1)
        metrics[f'{prefix}_area_pos_sim'] = pos_sim
        metrics[f'{prefix}_area_size'] = pos_weights.sum(-1)
        pos_sims[name] = pos_sim
        ms_loss.neg.append(pred[f'{prefix}_masked_expsum'])

      # Add the positive edges to the loss.
      if 'a-p' in pos_sims:
        # A-CC positives
        ms_loss.pos.append(ms_loss.pos_scale(pos_sims['a-p']))
        ms_loss.num_positives += 1
      if 'query-a' in pos_sims:
        ms_loss.pos.append(ms_loss.pos_scale(pos_sims['query-a']))
      if 'query-p' in pos_sims:
        ms_loss.pos.append(ms_loss.pos_scale(pos_sims['query-p']))
      ms_loss.num_positives += 2

      ms_pos, ms_neg = ms_loss.compute()
      losses['query_aerial_p_multi_similarity'] = ms_pos + ms_neg
      losses['query_aerial_p_multi_similarity_pos'] = ms_pos
      losses['query_aerial_p_multi_similarity_neg'] = ms_neg

      metrics['query_aerial_p_valid_negatives'] = ms_loss.num_negatives
      metrics['query_aerial_p_valid_positives'] = ms_loss.num_positives
    return losses, metrics

  def loss_metrics_function(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      model_params: jnp.ndarray | None = None,
      model_state: Any | None = None,
      learning_rates: Mapping[str, float] | None = None,
  ) -> types.LossMetricsTuple:
    losses, metrics = self._loss_metrics_function(pred, batch)
    losses['total'] = types.compose_loss(losses, self.config.compose_loss)
    return losses, metrics
