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

from etils.array_types import FloatArray
import flax.linen as nn
import jax
import numpy as np
import jax.numpy as jnp
import jaxtyping as jt
import ml_collections

from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.models import base
from scaling_geoloc.scenic.models import vit
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
  batch = misc.filter_batch_for_jit(batch)
  batch_views = jax.tree.map(
      lambda x: x[:, 0],
      batch['query']['views'],
  )

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
            self.config.get('crop', 'none'),
            self.config.get('crop_size', (224, 224)),
        )
    )

    temperature = self.config.get('temperature', 0.1)
    self.temperature = jnp.array(temperature)

    self.sv_temperature = jnp.array(
        self.config.get('cc_temperature', self.config.get('temperature', 0.1))
    )
    self.aerial_temperature = jnp.array(
        self.config.get('cc_temperature', self.config.get('temperature', 0.1))
    )

    self.cell_code_dim = self.config.get(
        'cell_code_dim', self.backbone.output_dim()
    )

  @nn.compact
  def get_cell_codes(self, cell_code_dim: int) -> FloatArray['N D']:
    num_cells = self.num_cells
    if self.config.get('shard_codes', False) and not self.is_initializing():
      num_cells = num_cells // jax.device_count()
    init_fn = nn.initializers.normal(stddev=1 / np.sqrt(cell_code_dim))
    cell_codes = self.param(
        'cell_codes',
        init_fn,
        (num_cells, cell_code_dim),
        self.dtype,
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
      # @TODO: Probably only need this one.
      return distributed.sharded_similarity(logits, cell_codes, normalize=normalize)
    else:
      return misc.compute_similarity(
          logits,
          cell_codes,
          normalize=normalize,
      )[..., 0]

  def encode_multiview(
      self,
      batch: types.Batch,
      *args,
      **kwargs,
  ) -> tuple[jt.Float[jt.Array, '... D'], jt.Float[jt.Array, '... L D']]:
    """Interface to backbone encoder."""
    kwargs['normalize'] = self.config.get('l2_normalize', False)
    kwargs['normalize_per_head'] = self.config.get(
        'l2_normalize_per_head', False
    )
    logits, tokens = self.backbone.encode_multiview(
        batch['image_pixels'], *args, **kwargs
    )
    return logits, tokens

  @nn.compact
  def encode_aerial(
      self,
      aerial_batch: types.Batch,
      *args,
      **kwargs,
  ) -> tuple[jt.Float[jt.Array, '... N D'], jt.Float[jt.Array, '... N L D']]:
    """Interface to aerial backbone encoder."""
    kwargs['normalize'] = self.config.get('l2_normalize', False)
    kwargs['normalize_per_head'] = self.config.get(
        'l2_normalize_per_head', False
    )
    aerial_pixels = aerial_batch['image_pixels']
    logits, tokens = self.aerial_backbone.encode_multiview(
        aerial_pixels, *args, **kwargs
    )
    return logits, tokens

  def get_image_pixels(
      self, batch: types.Batch, train: bool
  ) -> tuple[FloatArray['... H W C'], FloatArray['... N G G C']]:
    """Preprocess the image pixels (crop, augment)."""
    batch = flatten_batch(batch, stride=None)
    image_pixels = batch['image_pixels']
    aerial_pixels = batch['aerial']['rgb']

    # Crop to multiple of patch size to avoid boundary effects.
    image_pixels = misc.crop_to_multiple(
        image_pixels,
        self.config.vit.get('patches', {}).get('size', [16])[0],
    )
    aerial_pixels = misc.crop_to_multiple(
        aerial_pixels,
        self.config.vit_aerial.get('patches', {}).get('size', [16])[0],
    )

    if train and self.config.get('do_augment', False):
      # Augment images
      rng = self.make_rng('augment')
      do_padding_augment = self.config.get('do_padding_augment', False)
      do_stretching_augment = self.config.get('do_stretching_augment', False)
      image_pixels = get_augment_fn(do_padding_augment, do_stretching_augment)(
          image_pixels, rng
      )

      rng_a = self.make_rng('augment_aerial')
      aerial_pixels = get_augment_fn(False)(aerial_pixels, rng_a)

    if train and self.config.get('crop', 'none') != 'none':
      image_pixels = self.crop_fn(image_pixels, self.make_rng('crop'))
    return image_pixels, aerial_pixels

  def stochastic_probs(
      self,
      logits: FloatArray['B D'],
      source_latlngs: FloatArray['B 2'],
      target_logits: FloatArray['B D'],
      target_latlngs: FloatArray['B 2'],
      batch: types.Batch,
      axis_name: str = 'batch',
      target_name: str = 'aerial',
  ) -> types.Predictions:
    """Compute similarity between SV and aerial logits within batch (training)."""
    batch_size = logits.shape[0]
    if not self.is_initializing():
      target_logits_in_batch = jax.lax.all_gather(
          target_logits, axis_name=axis_name, tiled=True
      )  # NB x D
    else:
      target_logits_in_batch = target_logits

    cell_idx = jnp.asarray(batch['cell_idx'])
    cell_centers_ll = jnp.asarray(target_latlngs)
    if self.is_initializing():
      all_cell_idx_in_batch = cell_idx
      all_cell_centers_ll_in_batch = cell_centers_ll
    else:
      all_cell_idx_in_batch = jax.lax.all_gather(
          cell_idx, axis_name=axis_name, tiled=True
      )  # B x N
      all_cell_centers_ll_in_batch = jax.lax.all_gather(
          cell_centers_ll, axis_name=axis_name, tiled=True
      )  # B x N x 2

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

    if not self.is_initializing():
      gt_index = jax.lax.axis_index(axis_name) * batch_size + gt_index

    # B x NB
    # same_cell = cell_idx[:, None] == all_cell_idx[None]
    distance_meters = misc.haversine_distance(
        source_latlngs[:, None], all_cell_centers_ll[None]
    )[..., 0]
    same_cell = distance_meters < self.config.get('same_cell_threshold', 200.0)
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
  ) -> types.Predictions:
    image_pixels, aerial_pixels = self.get_image_pixels(batch, train)
    batch = flatten_batch(batch, stride=None)
    logits, _ = self.encode_multiview(
        {'image_pixels': image_pixels},
        train=train,
        sow_attention=self.config.get('sow_attention', False) and not train,
    )

    aerial_batch = {
        'image_pixels': aerial_pixels[..., :3],
    }
    if 'cell_idx' in batch:
      aerial_batch['cell_idxs'] = batch['cell_idx'][:, None]
    aerial_logits, _ = self.encode_aerial(
        aerial_batch,
        train=train,
        sow_attention=self.config.get('sow_attention', False) and not train,
    )

    assert aerial_logits.ndim == 3  # B x N x D

    aerial_logits_p = {
        'max': lambda x: jnp.max(x, axis=-2),
        'mean': lambda x: jnp.mean(x, axis=-2),
    }[self.config.aerial_pooling](
        aerial_logits
    )  # B x D

    pred = {
        'logits': logits,
        'aerial_logits': aerial_logits,
        'aerial_logits_p': aerial_logits_p,
        'augmented_pixels': image_pixels,
        'augmented_aerial_pixels': aerial_pixels[..., :3],
    }

    # Stochastic probability from SV to all aerial in batch.
    pred['sv_aerial_stochastic_probs'] = self.stochastic_probs(
        logits,
        batch['latlng'],
        aerial_logits_p,
        batch['aerial']['center_ll'],
        batch,
    )
    # Inverse probability from aerial to all SV in batch.
    aerial_sv_probs = self.stochastic_probs(
        aerial_logits_p,
        batch['aerial']['center_ll'],
        logits,
        batch['latlng'],
        batch,
        target_name='aerialsv',
    )
    pred['aerial_sv_stochastic_probs'] = aerial_sv_probs
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
    super().__init__(config, dataset_meta_data, dtype)
    config = self.config.copy_and_resolve_references().unlock()
    aerial_patch = self.config.vit_aerial
    config.vit_aerial = config.vit.copy_and_resolve_references()
    config.vit_aerial.update(aerial_patch)
    self.config = config
    self.flax_model = self.build_flax_model()

  def build_flax_model(self) -> nn.Module:
    return GeoLocalizer(
        self.config, self.dtype, cell_centers=jnp.array([[]])
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
  ) -> types.LossMetricsTuple:
    sv_aerial_losses, sv_aerial_metrics = self.stochastic_loss_metrics_function(
        pred['sv_aerial_stochastic_probs'], batch
    )
    aerial_sv_losses, aerial_sv_metrics = self.stochastic_loss_metrics_function(
        pred['aerial_sv_stochastic_probs'], batch, prefix='aerial_sv_'
    )
    losses = {**sv_aerial_losses, **aerial_sv_losses}
    metrics = {
        **sv_aerial_metrics,
        **aerial_sv_metrics,
    }
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
