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

"""Cell code retrieval model."""

from collections.abc import Sequence
from typing import Any, Mapping, Type, Union

from etils.array_types import FloatArray
from etils.array_types import IntArray
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxtyping as jt
import ml_collections
import numpy as np

from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.configs import types as config_types
from scaling_geoloc.scenic.models import base
from scaling_geoloc.scenic.models import convnext
from scaling_geoloc.scenic.models import intrinsics_encoding
from scaling_geoloc.scenic.models import vit
from scaling_geoloc.utils import augmentations
from scaling_geoloc.utils import misc
from scaling_geoloc.utils import plot


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


def get_backbone_class(
    backbone_type: str,
) -> Union[Type[vit.ViTBackbone], Type[convnext.ConvNeXtBackbone]]:
  if backbone_type == config_types.BackboneTypes.VIT:
    return vit.ViTBackbone
  elif backbone_type == config_types.BackboneTypes.CONVNET:
    return convnext.ConvNeXtBackbone
  else:
    raise ValueError(f'Unknown backbone type: {backbone_type}')


# https://github.com/fferflo/statewide-visual-geolocalization/blob/main/geoloc/loss.py
def decoupled_log_softmax(
    x: FloatArray,
    axis: int = -1,
    where: Any | None = None,
) -> FloatArray:
  """Decoupled log softmax similar to Fervers et al."""
  if where is None:
    where = jnp.ones_like(x, dtype=jnp.bool_)
  x = jnp.where(where, x, 0.0)

  # Nominator
  pos_x = x

  neg_x_max = jax.lax.stop_gradient(
      jnp.max(x, axis, where=where, initial=-jnp.inf, keepdims=True)
  )
  neg_x = (
      jnp.log(jnp.sum(jnp.exp(x - neg_x_max), axis, where=where, keepdims=True))
      + neg_x_max
  )

  # Denominator without positive term
  x_max = jax.lax.stop_gradient(jnp.maximum(pos_x, neg_x))
  neg_x = jnp.exp(neg_x - x_max) - jnp.exp(pos_x - x_max)
  neg_x = jnp.maximum(neg_x, 1e-40)
  neg_x = jnp.log(neg_x) + x_max

  return jnp.where(where, pos_x - neg_x, 0.0)


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


class SVAerialRetrieval(nn.Module):
  """SV-aerial retrieval model."""

  config: ml_collections.ConfigDict
  dtype: jnp.dtype
  cell_tree: Sequence[IntArray['N K']] | None = None
  cell_centers: FloatArray['N 2'] | None = None

  def setup(self):
    self.class_backbone = get_backbone_class(
        self.config.vit.get('backbone_type', config_types.BackboneTypes.VIT)
    )
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
    if self.config.get('learned_temperature', False):
      self.temperature = self.param(
          'temperature',
          nn.initializers.constant(temperature),
          (),
          self.dtype,
      )
    else:
      self.temperature = jnp.array(temperature)

  def encode_multiview(
      self,
      batch: types.Batch,
      *args,
      **kwargs,
  ) -> tuple[jt.Float[jt.Array, '... D'], jt.Float[jt.Array, '... L D']]:
    """Interface to backbone encoder."""
    if self.config.get('mask_ratio', None) is not None and kwargs['train']:
      mask = misc.token_mask(
          self.make_rng('droptok'),
          image_shape=batch['image_pixels'].shape,
          patch_size=self.config.vit.get('patches', {}).get('size', [16, 16]),
          mask_ratio=self.config.mask_ratio,
          mask_prob=self.config.get('mask_prob', 1.0),
      )
      kwargs['mask'] = mask
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
    if self.config.get('embed_aerial_coords', None):
      aerial_pixels = jnp.concatenate(
          [
              aerial_batch['image_pixels'],
              aerial_batch['latlng_pixels'],
          ],
          axis=-1,
      )
    else:
      aerial_pixels = aerial_batch['image_pixels']
    logits, tokens = self.aerial_backbone.encode_multiview(
        aerial_pixels, *args, **kwargs
    )

    if self.config.get('center_aerial_logits', False):
      logits = (
          logits
          - self.variable(
              'model_stats',
              'center_aerial_logits',
              lambda: jnp.zeros([1, 1, logits.shape[-1]], dtype=logits.dtype),
          ).value
      )
      if self.config.get('l2_normalize'):
        logits = misc.l2_normalize(logits)
    return logits, tokens

  def encode_and_pool_aerial(
      self,
      aerial_batch: types.Batch,
      *args,
      **kwargs,
  ) -> jt.Float[jt.Array, '... D']:
    """Interface to aerial backbone encoder."""
    logits, _ = self.encode_aerial(aerial_batch, *args, **kwargs)
    return {
        'max': lambda x: jnp.max(x, axis=-2),
        'mean': lambda x: jnp.mean(x, axis=-2),
    }[self.config.aerial_pooling](logits)

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

    if self.config.get('embed_aerial_coords', None):
      ecef_pixels = jax.vmap(jax.vmap(misc.aerial_pixel_coords))(
          aerial_pixels,
          batch['aerial']['resolution'],
          batch['aerial']['ecef_t_grid'],
          jnp.broadcast_arrays(
              batch['aerial']['upright'][:, None], batch['aerial']['resolution']
          )[0],
      )
      latlng_pixels = misc.ecef_to_latlngalt(ecef_pixels)[..., :2]
      aerial_pixels = jnp.concatenate([aerial_pixels, latlng_pixels], axis=-1)

    if train and self.config.get('crop', 'none') != 'none':
      image_pixels = self.crop_fn(image_pixels, self.make_rng('crop'))
    return image_pixels, aerial_pixels

  def maybe_read_from_memory(
      self, to_memorize: dict[str, Any]
  ) -> dict[str, Any]:
    """Read from memory if available."""
    if self.config.get('memorize_steps', None):
      n_steps = self.config.memorize_steps
      n_proc = jax.process_count()  # Total number of devices.
      memorized_variables = {
          k: self.variable(
              'model_stats',
              k,
              lambda: -jnp.ones((n_steps, v.shape[0] * n_proc) + v.shape[1:], dtype=v.dtype),  # pylint: disable=cell-var-from-loop
          )
          for k, v in to_memorize.items()
      }

      memorized_variables_f = {
          k: v.value.reshape(-1, *v.value.shape[2:])
          for k, v in memorized_variables.items()
      }
      return {
          k: jnp.concatenate([v, memorized_variables_f[k]], axis=0)
          for k, v in to_memorize.items()
      }
    else:
      return to_memorize

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
    updated_variables = self.maybe_read_from_memory(to_memorize)
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

    if self.config.get('l2_normalize', False):
      similarity_l2 = similarity
    else:
      similarity_l2 = misc.compute_similarity(
          logits[:, None],
          aerial_logits[None],
          normalize=True,
      )[..., 0]

    valid = (~duplicates) & (all_cell_idx[None] != -1)
    log_softmax_fn = (
        decoupled_log_softmax
        if self.config.get('decouple_softmax', False)
        else nn.log_softmax
    )
    log_probs = log_softmax_fn(
        similarity / self.temperature,
        axis=-1,
        where=valid,
    )
    probs = jnp.exp(log_probs)

    if self.config.get('label_smoothing', 0.0) > 0.0:
      weights = jnp.ones_like(log_probs) / log_probs.shape[-1]
      weights = weights.at[jnp.arange(batch_size), gt_index].set(
          1 - self.config.get('label_smoothing', 0.0)
      )
      gt_cell_log_prob = (weights * log_probs).sum(-1)
    else:
      gt_cell_log_prob = jnp.take_along_axis(
          log_probs, gt_index[:, None], axis=-1
      )[:, 0]
    gt_cell_prob = jnp.exp(gt_cell_log_prob)

    # Currently use 1.
    return {
        'log_probs': log_probs,
        'probs': probs,
        'similarity': similarity,
        'similarity_l2': similarity_l2,
        'duplicates': duplicates,
        'same_cell': same_cell,
        'gt_index': gt_index,
        'all_cell_idx': all_cell_idx,
        f'all_{target_name}_logits': target_logits,
        f'all_{target_name}_in_batch': target_logits_in_batch,
        'gt_cell_prob': gt_cell_prob,
        'gt_cell_log_prob': gt_cell_log_prob,
        'gt_similarity': jnp.take_along_axis(
            similarity, gt_index[:, None], axis=-1
        )[..., 0],
        'best_index': jnp.argmax(probs, axis=-1),
        'temperature': self.temperature[None].repeat(batch_size, axis=0),
    }

  def _get_extra_tokens(
      self, batch: types.Batch
  ) -> jt.Float[jt.Array, 'B M C'] | None:
    """Create extra tokens for the backbone."""
    return intrinsics_encoding.get_intrinsics_tensor(
        self.config.get('encode_intrinsics', []), batch.get('camera', {})
    )

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
    logits, tokens = self.encode_multiview(
        {'image_pixels': image_pixels},
        train=train,
        sow_attention=self.config.get('sow_attention', False) and not train,
        extra_tokens=self._get_extra_tokens(batch),
    )

    aerial_batch = {
        'image_pixels': aerial_pixels[..., :3],
    }
    # Check if latlng pixels exist.
    if aerial_pixels.shape[-1] == 5:
      aerial_batch['latlng_pixels'] = aerial_pixels[..., -2:]
    if 'cell_idx' in batch:
      aerial_batch['cell_idxs'] = batch['cell_idx'][:, None]
    aerial_logits, aerial_tokens = self.encode_aerial(
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

    idx_patches_sv_tokens = self.backbone.get_patch_token_indices(
        image_pixels.shape[-3], image_pixels.shape[-2]
    )
    idx_patches_aerial_tokens = self.aerial_backbone.get_patch_token_indices(
        aerial_pixels.shape[-3], aerial_pixels.shape[-2]
    )
    pred = {
        'logits': logits,
        'aerial_logits': aerial_logits,
        'aerial_logits_p': aerial_logits_p,
        'augmented_pixels': image_pixels,
        'augmented_aerial_pixels': aerial_pixels[..., :3],
    }

    # tokens are valid only for vit backbones
    if idx_patches_sv_tokens is not None:
      start_index, end_index = idx_patches_sv_tokens
      pred['tokens'] = tokens[..., start_index:end_index, :]

    if idx_patches_aerial_tokens is not None:
      start_index, end_index = idx_patches_aerial_tokens
      pred['aerial_tokens'] = aerial_tokens[..., start_index:end_index, :]

    if self.config.get('embed_aerial_coords', None):
      pred['aerial_coords'] = aerial_pixels[..., 3:]

    if self.config.get('compute_stochastic_probs', True):
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

    if self.config.get('compute_stochastic_self_probs', False):
      pred['sv_sv_stochastic_probs'] = self.stochastic_probs(
          logits,
          batch['latlng'],
          logits,
          batch['latlng'],
          batch,
          target_name='sv_sv',
      )
      pred['aerial_aerial_stochastic_probs'] = self.stochastic_probs(
          aerial_logits_p,
          batch['aerial']['center_ll'],
          aerial_logits_p,
          batch['aerial']['center_ll'],
          batch,
          target_name='aerial_aerial',
      )
    return pred


class SVAerialRetrievalModel(base.BaseModel):
  """Trainer-facing wrapper for the cell code retrieval model."""

  required_rngs: tuple[str, ...] = (
      'masking_rng',
      'dropout',
      'augment',
      'augment_aerial',
      'droptok',
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
    return SVAerialRetrieval(
        self.config, self.dtype, cell_centers=jnp.array([[]])
    )

  def init_state(
      self, model_params: types.ModelParams, model_state: types.ModelState
  ) -> tuple[types.ModelParams, types.ModelState]:
    class_backbone = get_backbone_class(
        self.config.vit.get('backbone_type', config_types.BackboneTypes.VIT)
    )
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

  def update_state(
      self,
      pred: types.Predictions,
      model_params: types.ModelParams,
      model_state: types.ModelState,
      learning_rates: types.LearningRates,
  ) -> tuple[types.ModelParams, types.ModelState]:
    if self.config.get('center_aerial_logits', False):
      logits_p = pred['sv_aerial_stochastic_probs']['all_aerial_logits'].mean(
          (0, 1), keepdims=True
      )
      momentum = self.config.get('center_aerial_logits_momentum', 0.99)
      new_center = logits_p * (1 - momentum) + model_state['model_stats'][
          'center_aerial_logits'
      ] * (momentum)
      model_state['model_stats']['center_aerial_logits'] = new_center
    if self.config.get('memorize_steps', None):
      n_steps = self.config.memorize_steps
      for k in model_state['model_stats'].keys():
        model_state['model_stats'][k] = pred[k].reshape(
            n_steps + 1, -1, *model_state['model_stats'][k].shape[2:]
        )[:n_steps]
    return model_params, model_state

  def stochastic_loss_metrics_function(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      prefix: str = '',
  ) -> types.LossMetricsTuple:
    losses, metrics = {}, {}
    batch = flatten_batch(
        batch, stride=self.config.vit.get('patches', {}).get('size', [16])[0]
    )
    losses[f'{prefix}nll'] = -pred['gt_cell_log_prob']

    t = pred['temperature'][0]
    losses[f'{prefix}info_dce'] = -pred[
        'gt_similarity'
    ] / t + jsp.special.logsumexp(
        pred['similarity'] / t, axis=-1, where=~pred['same_cell']
    )
    if self.config.get('l2_normalize', True):
      losses[f'{prefix}dist_l2'] = 1 - pred['gt_similarity']
    metrics[f'{prefix}dist_l2_avg'] = pred['similarity_l2'].mean(-1)

    metrics[f'{prefix}precision'] = (
        batch['cell_idx'] == pred['all_cell_idx'][pred['best_index']]
    ).astype(jnp.float32)
    metrics['duplicates'] = pred['duplicates'].sum(-1)

    max_k = min(pred['probs'].shape[-1], 100)
    pred_assignment = jax.lax.top_k(pred['probs'], k=max_k)[1]

    is_correct = pred_assignment == pred['gt_index'][:, None]
    metrics[f'{prefix}false_positives'] = jnp.cumprod(
        1 - is_correct, axis=-1
    ).sum(-1)
    for k in [1, 5, 20, max_k]:
      metrics[f'{prefix}recall_top{k:03d}'] = jnp.any(
          is_correct[:, :k], axis=-1
      )
      metrics[f'{prefix}_temperature'] = pred['temperature'][0]
    return losses, metrics

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

  def visualize(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      max_num_plots: int = 64,
      max_images_per_row: int = 8,
      plot_attention: bool = False,
      **kwargs,
  ) -> Mapping[str, FloatArray['B H W 3']]:

    def make_image_grid(x):
      h, w = x.shape[-3:-1]
      return jnp.concatenate(
          jnp.concatenate(
              x.reshape(-1, max_images_per_row, h, w, 3),
              axis=-2,
          ),
          axis=-3,
      )[None]

    def visualize_features(
        features: FloatArray['B N D'],
        image_pixels: FloatArray['B H W 3'],
        patch_size: tuple[int, int],
    ):
      h, w = image_pixels.shape[-3:-1]
      ph, pw = patch_size
      pca_sep = np.stack([misc.features_to_rgb(x) for x in features], 0)
      pca = misc.features_to_rgb(features, normalize=False)

      def like_image(x):
        return jax.image.resize(
            misc.sequence_to_image(x, h // ph, w // pw),
            image_pixels.shape,
            jax.image.ResizeMethod.NEAREST,
        )

      return jnp.concatenate(
          [
              image_pixels,
              like_image(pca_sep),
              like_image(pca),
          ],
          axis=-2,
      )

    visualizations = {}
    batch = flatten_batch(batch, stride=self.config.vit.patches.size[0])
    batch = jax.tree.map(lambda x: x[:max_num_plots], batch)
    pred = jax.tree.map(lambda x: x[:max_num_plots], pred)

    visualizations['aerial'] = make_image_grid(
        visualize_features(
            pred['aerial_tokens'][:, 0],
            batch['aerial']['rgb'][:, 0],
            self.config.vit_aerial.patches.size,
        )
    )
    visualizations['sv'] = make_image_grid(
        visualize_features(
            pred['tokens'],
            batch['image_pixels'],
            self.config.vit.patches.size,
        )
    )

    if 'intermediates' in pred and plot_attention:
      last_block = str(self.config.vit.num_layers - 1)
      representation = pred['intermediates']['backbone']
      attn_weights = representation['encoderblock_' + last_block][
          'MultiHeadDotProductAttention_0'
      ]['attention_weights'][0]

      ph, pw = self.config.vit.patches.size
      image_pixels = pred['augmented_pixels']
      bs, h, w = image_pixels.shape[:-1]
      # we look only at the attention of the CLS token
      raw_attentions = attn_weights[:, :, 0, 1:].reshape(
          bs, -1, h // ph, w // pw
      )

      # Scale to [0, 1] per head.
      norm_attentions = raw_attentions / raw_attentions.max(
          axis=(-2, -1), keepdims=True
      )
      attentions = plot.apply_colormap(norm_attentions, cmap='turbo')
      attentions = jax.image.resize(
          attentions,
          attentions.shape[:-3] + (h, w, 3),
          jax.image.ResizeMethod.NEAREST,
      )
      viz_att = jnp.concatenate([image_pixels[:, None], attentions], axis=-4)

      # viz_att = jax.lax.collapse(viz_att[:, :, 0], 1, -4)
      viz_att = jnp.concatenate(viz_att.swapaxes(0, 1), axis=-2)
      visualizations['attention'] = viz_att
      # Visualize sum of attention weights
      attentions = norm_attentions.mean(-3)
      attentions = plot.apply_colormap(attentions, cmap='turbo')
      attentions = jax.image.resize(
          attentions,
          image_pixels.shape,
          jax.image.ResizeMethod.NEAREST,
      )
      viz_att = jnp.concatenate([image_pixels, attentions], axis=-3)
      visualizations['attention_sum'] = viz_att
    return visualizations
