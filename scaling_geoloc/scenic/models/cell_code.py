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

from collections.abc import Mapping, Sequence
from typing import Any

from absl import logging
import dm_pix
from etils.array_types import FloatArray
from etils.array_types import IntArray
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt
import ml_collections
import numpy as np
import optax
from scenic.train_lib import train_utils

from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.models import base
from scaling_geoloc.scenic.models import layers
from scaling_geoloc.scenic.models import vit
from scaling_geoloc.utils import augmentations
from scaling_geoloc.utils import misc
from scaling_geoloc.utils import plot
from scaling_geoloc.viz import vmaps


augment: augmentations.AugmentationFn = augmentations.per_image(
    augmentations.sequential([
        augmentations.random_color_jitter(0.8, 0.1, 0.4, 0.2, 0.1),
        augmentations.random_grayscale(0.2),
        augmentations.random_gaussian_blur(1.0, 0.1, 2.0),
        augmentations.random_solarize(0.1, 0.5),
    ])
)


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


class CellCodeRetrieval(nn.Module):
  """Cell code retrieval model."""

  config: ml_collections.ConfigDict
  num_cells: int
  dtype: jnp.dtype
  cell_tree: Sequence[IntArray['N K']] | None = None
  cell_centers: FloatArray['N 2'] | None = None

  def setup(self):
    self.backbone = vit.ViTBackbone.from_config(self.config.vit, self.dtype)
    self.crop_fn = augmentations.get_crop_fn(
        self.config.get('crop', 'none'),
        self.config.get('crop_size', (224, 224)),
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
      self.temperature = jnp.array(temperature, dtype=self.dtype)

    self.cell_code_dim = self.config.get(
        'cell_code_dim', self.backbone.output_dim()
    )

  @nn.compact
  def _get_cell_codes(self) -> FloatArray['N D']:
    """Returns the cell codes. Builds them if cell_tree is available."""
    cell_code_dim = self.cell_code_dim
    def _init_cell_code(name: str, ncells: int, cell_code_dim: int):
      if self.config.get('ll_mlp', None):
        return layers.MLP(
            config=self.config.ll_mlp,
            dtype=self.dtype,
        )(self.cell_centers)
      else:
        return self.param(
            name,
            nn.initializers.normal(stddev=1 / np.sqrt(cell_code_dim)),
            (ncells, cell_code_dim),
            self.dtype,
        )

    if self.cell_tree is None:
      # One cell code for each cell in the finest level.
      if self.config.get('shard_codes', False) and not self.is_initializing():
        num_cells = self.num_cells // jax.device_count()
      else:
        num_cells = self.num_cells
      if self.config.get('ema_parameterfree', False):
        assert self.config.get('cell_codes_momentum')
        cell_codes_var = self.variable(
            'model_stats',
            'ema_cell_codes',
            jnp.zeros,
            shape=(num_cells, cell_code_dim),
        )
        return cell_codes_var.value
      elif self.config.get('cell_codes_momentum'):
        cell_codes = _init_cell_code('cell_codes', num_cells, cell_code_dim)
        ema_cell_codes = self.variable(
            'model_stats', 'ema_cell_codes', jnp.zeros, shape=cell_codes.shape
        )
        momentum = self.config.cell_codes_momentum
        cell_codes = (
            momentum * ema_cell_codes.value + (1 - momentum) * cell_codes
        )
        return cell_codes
      else:
        cell_codes = _init_cell_code('cell_codes', num_cells, cell_code_dim)
      return cell_codes
    else:
      assert not self.config.get('shard_codes', False)
      # One cell code for each cell in each level.
      d = cell_code_dim // len(self.cell_tree)
      height = len(self.cell_tree)
      last_d = cell_code_dim - (height - 1) * d
      cell_code_tree = [
          _init_cell_code(
              f'cell_codes{i}', x.shape[0], d if i < height - 1 else last_d
          )
          for i, x in enumerate(self.cell_tree)
      ]
      # Gather cell codes at each level.
      tree_idxs = self.cell_tree[-1]
      return jnp.concatenate(
          [
              cell_code_tree[i][tree_idxs[:, i]]
              for i in range(len(cell_code_tree))
          ],
          axis=-1,
      )

  def get_cell_codes(self) -> FloatArray['N D']:
    """Returns the cell codes."""
    cell_codes = self._get_cell_codes()
    if self.config.get('l2_normalize_per_head', False):
      cell_codes = misc.l2_normalize_per_head_vit(
          cell_codes, self.config.vit.head
      )
    if self.config.get('l2_normalize', False):
      cell_codes = misc.l2_normalize(cell_codes)
    return cell_codes

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
          patch_size=self.config.vit.patches.size,
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

  def get_image_pixels(
      self, batch: types.Batch, train: bool
  ) -> FloatArray['... H W C']:
    """Preprocess the image pixels (undistort, crop, augment)."""
    batch = flatten_batch(batch, stride=None)
    image_pixels = batch['image_pixels']
    undistort_p = self.config.get('undistort_prob', 0.0)
    cam = types.create_jaxcam(batch['cam_t_w'], batch['camera'])

    # Undistort images.
    if undistort_p > 0.9999:
      image_pixels, _ = jax.vmap(misc.undistort_image)(image_pixels, cam)
    elif train and undistort_p > 0.0:

      def maybe_undistort(rng, pixels, cam):
        return jnp.where(
            jax.random.uniform(rng) < undistort_p,
            # We currently use NN interpolation for undistortion.
            misc.undistort_image(pixels, cam, order=0)[0],
            pixels,
        )

      def undistort_batch(rng, pixels, cameras):
        rngs = jax.random.split(rng, pixels.shape[0])
        return jax.vmap(maybe_undistort)(rngs, pixels, cameras)

      image_pixels = undistort_batch(
          self.make_rng('undistort'), image_pixels, cam
      )

    if self.config.get('crop_borders', None) is not None:
      top, bottom, left, right = self.config.crop_borders
      h, w = image_pixels.shape[-3:-1]
      image_pixels = image_pixels[..., top : h - bottom, left : w - right, :]
    if self.config.get('center_crop', None) is not None:
      # Center crop to avoid boundary effects (like cars on the edge).
      image_pixels = dm_pix.center_crop(
          image_pixels, *self.config.center_crop, channel_axis=-1
      )
    if train:
      image_pixels = self.crop_fn(image_pixels, self.make_rng('crop'))
    # Crop to multiple of patch size to avoid boundary effects.
    image_pixels = misc.crop_to_multiple(
        image_pixels, self.config.vit.patches.size[0]
    )

    if train and self.config.get('do_augment', False):
      # Augment images
      rng = self.make_rng('augment')
      image_pixels = augment(image_pixels, rng)

    return image_pixels

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

  @nn.compact
  def __call__(
      self, batch: types.Batch, *, train: bool, debug: bool = False
  ) -> types.Predictions:
    # Augment pixels.
    image_pixels = self.get_image_pixels(batch, train)
    batch = flatten_batch(batch, stride=None)
    # Get the image embeddings.
    logits, tokens = self.encode_multiview(
        {'image_pixels': image_pixels},
        train=train,
        sow_attention=self.config.get('sow_attention', False) and not train,
    )
    if logits.ndim == 3:
      logits = logits[:, 0]

    # Get the cell codes.
    cell_codes = self.get_cell_codes()

    # Possible freeze the cell codes that are not in the batch.
    if (
        self.config.get('freeze_missing_cell_codes', False)
        and train
        and not self.is_initializing()
    ):
      frozen_cell_codes = jax.lax.stop_gradient(cell_codes)
      all_cell_idx = jax.lax.all_gather(
          batch['cell_idx'], axis_name='batch', tiled=True
      )
      do_freeze = jnp.ones(cell_codes.shape[0], dtype=jnp.bool_)
      # Do not freeze the cell codes that are in the batch.
      if self.config.get('shard_codes', False):
        shard_cell_idx = (
            all_cell_idx - jax.lax.axis_index('batch') * cell_codes.shape[0]
        )
        shard_cell_idx = jnp.where(
            shard_cell_idx < 0,
            cell_codes.shape[0],
            shard_cell_idx,
        )
      else:
        shard_cell_idx = all_cell_idx
      do_freeze = do_freeze.at[shard_cell_idx].set(False, mode='drop')
      cell_codes = jnp.where(
          do_freeze[..., None], frozen_cell_codes, cell_codes
      )

    # Compute the similarity to the cell codes.
    similarity = self.compute_similarity_to_cell_codes(
        logits,
        cell_codes,
        normalize=False,
    )

    # Select the best cell index.
    best_cell_idx = jnp.argmax(similarity, axis=-1)

    # Prepare the predictions.
    pred = {
        'logits': logits,
        'tokens': tokens,
        'best_cell_idx': best_cell_idx,
        'cell_similarity': similarity,
        'augmented_pixels': image_pixels,
        'temperature': self.temperature[None].repeat(logits.shape[0], axis=0),
    }

    # Regress the lat/lng of the best cell.
    if self.config.get('regress_latlng'):
      lat_lng = nn.Dense(2, name='regressor')(logits)
      pred['lat_lng'] = lat_lng

    # Return the ground truth cell index if available in batch.
    if 'cell_idx' in batch:
      pred['gt_cell_idx'] = batch['cell_idx']

    return pred


class CellCodeRetrievalModel(base.BaseModel):
  """Trainer-facing wrapper for the cell code retrieval model."""

  required_rngs: tuple[str, ...] = (
      'masking_rng',
      'dropout',
      'augment',
      'crop',
      'undistort',
      'droptok',
  )

  def build_flax_model(self) -> nn.Module:
    self.s2cell_ids = self.dataset_meta_data['s2cell_ids']

    self.cell_tree = [
        jnp.asarray(x) for x in self.dataset_meta_data.get('cell_tree', [])
    ]

    self.s2cell_centers = jnp.asarray(
        self.dataset_meta_data['s2cell_centers_ecef'], dtype=self.dtype
    )

    self.s2cell_centers_ll = jnp.asarray(
        self.dataset_meta_data['s2cell_centers_ll'], dtype=self.dtype
    )

    return CellCodeRetrieval(
        num_cells=len(self.s2cell_ids),
        config=self.config,
        dtype=self.dtype,
        cell_tree=self.cell_tree if self.config.get('use_tree_codes') else None,
        cell_centers=self.s2cell_centers_ll,
    )

  def init_state(
      self, model_params: types.ModelParams, model_state: types.ModelState
  ) -> tuple[types.ModelParams, types.ModelState]:
    model_params['backbone'] = vit.ViTBackbone.maybe_load_params(
        model_params['backbone'],
        self.config.vit.get('pretrained_weights', None),
        self.config.vit.get('pretrained_checkpoint', None),
    )
    return model_params, model_state

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
      for k, v in model_state['model_stats'].items():
        all_cell_idx = jax.lax.all_gather(
            pred['gt_cell_idx'], axis_name='batch', tiled=True
        )
        all_logits = jax.lax.all_gather(
            pred['logits'], axis_name='batch', tiled=True
        )
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
          if self.config.get('ema_parameterfree', False):
            updates = jnp.zeros_like(v)
            counts = jnp.zeros_like(v[:, 0], dtype=jnp.int32)
            updates = updates.at[shard_cell_idx].add(all_logits, mode='drop')
            counts = counts.at[shard_cell_idx].add(1, mode='drop')
            updates = updates / counts[..., None].clip(min=1.0)
            model_state['model_stats'][k] = (
                momentum * v + (1 - momentum) * updates
            )
          else:
            model_state['model_stats'][k] = (
                momentum * v
                + (1 - momentum) * model_params[k.replace('ema_', '')]
            )
    return model_params, model_state

  def distance_metrics(
      self, predicted_cell_idx: Any, gt_cell_idx: Any
  ) -> types.MetricsDict:
    best_cell_centers = self.s2cell_centers[predicted_cell_idx]
    gt_cell_centers = self.s2cell_centers[gt_cell_idx]
    distance_to_gt = jnp.linalg.norm(
        best_cell_centers - gt_cell_centers, axis=-1
    )
    metrics = misc.distance_metrics(distance_to_gt)
    return metrics

  def loss_metrics_function(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      model_params: jnp.ndarray | None = None,
      model_state: Any | None = None,
      learning_rates: types.LearningRates | None = None,
  ) -> types.LossMetricsTuple:
    losses, metrics = {}, {}
    batch = flatten_batch(batch, stride=self.config.vit.patches.size[0])
    # Align to cell index of the camera.
    gt_cell_idx = batch['cell_idx']  # B

    best_cell_idx = pred['best_cell_idx']
    cell_log_probs = nn.log_softmax(
        pred['cell_similarity'] / pred['temperature'][0], axis=-1
    )
    cell_probs = jnp.exp(cell_log_probs)

    if 'lat_lng' in pred:
      lat_lng = pred['lat_lng']
      losses['lat_lng_l2'] = jnp.sum((lat_lng - batch['latlng']) ** 2, axis=-1)

    gt_log_probs = jnp.take_along_axis(
        cell_log_probs, gt_cell_idx[:, None], axis=-1
    )[:, 0]
    losses['cell_nll'] = -gt_log_probs
    metrics['gt_cell_prob'] = jnp.exp(gt_log_probs)
    # For visualization purposes.
    pred['gt_cell_prob'] = jnp.exp(gt_log_probs)

    # Align to cell index where the camera has largest overlap.
    best_frustum_cell_idx = batch['frustum_cell_idx'][:, 0]  # B
    best_f_log_probs = jnp.take_along_axis(
        cell_log_probs, best_frustum_cell_idx[:, None], axis=-1
    )[:, 0]
    losses['best_frustum_nll'] = -best_f_log_probs
    metrics['best_frustum_cell_prob'] = jnp.exp(best_f_log_probs)

    # Compute (weighted) similarity to all cells covered by the frustum.
    frustum_cell_idxs = batch['frustum_cell_idx']
    frustum_log_probs = jnp.take_along_axis(
        cell_log_probs, frustum_cell_idxs, axis=-1
    )
    frustum_log_probs = jnp.where(
        frustum_cell_idxs > -1, frustum_log_probs, -1.0e8
    )
    frustum_overlap = batch['frustum_overlap'] * (
        batch['frustum_overlap'] > 0.01
    )
    frustum_overlap = frustum_overlap / frustum_overlap.sum(
        axis=-1, keepdims=True
    )
    frustum_probs = jnp.exp(frustum_log_probs)
    frustum_prob = frustum_probs.sum(axis=-1)
    frustum_max_prob = frustum_probs.max(axis=-1)

    losses['frustum_nll'] = -misc.log_safe(frustum_prob)
    losses['frustum_max_nll'] = -misc.log_safe(frustum_max_prob)
    losses['frustum_weighted_nll'] = -misc.masked_mean(
        frustum_log_probs, frustum_overlap
    )
    metrics['frustum_prob'] = frustum_prob

    n_cell_idxs = batch['neighbor_cell_idx']
    n_log_probs = jnp.take_along_axis(cell_log_probs, n_cell_idxs, axis=-1)
    n_log_probs = jnp.where(n_cell_idxs > -1, n_log_probs, -1.0e8)
    n_probs = jnp.exp(n_log_probs)
    n_prob = n_probs.sum(axis=-1)
    losses['neighbor_nll'] = -misc.log_safe(n_prob)

    # Similarity-based losses.
    if 'sig' in self.config.compose_loss:
      sim = pred['cell_similarity']
      gt_sim = jnp.take_along_axis(sim, gt_cell_idx[:, None], axis=-1)[:, 0]
      losses['cell_sig_nll'] = -nn.log_sigmoid(
          gt_sim / self.config.get('cell_sig_temperature', 1.0)
      )

      losses['cell_sig_bce_balanced'] = (
          misc.weighted_sigmoid_binary_cross_entropy(
              sim / self.config.get('cell_sig_temperature', 1.0),
              (jnp.arange(sim.shape[1])[None] == gt_cell_idx[:, None]),
          )
      )
      losses['cell_sig_bce'] = optax.sigmoid_binary_cross_entropy(
          sim / self.config.get('cell_sig_temperature', 1.0),
          (jnp.arange(sim.shape[1])[None] == gt_cell_idx[:, None]),
      ).mean(-1)

    if 'multi_similarity' in self.config.compose_loss:
      ms_pos, ms_neg = misc.multi_similarity_loss(
          pred['cell_similarity'],
          gt_cell_idx[:, None],
          alpha=2.0,
          beta=100.0,
          base=0.2,
      )
      losses['multi_similarity_pos'] = ms_pos
      losses['multi_similarity_neg'] = ms_neg
      losses['multi_similarity'] = (ms_pos + ms_neg) * 0.5

    # Hierarchical loss.
    if 'hierarchical_nll' in self.config.compose_loss:
      if 'frustum' in self.config.compose_loss:
        to_compute = {
            '_frustum': batch['frustum_cell_idx'],
        }
      else:
        to_compute = {
            '': batch['cell_idx'][..., None],
        }

      for name, batch_idxs in to_compute.items():
        gt_mask_per_level = misc.tree_mask(self.cell_tree[-1], batch_idxs)
        gt_prob_tree = (gt_mask_per_level * cell_probs[..., None]).sum(-2)
        gt_log_prob_tree = misc.log_safe(gt_prob_tree)
        for level in range(len(self.cell_tree)):
          metrics[f'z_gt_prob{name}_{level:02d}'] = gt_prob_tree[..., level]
          metrics[f'z_cell_nll{name}_{level:02d}'] = -gt_log_prob_tree[
              ..., level
          ]
          losses[f'hierarchical_nll{name}_{level:02d}'] = -(
              gt_log_prob_tree[..., level]
          ).mean(-1)
        losses[f'hierarchical_nll{name}'] = -gt_log_prob_tree.mean(-1)
        weights = misc.weights_from_tree(gt_prob_tree)
        losses[f'hierarchical_nll_weighted{name}'] = -(
            jax.lax.stop_gradient(weights) * gt_log_prob_tree
        ).mean(-1)
        losses[f'hierarchical_nll_weightedsq{name}'] = -(
            jax.lax.stop_gradient(weights) ** 2 * gt_log_prob_tree
        ).mean(-1)
        losses[f'hierarchical_nll_weighted_grad{name}'] = -(
            weights * gt_log_prob_tree
        ).mean(-1)

    x_ll = batch['latlng']
    g_ll = self.s2cell_centers_ll[gt_cell_idx]

    if 'siglip_nll' in self.config.compose_loss:
      gt_mask = jnp.arange(cell_probs.shape[-1])[None] == gt_cell_idx[:, None]
      z_gt = 2 * gt_mask - 1.0
      losses['siglip_nll'] = misc.log_safe(
          1
          + jnp.exp(
              -z_gt
              * (pred['cell_similarity'] - pred['similarity_bias'][0])
              / pred['temperature'][0]
          )
      ).sum(-1)

    # Haversine smoothing.
    if 'haversine_nll' in self.config.compose_loss:
      all_distances = misc.haversine_distance(
          self.s2cell_centers_ll[None], x_ll[:, None]
      )[..., 0]
      distance_to_gt = misc.haversine_distance(g_ll, x_ll)
      delta_distances = all_distances - distance_to_gt
      haversine_weights = misc.distances_to_weights(
          delta_distances / self.config.get('haversine_temperature', 75_000)
      )
      losses['haversine_nll'] = -(haversine_weights * cell_log_probs).sum(
          -1
      ) / haversine_weights.sum(-1).clip(min=1.0e-3)

      # Compute weighted haversine loss over a set of temperatures (radiis).
      if self.config.get('haversine_temperatures', None):
        temperatures = jnp.array(self.config.haversine_temperatures)[
            None, :, None
        ]
        probs = cell_probs[:, None]
        tree_weights = misc.distances_to_weights(
            delta_distances[:, None] / temperatures, clip_min=1.0e-3
        )
        tree_prob = (tree_weights * probs).sum(-1)  # B x H
        for i, temperature in enumerate(temperatures):
          losses[f'haversine_nllp_{temperature}'] = -misc.log_safe(
              tree_prob[..., i]
          )
        haversine_mask = delta_distances[:, None] < temperatures
        haversine_mask_probs = (haversine_mask * probs).sum(-1)

        weights = jnp.roll(tree_prob, 1, axis=-1)
        weights = weights.at[:, 0].set(1.0)
        losses['haversine_nll_weighted'] = -(
            jax.lax.stop_gradient(weights)
            * (tree_weights * misc.log_safe(probs)).sum(-1)
        ).sum(-1)
        losses['haversine_nll_mask_weighted'] = -(
            jax.lax.stop_gradient(weights) * misc.log_safe(haversine_mask_probs)
        ).sum(-1)
        losses['haversine_nll_mask_weightedsq'] = -(
            jax.lax.stop_gradient(weights) ** 2
            * misc.log_safe(haversine_mask_probs)
        ).sum(-1)
        losses['haversine_nll_mask_all'] = -(
            misc.log_safe(haversine_mask_probs)
        ).sum(-1)

    if self.config.get('compute_expected_location', False):
      # Compute expected location and covariance.
      def weighted_sum(x, weights, axis=-1):
        return (x * weights).sum(axis=axis)

      cell_probs = jnp.exp(cell_log_probs)
      expected_location_ll = weighted_sum(
          self.s2cell_centers_ll[None], cell_probs[..., None], axis=1
      )
      expected_location_cov_ll = misc.batched_cov(
          self.s2cell_centers_ll[None] - expected_location_ll[:, None],
          weights=cell_probs,
      )
      expected_location_std = jnp.sqrt(
          jnp.trace(expected_location_cov_ll, axis1=-2, axis2=-1)
      )
      expected_haversine_var = weighted_sum(
          misc.haversine_distance(
              self.s2cell_centers_ll[None], expected_location_ll[:, None]
          )[..., 0],
          cell_probs,
          axis=-1,
      )
      expected_distance_to_gt = misc.haversine_distance(
          expected_location_ll, x_ll
      )[..., 0]
      losses['expected_haversine_distance_gt'] = expected_distance_to_gt
      losses['expected_ll_distance_gt_squared'] = (
          (expected_location_ll - x_ll) ** 2
      ).sum(-1)
      losses['expected_ll_distance_gt_l1'] = (
          jnp.abs(expected_location_ll - x_ll)
      ).sum(-1)
      metrics = {
          'z_expected_ll_distance': jnp.sqrt(
              losses['expected_ll_distance_gt_squared']
          ),
          'z_expected_std_ll': expected_location_std,
          'z_expected_haversine_std': jnp.sqrt(expected_haversine_var),
          **misc.distance_metrics(expected_distance_to_gt, 'z_expected_'),
          **metrics,
      }
    # Get total loss.
    losses['total'] = types.compose_loss(losses, self.config.compose_loss)

    metrics['precision'] = (gt_cell_idx == best_cell_idx).astype(jnp.float32)
    metrics['sv_precision'] = metrics['precision']

    # Recall metrics.
    top100 = jax.lax.approx_max_k(jax.lax.stop_gradient(cell_probs), k=100)[1]
    is_correct100 = gt_cell_idx[:, None] == top100
    metrics['false_positives'] = jnp.cumprod(
        1 - (gt_cell_idx[:, None] == top100), axis=-1
    ).sum(-1)
    for k in [1, 5, 10, 20, 50, 100]:
      metrics[f'recall@{k}'] = is_correct100[:, :k].sum(-1)
    metrics['is_neighbor'] = jax.vmap(lambda x, y: jnp.isin(x, y).any())(
        best_cell_idx, batch['neighbor_cell_idx']
    )
    metrics['sv_is_neighbor'] = metrics['is_neighbor']
    metrics['is_best_frustum'] = (
        batch['frustum_cell_idx'][:, 0] == best_cell_idx
    ).astype(jnp.float32)
    metrics['temperature'] = pred['temperature']
    metrics['sv_temperature'] = metrics['temperature']

    metrics = {
        **metrics,
        **self.distance_metrics(best_cell_idx, gt_cell_idx),
    }

    return losses, metrics

  def visualize(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      max_num_plots: int = 64,
      max_images_per_row: int = 8,
      border_width: int = 5,
      plot_attention: bool = False,
      **kwargs,
  ) -> Mapping[str, FloatArray['B H W 3']]:
    visualizations = {}
    batch = flatten_batch(batch, stride=self.config.vit.patches.size[0])
    batch = jax.tree.map(lambda x: x[:max_num_plots], batch)
    pred = jax.tree.map(lambda x: x[:max_num_plots], pred)

    red = np.array([1.0, 0.0, 0.0])
    green = np.array([0.0, 1.0, 0.0])
    yellow = np.array([1.0, 1.0, 0.0])
    if self.config.get('use_patch_probs', False):
      best_cell_idx = pred['patch_best_cell_idx']
    else:
      best_cell_idx = pred['best_cell_idx']
    is_correct = best_cell_idx == batch['cell_idx']
    is_neighbor = np.apply_along_axis(
        lambda x: np.isin(x[0], x[1:]).any(),
        -1,
        np.concatenate(
            [
                best_cell_idx[..., None],
                batch['neighbor_cell_idx'],
            ],
            axis=-1,
        ),
    )
    # 0 is wrong, 1 is neighbor, 2 is correct
    labels = np.where(is_correct, 2, is_neighbor)
    # image_pixels = batch['image_pixels']
    image_pixels = pred['augmented_pixels']
    labelled_images = misc.label_image_border(
        image_pixels,
        labels,
        [red, yellow, green],
        border_width=border_width,
    )

    ph, pw = self.config.vit.patches.size
    bs, h, w = image_pixels.shape[:-1]

    def make_image_grid(x):
      return jnp.concatenate(
          jnp.concatenate(
              x.reshape(-1, max_images_per_row, h, w, 3),
              axis=-2,
          ),
          axis=-3,
      )[None]

    visualizations['correct_cell'] = make_image_grid(labelled_images)

    features = pred['tokens'][..., 1:, :]
    pca_sep = np.stack([misc.features_to_rgb(x) for x in features], 0)
    pca = misc.features_to_rgb(features, normalize=False)

    def like_image(x):
      return jax.image.resize(
          misc.sequence_to_image(x, h // ph, w // pw),
          image_pixels.shape,
          jax.image.ResizeMethod.NEAREST,
      )

    visualizations['features_pca_per_image'] = make_image_grid(
        like_image(pca_sep)
    )
    visualizations['features_pca'] = make_image_grid(like_image(pca))

    if 'intermediates' in pred and plot_attention:
      last_block = str(self.config.vit.num_layers - 1)
      representation = pred['intermediates']['backbone']
      attn_weights = representation['encoderblock_' + last_block][
          'MultiHeadDotProductAttention_0'
      ]['attention_weights'][0]

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

  def visualize_map(
      self,
      pred: types.Predictions,
      _: types.Batch,
      batch_indices: tuple[int, ...] = (),
      width: int = 2000,
      height: int = 900,
      zoom: int = 3,
      model_params: jnp.ndarray | None = None,
      offset: int = 0,
      cell_codes: jnp.ndarray | None = None,
      plot_counts: bool = False,
      plot_pca: bool = True,
      marker_size: int = 5,
      max_num_points: int = 400000,
      normalize_pca: bool = False,
      min_cell_count: int = 100,
  ) -> vmaps.GeoPlotter:
    # batch = flatten_batch(batch, stride=self.config.vit.patches.size[0])
    plotter = vmaps.GeoPlotter(zoom=zoom, height=height, width=width)
    s2cell_tokens = self.dataset_meta_data['s2cell_tokens']
    s2cell_token_counts = np.array(self.dataset_meta_data['s2cell_counts'])
    s2cell_mask = np.where(s2cell_token_counts > min_cell_count)[0]
    s2cell_mask = np.random.choice(
        s2cell_mask, min(max_num_points, len(s2cell_mask)), replace=False
    )
    cell_codes = (
        cell_codes if cell_codes is not None else model_params['cell_codes']
    )
    cell_code_pca = np.array(
        misc.features_to_rgb(
            cell_codes[s2cell_mask],
            normalize=normalize_pca,
            offset=offset,
        )
    )

    if plot_counts:
      count_color = plot.apply_colormap(
          s2cell_token_counts.astype(np.float32),
          cmap='turbo_r',
      )
      plotter.plot_s2cells(
          s2cell_tokens[s2cell_mask],
          plot_polygons=False,
          color=count_color[s2cell_mask],
          name='counts',
          labels=s2cell_token_counts[s2cell_mask],
          marker_size=marker_size,
      )

    if plot_pca:
      plotter.plot_s2cells(
          s2cell_tokens[s2cell_mask],
          plot_polygons=False,
          color=cell_code_pca,
          name='cell_codes',
          marker_size=marker_size,
      )
    for batch_idx in enumerate(batch_indices):
      ref_pred = jax.tree.map(lambda x, bid=batch_idx: x[bid], pred)
      s2cell_token = self.dataset_meta_data['s2cell_tokens'][
          ref_pred['best_cell_idx']
      ]
      sim_color = plot.apply_colormap(
          ref_pred['cell_similarity'],
          vmin=0.0,
          vmax=1.0,
          cmap='turbo_r',
      )

      plotter.plot_s2cells(
          s2cell_tokens[s2cell_mask],
          plot_polygons=False,
          color=sim_color[s2cell_mask],
          name=f'similarity_{s2cell_token}',
      )

      plotter.plot_s2cells(
          [s2cell_token],
          plot_polygons=False,
          color='black',
          marker_size=25,
          name=f'similarity_{s2cell_token}',
      )

    return plotter

  def get_cell_codes(
      self,
      train_state: train_utils.TrainState,
  ) -> FloatArray['N D']:
    return jax.pmap(self.inference_fn('get_cell_codes'), axis_name='batch')(
        train_state
    )

  def build_s2cell_database(
      self, train_state: train_utils.TrainState  # unreplicated
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
    if is_replicated:
      target_dims = [(jax.local_device_count(), *x) for x in target_dims]
    return target_dims

  def unreplicate_state(
      self, train_state: train_utils.TrainState
  ) -> train_utils.TrainState:
    # pytype: disable=attribute-error
    logging.info(
        'unreplicate_state target_shapes: %s %s',
        self.get_target_shapes(is_replicated=True),
        train_state.params['cell_codes'].shape,
    )
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
    logging.info(
        'replicate_state target_shapes: %s %s',
        self.get_target_shapes(is_replicated=False),
        train_state.params['cell_codes'].shape,
    )
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
      return jax.tree.map(
          misc.pmean_or_divide_from_shape_fn(grad['cell_codes'].shape), grad
      )
    else:
      return jax.lax.pmean(grad, axis_name='batch')

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
