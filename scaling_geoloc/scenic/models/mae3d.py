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

"""3D MAE."""

from typing import Any, Mapping

# Assumtions: query == map

# Approach 1: Masked 2D-3D
# Step 1: Masked encode --> tokens + cls token
# Step 2: Lift tokens to 3D (e.g. via LiDAR) --> 3D tokens
# Step 3: 2D coordinate grid with mask tokens + cls token -> catt to 3D tokens
# Step 4: RGB Reconstruction loss

# Approach 2: Masked cls token, Unmasked 2D-3D --> learns visibility
# Step 1: Unmasked encode --> 2D tokens
# Step 2: Lift tokens to 3D (e.g. via LiDAR) --> 3D tokens
# Step 3: Masked encode --> cls token (Should be very high mask rate)
# Step 4: 2D coordinate grid with mask tokens + cls token -> catt to 3D tokens
# Step 5: RGB Reconstruction loss

# Approach 3: Masked cls token,
# Step 1: Unmasked encode --> 2D tokens
# Step 2: Lift tokens to 3D (e.g. via LiDAR) --> 3D tokens
# Step 3: Masked encode --> cls token (Should be very high mask rate)
# Step 4: 3D coordinates enc + mask token + cls token -> catt to 3D tokens
# Step 5: RGB Reconstruction loss

# Approach 4: Masked encoding, 3D-3D attn -> decoding
# Step 1: Masked encode --> tokens + cls token
# Step 2: Lift tokens to 3D (e.g. via LiDAR) --> 3D tokens
# Step 3: 3D coordinates (C-frame) enc + mask token + cls token --> decode
# Step 4: RGB Reconstruction loss

# We use Approach 4 which learns 3D reasoning and implicitly the visibility.

from etils.array_types import FloatArray
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxcam
import ml_collections

from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.models import base
from scaling_geoloc.scenic.models import vit
from scaling_geoloc.utils import gaussians
from scaling_geoloc.utils import misc
from scaling_geoloc.utils import plot


class MAE3D(nn.Module):
  """Encoder."""

  config: ml_collections.ConfigDict
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.encoder = vit.ViTBackbone.from_config(self.config.encoder)
    self.decoder = vit.Decoder(**self.config.decoder)

  @nn.compact
  def encode(
      self, x: FloatArray['B N H W C'], *, train: bool, seqlen: int
  ) -> FloatArray['B N L+1 D']:
    pass

  @nn.compact
  def add_3d_posemb(
      self, x: FloatArray['B N L D'], map_gaussians: gaussians.Gaussians
  ) -> FloatArray['B N L D']:
    y = map_gaussians[0]
    y = nn.Dense(512, name='posemb_encode0')(y)
    y = nn.GroupNorm(32)(y)
    y = nn.Dense(x.shape[-1], name='posemb_encode1')(y)
    y = nn.relu(y)
    y = nn.GroupNorm(32)(y)
    return x + y

  @nn.compact
  def __call__(
      self,
      batch: types.Batch,
      *,
      train: bool,
      debug: bool = False,
  ) -> types.Predictions:
    ph, pw = self.config.encoder.patches.size
    batch = misc.filter_batch_for_jit(batch)
    batch = jax.tree.map(
        lambda x, y: jnp.concatenate([x, y], axis=1),
        batch['map']['views'],
        batch['query']['views'],
    )
    map_images = batch['image_pixels']
    mask_points = batch['lidar_mask']
    if 'w_lidar_points' in batch:
      w_points = batch['w_lidar_points']
    else:
      cameras = types.create_jaxcam(
          batch['cam_t_w'],
          batch['camera'],
          is_fisheye=True,
          use_inverted_distortion=True,
      )
      h, w = map_images.shape[-3:-1]
      pixels = misc.grid_indices(h, w, map_images.shape[:-3]) + 0.5
      depth = batch['lidar_depth'].astype(pixels.dtype)
      # If the depth is missing, we set it to a large value.
      # In patches with enough depth this will be ignored.
      # If a patch has no valid depth, the dummy depth will be used.
      if self.config.get('fill_missing_depth'):
        depth = jnp.where(mask_points, depth, 1000.0)
      # Batch first two dimensions B x N
      w_points = jax.vmap(jax.vmap(jaxcam.pixels_to_points))(
          cameras, pixels, depth[..., 0]
      )
      w_points = w_points[..., :3] / w_points[..., -1:]

    map_images = misc.crop_to_multiple(map_images, ph)
    w_points = misc.crop_to_multiple(w_points, ph)
    mask_points = misc.crop_to_multiple(mask_points, ph)
    w_points_patched = misc.patchify(
        w_points, *self.config.encoder.patches.size
    )
    map_gaussians = gaussians.gaussians_from_points(
        w_points_patched,
        misc.patchify(mask_points, *self.config.encoder.patches.size),
    )
    # Manually set missing gaussians.
    if self.config.get('fill_missing_depth'):
      map_gaussians = (
          map_gaussians[0],
          map_gaussians[1],
          jnp.where(
              map_gaussians[2][..., None] == 0,
              map_gaussians[0],
              w_points_patched.mean(-2),
          ),
      )
    logits, backbone_output, is_masked = self.encoder.encode_masked(
        map_images,
        self.config.masking_rate,
        train=train,
        debug=debug,
    )
    x_cls, x_patch = backbone_output[..., 0, :], backbone_output[..., 1:, :]
    num_tokens = x_patch.shape[-2]
    b, n = map_images.shape[:2]

    x_cloud = self.add_3d_posemb(x_patch, map_gaussians)  # b x n x l x cin
    x_cloud = jax.lax.collapse(x_cloud, 1, -1)  # b x nl x cin
    x = self.decoder(x_cloud, train=train)  # b x nl x cout

    reconstructed = nn.Dense(ph * pw * 3, name='decoded_patches')(x)
    reconstructed = reconstructed.reshape(
        b,
        n,
        num_tokens,
        reconstructed.shape[-1],
    )

    priors = jax.lax.collapse(
        misc.reshape_image_to_patches(map_images, ph, pw), -3
    )

    pred = {
        'reconstructed': reconstructed,  # b x n x l x (p*p*3)
        'priors': priors,  # b x n x l x (p*p*3)
        'images': map_images,  # b x n x h x w x 3
        'is_masked': is_masked,  # b x n x l x 1
        'map_features': x,  # b x n x l x cout
        'cls_tokens': x_cls,  # b x n x cin
        'masked_patch_tokens': x_patch,  # b x n x l x cin
        'logits': logits,  # b x n x chead
    }

    if not train:
      # Run encoding without masking for visualization purposes.
      pred['patch_tokens_full'] = self.encoder.encode_multiview(
          map_images, train=False
      )[1][..., 1:, :]
    return pred


class MAE3DModel(base.BaseModel):
  """Trainer facing wrapper for MAE3D."""

  required_rngs: tuple[str, ...] = ('masking_rng', 'dropout')

  def build_flax_model(self) -> nn.Module:
    return MAE3D(
        config=self.config,
        dtype=self.dtype,
    )

  def loss_metrics_function(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      model_params: jnp.ndarray | None = None,
      model_state: Any | None = None,
      learning_rates: Mapping[str, float] | None = None,
  ) -> types.LossMetricsTuple:
    losses, metrics = {}, {}
    reconstructed = pred['reconstructed']
    priors = pred['priors']

    norm_axes = (-1, -2, -3)
    mse = (reconstructed - priors) ** 2
    losses['total_mse'] = mse.mean(norm_axes)
    losses['masked_mse'] = misc.masked_mean(
        mse, pred['is_masked'].astype(jnp.float32), axis=norm_axes
    )
    losses['unmasked_mse'] = misc.masked_mean(
        mse, 1.0 - pred['is_masked'].astype(jnp.float32), axis=norm_axes
    )
    losses['total'] = losses['total_mse']
    metrics['masking_rate'] = pred['is_masked'].mean(norm_axes)
    return losses, metrics

  def visualize(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      max_num_plots: int = 64,
  ) -> Mapping[str, Any]:
    ph, pw = self.config.encoder.patches.size
    batch = misc.filter_batch_for_jit(batch)
    batch = jax.tree.map(
        lambda x, y: jnp.concatenate([x, y], axis=1),
        batch['map']['views'],
        batch['query']['views'],
    )
    images = misc.crop_to_multiple(
        batch['image_pixels'],
        ph,
    )
    if max_num_plots > images.shape[0]:
      max_num_plots = images.shape[0]
    images = images[:max_num_plots]

    h, w = images.shape[-3:-1]
    rec_images = misc.reshape_patches_to_image(
        pred['reconstructed'][:max_num_plots].reshape(
            images.shape[:2] + (-1, ph, pw, 3)
        ),
        h // ph,
        w // pw,
    )

    images_with_depth = misc.alpha_blend(
        misc.to_grayscale(images),
        misc.crop_to_multiple(
            plot.apply_colormap(
                batch['lidar_depth'][:max_num_plots, ..., 0],
                batch['lidar_mask'][:max_num_plots, ..., 0],
                cmap='turbo_r',
                vmin=0.0,
                vmax=30.0,
            ),
            ph,
        ),
        0.8,
    )

    visualizations = {}
    side_by_side = jnp.concatenate(
        [images, rec_images, images_with_depth], axis=-3
    )

    visualizations['reconstruction'] = jnp.concatenate(
        side_by_side.swapaxes(0, 1), axis=-2
    )

    if 'patch_tokens_full' in pred:
      visualizations['features_pca'] = jnp.concatenate(
          misc.features_to_rgb(
              misc.sequence_to_image(
                  pred['patch_tokens_full'][:max_num_plots],
                  h // ph,
                  w // pw,
              )
          ).swapaxes(0, 1),
          -2,
      )
    return visualizations
