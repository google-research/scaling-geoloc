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

"""Multiview MAE model from fomo."""

from collections.abc import Mapping

from etils.array_types import FloatArray
from etils.array_types import IntArray
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.projects.dino import vit as vit_dino

from google3.location.visualmapping.experimental.fomo.scenic import trainer as fomo_trainer
from google3.location.visualmapping.experimental.fomo.scenic.models import mae_model
from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.models import base
from scaling_geoloc.utils import grids
from scaling_geoloc.utils import misc
from scaling_geoloc.utils import plot
from scaling_geoloc.viz import viz3d


class _MaskedAutoEncoder(mae_model.MaskedAutoEncoder):
  """Masked Auto Encoder wrapper."""

  def encode_masked(
      self,
      image: FloatArray['B H W c'],
      *,
      train: bool,
      debug: bool = False,
  ) -> tuple[
      FloatArray['B T C'],
      FloatArray['B C'],
      FloatArray['B M'],
  ]:
    batch_size = image.shape[0]
    n_tokens = self.n_tokens
    to_encode, to_mask = mae_model.utils.shuffle_and_partition(
        n_batch=batch_size,
        n_tokens=n_tokens - 1,
        n_masked=self.n_masked,
        rng=self.make_rng('masking_rng'),
    )
    encoded_tokens = self._encode_image(
        image, encode_patch_indices=to_encode, train=train
    )
    x_rep = encoded_tokens[:, 0]

    all_rows = jnp.arange(batch_size).reshape(batch_size, 1)
    x = jnp.zeros((batch_size, n_tokens, self.hidden_size), dtype=image.dtype)
    x = x.at[all_rows, to_encode].set(encoded_tokens)
    x = x.at[all_rows, to_mask].set(self._shared_mask_token)

    # Do not decode here
    return x, x_rep, to_mask

  def encode_decode_masked(
      self, image: FloatArray['B H W c'], *, train: bool, debug: bool = False
  ):
    x, x_rep, to_mask = self.encode_masked(image, train=train, debug=debug)
    x_with_pos_emb = self._decoder_posembed(x[:, 1:])
    x_hat = self._decoder(x_with_pos_emb, train=train)

    return x_hat, x_rep, to_mask[:, 1:] - 1


class MultiViewMaskedAutoEncoder(nn.Module):
  """Masked Auto Encoder model."""

  config: ml_collections.ConfigDict
  dtype: jnp.dtype = jnp.float32

  def to_multiple(self, x: FloatArray['... H W C']) -> FloatArray['... h w C']:
    return misc.crop_to_multiple(x, self.config.patch_height)

  def preprocess_pixels(
      self, image_pixels: FloatArray['B N H W C']
  ) -> FloatArray['B*N h w C']:
    """Flatten, normalize and enforce size to be a multiple of patch size."""
    image_pixels_flat = (
        jnp.reshape(image_pixels, (-1, *image_pixels.shape[-3:])) * 2.0 - 1.0
    )
    return self.to_multiple(image_pixels_flat)

  def bn_to_bxn(
      self, n: int, *x_bn: FloatArray['B*N h w C']
  ) -> FloatArray['B N h w C']:
    def to_bxn(x):
      return x.reshape(-1, n, *x.shape[1:])

    if len(x_bn) == 1:
      return to_bxn(x_bn[0])
    return [to_bxn(x) for x in x_bn]

  def map_to_query(
      self,
      grid_id_query: FloatArray['B M L S'],
      grid_id_map: FloatArray['B N L S'],
      features_map: FloatArray['B N L C'],
  ) -> tuple[FloatArray['B M L C'], FloatArray['B M L 1']]:
    x_warped_q, occupancy_q = grids.gather_features_from_map_batched(
        grid_id_query,
        grid_id_map,
        features_map,
    )
    occupancy_q = occupancy_q[..., None]
    return x_warped_q, occupancy_q

  def get_grid_ids_per_patch(
      self, batch: types.Batch, topk: int = 8
  ) -> IntArray['B N L S']:
    if 'grid_3d_id_best' in batch:
      return jax.lax.collapse(batch['grid_3d_id_best'], -3, -1)
    else:
      ph, pw = self.config.patch_height, self.config.patch_width
      grid_ids = misc.reshape_image_to_patches(
          self.to_multiple(batch['grid_3d_id']), ph, pw
      )
      grid_ids_best, _ = grids.gather_top_k_ids(
          grid_ids, top_k=topk, start_dim=-3
      )  # b x m x l x k
      return grid_ids_best

  @nn.compact
  def __call__(
      self, batch: types.Batch, *, train: bool, debug: bool = False
  ) -> types.Predictions:
    del debug
    query, maps = batch['query']['views'], batch['map']['views']

    query_pixels = query['image_pixels']
    map_pixels = maps['image_pixels']

    n_query = query_pixels.shape[-4]
    n_map = map_pixels.shape[-4]

    query_pixels_flat = self.preprocess_pixels(query_pixels)
    map_pixels_flat = self.preprocess_pixels(map_pixels)

    h, w, c = query_pixels_flat.shape[-3:]
    nh, nw = h // self.config.patch_height, w // self.config.patch_width
    n_tokens = nh * nw
    n_masked = np.floor(n_tokens * self.config.masking_rate).astype('int32')

    mae = _MaskedAutoEncoder(
        input_kernel=None,
        encoder_config=self.config.encoder,
        decoder_config=self.config.decoder,
        patches_hw=(self.config.patch_height, self.config.patch_width),
        hidden_size=self.config.hidden_size,
        dropout_rate=self.config.dropout_rate,
        attention_dropout_rate=self.config.attention_dropout_rate,
        stochastic_droplayer_rate=self.config.stochastic_droplayer_rate,
        n_tokens=n_tokens + 1,
        n_masked=n_masked,
        num_image_channels=c,
        dtype=self.dtype,
    )

    # Encode map
    encoded_m = mae._encode_image(map_pixels_flat, train=train)  # bn x 1+l x c

    if self.config.get('encoding_dim'):
      project = vit_dino.ProjectionHead(
          output_dim=self.config.encoding_dim, normalize_last_layer=True
      )
    else:
      project = lambda x, train: x

    features_m = project(encoded_m, train)  # With class token

    if self.config.stop_gradient_map:
      features_m = jax.lax.stop_gradient(features_m)
    features_m = self.bn_to_bxn(n_map, features_m)

    if self.config.subtract_center_from_map:
      features_m = features_m - features_m.mean((-4, -3, -2), keepdims=True)

    # Encode query
    encoded_q, encoded_rep_q, mask_q = mae.encode_masked(
        query_pixels_flat, train=train
    )
    features_q = project(encoded_q, train)  # With class token
    features_with_pos_emb_q = mae._decoder_posembed(encoded_q[:, 1:])
    x_hat_q = mae._decoder(features_with_pos_emb_q, train=train)

    mask_q = mask_q[:, 1:] - 1

    # Create boolean mask for tokens.
    tokens_masked_binary = jnp.zeros_like(x_hat_q[..., :1], dtype=bool)
    batch_size = query_pixels.shape[0]
    all_rows = jnp.arange(batch_size).reshape(batch_size, 1)
    tokens_masked_binary = tokens_masked_binary.at[all_rows, mask_q].set(True)

    features_q, encoded_rep_q, mask_q, x_hat_q, tokens_masked_binary = (
        self.bn_to_bxn(
            n_query,
            features_q,
            encoded_rep_q,
            mask_q,
            x_hat_q,
            tokens_masked_binary,
        )
    )

    pred = {
        'x_hat_query': x_hat_q,
        'x_rep_query': encoded_rep_q,
        'tokens_masked_query': mask_q,
        'query_pixels': self.bn_to_bxn(n_query, query_pixels_flat),
        'tokens_masked_binary_query': tokens_masked_binary,
        'features_map': features_m[..., 1:, :],
        'features_query': features_q[..., 1:, :],
        'map_pixels': self.bn_to_bxn(n_map, map_pixels_flat),
    }

    if self.config.align is None:
      return pred
    elif self.config.align == 'image':
      features_m_to_q, occupancy_q = self.map_to_query(
          self.get_grid_ids_per_patch(query),
          self.get_grid_ids_per_patch(maps),
          features_m[..., 1:, :],  # remove class token
      )

      do_align = (occupancy_q > 0) * (~tokens_masked_binary)

      similarity_q = (
          misc.l2_normalize(features_q[..., 1:, :])
          * misc.l2_normalize(features_m_to_q)
      ).sum(-1, keepdims=True)

      center_axes = -2
      center_q = misc.l2_normalize(features_q[..., 1:, :]).mean(
          axis=center_axes, keepdims=True
      )
      return {
          **pred,
          'do_align': do_align,
          'features_warped_query': features_m_to_q,
          'occupancy_query': occupancy_q,
          'similarity': similarity_q,
          'center_q': center_q,
      }
    elif self.config.align == 'grid':
      grid_id_query = self.get_grid_ids_per_patch(query)
      grid_id_query = jnp.where(tokens_masked_binary, -1, grid_id_query)
      grid_id_map = self.get_grid_ids_per_patch(maps)
      warped = grids.gather_features_via_grid_batched(
          grid_id_query,
          grid_id_map,
          features_m[..., 1:, :],
          max_num_cells=1000,
          max_num_observations=40,
      )

      features_m_grid = warped['features_grid']
      occupancy_m_grid = warped['occupancy_grid'][..., None]
      occupancy_q = warped['occupancy_query'][..., None]
      features_m_to_q = warped['features_query']
      grid_3d_ids = warped['grid_ids'][..., None]

      features_q_grid, occupancy_q_grid = (
          grids.aggregate_features_in_grid_batched(
              warped['grid_ids'], grid_id_query, features_q[..., 1:, :], 40
          )
      )
      occupancy_q_grid = occupancy_q_grid[..., None]

      similarity_grid = (
          misc.l2_normalize(features_q_grid)
          * misc.l2_normalize(features_m_grid)
      ).sum(-1, keepdims=True)
      do_align_grid = (occupancy_q_grid > 0) & (occupancy_m_grid > 0)

      similarity_q = (
          misc.l2_normalize(features_q[..., 1:, :])
          * misc.l2_normalize(features_m_to_q)
      ).sum(-1, keepdims=True)
      do_align_q = (~tokens_masked_binary) & (occupancy_q > 0)

      return {
          **pred,
          'similarity_grid': similarity_grid,
          'do_align_grid': do_align_grid,
          'features_q_grid': features_q_grid,
          'features_m_grid': features_m_grid,
          'grid_id_query': grid_id_query,
          'unique_grid_id_query': warped['unique_grid_id_query'],
          'occupancy_q_grid': occupancy_q_grid,
          'occupancy_m_grid': occupancy_m_grid,
          'grid_3d_ids': grid_3d_ids,
          'similarity': similarity_q,
          'do_align': do_align_q,
      }

    else:
      raise ValueError(f'Unknown align mode: {self.config.align}')


class MultiViewMaskedAutoEncoderModel(base.BaseModel):
  """Masked Auto Encoder model."""

  required_rngs: tuple[str, ...] = ('masking_rng', 'dropout')

  def build_flax_model(self) -> nn.Module:
    return MultiViewMaskedAutoEncoder(
        config=self.config,
        dtype=self.dtype,
    )

  def loss_metrics_function(
      self,
      pred: types.Predictions,
      data: types.Batch,
      model_params: jnp.ndarray | None = None,
  ) -> types.LossMetricsTuple:
    x, weights = fomo_trainer._patch_inputs(  # pylint: disable=protected-access
        jax.lax.collapse(pred['query_pixels'], -5, -3),
        jax.lax.collapse(pred['tokens_masked_query'], -3, -1),
        patch_height=self.config.patch_height,
        patch_width=self.config.patch_width,
        include_unmasked_tokens_in_loss=self.config.include_unmasked_tokens_in_loss,
    )
    loss = fomo_trainer.regression_loss_with_token_masking(
        predictions=jax.lax.collapse(pred['x_hat_query'], -4, -2),
        targets=x,
        weights=weights,
    )

    losses = {'reconstruction': loss}
    metrics = {}

    if 'similarity' in pred:
      mse = 1.0 - pred['similarity']
      losses['alignment'] = misc.masked_mean(
          mse, pred['do_align'], axis=(-3, -2, -1)
      )
      metrics['can_align'] = pred['do_align'].sum((-3, -2, -1))

    if 'matching' in self.config.compose_loss:
      losses['grid_matching'] = grids.grid_matching_loss(
          pred['features_q_grid'],
          pred['features_m_grid'],
          pred['do_align_grid'],
      )

      losses['query_to_grid_matching'] = grids.query_to_grid_matching_loss(
          pred['unique_grid_id_query'],
          pred['features_query'],
          pred['features_m_grid'],
          pred['do_align_grid'],
      )

    if 'similarity_grid' in pred:
      losses['alignment_grid'] = misc.masked_mean(
          1.0 - pred['similarity_grid'],
          pred['do_align_grid'],
          axis=(-2, -1),
      )

    if 'features_warped_query' in pred:
      ce = misc.cross_entropy(
          pred['features_query'] / self.config.patch_temperature,
          pred['features_warped_query'] / self.config.patch_temperature,
      )
      losses['cross_entropy'] = misc.masked_mean(
          ce, pred['do_align'], axis=(-3, -2, -1)
      )

    if 'center_q' in pred:
      losses['center_q_max'] = jnp.abs(pred['center_q']).max(-1).mean(axis=-1)

    metrics['can_align'] = pred['do_align'].sum((-3, -2, -1))
    losses['alignment_raw'] = misc.masked_mean(
        1.0 - pred['similarity'], pred['do_align'], axis=(-3, -2, -1)
    )

    features_map_l2 = misc.l2_normalize(pred['features_map'])
    metrics['features_map_l2_max'] = (
        jnp.abs(features_map_l2)
        .max(-1, keepdims=True)
        .mean(
            axis=(-3, -2, -1),
        )
    )
    std = jnp.std(features_map_l2, axis=-2)  # over the tokens in an image
    metrics['features_map_l2_stdnorm'] = jnp.linalg.norm(std, axis=-1).mean(
        axis=-1
    )
    losses['total'] = types.compose_loss(losses, self.config.compose_loss)
    return losses, metrics

  def image_like(self, pred: types.Predictions) -> types.Predictions:
    h, w = pred['query_pixels'].shape[-3:-1]
    nh, nw = h // self.config.patch_height, w // self.config.patch_width
    nt = nh * nw

    return {
        k: misc.sequence_to_image(v, nh, nw) if v.shape[-2] == nt else v
        for k, v in pred.items()
    }

  def visualize(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      max_num_plots: int = 4,
      similarity_range: tuple[float, float] = (0.0, 1.0),
      **kwargs,
  ) -> Mapping[str, FloatArray['B H W 3']]:
    visualizations = {}
    visualizations['fomo'] = fomo_trainer._compose_images(  # pylint: disable=protected-access
        dataset_utils.shard(jax.lax.collapse(pred['query_pixels'], -5, -3)),
        dataset_utils.shard(jax.lax.collapse(pred['x_hat_query'], -4, -2)),
        config=ml_collections.ConfigDict(
            dict(
                use_passthrough_input_for_unmasked_tokens=self.config.use_passthrough_input_for_unmasked_tokens,
                include_unmasked_tokens_in_loss=self.config.include_unmasked_tokens_in_loss,
                model=dict(
                    patch_height=self.config.patch_height,
                    patch_width=self.config.patch_width,
                ),
                max_images_summary=max_num_plots,
            ),
        ),
        masked_token_indices=dataset_utils.shard(
            jax.lax.collapse(pred['tokens_masked_query'], -3, -1),
        ),
    )

    pred = self.image_like(pred)

    if 'similarity' in pred:
      similarity_image = plot.apply_colormap(
          pred['similarity'][..., 0],
          cmap='turbo_r',
          vmin=similarity_range[0],
          vmax=similarity_range[1],
      )

      similarity_image = jax.image.resize(
          similarity_image,
          pred['query_pixels'].shape,
          method=jax.image.ResizeMethod.NEAREST,
      )

      mask = jax.image.resize(
          pred['do_align'].astype(jnp.float32),
          pred['query_pixels'].shape,
          method=jax.image.ResizeMethod.NEAREST,
      )

      similarity_overlay = misc.alpha_blend(
          misc.to_grayscale(pred['query_pixels'] * 0.5 + 0.5),
          similarity_image,
          mask * 0.7,
      )

      features_map = pred['features_map']
      pca_map = misc.features_to_rgb(features_map)
      pca_images = jax.image.resize(
          pca_map,
          pred['map_pixels'].shape,
          method=jax.image.ResizeMethod.NEAREST,
      )

      map_images = pred['map_pixels'] * 0.5 + 0.5

      map_pca_images = jnp.concatenate(
          [similarity_overlay, pca_images, map_images], axis=-4
      )
      visualizations['map_pca'] = jnp.concatenate(
          map_pca_images.swapaxes(0, 1), -2
      )[:max_num_plots]

    return visualizations

  def visualize_3d(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      batch_idx: int = 0,
      width: int = 1000,
      height: int = 600,
      **kwargs,
  ) -> viz3d.Plotter3D:
    batch_i = jax.tree.map(lambda x: x[batch_idx], batch)
    pred_i = jax.tree.map(lambda x: x[batch_idx], pred)

    grid: grids.Grid3D = self.dataset_meta_data['grid_3d']
    plotter = viz3d.Plotter3D(
        show_axes=True,
        scene_t_w=batch_i['scene_t_w'],
        height=height,
        width=width,
        **kwargs,
    )

    for collection_idx in ['query', 'map']:
      scene_i = batch_i[collection_idx]
      w_t_cam = jnp.stack(
          [
              jnp.linalg.inv(cami_t_w)
              for cami_t_w in scene_i['views']['cam_t_w']
          ],
          0,
      )
      intrinsics = scene_i['views']['camera']['K']
      color = 'black' if collection_idx == 'query' else 'green'
      for i in range(w_t_cam.shape[0]):
        plotter.add_camera(
            w_t_cam[i],
            intrinsics[i],
            legendgroup=collection_idx,
            name=f'{collection_idx}/{i}',
            color=color,
            size=20.0,
        )

    grid_3d_ids = pred_i['grid_3d_ids']
    valid = pred_i['do_align_grid']
    plotter.add_voxel_grid(
        grid.id_to_index(grid_3d_ids[valid]) * grid.cell_size,
        jnp.linalg.inv(batch_i['grid_t_w']),
        grid.cell_size,
        mesh_color=plot.apply_colormap(
            pred_i['similarity_grid'][valid], vmin=0.0, vmax=1.0
        ),
    )
    return plotter
