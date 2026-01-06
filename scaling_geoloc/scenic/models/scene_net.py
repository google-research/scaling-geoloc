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

"""Occupancy network with voxel rendering."""

from collections.abc import Mapping
import functools
from typing import Any
from etils.array_types import Array
from etils.array_types import FloatArray
import flax.linen as nn
import jax
from jax.experimental import sparse as jax_sparse
import jax.numpy as jnp
import jaxcam
import numpy as np
from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.models import dino
from scaling_geoloc.utils import geometry
from scaling_geoloc.utils import grids
from scaling_geoloc.utils import misc
from scaling_geoloc.utils import plot
from scaling_geoloc.utils import rendering
from scaling_geoloc.viz import viz3d
from google3.location.visualmapping.experimental.lemming.models import multiview_encoder


gather_observations = jax.vmap(
    geometry.project_points, in_axes=(0, None), out_axes=0
)

batched_gather_observations = jax.vmap(gather_observations, in_axes=(0, 0))

batched_render_views_from_grid = jax.vmap(
    rendering.render_views_from_grid,
    in_axes=(0, 0, None, None, None),  # Map over batch dims.
)

batched_render_views_from_grid_with_prior = jax.vmap(
    rendering.render_views_from_grid,
    in_axes=(0, 0, None, None, 0),  # Map over batch dims.
)


@jax.checkpoint
@functools.partial(jax.vmap, in_axes=0)  # batch
@functools.partial(jax.vmap, in_axes=0, out_axes=0)  # views
def interpolate_views_all(
    array: FloatArray['H W D'], points: FloatArray['N 2']
) -> FloatArray['N D']:
  """Interpolation of all observations in all views."""
  interp, _ = grids.interpolate_nd(array, points)
  return interp


def make_sparse_grid(
    f_grid: grids.FeatureGrid,
) -> tuple[jax_sparse.BCOO, jax_sparse.BCOO]:
  """Create sparse grid with occupancy."""
  grid_features_sparse = jax.vmap(grids.sparse_volume)(
      f_grid.grid, f_grid.grid_ids, f_grid.features
  )

  grid_occupancies_sparse = jax.vmap(grids.sparse_volume)(
      f_grid.grid, f_grid.grid_ids, f_grid.opacities
  )
  return grid_features_sparse, grid_occupancies_sparse


def _sparse_sum(x: jax_sparse.BCOO, axis: int) -> Array:
  return jax_sparse.sparsify(functools.partial(jnp.sum, axis=axis))(x).todense()


class SceneNet(dino.ViTDINO):
  """Occupancy network with voxel rendering."""

  @nn.compact
  def __call__(
      self,
      batch: types.Batch,
      train: bool = False,
      debug: bool = False,
  ) -> types.Predictions:

    views = {}

    # encode images
    batch = misc.to_flat_batch(batch, stride=self.config.vit.patches.size[0])

    grid = grids.Grid3D.from_bounding_box(
        batch['map']['lidar_extent_meters'],
        self.config.grid_extent,
    )

    views['student_views'] = batch['image_pixels']
    views['teacher_views'] = views['student_views']

    if train:
      encode = jax.checkpoint(self.encode, static_argnums=(1, 2))
    else:
      encode = self.encode
    pred = encode(views, train, debug)
    bs, _, h, w = batch['image_pixels'].shape[:-1]
    ph, pw = self.config.vit.patches.size
    nh, nw = h // ph, w // pw
    features = pred['student_tokens'][..., 1:, :]
    f_dim = features.shape[-1]
    # Resize features to image size.

    up_factor = self.config.get('learned_upsample_factor', 4)
    if up_factor > 1:
      out_channels = f_dim * up_factor**2
      up_net = nn.Sequential([
          nn.Dense(f_dim),
          nn.relu,
          nn.Dense(out_channels),
      ])
      features_image = up_net(features)
      features_image = jnp.reshape(
          features_image,
          features_image.shape[:-1] + (up_factor, up_factor, f_dim),
      )
      features_image = misc.reshape_patches_to_image(
          features_image, nh, nw, image_like=False
      )
    else:
      features_image = misc.sequence_to_image(features, nh, nw)

    if self.config.get('upsample_features', False):
      features_image = jax.image.resize(
          features_image,
          pred['student_views'].shape[:-1] + features_image.shape[-1:],
          method='linear',
      )

    if self.config.get('weighted_fusion', True):
      fusion_mlp = nn.Sequential([
          nn.Dense(f_dim),
          nn.relu,
          nn.Dense(f_dim + self.config.num_depth_bins),
      ])
      features_image = fusion_mlp(features_image)

    # Create cameras.
    jcamera_grid = types.create_jaxcam(
        batch['cam_t_w'] @ jnp.linalg.inv(batch['grid_t_w'][:, None]),
        batch['camera'],
        is_fisheye=True,
        use_inverted_distortion=False,
    )

    # Create (sparse) grid.
    if self.config.get('query_from_lidar', True):
      grid_xyz = jax.vmap(misc.lidar_depth_to_xyz)(
          batch['lidar_depth'], jcamera_grid
      )
      grid_ids, _ = jax.vmap(grids.voxelize_points, in_axes=(0, 0, 0))(
          grid_xyz,
          batch['lidar_mask'],
          grid,
      )

      # Extract top-k unique grid ids.
      unique_grid_ids = grids.extract_topk_unique_ids(
          grid_ids, self.config.max_num_voxels
      )

      valid_grid_ids = unique_grid_ids > -1
      if self.config.get('pad_grid', False):
        pad_rng = jax.random.split(self.make_rng('pad_grid'), bs)
        unique_grid_ids = jax.vmap(grids.fill_missing_ids, in_axes=(0, 0, 0))(
            pad_rng, grid, unique_grid_ids
        )

    else:
      # Take full (dense) grid.
      all_grid_ids = jax.vmap(lambda g: g.index_to_id(g.flat_grid_index()))(
          grid
      )
      unique_grid_ids = all_grid_ids
      valid_grid_ids = unique_grid_ids > -1

    pred = pred | {
        'valid_grid_ids': valid_grid_ids,
        'grid_ids_query': unique_grid_ids,
        'grid_index_query': jax.vmap(lambda grid, v: grid.id_to_index(v))(
            grid, unique_grid_ids
        ),
    }

    pred['grid_xyz_query'] = jax.vmap(lambda grid, v: grid.index_to_xyz(v))(
        grid, pred['grid_index_query']
    )

    # Gather observations.
    p2d_xy, visible, depth = batched_gather_observations(
        jcamera_grid, pred['grid_xyz_query']
    )
    fh, fw = features_image.shape[-3:-1]
    sh, sw = h / fh, w / fw  # strides
    p2d_ij = jnp.flip(p2d_xy, axis=-1) / jnp.array([sh, sw], dtype=p2d_xy.dtype)

    # Dense interpolation in Image.
    features_proj = interpolate_views_all(features_image, p2d_ij)

    if features_proj.shape[-1] > f_dim:
      features_proj, scores_scales = jnp.split(features_proj, [f_dim], axis=-1)
      scores_proj = multiview_encoder.interpolate_depth_score(
          scores_scales, depth[..., 0], self.config.depth_min_max
      ).swapaxes(-1, -2)
    else:
      scores_proj = None

    f_pooled, valid = multiview_encoder.pool_multiview_features(
        features_proj.swapaxes(-2, -3),
        visible.swapaxes(-2, -3)[..., 0],
        scores_proj,
        self.config.get('fusion_add_minmax', False),
        self.config.get('fusion_use_variance', False),
    )

    f_grid = f_pooled
    f_grid = jnp.where(valid[..., None], f_grid, 0)
    # Render features back to images.
    grid_id = pred['grid_ids_query']
    jcam = types.create_jaxcam(
        batch['cam_t_w'],
        batch['camera'],
        is_fisheye=True,
        use_inverted_distortion=True,
    )

    if self.config.get('predict_opacities', False):
      opacity_head = nn.Sequential(
          [
              nn.Dense(f_grid.shape[-1]),
              nn.relu,
              nn.Dense(1),
          ],
          name='opacity_head',
      )
      opacities = nn.sigmoid(opacity_head(f_grid))
    else:
      # opacities = jnp.ones_like(grid_id[..., None], dtype=f_grid.dtype)
      opacities = valid_grid_ids[..., None].astype(f_grid.dtype)

    # Build feature grid.
    feature_grid = grids.FeatureGrid(
        grid=grid,
        grid_t_w=batch['grid_t_w'],
        grid_ids=grid_id,
        features=f_grid,
        opacities=opacities,
    )

    feature_grid_sparse, occupancy_grid_sparse = make_sparse_grid(feature_grid)

    bev_features = _sparse_sum(feature_grid_sparse, -2)
    bev_occupancies = _sparse_sum(occupancy_grid_sparse, -2)
    bev_features = bev_features / bev_occupancies.clip(min=1.0)

    pred = pred | {
        'bev_features': bev_features,
        'bev_occupancies': bev_occupancies,
        'image_features': features_image,
        'grid_features': f_grid,
    }

    render_fn, depth_prior = batched_render_views_from_grid, None
    depth_range = jnp.linspace(*self.config.depth_range)
    if self.config.get('use_depth_prior', True):
      render_fn = batched_render_views_from_grid_with_prior
      depth_prior = batch['lidar_depth']
    else:
      depth_range = 10**depth_range

    rendered_views = render_fn(
        jcam,
        feature_grid,
        (h, w, self.config.render_stride),
        depth_range,
        depth_prior,
    )

    rendered_views['expected_depth'] = jax.vmap(
        jax.vmap(jaxcam.world_points_to_local_points)
    )(jcam, rendered_views['expected_xyz'])[..., -1:]

    if self.config.get('add_sky_token', True):
      # Alpha blend with sky token.
      shared_sky_token = self.param(
          'shared_sky_token',
          nn.initializers.normal(stddev=0.02),
          rendered_views['expected_feature'].shape[-1:],
      )
      alpha = rendered_views['alpha']
      rendered_features = (
          alpha * rendered_views['expected_feature']
          + (1 - alpha) * shared_sky_token
      )
    else:
      rendered_features = rendered_views['expected_feature']

    strided_image_features = jax.image.resize(
        features_image,
        rendered_features.shape,
        method='linear',
    )

    pred = (
        pred
        | rendered_views
        | dict(
            feature_grid=feature_grid,
            strided_image_features=strided_image_features,
            expected_feature=rendered_features,
            jcam=jcam,
        )
    )

    return pred


class SceneNetModel(dino.ViTDINOModel):
  """Trainer facing wrapper for ViTDINO."""

  required_rngs: tuple[str, ...] = ('droptok', 'dropout', 'pad_grid')

  def build_flax_model(self) -> nn.Module:
    return SceneNet(
        config=self.config,
        dtype=self.dtype,
    )

  def update_state(
      self,
      pred: types.Predictions,
      model_params: types.ModelParams,
      model_state: types.ModelState,
      learning_rates: types.LearningRates,
  ) -> tuple[types.ModelParams, types.ModelState]:
    return model_params, model_state

  def loss_metrics_function(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      model_params: jnp.ndarray | None = None,
      model_state: Any | None = None,
      rate_params: Mapping[str, float] | None = None,
  ) -> types.LossMetricsTuple:
    # assert rate_params is not None
    batch = misc.to_flat_batch(batch, stride=self.config.vit.patches.size[0])
    losses, metrics = {}, {}

    s_features = pred['strided_image_features']
    r_features = pred['expected_feature']
    r_valid = pred['any_hit']  # Only supervise where at least one voxel is hit.
    sr_sim_l2 = misc.compute_similarity(s_features, r_features, normalize=True)[
        ..., 0
    ]
    sr_sim = misc.compute_similarity(s_features, r_features, normalize=False)[
        ..., 0
    ]

    losses['rendered_feature_similarity_l2'] = 1 - misc.masked_bmean(
        sr_sim_l2, r_valid[..., 0]
    )
    losses['rendered_feature_similarity'] = 1 - misc.masked_bmean(
        sr_sim, r_valid[..., 0]
    )

    r_depth = pred['expected_depth']
    gt_depth = jax.image.resize(batch['lidar_depth'], r_depth.shape, 'nearest')
    gt_mask = jax.image.resize(batch['lidar_mask'], r_depth.shape, 'nearest')
    gt_xyz = jax.image.resize(
        misc.batched_lidar_depth_to_xyz(batch['lidar_depth'], pred['jcam']),
        r_depth.shape[:-1] + (3,),
        'nearest',
    )

    eps = 1.0e-3

    losses['log_depth_l1'] = misc.masked_bmean(
        jnp.abs(
            jnp.log(r_depth.clip(min=eps)) - jnp.log(gt_depth.clip(min=eps))
        ),
        gt_mask & r_valid,
    )
    losses['depth_l1'] = misc.masked_bmean(
        jnp.abs(r_depth - gt_depth), gt_mask & r_valid
    )
    losses['depth_l2'] = misc.masked_bmean(
        jnp.square(r_depth - gt_depth), gt_mask & r_valid
    )

    metrics['num_valid_voxels'] = jnp.sum(pred['valid_grid_ids'], axis=-1)

    losses['alpha_nll'] = misc.masked_bmean(
        -jnp.log(pred['alpha']), gt_mask & r_valid
    )
    metrics['alpha_valid'] = misc.masked_bmean(pred['alpha'], gt_mask & r_valid)

    metrics['grid_valid_opacities'] = misc.masked_bmean(
        pred['feature_grid'].opacities, pred['valid_grid_ids']
    )
    metrics['grid_invalid_opacities'] = misc.masked_bmean(
        pred['feature_grid'].opacities, ~pred['valid_grid_ids']
    )

    def xyz_to_voxel_id(feat_grid, w_xyz):
      grid_xyz = jnp.einsum(
          'ij,...j->...i', feat_grid.grid_t_w[:3, :], misc.to_homogeneous(w_xyz)
      )
      grid = feat_grid.grid
      return grid.index_to_id(grid.xyz_to_index(grid_xyz))[..., None]

    feat_grid = pred['feature_grid']
    gt_voxel_ids = jax.vmap(xyz_to_voxel_id)(feat_grid, gt_xyz)
    r_voxel_ids = jax.vmap(xyz_to_voxel_id)(feat_grid, pred['expected_xyz'])
    metrics['correct_voxel'] = misc.masked_bmean(
        (gt_voxel_ids == r_voxel_ids), gt_mask & r_valid
    )
    opc = pred['feature_grid'].opacities
    voxel_pos = misc.masked_bmean(-jnp.log(opc), pred['valid_grid_ids'])
    voxel_neg = misc.masked_bmean(-jnp.log(1 - opc), ~pred['valid_grid_ids'])
    losses['voxel_nll'] = 0.5 * (voxel_pos + voxel_neg)
    losses['total'] = types.compose_loss(losses, self.config.compose_loss)
    return losses, metrics

  def eval_loss_metrics_function(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      model_params: jnp.ndarray | None = None,
  ) -> types.LossMetricsTuple:
    return self.loss_metrics_function(pred, batch, model_params=model_params)

  def visualize(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      max_num_plots: int = 8,
      stack_axis: int = -2,
      vmax: float = 20.0,
      **kwargs,
  ) -> Mapping[str, FloatArray['B H W 3']]:
    visualizations = {}

    batch = misc.to_flat_batch(batch, stride=self.config.vit.patches.size[0])
    images = pred['student_views']
    h, w = images.shape[-3:-1]

    def resize(x, method):
      return jax.image.resize(
          x,
          x.shape[:-3] + (h, w) + x.shape[-1:],
          method=method,
      )

    s_features = pred['strided_image_features']
    r_features = pred['expected_feature']
    r_valid = pred['any_hit']
    sr_sim = misc.compute_similarity(s_features, r_features, normalize=True)[
        ..., 0
    ]

    s_features = resize(misc.features_to_rgb(s_features), 'nearest')
    r_features = resize(
        misc.features_to_rgb(r_features, mask=r_valid[..., 0]),
        'nearest',
    )

    sr_sim = plot.apply_colormap(sr_sim, cmap='turbo_r', vmin=0.0, vmax=1.0)
    sr_sim = misc.alpha_blend(0.0, sr_sim, r_valid)
    sr_sim = resize(sr_sim, 'nearest')

    r_depth_cmap = plot.apply_colormap(
        pred['expected_depth'][..., 0], cmap='turbo_r', vmin=0.0, vmax=vmax
    )
    r_depth_cmap = resize(r_depth_cmap, 'nearest')

    depth_errmap = plot.apply_colormap(
        jnp.abs(
            resize(pred['expected_depth'], 'nearest') - batch['lidar_depth']
        )[..., 0],
        cmap='turbo',
        vmin=0.0,
        vmax=5.0,
        mask=(batch['lidar_mask'] & resize(r_valid, 'nearest'))[..., 0],
    )

    gt_depth_map = plot.apply_colormap(
        batch['lidar_depth'][..., 0], cmap='turbo_r', vmin=0.0, vmax=vmax
    )
    images = misc.alpha_blend(images, gt_depth_map, batch['lidar_mask'])

    visualizations['features'] = jnp.concatenate(
        jnp.concatenate(
            [
                images,
                s_features,
                r_features,
                sr_sim,
                r_depth_cmap,
                depth_errmap,
            ],
            axis=-3,
        ).swapaxes(0, 1),
        axis=-2,
    )[:max_num_plots]

    bev_features = pred['bev_features']
    bev_occupancies = pred['bev_occupancies']

    visualizations['bev_features'] = misc.features_to_rgb(
        bev_features[:max_num_plots],
        mask=bev_occupancies[:max_num_plots, ..., 0] > 0.0,
    )

    visualizations['bev_occupancies'] = bev_occupancies.repeat(3, -1)[
        :max_num_plots
    ]

    return visualizations

  def visualize_3d(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      batch_idx: int = 0,
      width: int = 1000,
      height: int = 600,
      max_num_voxels: int = 6_000,
      max_num_points: int = 100_000,
      plot_semantic_points: bool = False,
      add_voxels: bool = False,
  ) -> viz3d.Plotter3D:
    batch = misc.to_flat_batch(batch, stride=self.config.vit.patches.size[0])
    batch_i = jax.tree.map(lambda x: x[batch_idx], batch)
    pred_i = jax.tree.map(lambda x: x[batch_idx], pred)

    grid = pred_i['feature_grid'].grid
    plotter = viz3d.Plotter3D(
        height=height,
        width=width,
        show_axes=True,
        scene_t_w=batch_i['grid_t_w'],
    )

    all_grid_ids = pred_i['feature_grid'].grid_ids
    grid_mask = (all_grid_ids > -1) & (
        pred_i['feature_grid'].opacities[..., 0] > 0
    )
    grid_features = pred_i['feature_grid'].features
    features_pca = misc.features_to_rgb(grid_features[grid_mask])

    grid_index = grid.id_to_index(all_grid_ids[grid_mask])
    if add_voxels:
      vox_sample = np.random.choice(
          grid_mask.sum(),
          min(grid_mask.sum(), max_num_voxels),
          replace=False,
      )
      plotter.add_voxel_grid(
          grid_index[vox_sample] * grid.cell_size,
          np.linalg.inv(batch_i['grid_t_w']),
          grid.cell_size,
          mesh_color=features_pca[vox_sample],
      )

    grid_xyz = grid.index_to_xyz(grid_index)
    w_xyz = np.einsum(
        'ij,nj->ni',
        np.linalg.inv(batch_i['grid_t_w'])[:3, :],
        misc.to_homogeneous(grid_xyz),
    )
    plotter.add_points(
        w_xyz,
        color=features_pca,
        name='grid_features',
        max_num_samples=max_num_points,
        size=4,
        colorscale=None,
    )

    plotter.add_points(
        w_xyz,
        color=pred_i['feature_grid'].opacities[grid_mask, 0] > 0.5,
        name='opacities',
        max_num_samples=max_num_points,
        size=4,
        visible='legendonly',
        colorscale='greys',
        cmin=0.0,
        cmax=1.0,
    )

    plotter.add_cameras(
        np.linalg.inv(batch_i['cam_t_w']),
        batch_i['camera']['K'],
        size=10.0,
        color='black',
    )

    if plot_semantic_points:
      jcamera = types.create_jaxcam(
          batch_i['cam_t_w'],
          batch_i['camera'],
          is_fisheye=True,
          use_inverted_distortion=True,
      )
      w_points = misc.lidar_depth_to_xyz(batch_i['lidar_depth'], jcamera)
      w_mask = batch_i['lidar_mask'][..., 0]

      plotter.add_labelled_points(
          w_points[w_mask],
          batch_i['lidar_semantic_labels'][w_mask, 0],
          label_convert_fn=lambda idx: types.SEMANTIC_INDEX_TO_NAME[idx],
          max_num_samples=50_000,
          legendgroup='lidar',
      )

    return plotter
