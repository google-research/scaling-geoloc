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

"""Dino multiview with distillation."""

from collections.abc import Mapping
from typing import Any
from etils.array_types import FloatArray
from etils.array_types import IntArray
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.models import dino
from scaling_geoloc.utils import gaussians
from scaling_geoloc.utils import grids
from scaling_geoloc.utils import misc
from scaling_geoloc.utils import plot
from scaling_geoloc.viz import viz3d


class ViTMultiview(dino.ViTDINO):
  """DINO multiview with distillation."""

  def encode_multiview(
      self,
      images: FloatArray['... H W C'],
      *args,
      **kwargs,
  ) -> tuple[jt.Float[jt.Array, '... D'], jt.Float[jt.Array, '... L D']]:
    """Interface to backbone encoder."""
    return self.student.encode_multiview(images, *args, **kwargs)

  def encode_image(
      self, image: jt.Float[jt.Array, 'B H W C'], *, train: bool
  ) -> tuple[jt.Float[jt.Array, 'B h w c'], jt.Float[jt.Array, 'B c']]:
    """Interface to FOMO evals."""
    _, bb_output = self.encode_multiview(image, train=train)
    cls_tok = bb_output[..., 0, :]
    h, w = image.shape[-3:-1]
    ph, pw = self.config.vit.patches.size
    nh, nw = h // ph, w // pw
    patch_tok = misc.sequence_to_image(bb_output[..., 1:, :], nh, nw)
    return patch_tok, cls_tok

  def __call__(
      self,
      batch: types.Batch,
      train: bool = False,
      debug: bool = False,
  ) -> types.Predictions:

    views = {}
    teacher_batch = (
        batch
        if self.config.align_to == 'map+query'
        else batch[self.config.align_to]
    )
    views['teacher_views'] = misc.concat_all(
        'image_pixels',
        teacher_batch,
        axis=-4,
    )

    views['student_views'] = misc.concat_all(
        'image_pixels', batch['query'], axis=-4
    )
    return self.encode(views, train=train, debug=debug)


class ViTMultiviewModel(dino.ViTDINOModel):
  """Trainer facing wrapper for ViTDINO."""

  required_rngs: tuple[str, ...] = ('droptok', 'dropout')

  def build_flax_model(self) -> nn.Module:
    return ViTMultiview(
        config=self.config,
        dtype=self.dtype,
    )

  def get_grid_ids_per_patch(
      self,
      batch: types.Batch,
      topk: int = 8,
      force_compute: bool = False,
  ) -> IntArray['B N L S']:
    if 'grid_3d_id_best' in batch and not force_compute:
      return jax.lax.collapse(batch['grid_3d_id_best'], -3, -1)
    else:
      ph, pw = self.config.vit.patches.size
      grid_ids = misc.reshape_image_to_patches(
          misc.crop_to_multiple(batch['grid_3d_id'], ph), ph, pw
      )
      grid_ids_best, _ = grids.gather_top_k_ids(
          grid_ids, top_k=topk, start_dim=-3
      )  # b x m x l x k
      return grid_ids_best

  def loss_metrics_function(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      model_params: jnp.ndarray | None = None,
      model_state: Any | None = None,
      rate_params: Mapping[str, float] | None = None,
  ) -> types.LossMetricsTuple:
    # assert rate_params is not None

    norm_axes = (-1, -2, -3)
    losses, metrics = {}, {}
    query = batch['query']['views']
    nquery = query['image_pixels'].shape[-4]
    if self.config.align_to == 'map+query':
      maps = jax.tree.map(
          lambda *xy: jnp.concatenate(xy, axis=1),
          batch['map'],
          batch['query'],
      )['views']
    else:
      maps = batch[self.config.align_to]['views']
    map_teacher_patch_logits = pred['teacher_patch_logits']
    if self.config.align == 'voxels':
      grid_id_query = self.get_grid_ids_per_patch(query)
      grid_id_map = self.get_grid_ids_per_patch(maps)

      pred['grid_id_query'] = grid_id_query
      pred['grid_id_map'] = grid_id_map

      query_target_logits, occupancy_q = grids.gather_features_from_map_batched(
          grid_id_query,
          grid_id_map,
          map_teacher_patch_logits,
      )

      occupancy_q = occupancy_q[..., None]
      align_weights = (occupancy_q > 0).astype(jnp.float32)
    elif self.config.align == 'gaussians':
      query['w_lidar_points'] = misc.batched_lidar_depth_to_xyz(
          query['lidar_depth'],
          types.create_jaxcam(
              query['cam_t_w'],
              query['camera'],
              use_inverted_distortion=True,
          ),
      )
      query_gaussians = gaussians.gaussians_from_points(
          misc.patchify(query['w_lidar_points'], *self.config.vit.patches.size),
          misc.patchify(query['lidar_mask'], *self.config.vit.patches.size),
      )

      maps['w_lidar_points'] = misc.batched_lidar_depth_to_xyz(
          maps['lidar_depth'],
          types.create_jaxcam(
              maps['cam_t_w'],
              maps['camera'],
              use_inverted_distortion=True,
          ),
      )

      map_gaussians = gaussians.gaussians_from_points(
          misc.patchify(maps['w_lidar_points'], *self.config.vit.patches.size),
          misc.patchify(maps['lidar_mask'], *self.config.vit.patches.size),
      )

      pred['query_gaussians'] = query_gaussians
      pred['map_gaussians'] = map_gaussians

      weights = gaussians.batched_probabilities_between_gaussians(
          query_gaussians,
          map_gaussians,
      )[
          ..., None
      ]  # b x nq x l x (nm+nq) x l x 1

      if self.config.get('clip_self_weight') is not None:
        # clip weight of self-self alignment
        q_indices = jnp.arange(0, nquery)[::-1]
        weights = weights.at[:, q_indices, :, -q_indices[::-1] - 1, :, :].set(
            weights[:, q_indices, :, -q_indices[::-1] - 1, :, :].clip(
                max=self.config['clip_self_weight']
            )
        )

      occupancy = weights.sum((-3, -2))
      query_target_logits = misc.masked_mean(
          map_teacher_patch_logits[:, None, None], weights, axis=(-3, -2)
      )

      pred['weights'] = weights
      pred['occupancy'] = occupancy
      align_weights = (occupancy > 0.05) * (1 + occupancy)
    else:
      raise ValueError(f'Unknown align mode: {self.config.align}')

    pred['do_align'] = align_weights
    pred['query_target_logits'] = query_target_logits

    aligments = self.compute_alignment_metrics(
        pred['student_patch_logits'], query_target_logits
    )
    for k, v in aligments.items():
      pred[k] = v
    alignment_losses = {
        k: misc.masked_mean(v, align_weights, axis=norm_axes)
        for k, v in aligments.items()
    }
    metrics['do_align'] = align_weights.mean(axis=norm_axes)

    losses = {
        **losses,
        **alignment_losses,
    }
    losses['total'] = types.compose_loss(losses, self.config.compose_loss)
    return losses, metrics

  def compute_alignment_metrics(
      self,
      source_logits: FloatArray['... L D'],
      target_logits: FloatArray['... L D'],
  ) -> types.Predictions:

    # cross MiM
    similarity = (source_logits * target_logits).sum(-1, keepdims=True)

    mse = ((source_logits - target_logits) ** 2).sum(-1, keepdims=True)

    mse_l2 = (
        (misc.l2_normalize(source_logits) - misc.l2_normalize(target_logits))
        ** 2
    ).sum(-1, keepdims=True)

    similarity_l2 = (
        misc.l2_normalize(source_logits) * misc.l2_normalize(target_logits)
    ).sum(-1, keepdims=True)

    pred = {}

    pred['similarity'] = similarity
    pred['similarity_l2'] = similarity_l2
    pred['mse'] = mse
    pred['mse_l2'] = mse_l2

    return pred

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
      match_threshold: float = 0.2,
      **kwargs,
  ) -> Mapping[str, FloatArray['B H W 3']]:
    visualizations = {}

    def stack(x):
      return jnp.concat(
          jax.lax.collapse(x, 1, -3).swapaxes(-5, -4),
          axis=stack_axis,
      )

    h, w = pred['student_views'].shape[-3:-1]
    ph, pw = self.config.vit.patches.size

    teacher_views = stack(pred['teacher_views'])

    all_views = stack(
        jnp.concatenate(
            [pred['student_views'], pred['teacher_views']], axis=-4
        ),
    )

    patch_tokens = jnp.concatenate(
        [
            pred['student_tokens'][..., 1:, :],
            pred['teacher_tokens'][..., 1:, :],
        ],
        axis=-3,
    )

    patch_pca = misc.features_to_rgb(patch_tokens)
    patch_pca = stack(misc.sequence_to_image(patch_pca, h // ph, w // pw))
    patch_pca = jax.image.resize(
        patch_pca, all_views.shape, jax.image.ResizeMethod.NEAREST
    )

    similarity_image = misc.sequence_to_image(
        plot.apply_colormap(
            pred['similarity_l2'][..., 0],
            cmap='turbo_r',
            vmin=0.0,
            vmax=1.0,
        ),
        h // ph,
        w // pw,
    )

    similarity_image = jax.image.resize(
        similarity_image,
        pred['student_views'].shape,
        method=jax.image.ResizeMethod.NEAREST,
    )

    mask = jax.image.resize(
        misc.sequence_to_image(
            pred['do_align'].astype(jnp.float32), h // ph, w // pw
        ),
        pred['student_views'].shape,
        method=jax.image.ResizeMethod.NEAREST,
    )

    similarity_overlay = misc.alpha_blend(
        misc.to_grayscale(pred['student_views'] * 0.5 + 0.5),
        similarity_image,
        (mask > 0) * 0.5,
    )

    similarity_overlay = stack(similarity_overlay)

    visualizations['similarity'] = jnp.concatenate(
        [
            jnp.concatenate([similarity_overlay, teacher_views], axis=-2),
            patch_pca,
        ],
        axis=-3,
    )[:max_num_plots]

    if match_threshold > 0 and pred['student_views'].shape[1] > 1:
      n = pred['student_views'].shape[1]
      images = pred['student_views'].swapaxes(1, 0)
      images = jnp.concatenate(
          [jnp.repeat(images[-1:], n - 1, 0), images[:-1]], axis=-2
      ).swapaxes(1, 0)
      features = misc.sequence_to_image(
          pred['student_tokens'][..., 1:, :], h // ph, w // pw
      )
      matches = misc.match_features(
          misc.l2_normalize(features[:, -1:]),
          misc.l2_normalize(features[:, :-1]),
          threshold=match_threshold,
      )
      matching_images = misc.batch_draw_matches(
          images,
          matches['keypoints_i'] * ph,
          matches['keypoints_j'] * pw,
          matches['matches_i'],
          matches['matching_scores_i'],
          draw='points',
      )
      visualizations['correspondences'] = stack(
          jnp.concatenate(matching_images.swapaxes(1, 0), axis=-2)
      )

    return visualizations

  def visualize_3d(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      batch_idx: int = 0,
      width: int = 1000,
      height: int = 600,
      plot_semantics: bool = False,
      max_num_samples: int = 100_000,
      **kwargs,
  ) -> viz3d.Plotter3D:
    batch_i = jax.tree.map(lambda x: x[batch_idx], batch)
    pred_i = jax.tree.map(lambda x: x[batch_idx], pred)

    grid: grids.Grid3D = self.dataset_meta_data['grid_3d']
    plotter = viz3d.Plotter3D(
        show_axes=True,
        height=height,
        width=width,
        **kwargs,
    )

    for collection_idx in ['query', 'map']:
      scene_i = batch_i[collection_idx]
      w_t_cam = np.linalg.inv(scene_i['views']['cam_t_w'].astype(np.float64))
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

      w_mask = scene_i['views']['lidar_mask']

      if 'w_lidar_points' in scene_i['views']:
        w_points = scene_i['views']['w_lidar_points']
      else:
        w_points = np.einsum(
            '...ij,...hwj->...hwi',
            w_t_cam[..., :3, :].astype(np.float64),
            misc.to_homogeneous(scene_i['views']['lidar_points']).astype(
                np.float64
            ),
        )

      if plot_semantics:
        semantics = scene_i['views']['lidar_semantic_labels'][..., 0]
        plotter.add_labelled_points(
            w_points[w_mask[..., 0]],
            semantics[w_mask[..., 0]],
            label_convert_fn=lambda idx: types.SEMANTIC_INDEX_TO_NAME[idx],
            max_num_samples=max_num_samples,
            legendgroup=collection_idx,
        )
      else:
        plotter.add_points(
            w_points[w_mask[..., 0]],
            max_num_samples=max_num_samples,
            name=collection_idx,
        )

      if self.config.align == 'gaussians':
        scene_gaussians = pred_i[f'{collection_idx}_gaussians']
        scene_centers, scene_cov, occ = scene_gaussians
        w_centers = np.einsum(
            '...ij,...hwj->...hwi',
            misc.inv_pose(plotter.scene_t_w)[..., :3, :],
            misc.to_homogeneous(scene_centers),
        )

        var = jnp.diagonal(scene_cov[occ > 0], axis1=-2, axis2=-1)

        std = jnp.sqrt(var)
        errors = {
            'error_x': {'array': std[..., 0], 'color': 'red'},
            'error_y': {'array': std[..., 1], 'color': 'green'},
            'error_z': {'array': std[..., 2], 'color': 'blue'},
        }

        plotter.add_points(
            w_centers[occ > 1],
            'cyan',
            size=2,
            name=f'{collection_idx}_gaussians',
            **errors,
        )

      elif self.config.align == 'voxels':
        teacher_pca = misc.features_to_rgb(pred_i['teacher_tokens'][..., 1:, :])

        warped = grids.gather_features_via_grid(
            pred_i['grid_id_query'],
            pred_i['grid_id_map'],
            teacher_pca,
            max_num_cells=10000,
            max_num_observations=100,
        )
        grid_3d_ids = warped['grid_ids']
        valid = warped['occupancy_grid'] > 0
        pca = warped['features_grid'][valid]
        plotter.add_voxel_grid(
            grid.id_to_index(grid_3d_ids[valid]) * grid.cell_size,
            jnp.linalg.inv(batch_i['grid_t_w']),
            grid.cell_size,
            mesh_color=pca,
        )

    return plotter
