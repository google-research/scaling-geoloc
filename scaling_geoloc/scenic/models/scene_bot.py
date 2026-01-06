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

"""iBOT for scenes."""

from collections.abc import Mapping
from typing import Any
from etils.array_types import FloatArray
import flax.linen as nn
import jax
import jax.numpy as jnp
from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.models import dino
from scaling_geoloc.utils import gaussians
from scaling_geoloc.utils import misc


def _compute_gaussians(
    batch: types.Batch,
    patch_size: tuple[int, int],
) -> gaussians.Gaussians:
  """Computes gaussians from lidar points."""
  w_lidar_points = misc.batched_lidar_depth_to_xyz(
      batch['lidar_depth'],
      types.create_jaxcam(
          batch['cam_t_w'],
          batch['camera'],
          use_inverted_distortion=True,
      ),
  )
  w_gaussians = gaussians.gaussians_from_points(
      misc.patchify(w_lidar_points, *patch_size),
      misc.patchify(batch['lidar_mask'], *patch_size),
  )
  return w_gaussians


class SceneBot(dino.ViTDINO):
  """iBOT for scenes."""

  def __call__(
      self,
      batch: types.Batch,
      train: bool = False,
      debug: bool = False,
  ) -> types.Predictions:
    batch = misc.filter_batch_for_jit(batch)
    views = {}

    if self.config.align_to == 'map+query':
      maps = jax.tree.map(
          lambda x, y: jnp.concatenate([x, y], axis=1),
          batch['map'],
          batch['query'],
      )['views']
    else:
      maps = batch[self.config.align_to]['views']
    query = maps
    # query = batch['query']['views']
    views['student_views'] = query['image_pixels']
    views['teacher_views'] = maps['image_pixels']

    pred = self.encode(views, train=train, debug=debug)
    pred['query'] = query
    pred['maps'] = maps

    if not train:
      del pred['teacher_patch_logits']
      del pred['student_patch_logits']

    return pred


class SceneBotModel(dino.ViTDINOModel):
  """Trainer facing wrapper for ViTDINO."""

  required_rngs: tuple[str, ...] = ('droptok', 'dropout')

  def build_flax_model(self) -> nn.Module:
    return SceneBot(
        config=self.config,
        dtype=self.dtype,
    )

  def loss_metrics_function(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      model_params: jnp.ndarray | None = None,
      model_state: Any | None = None,
      learning_rates: types.LearningRates | None = None,
  ) -> types.LossMetricsTuple:
    # assert learning_rates is not None
    teacher_cls_logits = pred['teacher_cls_logits']
    teacher_cls_logits -= model_state['model_stats']['center']
    teacher_cls_logits /= learning_rates['teacher_temperature']
    student_cls_logits = (
        pred['student_cls_logits'] / learning_rates['student_temperature']
    )

    losses = {}
    # b x q x d, b x m x d -> b x q x m x 1
    cls_global = misc.cross_entropy(
        jnp.expand_dims(student_cls_logits, -2),
        jnp.expand_dims(teacher_cls_logits, -3),
    )

    batch_dims = teacher_cls_logits.shape[:-2]
    # Avoid supervising self-self.
    cls_weights = 1.0 - jnp.expand_dims(
        jnp.eye(cls_global.shape[-2])[..., None], range(len(batch_dims))
    )
    losses['cls_global'] = misc.masked_bmean(cls_global, cls_weights)

    if 'student_local_views' in pred:
      # b x q x 4 x d, b x m x d -> b x q x 4 x x m x d --> all-to-all
      student_local_cls_logits = (
          pred['student_local_cls_logits']
          / learning_rates['student_temperature']
      )
      ce_local_to_global = misc.cross_entropy(
          jnp.expand_dims(student_local_cls_logits, -3),
          jnp.expand_dims(teacher_cls_logits, -2),
      )  # shape b x ... x nlocal x nglobal x n
      losses['cls_local'] = jax.lax.collapse(ce_local_to_global, 1).mean(-1)

    nquery = pred['student_views'].shape[-4]

    query_gaussians = _compute_gaussians(
        pred['query'], self.config.vit.patches.size
    )
    map_gaussians = _compute_gaussians(
        pred['maps'], self.config.vit.patches.size
    )

    weights = gaussians.batched_probabilities_between_gaussians(
        query_gaussians,
        map_gaussians,
    )[..., None]

    if self.config.get('clip_self_weight') is not None:
      # clip weight of self-self alignment
      q_indices = jnp.arange(0, nquery)[::-1]
      weights = weights.at[:, q_indices, :, -q_indices[::-1] - 1, :, :].set(
          weights[:, q_indices, :, -q_indices[::-1] - 1, :, :].clip(
              max=self.config['clip_self_weight']
          )
      )

    occupancy = weights.sum((-3, -2))
    norm_axes = (-1, -2, -3)
    if 'student_patch_logits' in pred:
      # self MiM
      student_patch_logits = pred['student_patch_logits']
      student_patch_logits /= learning_rates['student_temperature']
      teacher_patch_logits = pred['teacher_patch_logits']
      teacher_patch_logits -= model_state['model_stats']['patch_center']
      teacher_patch_logits /= learning_rates['teacher_temperature']
      ce_patch_global = misc.cross_entropy(
          student_patch_logits,
          teacher_patch_logits[:, -nquery:],
      )
      losses['patch_global'] = ce_patch_global.mean(norm_axes)
      map_teacher_patch_logits = teacher_patch_logits

      query_target_logits = misc.masked_mean(
          map_teacher_patch_logits[:, None, None], weights, axis=(-3, -2)
      )

      # cross MiM
      ce_patch_cross = misc.cross_entropy(
          student_patch_logits, query_target_logits
      )

      losses['patch_cross'] = misc.masked_mean(
          ce_patch_cross, occupancy > 0, axis=norm_axes
      )
      if 'student_token_mask' in pred:
        losses['patch_mask'] = misc.masked_mean(
            ce_patch_global, pred['student_token_mask'], axis=norm_axes
        )
        losses['patch_mask_cross'] = misc.masked_mean(
            ce_patch_cross,
            pred['student_token_mask'] & (occupancy > 0),
            axis=norm_axes,
        )

    student_feats = pred['student_tokens'][..., 1:, :]
    teacher_feats = pred['teacher_tokens'][..., 1:, :]
    target_feats = misc.masked_mean(
        teacher_feats[:, None, None], weights, axis=(-2, -3)
    )

    losses['patch_cross_mse'] = misc.masked_mean(
        0.5
        - 0.5
        * misc.compute_similarity(
            student_feats,
            target_feats,
            normalize=False,
        ),
        occupancy > 0,
        axis=norm_axes,
    )

    losses['patch_cross_mse_l2'] = misc.masked_mean(
        0.5
        - 0.5
        * misc.compute_similarity(
            student_feats,
            target_feats,
            normalize=True,
        ),
        occupancy > 0,
        axis=norm_axes,
    )

    losses['student_patch_cross_mse_l2'] = misc.masked_mean(
        0.5
        - 0.5
        * misc.compute_similarity(
            student_feats,
            misc.masked_mean(
                student_feats[:, None, None], weights, axis=(-2, -3)
            ),
            normalize=True,
        ),
        occupancy > 0,
        axis=norm_axes,
    )

    losses['patch_cross_ce_feats'] = misc.masked_mean(
        misc.cross_entropy(student_feats, target_feats),
        occupancy > 0,
        axis=norm_axes,
    )

    losses['st_similarity_l2'] = jnp.mean(
        misc.compute_similarity(
            pred['student_tokens'][..., 1:, :],
            pred['teacher_tokens'][..., 1:, :],
            normalize=True,
        ),
        axis=norm_axes,
    )

    # Enforce uniqueness of patch tokens.
    bmean = lambda x: jax.lax.collapse(x, 1).mean(-1)
    patch_tokens = pred['student_tokens'][..., 1:, :]
    self_similarity = jnp.einsum(
        '...md,...nd->...mn', patch_tokens, patch_tokens
    )
    self_similarity = nn.log_softmax(self_similarity, axis=-1)
    losses['self_similarity_nll'] = bmean(
        -self_similarity.diagonal(axis1=-2, axis2=-1)
    )
    st_similarity = jnp.einsum(
        '...md,...nd->...mn', student_feats, target_feats
    )
    st_similarity = nn.log_softmax(st_similarity, axis=-1)
    losses['st_similarity_nll'] = bmean(
        -st_similarity.diagonal(axis1=-2, axis2=-1)
    )

    losses['total'] = types.compose_loss(losses, self.config.compose_loss)
    return losses, {}

  def eval_loss_metrics_function(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      model_params: jnp.ndarray | None = None,
  ) -> types.LossMetricsTuple:
    return {}, {}

  def visualize(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      max_num_plots: int = 4,
      stack_axis: int = -2,
      match_threshold: float = 0.2,
      show_gt_matches: bool = True,
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

    student_views = stack(pred['student_views'])
    teacher_views = stack(pred['teacher_views'])

    student_pca = misc.features_to_rgb(pred['student_tokens'][..., 1:, :])
    teacher_pca = misc.features_to_rgb(pred['teacher_tokens'][..., 1:, :])
    student_pca = stack(misc.sequence_to_image(student_pca, h // ph, w // pw))
    teacher_pca = stack(misc.sequence_to_image(teacher_pca, h // ph, w // pw))
    student_pca = jax.image.resize(
        student_pca, student_views.shape, jax.image.ResizeMethod.NEAREST
    )
    teacher_pca = jax.image.resize(
        teacher_pca, teacher_views.shape, jax.image.ResizeMethod.NEAREST
    )

    visualizations['student_feats'] = jnp.concatenate(
        [student_views, student_pca], axis=-3
    )[:max_num_plots]
    visualizations['teacher_feats'] = jnp.concatenate(
        [teacher_views, teacher_pca], axis=-3
    )[:max_num_plots]

    if match_threshold > 0 and pred['teacher_views'].shape[1] > 1:
      # Pairwise matching
      n = pred['teacher_views'].shape[1]
      images = pred['teacher_views'].swapaxes(1, 0)

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

      if show_gt_matches:
        query_gaussians = _compute_gaussians(
            pred['query'], self.config.vit.patches.size
        )
        map_gaussians = _compute_gaussians(
            pred['maps'], self.config.vit.patches.size
        )
        gt_similarity = gaussians.batched_probabilities_between_gaussians(
            query_gaussians, map_gaussians
        )
        gt_similarity = gt_similarity.swapaxes(-2, -3)[..., -1, :-1, :, :]
        gt_matches = misc.matches_from_similarity(
            gt_similarity, threshold=0.001, mutual_check=False
        )
        matching_images = misc.batch_draw_matches(
            images,
            matches['keypoints_i'] * ph,
            matches['keypoints_j'] * pw,
            gt_matches['matches_i'],
            jnp.where(
                matches['matches_i'] == gt_matches['matches_i'],
                1.0,
                -1.0,
            )
            * (gt_matches['matches_i'] > -1),
            draw='points+lines',
            color_by_coords=False,
        )
      else:
        matching_images = misc.batch_draw_matches(
            images,
            matches['keypoints_i'] * ph,
            matches['keypoints_j'] * pw,
            matches['matches_i'],
            matches['matching_scores_i'],
            draw='points+lines',
        )
      visualizations['correspondences'] = stack(
          jnp.concatenate(matching_images.swapaxes(1, 0), axis=-2)
      )[:max_num_plots]
    return visualizations
