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

"""Dino reimplementation."""

from collections.abc import Mapping
import copy
from typing import Any
from absl import logging
from etils.array_types import FloatArray
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt
import ml_collections
from scenic.train_lib import optimizers
from scaling_geoloc.scenic import load_params
from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.models import base
from scaling_geoloc.scenic.models import vit
from scaling_geoloc.utils import augmentations
from scaling_geoloc.utils import misc
from scaling_geoloc.utils import plot

augment_global0: augmentations.AugmentationFn = augmentations.per_image(
    augmentations.sequential([
        augmentations.random_color_jitter(0.8, 0.1, 0.4, 0.2, 0.1),
        augmentations.random_grayscale(0.2),
        augmentations.random_gaussian_blur(1.0, 0.1, 2.0),
    ])
)

augment_global1: augmentations.AugmentationFn = augmentations.per_image(
    augmentations.sequential([
        augmentations.random_color_jitter(0.8, 0.1, 0.4, 0.2, 0.1),
        augmentations.random_grayscale(0.2),
        augmentations.random_gaussian_blur(0.1, 0.1, 2.0),
        augmentations.random_solarize(0.2, 0.5),
    ])
)

augment_local: augmentations.AugmentationFn = augmentations.per_image(
    augmentations.sequential([
        augmentations.random_color_jitter(0.8, 0.1, 0.4, 0.2, 0.1),
        augmentations.random_grayscale(0.2),
        augmentations.random_gaussian_blur(0.5, 0.1, 2.0),
    ])
)


class ViTDINO(nn.Module):
  """DINO with student and teacher."""

  config: ml_collections.ConfigDict
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.student = vit.ViTBackbone.from_config(self.config.vit, self.dtype)
    self.teacher = vit.ViTBackbone.from_config(self.config.vit, self.dtype)

  def encode_multiview(
      self,
      images: FloatArray['... H W C'],
      *args,
      **kwargs,
  ) -> tuple[jt.Float[jt.Array, '... D'], jt.Float[jt.Array, '... L D']]:
    """Interface to backbone encoder."""
    return self.teacher.encode_multiview(images, *args, **kwargs)

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

  def bn_to_bxn(
      self, n: int, *x_bn: FloatArray['B*N ...']
  ) -> FloatArray['B N ...']:
    def to_bxn(x):
      return x.reshape(-1, n, *x.shape[1:])

    if len(x_bn) == 1:
      return to_bxn(x_bn[0])
    return [to_bxn(x) for x in x_bn]

  @nn.compact
  def encode(
      self,
      batch: types.Batch,
      train: bool = False,
      debug: bool = False,
  ) -> types.Predictions:
    del debug
    pred = {}
    if self.config.get('mask_ratio', 0.0) > 0:
      pred['student_token_mask'] = misc.token_mask(
          self.make_rng('droptok'),
          image_shape=batch['student_views'].shape,
          patch_size=self.config.vit.patches.size,
          mask_ratio=self.config.mask_ratio,
          mask_prob=self.config.get('mask_prob', 1.0),
      )
    student_logits, student_tokens = self.student.encode_multiview(
        batch['student_views'], train=train, mask=pred.get('student_token_mask')
    )
    teacher_logits, teacher_tokens = jax.lax.stop_gradient(
        self.teacher.encode_multiview(batch['teacher_views'], train=False)
    )

    pred = {
        **pred,
        'student_cls_logits': student_logits,
        'teacher_cls_logits': teacher_logits,
        'student_tokens': student_tokens,
        'teacher_tokens': teacher_tokens,
    }

    if 'student_local_views' in batch:
      student_local_logits, _ = self.student.encode_multiview(
          batch['student_local_views'], train=train
      )
      pred['student_local_cls_logits'] = student_local_logits

    if self.config.vit.pooling == 'none':
      pred = {
          k: v[..., 0, :] if 'cls_logits' in k else v for k, v in pred.items()
      }
      pred['student_patch_logits'] = student_logits[..., 1:, :]
      pred['teacher_patch_logits'] = teacher_logits[..., 1:, :]
    pred = {**pred, **batch}
    return pred

  def __call__(
      self,
      batch: types.Batch,
      train: bool = False,
      debug: bool = False,
  ) -> types.Predictions:

    multicrops = {}
    if train or self.config.get('eval_crops'):
      if self.config.get('stack_crops'):
        # Crops are provided, e.g. for imagenet.
        stacked_batch = {
            'global_views': jnp.stack(
                [
                    misc.concat_all('global_view0', batch, axis=-4),
                    misc.concat_all('global_view1', batch, axis=-4),
                ],
                axis=-4,
            ),
        }
        if self.config.num_local_views > 0:
          stacked_batch['local_views'] = jnp.stack(
              [
                  misc.concat_all(f'local_view{i}', batch, axis=-4)
                  for i in range(self.config.num_local_views)
              ],
              axis=-4,
          )
        batch = stacked_batch

      if self.config.get('do_augment'):
        _, rng0, rng1, rng_local = jax.random.split(self.make_rng('augment'), 4)
        batch['global_views'] = jnp.stack(
            [
                augment_global0(batch['global_views'][..., 0, :, :, :], rng0),
                augment_global1(batch['global_views'][..., 1, :, :, :], rng1),
            ],
            -4,
        )
        if 'local_views' in batch:
          batch['local_views'] = augment_local(batch['local_views'], rng_local)

      multicrops['student_views'] = misc.concat_all(
          'global_views', batch, axis=-5
      )
      assert multicrops['student_views'].shape[-4] == 2
      multicrops['teacher_views'] = multicrops['student_views']
      if self.config.num_local_views > 0:
        multicrops['student_local_views'] = misc.concat_all(
            'local_views', batch, axis=-5
        )
    else:
      multicrops['student_views'] = misc.concat_all(
          'image_pixels', batch, axis=-4
      )[:, :, None]
      multicrops['teacher_views'] = multicrops['student_views']

    return self.encode(multicrops, train=train, debug=debug)


class ViTDINOModel(base.BaseModel):
  """Trainer facing wrapper for ViTDINO."""

  required_rngs: tuple[str, ...] = ('droptok', 'dropout', 'augment')

  def build_flax_model(self) -> nn.Module:
    return ViTDINO(
        config=self.config.get('model', self.config),
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
    losses, metrics = {}, {}
    teacher_cls_logits = pred['teacher_cls_logits']
    teacher_cls_logits -= model_state['model_stats']['center']
    teacher_cls_logits /= learning_rates['teacher_temperature']
    student_cls_logits = (
        pred['student_cls_logits'] / learning_rates['student_temperature']
    )
    if self.config.get('all_to_all'):
      student_cls_logits = jax.lax.collapse(student_cls_logits, 1, -1)
      teacher_cls_logits = jax.lax.collapse(teacher_cls_logits, 1, -1)

    ce_global = misc.cross_entropy(
        jnp.expand_dims(student_cls_logits, -2),
        jnp.expand_dims(teacher_cls_logits, -3),
    )  # b x n x 2 x 2 x d

    batch_dims = teacher_cls_logits.shape[:-2]
    # Avoid supervising self-self.
    cls_weights = 1.0 - jnp.expand_dims(
        jnp.eye(ce_global.shape[-2])[..., None], range(len(batch_dims))
    )
    losses['cls_global'] = misc.masked_bmean(ce_global, cls_weights)
    teacher_cls_logits = pred['teacher_cls_logits']
    teacher_cls_logits -= model_state['model_stats']['center']
    teacher_cls_logits /= learning_rates['teacher_temperature']

    if (
        'student_local_cls_logits' in pred
        and pred['student_local_cls_logits'].size > 0
    ):
      student_local_cls_logits = (
          pred['student_local_cls_logits']
          / learning_rates['student_temperature']
      )

      ce_local_to_global = misc.cross_entropy(
          jnp.expand_dims(student_local_cls_logits, -3),
          jnp.expand_dims(teacher_cls_logits, -2),
      )  # shape b x ... x nlocal x nglobal x n

      losses['cls_local'] = jax.lax.collapse(ce_local_to_global, 1).mean(-1)
      n = student_cls_logits.shape[-2]
      nl, ml = ce_local_to_global.shape[-3:-1]
      n_global = n * (n - 1)
      n_local = nl * ml
      n_total = n_global + n_local
      losses['cls_all'] = (
          losses['cls_global'] * n_global + losses['cls_local'] * n_local
      )
      losses['cls_all'] /= n_total
    else:
      losses['cls_all'] = losses['cls_global']

    if 'student_patch_logits' in pred:
      teacher_patch_logits = pred['teacher_patch_logits']
      teacher_patch_logits -= model_state['model_stats']['patch_center']
      teacher_patch_logits /= learning_rates['teacher_temperature']
      ce_patch_global = misc.cross_entropy(
          pred['student_patch_logits'] / learning_rates['student_temperature'],
          teacher_patch_logits,
      )  # b x n x 2 x L x 1
      losses['patch_global'] = jax.lax.collapse(ce_patch_global, 1).mean(-1)

      if 'student_token_mask' in pred:
        losses['patch_mask'] = misc.masked_bmean(
            ce_patch_global, pred['student_token_mask']
        )

    # Enforce uniqueness of patch tokens.
    bmean = lambda x: jax.lax.collapse(x, 1).mean(-1)
    patch_tokens = pred['student_tokens'][..., 1:, :]
    self_similarity = jnp.einsum(
        '...md,...nd->...mn', patch_tokens, patch_tokens
    )
    self_similarity = nn.log_softmax(self_similarity, axis=-1)
    losses['self_similarity'] = bmean(
        -self_similarity.diagonal(axis1=-2, axis2=-1)
    )
    ph, pw = self.config.vit.patches.size
    h, w = pred['student_views'].shape[-3:-1]
    nh, nw = h // ph, w // pw
    grid_xy = misc.grid_indices(nh, nw, patch_tokens.shape[:-2])
    grid_xy = jax.lax.collapse(grid_xy, -3, -1)  # b ... x m x 2
    probs = jnp.exp(self_similarity)[..., None]
    expected_location = (probs * grid_xy[..., None, :, :]).sum(-2)
    mse_location = ((grid_xy - expected_location) ** 2).sum(-1)  # patch res.
    losses['self_expected_location_mse'] = bmean(mse_location)
    metrics['self_expected_location_norm'] = bmean(jnp.sqrt(mse_location))

    # Maximize distance between cls token and nn in batch.
    cls_tokens = jax.lax.collapse(pred['student_tokens'][..., 0, :], 0, -2)
    # Avoid regularizing to other global views from same image.
    cls_tokens = cls_tokens.swapaxes(0, 1)  # 2 x B x D
    self_similarity_l2 = jnp.einsum(
        'gmd,gnd->gmn',
        misc.l2_normalize(cls_tokens),
        misc.l2_normalize(cls_tokens),
    )
    ids = jnp.arange(cls_tokens.shape[-2])
    # Avoid regularizing to i==j.
    self_similarity_l2 = self_similarity_l2.at[..., ids, ids].set(-1.0)
    max_similarity_l2 = jnp.max(self_similarity_l2, axis=-1)
    min_distance_l2 = 0.5 - 0.5 * max_similarity_l2  # G x M
    losses['koleo'] = -jnp.log(min_distance_l2 + 1.0e-8).mean(-1)  # G
    metrics['max_similarity_l2'] = max_similarity_l2.mean(-1).mean(
        keepdims=True
    )  # G

    metrics['teacher_cls_logits_std'] = jnp.std(
        jax.lax.collapse(teacher_cls_logits, 1, -1), axis=1
    ).mean(-1)
    metrics['teacher_patch_token_std'] = jnp.std(
        jax.lax.collapse(pred['teacher_tokens'][..., 1:, :], 1, -1), axis=1
    ).mean(axis=-1)
    metrics['teacher_patch_token_std_per_image'] = jnp.std(
        jax.lax.collapse(pred['teacher_tokens'][..., 1:, :], 1, -2), axis=-2
    ).mean(axis=(-1, -2))
    if 'teacher_patch_logits' in pred:
      metrics['teacher_patch_logits_std'] = jnp.std(
          jax.lax.collapse(pred['teacher_patch_logits'], 1, -1), axis=1
      ).mean(-1)
    losses['total'] = types.compose_loss(losses, self.config.compose_loss)
    return losses, metrics

  def eval_loss_metrics_function(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      model_params: jnp.ndarray | None = None,
  ) -> types.LossMetricsTuple:
    return {}, {}

  def scale_grad(
      self, grad: types.Grad, learning_rates: types.LearningRates
  ) -> types.Grad:
    """Manually scale the gradients of the model.

    This function is called in the trainer after the gradients are computed.

    Args:
      grad: The gradients of the model.
      learning_rates: The learning rates provided by the trainer.

    Returns:
      The scaled gradients.
    """
    # Freeze prototype layer in first couple of steps.
    prototype_grad_scale = learning_rates.get('prototype_grad_scale', None)
    return optimizers.tree_map_with_names(
        lambda param: param * prototype_grad_scale,
        grad,
        lambda name: 'prototypes' in name,
    )

  def init_state(
      self, model_params: types.ModelParams, model_state: types.ModelState
  ) -> tuple[types.ModelParams, types.ModelState]:
    pretrained_weights = self.config.vit.get('pretrained_weights')
    pretrained_checkpoint = self.config.vit.get('pretrained_checkpoint')
    if pretrained_weights is not None or pretrained_checkpoint is not None:
      if pretrained_weights:
        logging.info(
            'Loading pretrained teacher weights: %s', pretrained_weights
        )
      else:
        logging.info(
            'Loading pretrained teacher checkpoint: %s', pretrained_checkpoint
        )
      pretrained_params = flax.core.unfreeze(
          load_params.load_params(
              pretrained_weights,
              pretrained_checkpoint,
              model_params['student'],
          )
      )

      def check_shape_equal(x: FloatArray, y: FloatArray) -> bool:
        is_shape_equal = x.shape == y.shape
        if not is_shape_equal:
          raise ValueError(f'Shapes are not identical: {x.shape} != {y.shape}')
        return is_shape_equal

      jax.tree.map(
          check_shape_equal,
          model_params['student'],
          pretrained_params,
      )
      model_params['student'] = pretrained_params
    model_params['teacher'] = copy.deepcopy(model_params['student'])
    out_dim = self.config.vit.head_output_dim or self.config.vit.hidden_size
    model_state['model_stats']['center'] = jnp.zeros(out_dim)

    if self.config.vit.pooling == 'none':
      model_state['model_stats']['patch_center'] = jnp.zeros(out_dim)
    return model_params, model_state

  def update_state(
      self,
      pred: types.Predictions,
      model_params: types.ModelParams,
      model_state: types.ModelState,
      learning_rates: types.LearningRates,
  ) -> tuple[types.ModelParams, types.ModelState]:
    """Run EMA and update center."""
    momentum = learning_rates['params_momentum']

    model_params['teacher'] = jax.tree_util.tree_map(
        lambda s, t: momentum * t + (1 - momentum) * s,
        model_params['student'],
        model_params['teacher'],
    )

    batch_center = jnp.mean(
        jax.lax.collapse(pred['teacher_cls_logits'], 0, -1), axis=0
    )
    batch_center = jax.lax.pmean(batch_center, axis_name='batch')

    center = model_state['model_stats']['center']

    cm = self.config.get('center_momentum', learning_rates['center_momentum'])
    model_state['model_stats']['center'] = cm * center + (1 - cm) * batch_center

    if 'teacher_patch_logits' in pred:
      if 'patch_center' not in model_state['model_stats']:
        raise KeyError('patch_center was not initialized.')
      p_momentum = learning_rates.get('patch_center_momentum', momentum)
      p_center = model_state['model_stats']['patch_center']
      p_batch_center = jnp.mean(
          jax.lax.collapse(pred['teacher_patch_logits'], 0, -1), axis=0
      )
      p_batch_center = jax.lax.pmean(p_batch_center, axis_name='batch')
      model_state['model_stats']['patch_center'] = (
          p_momentum * p_center + (1 - p_momentum) * p_batch_center
      )

    if 'teacher_patch_logits' in pred:
      if 'patch_center' not in model_state['model_stats']:
        raise KeyError('patch_center was not initialized.')
      p_momentum = learning_rates.get('patch_center_momentum', momentum)
      p_center = model_state['model_stats']['patch_center']
      p_batch_center = jnp.mean(
          jax.lax.collapse(pred['teacher_patch_logits'], 0, -1), axis=0
      )
      p_batch_center = jax.lax.pmean(p_batch_center, axis_name='batch')
      model_state['model_stats']['patch_center'] = (
          p_momentum * p_center + (1 - p_momentum) * p_batch_center
      )

    return model_params, model_state

  def visualize(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      max_num_plots: int = 64,
      match_threshold: float = 0,
      stack_axis: int = -3,
      plot_attention: bool = False,
      **kwargs,
  ) -> Mapping[str, FloatArray['B H W 3']]:
    visualizations = {}

    bs = pred['student_views'].shape[:-3]
    h, w = pred['student_views'].shape[-3:-1]
    ph, pw = self.config.vit.patches.size

    def stack(x):
      return jnp.concat(
          jax.lax.collapse(x, 1, -3).swapaxes(-5, -4),
          axis=stack_axis,
      )

    def pstack(x):
      x_img = misc.sequence_to_image(x, h // ph, w // pw)
      x_img = jax.image.resize(
          x_img,
          x_img.shape[:-3] + (h, w, 3),
          'nearest',
      )
      return stack(x_img)

    teacher_views = stack(pred['teacher_views'])

    student_pca = misc.features_to_rgb(pred['student_tokens'][..., 1:, :])
    teacher_pca = misc.features_to_rgb(pred['teacher_tokens'][..., 1:, :])

    teacher_patch_logits = pred.get(
        'teacher_patch_logits', pred['teacher_tokens'][..., 1:, :]
    )
    student_patch_logits = pred.get(
        'student_patch_logits', pred['student_tokens'][..., 1:, :]
    )
    ce_patch_global = misc.cross_entropy(
        student_patch_logits / 0.1,
        teacher_patch_logits / 0.04,
    )  # b x n x 2 x L x 1

    ce_image = plot.apply_colormap(
        ce_patch_global[..., 0], cmap='turbo', vmin=0.0, vmax=10.0
    )

    token_similarity_l2 = (
        misc.l2_normalize(pred['student_tokens'][..., 1:, :])
        * misc.l2_normalize(pred['teacher_tokens'][..., 1:, :])
    ).sum(-1)

    token_sim_image = plot.apply_colormap(
        token_similarity_l2, cmap='turbo_r', vmin=0.0, vmax=1.0
    )

    student_pca = pstack(student_pca)
    teacher_pca = pstack(teacher_pca)
    ce_image = pstack(ce_image)
    token_sim_image = pstack(token_sim_image)

    visualizations['features'] = jnp.concatenate(
        [teacher_views, student_pca, ce_image, token_sim_image, teacher_pca],
        axis=-2,
    )

    if 'intermediates' in pred and plot_attention:
      last_block = str(self.config.vit.num_layers - 1)
      representation = pred['intermediates']['teacher']
      attn_weights = representation['encoderblock_' + last_block][
          'MultiHeadDotProductAttention_0'
      ]['attention_weights'][0]

      # we look only at the attention of the CLS token
      attentions = attn_weights[:, :, 0, 1:].reshape(*bs, -1, h // ph, w // pw)
      attentions = plot.apply_colormap(attentions, cmap='turbo')
      attentions = jax.image.resize(
          attentions,
          attentions.shape[:-3] + (h, w, 3),
          jax.image.ResizeMethod.NEAREST,
      )
      viz_att = jnp.concatenate(
          [pred['teacher_views'][..., None, :, :, :], attentions], axis=-4
      )

      viz_att = jax.lax.collapse(viz_att[:, :, 0], 1, -4)
      viz_att = jnp.concatenate(viz_att.swapaxes(0, 1), axis=-3)
      viz_att = jnp.concatenate(viz_att.swapaxes(0, 1), axis=-2)
      visualizations['teacher_attention'] = viz_att

    if match_threshold > 0 and pred['teacher_views'].shape[-4] == 2:
      # Matching between crops
      images = pred['teacher_views']
      images = jnp.concatenate(jnp.split(images, 2, axis=-4), axis=-2)[:, :, 0]
      features = misc.sequence_to_image(
          pred['teacher_tokens'][..., 1:, :], h // ph, w // pw
      )
      matches = misc.match_features(
          misc.l2_normalize(features[..., 0, :, :, :]),
          misc.l2_normalize(features[..., 1, :, :, :]),
          threshold=match_threshold,
      )
      matching_images = misc.batch_draw_matches(
          images,
          matches['keypoints_i'] * ph,
          matches['keypoints_j'] * pw,
          matches['matches_i'],
          matches['matching_scores_i'],
          draw='points+lines',
      )
      visualizations['correspondences'] = stack(matching_images)
    elif match_threshold > 0 and pred['teacher_views'].shape[1] > 1:
      # Pairwise matching
      n = pred['teacher_views'].shape[1]
      images = pred['teacher_views'].swapaxes(1, 0)
      images = jnp.concatenate(
          [jnp.repeat(images[-1:], n - 1, 0), images[:-1]], axis=-2
      ).swapaxes(1, 0)
      features = misc.sequence_to_image(
          pred['teacher_tokens'][..., 1:, :], h // ph, w // pw
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
          draw='points+lines',
      )
      visualizations['correspondences'] = stack(
          jnp.concatenate(matching_images.swapaxes(1, 0), axis=-2)
      )
    visualizations = {
        k: v[: min(v.shape[0], max_num_plots)]
        for k, v in visualizations.items()
    }
    return visualizations
