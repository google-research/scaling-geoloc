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

"""IJEPA reimplementation."""

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
from scenic.model_lib.layers import nn_layers
from scaling_geoloc.scenic import load_params
from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.models import base
from scaling_geoloc.scenic.models import vit
from scaling_geoloc.utils import misc
from scaling_geoloc.utils import plot


def smooth_l1(delta: FloatArray) -> FloatArray:
  return jnp.where(
      jnp.abs(delta) < 1, 0.5 * (delta) ** 2, jnp.abs(delta) - 0.5
  ).sum(-1, keepdims=True)


class ViTIJEPA(nn.Module):
  """IJEPA with student and teacher."""

  config: ml_collections.ConfigDict
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.student = vit.ViTBackbone.from_config(self.config.vit, self.dtype)
    self.teacher = vit.ViTBackbone.from_config(self.config.vit, self.dtype)
    self.predictor = vit.ViTBackbone.from_config(
        self.config.predictor, self.dtype
    )
    assert not self.predictor.conv_embedding

  def encode_multiview(
      self,
      images: FloatArray['... H W C'],
      *args,
      **kwargs,
  ) -> tuple[jt.Float[jt.Array, '... D'], jt.Float[jt.Array, '... L D']]:
    """Interface to backbone encoder."""
    return self.teacher.encode_multiview(images, *args, **kwargs)

  @nn.compact
  def encode(
      self,
      batch: types.Batch,
      train: bool = False,
      debug: bool = False,
  ) -> types.Predictions:
    del debug
    pred = {}

    ph, pw = self.config.vit.patches.size
    h, w = batch['student_views'].shape[-3:-1]
    nh, nw = h // ph, w // pw

    # Run context encoder where mask==True. Ignore else (zero padded).
    context_logits, context_tokens = self.student.encode_multiview(
        batch['student_views'][:, :, None],
        train=train,
        mask=~batch['student_token_mask'],  # ViT mask is inverted.
        mask_mode='ignore',
    )

    # Should only project where not masked.
    projected_tokens = nn_layers.IdentityLayer(name='project_predictor')(
        context_tokens[..., 1:, :]
    )

    n_predict = batch['predictor_token_mask'].shape[-3]
    # Add mask token where masked out.
    mask_tok = self.param(
        'mask_tok',
        nn.initializers.zeros,
        (1, 1, 1, 1, projected_tokens.shape[-1]),
        projected_tokens.dtype,
    )
    padded_tokens = jnp.where(
        batch['student_token_mask'],
        projected_tokens,
        mask_tok,
    ).repeat(n_predict, axis=-3)

    # Keep the union over both masks.
    all_mask = batch['predictor_token_mask'] | batch['student_token_mask']

    # This gathers and pads at the end, but we don't need the final pad anyway.
    predictor_logits, predictor_tokens = self.predictor.encode_multiview(
        misc.sequence_to_image(padded_tokens, nh, nw),
        train=train,
        mask=~all_mask,  # ViT mask is inverted.
        mask_mode='ignore',
    )

    predictor_tokens = nn.Dense(context_tokens.shape[-1], name='prototypes')(
        predictor_tokens
    )

    # Full forward pass of teacher.
    target_logits, target_tokens = jax.lax.stop_gradient(
        self.teacher.encode_multiview(
            batch['teacher_views'][:, :, None], train=False
        )
    )

    pred = {
        **pred,
        'student_cls_logits': context_logits,
        'teacher_cls_logits': target_logits,
        'predictor_cls_logits': predictor_logits,
        'student_tokens': context_tokens,
        'teacher_tokens': target_tokens,
        'predictor_tokens': predictor_tokens,
        'is_predicted': batch['predictor_token_mask'],
    }

    pred = {**pred, **batch}
    return pred

  def __call__(
      self,
      batch: types.Batch,
      train: bool = False,
      debug: bool = False,
  ) -> types.Predictions:

    multicrops = {}
    batch = misc.filter_batch_for_jit(batch)
    if 'image_pixels' in batch:
      batch_flat = batch
    else:
      views = (batch['map']['views'],) if 'map' in batch else ()
      views = views + (batch['query']['views'],)
      batch_flat = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *views)

    if 'patch_masks_encoder' in batch_flat:
      multicrops['student_token_mask'] = batch_flat['patch_masks_encoder']
      multicrops['predictor_token_mask'] = batch_flat['patch_masks_predictor']

    multicrops['student_views'] = misc.crop_to_multiple(
        batch_flat['image_pixels'],
        self.config.vit.patches.size[0],
    )

    multicrops['teacher_views'] = multicrops['student_views']
    return self.encode(multicrops, train=train, debug=debug)


class ViTIJEPAModel(base.BaseModel):
  """Trainer facing wrapper for ViTIJEPA."""

  required_rngs: tuple[str, ...] = ('droptok', 'dropout', 'augment')

  def build_flax_model(self) -> nn.Module:
    return ViTIJEPA(
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
    losses, metrics = {}, {}
    h = pred['teacher_tokens'][..., 1:, :]
    z = pred['predictor_tokens'][..., 1:, :]
    is_predicted = pred['is_predicted']

    def add_losses(h, z, suffix=''):
      losses['l1' + suffix] = misc.masked_bmean(jnp.abs(h - z), is_predicted)
      losses['l2' + suffix] = misc.masked_bmean(
          0.5 * (h - z) ** 2, is_predicted
      )
      losses['soft_l1' + suffix] = misc.masked_bmean(
          jnp.log(1 + jnp.abs(h - z)), is_predicted
      )
      # Huber loss. Used in original IJEPA.
      losses['smooth_l1' + suffix] = misc.masked_bmean(
          smooth_l1(h - z),
          is_predicted,
      )

    add_losses(h, z)
    add_losses(misc.l2_normalize(h), misc.l2_normalize(z), suffix='_normalized')

    losses['total'] = types.compose_loss(losses, self.config.compose_loss)

    def bmean(x):
      return jax.lax.collapse(x, 1).mean(-1)

    metrics['is_predicted'] = bmean(is_predicted)
    metrics['is_context'] = bmean(pred['student_token_mask'])
    metrics['predict_or_context'] = bmean(
        pred['student_token_mask'] | is_predicted
    )
    metrics['overlap'] = bmean(pred['student_token_mask'] & is_predicted)
    return losses, metrics

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

    return model_params, model_state

  def visualize(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      max_num_plots: int = 64,
      match_threshold: float = 0,
      stack_axis: int = -3,
      plot_attention: bool = False,
      vmax: float = 1.0,
      **kwargs,
  ) -> Mapping[str, FloatArray['B H W 3']]:
    visualizations = {}
    pred = jax.tree.map(lambda x: x[:max_num_plots], pred)
    teacher_views = pred['teacher_views']
    h, w = teacher_views.shape[-3:-1]
    ph, pw = self.config.vit.patches.size
    nh, nw = h // ph, w // pw
    h = pred['teacher_tokens'][..., 1:, :]
    c = pred['student_tokens'][..., 1:, :]
    z = pred['predictor_tokens'][..., 1:, :]

    h = misc.l2_normalize(h)
    c = misc.l2_normalize(c)
    z = misc.l2_normalize(z)
    is_predicted = pred['is_predicted']
    is_context = pred['student_token_mask']

    teacher_pca = misc.sequence_to_image(misc.features_to_rgb(h), nh, nw)
    teacher_pca = jax.image.resize(
        teacher_pca,
        teacher_pca.shape[:-3] + teacher_views.shape[-3:],
        'nearest',
    )
    error_context = misc.alpha_blend(
        0,
        plot.apply_colormap(
            smooth_l1(h - c)[..., 0], vmin=0.0, vmax=vmax, cmap='turbo'
        ),
        is_context,
    )
    error_predictor = misc.alpha_blend(
        0,
        plot.apply_colormap(
            smooth_l1(h - z)[..., 0], vmin=0.0, vmax=vmax, cmap='turbo'
        ),
        is_predicted,
    )

    errors = misc.sequence_to_image(
        jnp.concatenate(
            [
                error_context,
                error_predictor,
            ],
            axis=-3,
        ),
        nh,
        nw,
    )
    errors = jax.image.resize(
        errors, errors.shape[:-3] + teacher_views.shape[-3:], 'nearest'
    )

    joint_figure = jnp.concatenate(
        [
            teacher_views[:, :, None],
            teacher_pca,
            errors,
        ],
        axis=-4,
    )
    joint_figure = jnp.concatenate(jnp.swapaxes(joint_figure, 0, 1), axis=-3)
    joint_figure = jnp.concatenate(jnp.swapaxes(joint_figure, 0, 1), axis=-2)
    visualizations['features'] = joint_figure

    return visualizations
