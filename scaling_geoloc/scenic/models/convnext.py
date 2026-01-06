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

"""ConvNeXt backbone for streetview and aerial images."""

from typing import Any, TypeAlias

from absl import logging
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt
import ml_collections
from scenic.projects.convnext import convnext

from scaling_geoloc.scenic import load_params
from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.models import heads
from scaling_geoloc.utils import misc


ValidIndex = jt.Int[jt.Array, 'L']

MaskArray: TypeAlias = jt.Bool[jt.Array, 'B L 1'] | jt.Int[jt.Array, 'B L']


class ConvNeXtBackbone(nn.Module):
  """ConvNeXt backbone."""

  head_output_dim: int | None
  head: str
  head_config: ml_collections.ConfigDict | None
  dtype: Any
  size: str = 'B'
  imagenet_normalization: bool = False
  freeze_backbone: bool = False
  pooling: str = 'gap'

  @classmethod
  def from_config(
      cls, config: ml_collections.ConfigDict, dtype: Any = jnp.float32
  ) -> 'ConvNeXtBackbone':
    return ConvNeXtBackbone(
        size=config.get('version', 'B'),
        freeze_backbone=config.get('freeze_backbone', False),
        head_output_dim=config.get('head_output_dim', None),
        head=config.get('head', 'projection'),
        imagenet_normalization=config.get('imagenet_normalization', False),
        pooling=config.get('pooling', 'gap'),
        head_config=config.get('head_config', None),
        dtype=dtype,
    )

  def setup(self):
    self.imagenet_mean = jnp.array([0.485, 0.456, 0.406])
    self.imagenet_std = jnp.array([0.229, 0.224, 0.225])

  @nn.compact
  def __call__(
      self,
      x: jt.Float[jt.Array, 'B H W C'],
      *,
      train: bool,
      debug: bool = False,
      pooling: None | str = None,
      sow_attention: bool = False,
      normalize: bool = False,
      normalize_per_head: bool = False,
  ) -> tuple[jt.Float[jt.Array, 'B L D'], jt.Float[jt.Array, 'B D']]:

    if self.imagenet_normalization:
      # Apply imagenet offset
      x = x.at[..., :3].set(
          jnp.apply_along_axis(
              lambda c: (c - self.imagenet_mean) / self.imagenet_std,
              -1,
              x[..., :3],
          )
      )

    backbone_output = convnext.ConvNeXt(num_outputs=None, size=self.size)(
        x, train=train
    )
    backbone_output = backbone_output['stage_4']  # stage 4 is the last layer
    backbone_output = jax.lax.collapse(backbone_output, 1, 3)  # BHWC -> B(H*W)C

    if self.freeze_backbone:
      backbone_output = jax.lax.stop_gradient(backbone_output)

    if not pooling:
      pooling = self.pooling

    if pooling == 'gap':
      x = jnp.mean(backbone_output, axis=1)
      x = nn.LayerNorm(name='norm')(x)
      backbone_output = jnp.concatenate([x[:, None], backbone_output], axis=1)
    else:
      raise ValueError('ConvNeXt only supports pooling=gap.')

    supported_heads = ['linear', 'projection', 'salad_nol2', 'salad']
    if self.head not in supported_heads:
      raise ValueError(f'ConvNeXt only supports heads: {supported_heads}.')

    x = heads.backbone_head(
        backbone_output,
        x,
        head_output_dim=self.head_output_dim,
        head=self.head,
        head_config=self.head_config,
        train=train,
        pooling=pooling,
        sow_attention=sow_attention,
        normalize_per_head=normalize_per_head,
        dtype=self.dtype,
    )

    if normalize:
      x = misc.l2_normalize(x)

    return x, backbone_output

  @nn.compact
  def encode_multiview(
      self,
      x: jt.Float[jt.Array, '... H W 3'],
      *,
      train: bool,
      mask: MaskArray | None = None,
      mask_mode: str = 'pad_token',
      debug: bool = False,
      pooling: None | str = None,
      sow_attention: bool = False,
      extra_tokens: jt.Float[jt.Array, '... M C'] | None = None,
      normalize: bool = False,  # logits
      normalize_per_head: bool = False,  # logits
  ) -> tuple[jt.Float[jt.Array, '... L D'], jt.Float[jt.Array, '... D']]:

    x_flat = jax.lax.collapse(x, 0, -3)

    logits_flat, backbone_output_flat = self(
        x_flat,
        train=train,
        debug=debug,
        pooling=pooling,
        sow_attention=sow_attention,
        normalize=normalize,
        normalize_per_head=normalize_per_head,
    )

    logits = logits_flat.reshape(x.shape[:-3] + logits_flat.shape[1:])
    backbone_output = backbone_output_flat.reshape(
        x.shape[:-3] + backbone_output_flat.shape[1:]
    )

    return logits, backbone_output

  def output_dim(self) -> int:
    """Returns the output dimension of the logits."""
    return self.head_output_dim or 0

  def get_patch_token_indices(self, height: int, width: int) -> None:
    """Returns none as ConvNeXt does not support tokenization."""
    return None

  @classmethod
  def maybe_load_params(
      cls,
      model_params: types.ModelParams,
      pretrained_weights: str | None = None,
      pretrained_checkpoint: str | None = None,
  ) -> Any:
    """Loads pretrained parameters."""
    if pretrained_weights is not None or pretrained_checkpoint is not None:
      if pretrained_weights:
        logging.info('Loading pretrained weights: %s', pretrained_weights)
      else:
        logging.info('Loading pretrained checkpoint: %s', pretrained_checkpoint)
      pretrained_params = flax.core.unfreeze(
          load_params.load_params(
              pretrained_weights,
              pretrained_checkpoint,
              model_params,
          )
      )

      def check_shape_equal(x, y) -> bool:
        is_shape_equal = x.shape == y.shape
        if not is_shape_equal:
          raise ValueError(f'Shapes are not identical: {x.shape} != {y.shape}')
        return is_shape_equal

      jax.tree.map(
          check_shape_equal,
          model_params,
          pretrained_params,
      )
      model_params = pretrained_params
    return model_params
