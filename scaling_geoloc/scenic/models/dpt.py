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

"""Dense Prediction Transformers (DPT) implementation in jax/flax.

Inspired by https://github.com/naver/croco and
https://github.com/huggingface/transformers/pull/17779
"""

from collections.abc import Callable, Sequence
import functools
from typing import Any
from etils.array_types import FloatArray
import flax.linen as nn
import jax
import jax.numpy as jnp
from scenic.model_lib.layers import nn_layers
from scenic.projects.baselines import vit
from scenic.projects.dpt.decoders import conv as conv_decoder
from scaling_geoloc.utils import misc

# L = patch_size**2 = h * w
ViTEncoding = FloatArray['B L C'] | FloatArray['B h w C']


class ViTEncoder(nn.Module):
  """ViT encoder."""

  mlp_dim: int
  num_layers: int
  num_heads: int
  patch_size: int
  hidden_size: int
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  return_layers: Sequence[int] = ()
  padding: str = 'VALID'
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      img: FloatArray['B H W 3'],
      image_like: bool = True,
      train: bool = False,
  ) -> tuple[ViTEncoding, list[ViTEncoding]]:
    """Applies Transformer model on the inputs."""

    if img.ndim != 4:
      raise ValueError(
          f'Encoder only supports 4D inputs, x.shape=: {img.shape}'
      )

    img = misc.pad_to_multiple(img, self.patch_size)

    ph, pw = self.patch_size, self.patch_size
    x = nn.Conv(
        self.hidden_size,
        (ph, pw),
        strides=(ph, pw),
        padding=self.padding,
        name='embedding',
    )(img)
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    x = vit.AddPositionEmbs(
        posemb_init=nn.initializers.normal(stddev=0.02), name='encoder_posembed'
    )(x)

    dtype = jax.dtypes.canonicalize_dtype(self.dtype)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=True)

    layers = []
    for lyr in range(self.num_layers):
      x = vit.Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
          dtype=dtype,
      )(x, deterministic=not train)
      if lyr in self.return_layers:
        layers.append(x)

    encoded = nn.LayerNorm(name='encoder_norm')(x)

    if image_like:
      h, w = img.shape[-3:-1]
      gh, gw = h // self.patch_size, w // self.patch_size
      encoded = misc.sequence_to_image(encoded, gh, gw)
      layers = [misc.sequence_to_image(layer, gh, gw) for layer in layers]
    return encoded, layers


# Adapted from google3/experimental/labs/media/spin/framework/dust3r/models.py
class DPTHead(nn.Module):
  """A DPTHead that takes encoder features and outputs dense predictions."""

  block_size: tuple[int, ...]
  block_scale: tuple[int, ...]
  feature_dim: int
  num_classes: int
  num_convs: int = 2
  padding: str = 'SAME'
  norm: Callable[[], Any] | None = None

  @nn.compact
  def __call__(
      self,
      image_features: list[FloatArray['B h w c']],
  ) -> FloatArray['B H W C']:
    num_layers = len(self.block_size)
    if len(image_features) != num_layers:
      raise ValueError(
          f'Expected {num_layers} image features, got {len(image_features)}'
      )
    if len(self.block_scale) != num_layers:
      raise ValueError(
          f'Expected {num_layers} upscale factors, got {len(self.block_scale)}'
      )
    features = {f'block_{i}': f for i, f in enumerate(image_features)}

    for i in range(num_layers):
      # Project
      features[f'block_{i}'] = nn.Conv(
          features=self.block_size[i],
          kernel_size=(1, 1),
          strides=(1, 1),
          padding=self.padding,
      )(features[f'block_{i}'])

      # Upscale
      block_scale = self.block_scale[i]
      features[f'block_{i}'] = nn.ConvTranspose(
          kernel_size=(2**block_scale, 2**block_scale),
          padding='SAME',
          strides=(2**block_scale, 2**block_scale),
          features=self.block_size[i],
      )(features[f'block_{i}'])

      # Project
      features[f'block_{i}'] = nn.Conv(
          features=self.feature_dim,
          kernel_size=(1, 1),
          strides=(1, 1),
          padding=self.padding,
      )(features[f'block_{i}'])

    features = conv_decoder.ConvDecoder(
        num_classes=self.num_classes,
        norm=self.norm,
        block_size=tuple([self.feature_dim] * (self.num_layers + 1)),
        last_norm=False,
        denoise=False,
        pool=False,
        num_convs=self.num_convs,
    )(features, train=False)
    return features


# Full DPT reimplementation


class Reassemble(nn.Module):
  """Reassembles the image from the patches."""

  factor: int
  channels: int
  padding: str = 'SAME'

  @nn.compact
  def __call__(self, features: FloatArray['B h w c']):
    # Projection
    features = nn.Conv(
        features=self.channels,
        kernel_size=(1, 1),
        padding=self.padding,
    )(features)

    if self.factor > 1:
      # Upsample
      features = nn.ConvTranspose(
          self.channels,
          kernel_size=(self.factor, self.factor),
          strides=(self.factor, self.factor),
          padding='SAME',
          kernel_dilation=(1, 1),
      )(features)
    elif self.factor < 1:
      # Downsample
      features = nn.Conv(
          self.channels,
          kernel_size=(3, 3),
          strides=(int(1 / self.factor), int(1 / self.factor)),
          padding='SAME',
      )(features)
    return features


class ResidualConvUnit(nn.Module):
  """A residual conv unit."""

  dim: int
  padding: str = 'SAME'
  norm: Callable[[], Any] | None = None
  use_bias: bool = True  # Disable with BatchNorm

  @nn.compact
  def __call__(self, x: FloatArray['B H W C']) -> FloatArray['B H W C']:
    y = nn.Conv(
        features=self.dim,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=self.padding,
        use_bias=self.use_bias,
    )(x)
    if self.norm is not None:
      y = self.norm()(y)
    y = nn.relu(y)

    y = nn.Conv(
        features=self.dim,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=self.padding,
        use_bias=self.use_bias,
    )(y)
    if self.norm is not None:
      y = self.norm()(y)
    y = nn.relu(y)
    return x + y


# From: https://github.com/jax-ml/jax/issues/11206
def resize_with_aligned_corners(
    image: jax.Array,
    shape: tuple[int, ...],
    method: str | jax.image.ResizeMethod,
    antialias: bool,
) -> jax.Array:
  """Alternative to jax.image.resize() with aligned corners (like torch)."""
  spatial_dims = tuple(
      i for i in range(len(shape)) if image.shape[i] != shape[i]
  )
  scale = jnp.array(
      [(shape[i] - 1.0) / (image.shape[i] - 1.0) for i in spatial_dims]
  )
  translation = -(scale / 2.0 - 0.5)
  return jax.image.scale_and_translate(
      image,
      shape,
      method=method,
      scale=scale,
      spatial_dims=spatial_dims,
      translation=translation,
      antialias=antialias,
  )


class Upsample(nn.Module):
  """Upsample 2x with a resize operation."""

  # conv, deconv, (any option that jax.image.scale_and_translate supports)
  method: str = 'conv'

  @nn.compact
  def __call__(self, x):
    if self.method == 'conv':
      norm = functools.partial(nn.GroupNorm, epsilon=1e-6)
      x = nn.ConvTranspose(
          features=256,
          kernel_size=(3, 3),
          strides=(self.scale, self.scale),
          padding='SAME',
      )(x)
      x = norm(x)
    elif self.method == 'deconv':
      norm = functools.partial(nn.GroupNorm, epsilon=1e-6)
      x = conv_decoder.DeConv3x3(
          features=256,
          norm=None,
          padding='SAME',
      )(x, train=True)
      x = norm(x)
    else:
      output_size = x.shape
      output_size = (
          output_size[0],
          output_size[1] * self.scale,
          output_size[2] * self.scale,
          output_size[3],
      )
      x = resize_with_aligned_corners(
          x, output_size, method=self.method, antialias=False
      )
    return x


class FusionBlock(nn.Module):
  """The feature fusion block of DPT."""

  dim: int
  out_dim: int | None = None
  padding: str = 'SAME'

  @nn.compact
  def __call__(
      self,
      features: FloatArray['B H/2 W/2 C'],
      residual: FloatArray['B H/2 w/2 c'] | None = None,
  ) -> FloatArray['B H W C']:

    if residual is not None:
      features = features + ResidualConvUnit(self.dim)(residual)

    features = ResidualConvUnit(self.dim)(features)
    features = Upsample(method='conv')(features)
    features = nn.Conv(
        self.dim if self.out_dim is None else self.out_dim,
        kernel_size=(1, 1),
        padding=self.padding,
    )(features)
    return features


# https://github.com/naver/croco/blob/743ee71a2a9bf57cea6832a9064a70a0597fcfcb/models/dpt_block.py#L280
class DPTNeck(nn.Module):
  """A DPTHead that takes encoder features and outputs dense predictions."""

  block_size: tuple[int, int, int, int] = (96, 192, 384, 768)
  block_scale: tuple[int | float, ...] = (3, 2, 1, 0)
  feature_dim: int = 256
  norm: Callable[..., jnp.ndarray] | None = None

  @nn.compact
  def __call__(
      self,
      image_features: list[FloatArray['B h w c']],
  ) -> FloatArray['B H W C']:
    if len(image_features) != 4:
      raise ValueError(f'Expected 4 layer as input, got {len(image_features)}')

    features = [
        Reassemble(
            self.block_scale[i], self.block_size[i], name=f'reassemble_{i}'
        )(image_features[i])
        for i in range(4)
    ]

    # Project features to a fixed dimension
    features = [
        nn.Conv(
            features=self.feature_dim,
            kernel_size=(3, 3),
            padding='SAME',
            use_bias=False,
            name=f'projection_{i}',
        )(features[i])
        for i in range(4)
    ]

    # Fusion stage
    path_3 = FusionBlock(self.feature_dim)(features[3])
    path_2 = FusionBlock(self.feature_dim)(path_3, residual=features[2])
    path_1 = FusionBlock(self.feature_dim)(path_2, residual=features[1])
    out = FusionBlock(self.feature_dim)(path_1, residual=features[0])

    out = nn_layers.IdentityLayer(name='pre_logits')(out)
    return out


class DPTRegressionHead(nn.Module):
  """A DPTHead that takes encoder features and outputs dense predictions."""

  in_dim: int
  out_dim: int
  hidden_dim: int | None = None
  padding: str = 'SAME'
  norm: Callable[[], Any] | None = None

  @nn.compact
  def __call__(
      self, features: FloatArray['B H/2 W/2 D']
  ) -> FloatArray['B H W C']:
    features = nn.Conv(
        features=self.in_dim,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='SAME',
        name='regression_conv0',
    )(features)

    if self.norm is not None:
      features = self.norm()(features)

    features = nn.relu(features)

    features = nn.Conv(
        features=self.hidden_dim or self.in_dim,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='SAME',
        name='regression_conv1',
    )(features)

    if self.norm is not None:
      features = self.norm()(features)

    features = nn.relu(features)

    features = nn.Conv(
        features=self.out_dim,
        kernel_size=(1, 1),
        strides=(1, 1),
        name='regression_conv2',
        padding=self.padding,
    )(features)

    return features
