# Copyright 2026 Google LLC.
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

"""UPerNet decoder from http://arxiv.org/abs/1807.10221.

Implementation from https://github.com/VMarsocci/pangaea-bench. Decoder for
semantic segmentation.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp


class AdaptiveAveragePooling2D(nn.Module):
  """Performs adaptive average pooling on the input tensor.

  The output size is fixed, and the pooling window is adjusted based on the
  input size. If the input is smaller than the output size, it's first
  upsampled.
  """

  output_size: tuple[int, int]

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    _, h, w, _ = inputs.shape
    target_h, target_w = self.output_size

    def _pool(x: jnp.ndarray) -> jnp.ndarray:
      _, pool_h, pool_w, _ = x.shape
      win_h, win_w = pool_h // target_h, pool_w // target_w
      trimmed = x[:, : target_h * win_h, : target_w * win_w, :]
      reshaped = jnp.reshape(
          trimmed, (-1, target_h, win_h, target_w, win_w, x.shape[-1])
      )
      return jnp.mean(reshaped, axis=(2, 4))

    def upsample_and_pool(x: jnp.ndarray) -> jnp.ndarray:
      resized = jax.image.resize(
          x,
          (x.shape[0], max(h, target_h), max(w, target_w), x.shape[3]),
          method='bilinear',
      )
      return _pool(resized)

    return jax.lax.cond(
        h < target_h or w < target_w, upsample_and_pool, _pool, inputs
    )


class PPM(nn.Module):
  """Pyramid Pooling Module (PPM).

  This module performs multi-scale pooling and upsampling to capture global
  context information, as used in PSPNet and UPerNet.
  """

  channels: int
  pool_scales: tuple[int, ...] = (1, 2, 3, 6)

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = True) -> list[jnp.ndarray]:
    ppm_outs = []
    for scale in self.pool_scales:
      pooled = AdaptiveAveragePooling2D(output_size=(scale, scale))(x)
      conv = nn.Conv(features=self.channels, kernel_size=(1, 1))(pooled)
      bn_out = nn.BatchNorm(use_running_average=not train)(conv)
      act_out = nn.relu(bn_out)
      ppm_outs.append(act_out)
    return ppm_outs


class UPerNetDecoder(nn.Module):
  """The UPerNet decoder module.

  This module takes a list of feature maps from a backbone and processes them
  through a Pyramid Pooling Module (PPM) and Feature Pyramid Network (FPN)
  structure to produce segmentation logits.
  """

  num_classes: int
  channels: int
  in_channels_list: list[int]

  @nn.compact
  def __call__(
      self, features: list[jnp.ndarray], train: bool = True
  ) -> jnp.ndarray:
    x = features[-1]
    psp_outs = PPM(channels=self.channels)(x, train=train)
    concatenated_psp = jnp.concatenate([x] + psp_outs, axis=-1)
    psp_bottleneck_out = nn.Sequential([
        nn.Conv(self.channels, kernel_size=(3, 3), padding='SAME'),
        nn.BatchNorm(use_running_average=not train),
        nn.relu,
    ])(concatenated_psp)
    laterals = [psp_bottleneck_out]
    for i in range(len(self.in_channels_list[:-1])):
      conv = nn.Sequential([
          nn.Conv(self.channels, kernel_size=(1, 1)),
          nn.BatchNorm(use_running_average=not train),
          nn.relu,
      ])(features[i])
      laterals.insert(0, conv)
    for i in range(len(laterals) - 1, 0, -1):
      upsampled = jax.image.resize(
          laterals[i], laterals[i - 1].shape, method='bilinear'
      )
      laterals[i - 1] += upsampled
    fpn_outs = []
    for lateral in laterals:
      conv = nn.Sequential([
          nn.Conv(self.channels, kernel_size=(3, 3), padding='SAME'),
          nn.BatchNorm(use_running_average=not train),
          nn.relu,
      ])(lateral)
      fpn_outs.append(conv)
    target_shape = fpn_outs[0].shape
    for i in range(1, len(fpn_outs)):
      fpn_outs[i] = jax.image.resize(
          fpn_outs[i], target_shape, method='bilinear'
      )
    concatenated_fpn = jnp.concatenate(fpn_outs, axis=-1)
    fused_features = nn.Sequential([
        nn.Conv(self.channels, kernel_size=(3, 3), padding='SAME'),
        nn.BatchNorm(use_running_average=not train),
        nn.relu,
    ])(concatenated_fpn)
    fused_features = nn.Dropout(0.1, deterministic=not train)(fused_features)
    return nn.Conv(self.num_classes, kernel_size=(1, 1))(fused_features)


class FeaturePyramidAdapter(nn.Module):
  """Adapts a single-resolution feature map to a feature pyramid.

  This module takes a high-level feature map (e.g., from a Vision Transformer)
  and generates a feature pyramid with multiple resolutions by upsampling
  and applying 1x1 convolutions.
  """

  embed_dim: int = 768

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> list[jnp.ndarray]:
    f4 = x
    f3 = nn.Conv(self.embed_dim, kernel_size=(1, 1))(
        jax.image.resize(
            f4,
            (f4.shape[0], f4.shape[1] * 2, f4.shape[2] * 2, f4.shape[3]),
            'bilinear',
        )
    )
    f2 = nn.Conv(self.embed_dim, kernel_size=(1, 1))(
        jax.image.resize(
            f3,
            (f3.shape[0], f3.shape[1] * 2, f3.shape[2] * 2, f3.shape[3]),
            'bilinear',
        )
    )
    f1 = nn.Conv(self.embed_dim, kernel_size=(1, 1))(
        jax.image.resize(
            f2,
            (f2.shape[0], f2.shape[1] * 2, f2.shape[2] * 2, f2.shape[3]),
            'bilinear',
        )
    )
    return [f1, f2, f3, f4]


class ViTUPerNet(nn.Module):
  """Vision Transformer-based UPerNet for semantic segmentation.

  This module combines a Vision Transformer backbone (features are expected
  to be provided as input) with a UPerNet-style decoder for dense prediction.
  """

  num_classes: int
  decoder_channels: int
  output_size: tuple[int, int]

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
    feature_pyramid = FeaturePyramidAdapter()(x)
    in_channels_list = [f.shape[-1] for f in feature_pyramid]
    logits = UPerNetDecoder(
        num_classes=self.num_classes,
        channels=self.decoder_channels,
        in_channels_list=in_channels_list,
    )(feature_pyramid, train=train)
    return jax.image.resize(
        logits,
        (logits.shape[0], *self.output_size, logits.shape[3]),
        'bilinear',
    )
