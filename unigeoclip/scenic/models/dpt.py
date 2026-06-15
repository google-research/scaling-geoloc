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

"""A Flax implementation of the DPT model for dense prediction tasks.

This module implements the Dense Prediction Transformer (DPT) architecture,
which leverages a Vision Transformer (ViT) backbone and a convolutional
decoder to generate dense predictions for tasks like depth estimation.
"""

from typing import Callable, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


Activation = Callable[[jnp.ndarray], jnp.ndarray]
Shape = Tuple[int, ...]
Initializer = Callable[[jnp.ndarray, Shape, jnp.dtype], jnp.ndarray]


class ConvModule(nn.Module):
  """A convolutional module with optional normalization and activation.

  Attributes:
    features: Number of output channels.
    kernel_size: Size of the convolution kernel.
    stride: Stride of the convolution.
    padding: Padding mode.
    dilation: Dilation rate.
    groups: Number of feature groups.
    use_bias: Whether to use a bias term.
    use_norm: Whether to use normalization layer.
    activation: Activation function.
    order: Order of operations ('conv', 'norm', 'act').
    kernel_init: Initializer for the convolution kernel.
    bias_init: Initializer for the bias.
  """

  features: int
  kernel_size: Tuple[int, int]
  stride: Tuple[int, int] = (1, 1)
  padding: str | int | Sequence[Tuple[int, int]] = 'SAME'
  dilation: Tuple[int, int] = (1, 1)
  groups: int = 1
  use_bias: bool = True
  use_norm: bool = True
  activation: Optional[Activation] = nn.relu
  order: Tuple[str, str, str] = ('conv', 'norm', 'act')
  kernel_init: Initializer = nn.initializers.kaiming_normal()
  bias_init: Initializer = nn.initializers.zeros

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      train: bool,
      activate: bool = True,
      norm: bool = True,
  ) -> jnp.ndarray:
    """Applies the convolutional module to the input.

    Args:
      x: Input tensor.
      train: Whether the model is in training mode.
      activate: Whether to apply the activation function.
      norm: Whether to apply batch normalization.

    Returns:
      The output tensor.
    """
    conv_layer = nn.Conv(
        features=self.features,
        kernel_size=self.kernel_size,
        strides=self.stride,
        padding=self.padding,
        kernel_dilation=self.dilation,
        feature_group_count=self.groups,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
    )
    norm_layer = nn.LayerNorm if self.use_norm else None

    for layer_type in self.order:
      if layer_type == 'conv':
        x = conv_layer(x)
      elif layer_type == 'norm':
        if norm and self.use_norm:
          assert norm_layer is not None
          x = norm_layer()(x)
      elif layer_type == 'act':
        if activate and self.activation is not None:
          x = self.activation(x)
      else:
        raise ValueError(f'Unsupported layer type: {layer_type}')
    return x


class UpConvHead(nn.Module):
  """Upsampling and convolutional head.

  Attributes:
    features: Number of features in the intermediate convolution.
    n_output_channels: Number of output channels.
    n_hidden_channels: Number of hidden channels.
  """

  features: int
  n_output_channels: int
  n_hidden_channels: int = 32

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies the upsampling head to the input tensor.

    Args:
      x: Input tensor.

    Returns:
      The upsampled output tensor.
    """
    assert x.shape[-1] == self.features
    x = nn.Conv(
        features=self.features // 2, kernel_size=(3, 3), padding='SAME'
    )(x)
    new_shape = (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3])
    x = jax.image.resize(x, shape=new_shape, method='bilinear')
    x = nn.Conv(
        features=self.n_hidden_channels, kernel_size=(3, 3), padding='SAME'
    )(x)
    x = nn.relu(x)
    x = nn.Conv(features=self.n_output_channels, kernel_size=(1, 1))(x)
    return x


class ReassembleBlocks(nn.Module):
  """Rearranges ViT patch features into a feature pyramid.

  **This version does NOT use a cls_token.**
  """

  in_channels: Sequence[int]
  out_channels: Sequence[int]
  use_norm: bool = False

  def setup(self):
    self.projects = [
        ConvModule(
            features=out_ch, kernel_size=(1, 1), activation=None, use_norm=False
        )
        for _, out_ch in zip(self.in_channels, self.out_channels, strict=True)
    ]
    self.resize_layers = [
        nn.ConvTranspose(
            features=self.out_channels[0], kernel_size=(4, 4), strides=(4, 4)
        ),
        nn.ConvTranspose(
            features=self.out_channels[1], kernel_size=(2, 2), strides=(2, 2)
        ),
        lambda x: x,
        nn.Conv(
            features=self.out_channels[3],
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='SAME',
        ),
    ]
    if self.use_norm:
      self.norm_layers = [nn.LayerNorm() for _ in self.in_channels]

  def __call__(
      self, inputs: Sequence[jnp.ndarray], train: bool
  ) -> Sequence[jnp.ndarray]:
    """Rearranges ViT patch features into a feature pyramid.

    Args:
      inputs: A Sequence of feature maps from the ViT backbone.
      train: Whether the model is in training mode.

    Returns:
      A Sequence of feature maps forming a feature pyramid.
    """
    out = []
    for i, x in enumerate(inputs):
      if self.use_norm:
        norm = self.norm_layers[i]
        x = norm(x)
      x = self.projects[i](x, train=train)
      x = self.resize_layers[i](x)
      out.append(x)
    return out


class PreActResidualConvUnit(nn.Module):
  """Pre-activation residual convolutional unit.

  Attributes:
    features: Number of output features.
  """

  features: int

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
    """Applies the pre-activation residual unit to the input.

    Args:
      x: Input tensor.
      train: Whether the model is in training mode.

    Returns:
      The output tensor after the residual connection.
    """
    conv_order = ('act', 'conv', 'norm')
    y = ConvModule(
        features=self.features,
        kernel_size=(3, 3),
        padding='SAME',
        use_bias=False,
        order=conv_order,
    )(x, train=train)
    y = ConvModule(
        features=self.features,
        kernel_size=(3, 3),
        padding='SAME',
        use_bias=False,
        order=conv_order,
    )(y, train=train)
    return x + y


class FeatureFusionBlock(nn.Module):
  """Feature fusion block.

  Attributes:
    features: Number of features.
    expand: Whether to expand the number of features.
  """

  features: int
  expand: bool = False

  @nn.compact
  def __call__(self, *inputs: jnp.ndarray, train: bool) -> jnp.ndarray:
    """Fuses feature maps.

    This block can take one or two inputs. If two inputs are provided, they
    are combined with a residual connection.

    Args:
      *inputs: One or two input tensors.
      train: Whether the model is in training mode.

    Returns:
      The fused and processed feature map.
    """
    x = inputs[0]
    if len(inputs) == 2:
      res = inputs[1]
      if x.shape != res.shape:
        res = jax.image.resize(res, x.shape, method='bilinear')
      res_processed = PreActResidualConvUnit(features=self.features)(
          res, train=train
      )
      x = x + res_processed
    elif len(inputs) > 2:
      raise ValueError(f'Too many inputs: {len(inputs)}')
    x = PreActResidualConvUnit(features=self.features)(x, train=train)
    new_shape = (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3])
    x = jax.image.resize(x, shape=new_shape, method='bilinear')
    out_features = self.features // 2 if self.expand else self.features
    x = ConvModule(
        features=out_features,
        kernel_size=(1, 1),
        activation=None,
        use_norm=False,
    )(x, train=train)
    return x


class DPTHead(nn.Module):
  """DPT decoder head.

  This module takes features from a ViT backbone and processes them through a
  series of reassembly, fusion, and upsampling blocks to produce a dense
  prediction.

  Attributes:
    in_channels: A list of input channel dimensions.
    channels: The number of channels in the fusion blocks.
    post_process_channels: A list of channel dimensions after reassembly.
    n_output_channels: The number of output channels for the final prediction.
    use_norm: Whether to use normalization in the reassembly blocks.
    expand_channels: Whether to expand channels in the final fusion block.
  """

  in_channels: Sequence[int]
  post_process_channels: Sequence[int]
  channels: int = 256
  n_output_channels: int = 256
  use_norm: bool = False
  expand_channels: bool = False

  def setup(self):
    self.reassemble_blocks = ReassembleBlocks(
        in_channels=self.in_channels,
        out_channels=self.post_process_channels,
        use_norm=self.use_norm,
    )
    self.convs = [
        ConvModule(
            features=self.channels,
            kernel_size=(3, 3),
            padding='SAME',
            activation=None,
            use_bias=False,
        )
        for _ in self.post_process_channels
    ]
    self.fusion_blocks = [
        FeatureFusionBlock(features=self.channels)
        for _ in self.post_process_channels
    ]
    self.project = ConvModule(
        features=self.channels, kernel_size=(3, 3), padding='SAME'
    )
    self.conv_depth = UpConvHead(
        features=self.channels, n_output_channels=self.n_output_channels
    )

  def __call__(self, inputs: Sequence[jnp.ndarray], train: bool) -> jnp.ndarray:
    """Forward pass of the DPT head.

    Args:
      inputs: A list of feature maps from the ViT backbone.
      train: Whether the model is in training mode.

    Returns:
      The final dense prediction.
    """
    x = self.reassemble_blocks(inputs, train=train)
    x = [conv(f, train=train) for conv, f in zip(self.convs, x, strict=True)]
    out = self.fusion_blocks[0](x[-1], train=train)
    for fuse, feature in zip(
        self.fusion_blocks[1:], reversed(x[:-1]), strict=True
    ):
      out = fuse(out, feature, train=train)
    out = self.project(out, train=train)
    out = self.conv_depth(out)
    return out
