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

"""Vision Transformer adapted for Anygeoloc. Can be used for aerial and sv."""

import dataclasses
import functools
from typing import Any, Final, Mapping, TypeAlias

from absl import logging
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt
import ml_collections
import numpy as np
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers

from unigeoclip.models.utils import attention
from unigeoclip.models.utils import norm
from unigeoclip import load_params
from scaling_geoloc import types
from scaling_geoloc.utils import misc

MaskArray: TypeAlias = jt.Bool[jt.Array, 'B L 1'] | jt.Int[jt.Array, 'B L']

_NORM_LAYER: Final[Mapping[str, type[nn.Module]]] = {
    'layernorm': nn.LayerNorm,
    'DyT': norm.DynamicTanhNorm,
}


class ViTBackbone(nn.Module):
  """Vision Transformer model for AnyGeoLoc.

  Attributes:

  mlp_dim: Dimension of the mlp on top of attention block.
  num_layers: Number of layers.
  num_heads: Number of self-attention heads.
  patches: Configuration of the patches extracted in the stem of the model.
  hidden_size: Size of the hidden state of the output of model's stem.
  head_hidden_dim: Dimension of the hidden layer in the projection mlp.
  head_bottleneck_dim: Dimension of the bottleneck.
  head_output_dim: Dimension of the output ("number of prototypes").
  normalize_last_layer: Normalize the last layer.
  use_bn: Whether or not to use batch normalization layers in the head.
  pos_interpolation_method: Method used to interpolate pos embeddings
  n_registers: Number of registers to use.
  pooling: Either CLS token or global average pooling.
  dropout_rate: Dropout rate.
  attention_dropout_rate: Dropout for attention heads.
  stochastic_depth: Stochastic depth.
  posembs: Positional embedding size.
  approximate_gelu: approximate parameter to GELU activation.
  dtype: JAX data type for activations.
  """

  mlp_dim: int
  num_layers: int
  num_heads: int
  patches: ml_collections.ConfigDict
  hidden_size: int
  head_hidden_dim: int
  head_bottleneck_dim: int
  head_output_dim: int
  normalize_last_layer: bool
  use_bn: bool
  pos_interpolation_method: str
  add_class_token: bool
  n_registers: int = 4
  pooling: str = 'tok'
  dropout_rate: float = 0.1
  norm_layer: type[nn.Module] = nn.LayerNorm
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  positional_embedding: str = 'learned'
  posembs: tuple[int, int] | None = None
  use_bias: bool = True
  approximate_gelu: bool = True
  pos_interpolation_antialias: bool = True
  pos_interpolation_offset: float = 0.0
  freeze_backbone: bool = False  # Freeze entire backbone.
  train_last_layers: int = -1  # Train last n layers (-1: train all).
  imagenet_normalization: bool = False
  conv_embedding: bool = True
  head: str = 'projection'  # 'projection' or 'linear'
  head_config: ml_collections.ConfigDict = dataclasses.field(
      default_factory=ml_collections.ConfigDict
  )
  dtype: Any = jnp.float32

  @classmethod
  def from_config(
      cls, config: ml_collections.ConfigDict, dtype: Any = jnp.float32
  ) -> 'ViTBackbone':
    return ViTBackbone(
        mlp_dim=config.mlp_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        patches=config.patches,
        hidden_size=config.hidden_size,
        head_hidden_dim=config.head_hidden_dim,
        head_bottleneck_dim=config.head_bottleneck_dim,
        head_output_dim=config.head_output_dim,
        normalize_last_layer=config.normalize_last_layer,
        use_bn=config.use_bn,
        add_class_token=config.get(
            'add_class_token', config.pooling in ('tok', 'none')
        ),
        n_registers=config.get('n_registers', 4),
        norm_layer=_NORM_LAYER[config.get('norm_layer', 'layernorm')],
        pooling=config.pooling,
        pos_interpolation_method=config.pos_interpolation_method,
        dropout_rate=config.dropout_rate,
        attention_dropout_rate=config.attention_dropout_rate,
        stochastic_depth=config.stochastic_depth,
        posembs=config.posembs,
        use_bias=config.use_bias,
        freeze_backbone=config.freeze_backbone,
        train_last_layers=config.get('train_last_layers', -1),
        imagenet_normalization=config.get('imagenet_normalization', False),
        pos_interpolation_antialias=config.get(
            'pos_interpolation_antialias', True
        ),
        pos_interpolation_offset=config.get('pos_interpolation_offset', 0.0),
        approximate_gelu=config.get('approximate_gelu', True),
        positional_embedding=config.get('positional_embedding', 'learned'),
        conv_embedding=config.get('conv_embedding', True),
        head=config.get('head', 'projection'),
        head_config=config.get('head_config', ml_collections.ConfigDict()),
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
      seqlen: int = -1,
      mask: MaskArray | None = None,
      mask_mode: str = 'pad_token',
      debug: bool = False,
      pooling: None | str = None,
      sow_attention: bool = False,
      extra_tokens: jt.Float[jt.Array, 'B M C'] | None = None,
      normalize: bool = False,  # logits
      normalize_per_head: bool = False,  # logits
  ) -> jt.Float[jt.Array, 'B L D']:
    height = x.shape[1]
    width = x.shape[2]
    if self.imagenet_normalization:
      # Apply imagenet offset
      x = x.at[..., :3].set(
          jnp.apply_along_axis(
              lambda c: (c - self.imagenet_mean) / self.imagenet_std,
              -1,
              x[..., :3],
          )
      )

    # Patch embedding
    x = nn.Conv(
        self.hidden_size,
        self.patches.size,
        strides=self.patches.size,
        padding='VALID',
        name='embedding',
        dtype=self.dtype,
    )(x)

    if mask is not None and mask.dtype == bool:
      # replace tokens with mask_tok where mask is True
      d = x.shape[-1]
      mask = mask.reshape(x.shape[:3] + (1,))
      if mask_mode == 'pad_token':
        mask_tok = self.param(
            'mask_tok', nn.initializers.zeros, (1, 1, 1, d), x.dtype
        )
      else:
        mask_tok = jnp.zeros_like(x)
      x = jnp.where(mask, mask_tok, x)

    # Add positional encodings.
    x = self.add_positional_encodings(x)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # Possibly drop some tokens.
    if seqlen > 0 and seqlen < h * w:
      droptok_rng = self.make_rng('droptok')
      i_keep = jax.random.permutation(droptok_rng, h * w)[:seqlen]
      x = jnp.take(x, i_keep, axis=1)

    # Add registers
    n_registers = self.n_registers
    if n_registers > 0:
      reg_tok = self.param(
          'reg', nn.initializers.zeros, (1, self.n_registers, c), x.dtype
      )
      reg_tok = jnp.tile(reg_tok, [n, 1, 1])
      x = jnp.concatenate([reg_tok, x], axis=1)

    # Add a "class" token.
    if self.add_class_token:
      cls_tok = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
      cls_tok = jnp.tile(cls_tok, [n, 1, 1])
      x = jnp.concatenate([cls_tok, x], axis=1)
      n_registers += 1

    # Setup attention mask.
    if mask is not None and mask.dtype == bool and mask_mode == 'ignore':
      if n_registers > 0:
        # Append cls token at start of mask.
        mask = jnp.concatenate(
            [jnp.ones_like(mask[:, 0:n_registers, :]), mask], axis=1
        )
      # mask true means remove --> attention_mask true means keep.
      # broadcast and invert the mask ('False' means 'ignore'). Shape: B 1 L L.
      attention_mask = ((~mask) & jnp.swapaxes(~mask, 1, 2))[:, None]

    else:
      attention_mask = None

    if extra_tokens is not None:
      extra_tokens = nn.Dense(self.hidden_size, name='extra_tokens_projection')(
          extra_tokens
      )
      x = jnp.concatenate([x, extra_tokens], axis=1)

    # Input Encoder.
    use_rope = self.positional_embedding == 'rope'
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          use_rope=use_rope,
          norm_layer=self.norm_layer,
          n_registers=n_registers,
          attention_dropout_rate=self.attention_dropout_rate,
          stochastic_depth=(lyr / max(self.num_layers - 1, 1))
          * self.stochastic_depth,
          use_bias=self.use_bias,
          approximate_gelu=self.approximate_gelu,
          name=f'encoderblock_{lyr}',
          dtype=jax.dtypes.canonicalize_dtype(self.dtype),
      )(
          x,
          deterministic=not train,
          height=height // self.patches.size[0],
          width=width // self.patches.size[1],
          sow_attention_weights=(lyr == self.num_layers - 1) and sow_attention,
          attention_mask=attention_mask,
      )
      if (
          self.train_last_layers >= 0
          and lyr + 1 == self.num_layers - self.train_last_layers
      ):
        x = jax.lax.stop_gradient(x)
    x = self.norm_layer(name='encoder_norm')(x)
    backbone_output = nn_layers.IdentityLayer(name='backbone_output')(x)

    if self.freeze_backbone:
      backbone_output = jax.lax.stop_gradient(backbone_output)

    if not pooling:
      pooling = self.pooling

    if pooling == 'tok':
      # Keep only the output class token.
      x = backbone_output[:, 0]
    elif pooling == 'gap':
      x = jnp.mean(backbone_output, axis=1)
      backbone_output = jnp.concatenate([x[:, None], backbone_output], axis=1)
    elif pooling == 'dense':
      return backbone_output[:, n_registers:]
    elif pooling == 'none':
      x = backbone_output
    else:
      raise ValueError(f'Unknown pooling {self.pooling}')

    if self.head_output_dim:
      if self.head == 'projection':
        x = ProjectionHead(
            hidden_dim=self.head_hidden_dim,
            bottleneck_dim=self.head_bottleneck_dim,
            output_dim=self.head_output_dim,
            normalize_last_layer=self.normalize_last_layer,
            use_bn=self.use_bn,
            approximate_gelu=self.approximate_gelu,
            name='output_projection',
        )(x, train)
      elif self.head == 'linear':
        x = nn.Dense(
            self.head_output_dim,
            dtype=self.dtype,
            name='output_projection',
        )(x)
      elif self.head == 'geoclip':
        x = nn.Sequential(
            [
                nn.Dense(768, dtype=self.dtype, name='geoclip_projection_0'),
                nn.relu,
                nn.Dense(
                    self.head_output_dim,
                    dtype=self.dtype,
                    name='geoclip_projection_1',
                ),
            ],
            name='output_projection',
        )(x)
      elif self.head == 'spatial_softmax':
        x = SpatialSoftmax(
            channels=self.head_output_dim,
            name='output_projection',
        )(backbone_output)
      elif self.head == 'fervers':
        pass  # Aggregate in encode_multiview.
      else:
        raise ValueError(f'Unknown head {self.head}')

      if normalize_per_head:
        x = misc.l2_normalize_per_head_vit(x, self.head)
    if normalize:
      x = misc.l2_normalize(x)
    return x

  @nn.compact
  def encode_multiview(
      self,
      x: jt.Float[jt.Array, '... H W C'],
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
    if mask is not None:
      if mask.dtype == bool:
        mask = jax.lax.collapse(mask, 0, -2)
      else:
        mask = jax.lax.collapse(mask, 0, -1)

    if extra_tokens is not None:
      extra_tokens = jax.lax.collapse(extra_tokens, 0, -2)

    return self(
        x_flat,
        train=train,
        mask=mask,
        mask_mode=mask_mode,
        debug=debug,
        pooling=pooling,
        sow_attention=sow_attention,
        extra_tokens=extra_tokens,
        normalize=normalize,
        normalize_per_head=normalize_per_head,
    )

  @nn.compact
  def encode_dense(
      self,
      x: jt.Float[jt.Array, '... H W C'],
      *,
      train: bool = False,
      mask: MaskArray | None = None,
      mask_mode: str = 'pad_token',
      debug: bool = False,
      sow_attention: bool = False,
      extra_tokens: jt.Float[jt.Array, '... M C'] | None = None,
      normalize: bool = False,  # logits
      normalize_per_head: bool = False,  # logits
  ) -> jt.Float[jt.Array, '... L D']:
    x = jax.lax.collapse(x, 0, -3)
    return self(
        x,
        train=train,
        mask=mask,
        mask_mode=mask_mode,
        debug=debug,
        pooling='dense',
        sow_attention=sow_attention,
        extra_tokens=extra_tokens,
        normalize=normalize,
        normalize_per_head=normalize_per_head,
    )

  def output_dim(self) -> int:
    """Returns the output dimension of the logits."""
    if self.head_output_dim:
      return self.head_output_dim
    else:
      return self.mlp_dim

  def add_positional_encodings(
      self, x: jt.Float[jt.Array, '...']
  ) -> jt.Float[jt.Array, '...']:
    """Returns x with the positional embedding added."""
    _, h, w, c = x.shape
    if self.positional_embedding == 'rope':
      return x
    elif self.positional_embedding == 'learned':
      assert self.posembs is not None
      posemb = self.param(
          'posembed_input',
          nn.initializers.normal(stddev=1 / np.sqrt(c)),
          (1, self.posembs[0], self.posembs[1], c),
          x.dtype,
      )
      # Optionally resize the positional encodings.
      if (h, w) != self.posembs:
        if self.pos_interpolation_offset:
          # Approximately replicate positional embedding interpolation used by
          # default in the official PyTorch implementation. This is necessarily
          # inexact because Jax bicubic interpolation differs from that in
          # PyTorch (go/jax-issues/15768); using method='lanczos3',
          # antialias=False and offset=0.1 gives the closest result to PyTorch.
          posemb = jax.image.scale_and_translate(
              posemb,
              (1, h, w, c),
              spatial_dims=(1, 2),
              scale=(jnp.array([h, w]) + self.pos_interpolation_offset)
              / jnp.array(self.posembs),
              translation=jnp.zeros([2]),
              method=self.pos_interpolation_method,
              antialias=self.pos_interpolation_antialias,
          )
        else:
          posemb = jax.image.resize(
              posemb,
              (1, h, w, c),
              self.pos_interpolation_method,
              antialias=self.pos_interpolation_antialias,
          )
      x = x + posemb
    elif self.positional_embedding == 'sinusoidal_2d':
      x = attention_layers.AddFixedSinCosPositionEmbedding()(x)

    return x

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


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of self-attention heads.
    dtype: The dtype of the computation (default: float32).
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
      the provided value.
    approximate_gelu: approximate parameter to GELU activation.

  Returns:
    output after transformer encoder block.
  """

  mlp_dim: int
  num_heads: int
  dtype: Any = jnp.float32
  dropout_rate: float = 0.1
  use_rope: bool = False
  n_registers: int = 0
  norm_layer: type[nn.Module] = nn.LayerNorm
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  use_bias: bool = True
  approximate_gelu: bool = True

  @nn.compact
  def __call__(
      self,
      inputs: jt.Float[jt.Array, 'B L C'],
      height: int,
      width: int,
      deterministic: bool,
      sow_attention_weights: bool = False,
      attention_mask: jt.Bool[jt.Array, 'B H L L'] | None = None,
  ) -> jt.Bool[jt.Array, 'B L C']:
    """Applies Encoder1DBlock module.

    Args:
      inputs: Input data.
      height: Height of the image.
      width: Width of the image.
      deterministic: Deterministic or not (to apply dropout).
      sow_attention_weights: Whether attention weights should be sowed.
      attention_mask: Ignore attention weights where mask is ``False``.

    Returns:
      Output after transformer encoder block.
    """
    # Attention block.
    assert inputs.ndim == 3
    x = self.norm_layer(dtype=self.dtype)(inputs)
    x = attention.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        use_rope=self.use_rope,
        norm_layer=self.norm_layer,
        n_registers=self.n_registers,
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        use_bias=self.use_bias,
    )(
        x,
        x,
        height=height,
        width=width,
        sow_weights=sow_attention_weights,
        mask=attention_mask,
    )
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = nn_layers.StochasticDepth(rate=self.stochastic_depth)(x, deterministic)
    x = x + inputs

    # MLP block.
    y = self.norm_layer(dtype=self.dtype)(x)
    y = attention_layers.MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=functools.partial(
            nn.gelu, approximate=self.approximate_gelu
        ),
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )(y, deterministic=deterministic)
    y = nn_layers.StochasticDepth(rate=self.stochastic_depth)(y, deterministic)
    return y + x


def norm_kernel_init_fn(rng, shape, dtype):
  """Initialize kernel with l2 normalized columns."""
  param = nn.linear.default_kernel_init(rng, shape, dtype)
  param = misc.l2_normalize(param, axis=0)
  return param


class ProjectionHead(nn.Module):
  """Projection head.

  Attributes:
    hidden_dim: Dimension of the hidden layer in the projection mlp.
    bottleneck_dim: Dimension of the bottleneck.
    output_dim: Dimension of the output ("number of prototypes").
    normalize_last_layer: Normalize the last layer of prototypes.
    use_bn: Use batch normalizations.
    approximate_gelu: approximate parameter to GELU activation.
    n_layers: Depth of the projection head.
  """

  hidden_dim: int = 2048
  bottleneck_dim: int = 256
  output_dim: int = 65536
  normalize_last_layer: bool = False
  use_bn: bool = False
  approximate_gelu: bool = True
  n_layers: int = 2

  @nn.compact
  def __call__(
      self, x: jt.Float[jt.Array, 'B L C'], train: bool
  ) -> jt.Float[jt.Array, 'B L D']:
    batch_norm = nn_layers.IdentityLayer
    if self.use_bn:
      batch_norm = functools.partial(
          nn.BatchNorm,
          use_running_average=not train,
          momentum=0.9,
          epsilon=1e-5,
      )
    for i in range(self.n_layers):
      x = nn.Dense(self.hidden_dim)(x)
      x = batch_norm()(x)
      x = nn.gelu(x, approximate=self.approximate_gelu)
      x = nn_layers.IdentityLayer(name=f'mlp_{i}')(x)
    x = nn.Dense(self.bottleneck_dim)(x)
    # Normalize.
    x = misc.l2_normalize(x)
    x = WeightNormDense(
        self.output_dim,
        use_bias=False,
        name='prototypes',
        kernel_init=norm_kernel_init_fn,
        # no normalize now means apply rescale
        apply_rescale=not self.normalize_last_layer,
    )(x)
    x = nn_layers.IdentityLayer(name='pre_logits')(x)
    return x


class WeightNormDense(nn.Dense):
  """Linear layer with weight normalized kernel."""

  apply_rescale: bool = False

  def param(self, name: str, *args, **kwargs):
    param = super().param(name, *args, **kwargs)
    if name == 'kernel':
      param = misc.l2_normalize(param, axis=0)

      if self.apply_rescale:
        scale = super().param(
            'weightnorm_scale',
            jax.nn.initializers.ones,
            (1, param.shape[-1]),
            self.param_dtype,
        )

        param *= scale
    return param


class SpatialSoftmax(nn.Module):
  """Spatial softmax."""

  channels: int
  heads: int = 64
  bias: bool = False

  @nn.compact
  def __call__(
      self, x: jt.Float[jt.Array, 'B N C']
  ) -> jt.Float[jt.Array, 'B C']:
    """Applies softmax to the spatial dimension."""
    attn = nn.Dense(self.heads, use_bias=self.bias)(x)
    attn = jax.nn.softmax(attn, axis=-2)
    value = nn.Dense(self.channels)(x)
    value = value.reshape(*value.shape[:-1], self.heads, -1)
    return (value * attn[..., None]).sum(axis=-3).reshape(*x.shape[:-2], -1)
