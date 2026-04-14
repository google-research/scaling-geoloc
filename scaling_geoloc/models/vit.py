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

"""Vision Transformer adapted for lama."""

import dataclasses
import functools
from typing import Any, Literal, TypeAlias
from absl import logging
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt
import ml_collections
import numpy as np
from scenic.model_lib.base_models import base_model
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers
from scaling_geoloc import types
from scaling_geoloc.models import heads
from scaling_geoloc.models import rope_encoding
from scaling_geoloc.utils import misc

ValidIndex = jt.Int[jt.Array, 'L']

MaskArray: TypeAlias = jt.Bool[jt.Array, 'B L 1'] | jt.Int[jt.Array, 'B L']


class ToTokenSequence(nn.Module):
  """Transform a batch of views into a sequence of tokens."""

  patches: ml_collections.ConfigDict
  pos_interpolation_method: str
  hidden_size: int
  num_heads: int
  add_class_token: bool = True
  num_storage_tokens: int = 0
  posembs: tuple[int, int] | None = None
  positional_embedding: str = 'learned'
  pos_interpolation_antialias: bool = True
  pos_interpolation_offset: float = 0.0
  conv_embedding: bool = True
  rope_config: ml_collections.ConfigDict = dataclasses.field(
      default_factory=ml_collections.ConfigDict
  )

  @nn.compact
  def add_positional_encodings(
      self, x: jt.Float[jt.Array, 'B H W C']
  ) -> jt.Float[jt.Array, 'B H W C']:
    """Add positional encodings to the input patch sequence."""
    _, h, w, c = x.shape
    if self.positional_embedding == 'learned':
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

  @nn.compact
  def __call__(
      self,
      x: jt.Float[jt.Array, 'B H W 3|3+P'],
      *,
      train: bool,
      seqlen: int = -1,
      mask: MaskArray | None = None,
      mask_mode: str = 'pad_token',
  ) -> tuple[
      jt.Float[jt.Array, 'B L C'],
      tuple[jt.Float[jt.Array, 'S_ROPE D'], jt.Float[jt.Array, 'S_ROPE D']]
      | None,
  ]:
    """Transform a batch of views into a sequence of tokens."""
    if x.shape[-1] == 3:
      pixel_coordinates = None
    else:
      x, pixel_coordinates = x[..., :3], x[..., 3:]
    fh, fw = self.patches.size
    if self.conv_embedding:
      # Extracting patches and then embedding is in fact a single convolution.
      x = nn.Conv(
          self.hidden_size,
          (fh, fw),
          strides=(fh, fw),
          padding='VALID',
          name='embedding',
      )(x)
    else:
      x = nn.Dense(self.hidden_size, name='embedding')(x)

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

    b, h, w, c = x.shape
    rope_embeds = None
    if self.rope_config.get('use_rope', False):
      rope_embeds = rope_encoding.RopePositionEmbedding(
          embed_dim=self.hidden_size,
          num_heads=self.num_heads,
          base=self.rope_config.get('base', 100.0),
          min_period=self.rope_config.get('min_period', None),
          max_period=self.rope_config.get('max_period', None),
          normalize_coords=self.rope_config.get('normalize_coords', 'separate'),
          shift_coords=self.rope_config.get('shift_coords', None),
          jitter_coords=self.rope_config.get('jitter_coords', None),
          rescale_coords=self.rope_config.get('rescale_coords', None),
          dtype=self.rope_config.get('dtype', x.dtype),
      )(h, w, deterministic=not train)
    else:
      x = self.add_positional_encodings(x)
    x = jnp.reshape(x, [b, h * w, c])

    if pixel_coordinates is not None:
      p = nn.Conv(
          self.hidden_size,
          (fh, fw),
          strides=(fh, fw),
          padding='VALID',
          name='pos_embed_conv',
      )(pixel_coordinates)
      p = nn.relu(p)
      p = nn.Dense(self.hidden_size, name='pos_embed_project')(p)
      skip_connection = self.param(
          'pos_embed_skip_connection', nn.initializers.zeros, (1, 1, 1), x.dtype
      )
      x = x + skip_connection * p.reshape(x.shape)
      # x = nn.Dense(self.hidden_size, name='pos_embed_project')(
      #     jnp.concatenate([x, p.reshape(x.shape)], axis=-1)
      # )

    if mask is not None and jnp.issubdtype(mask.dtype, jnp.integer):
      # Fixed size mask, drop remaining tokens.
      x = jnp.take_along_axis(x, mask[..., None], axis=-2)
    n, l, c = x.shape

    # Possibly drop some tokens.
    if seqlen > 0 and seqlen < l:
      droptok_rng = self.make_rng('droptok')
      i_keep = jax.random.permutation(droptok_rng, l)[:seqlen]
      x = jnp.take(x, i_keep, axis=1)

    # Add storage tokens.
    if self.num_storage_tokens > 0:
      storage_tok = self.param(
          'storage',
          nn.initializers.zeros,
          (1, self.num_storage_tokens, c),
          x.dtype,
      )
      storage_tok = jnp.tile(storage_tok, [n, 1, 1])
      x = jnp.concatenate([storage_tok, x], axis=1)

    # Add a "class" token.
    if self.add_class_token:
      cls_tok = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
      cls_tok = jnp.tile(cls_tok, [n, 1, 1])
      x = jnp.concatenate([cls_tok, x], axis=1)

    return x, rope_embeds


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
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  use_bias: bool = True
  approximate_gelu: bool = True

  @nn.compact
  def __call__(
      self,
      inputs: jt.Float[jt.Array, 'B L C'],
      deterministic: bool,
      sow_attention_weights: bool = False,
      attention_mask: jt.Bool[jt.Array, 'B H L L'] | None = None,
      rope_embeds: (
          tuple[jt.Float[jt.Array, 'S_ROPE D'], jt.Float[jt.Array, 'S_ROPE D']]
          | None
      ) = None,
  ) -> jt.Bool[jt.Array, 'B L C']:
    """Applies Encoder1DBlock module.

    Args:
      inputs: Input data.
      deterministic: Deterministic or not (to apply dropout).
      sow_attention_weights: Whether attention weights should be sowed.
      attention_mask: Ignore attention weights where mask is ``False``.
      rope_embeds: Optional rope embeddings to be used in attention.

    Returns:
      Output after transformer encoder block.
    """
    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        use_bias=self.use_bias,
        attention_fn=lambda q, k, v, **kwargs: rope_encoding.dot_product_attention(
            q, k, v, rope_embeds, **kwargs
        ),
    )(x, x, sow_weights=sow_attention_weights, mask=attention_mask)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = nn_layers.StochasticDepth(rate=self.stochastic_depth)(x, deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
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


class ViTBackbone(nn.Module):
  """Vision Transformer model for training.

  Attributes:

  mlp_dim: Dimension of the mlp on top of attention block.
  num_layers: Number of layers.
  num_heads: Number of self-attention heads.
  patches: Configuration of the patches extracted in the stem of the model.
  hidden_size: Size of the hidden state of the output of model's stem.
  head_output_dim: Dimension of the output ("number of prototypes").
  pos_interpolation_method: Method used to interpolate pos embeddings
  add_class_token: Whether to add a class token.
  num_storage_tokens: Number of storage/register tokens.
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
  head_output_dim: int
  pos_interpolation_method: str
  add_class_token: bool
  num_storage_tokens: int = 0
  pooling: str = 'tok'
  dropout_rate: float = 0.1
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
  normalization: Literal['none', 'imagenet', 'sat493m'] = 'imagenet'
  conv_embedding: bool = True
  head: str = 'projection'  # 'projection' or 'linear'
  head_config: ml_collections.ConfigDict = dataclasses.field(
      default_factory=ml_collections.ConfigDict
  )
  rope_config: ml_collections.ConfigDict = dataclasses.field(
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
        head_output_dim=config.head_output_dim,
        add_class_token=config.get(
            'add_class_token', config.pooling in ('tok', 'none')
        ),
        num_storage_tokens=config.get('num_storage_tokens', 0),
        pooling=config.pooling,
        pos_interpolation_method=config.pos_interpolation_method,
        dropout_rate=config.dropout_rate,
        attention_dropout_rate=config.attention_dropout_rate,
        stochastic_depth=config.stochastic_depth,
        posembs=config.posembs,
        use_bias=config.use_bias,
        freeze_backbone=config.freeze_backbone,
        train_last_layers=config.get('train_last_layers', -1),
        normalization=config.get(
            'normalization',
            'imagenet'
            if config.get('imagenet_normalization', False)
            else 'none',
        ),
        use_big_vision_backbone=config.get(
            'use_big_vision_backbone', config.get('siglip_normalization', False)
        ),
        pos_interpolation_antialias=config.get(
            'pos_interpolation_antialias', True
        ),
        pos_interpolation_offset=config.get('pos_interpolation_offset', 0.0),
        approximate_gelu=config.get('approximate_gelu', True),
        positional_embedding=config.get('positional_embedding', 'learned'),
        conv_embedding=config.get('conv_embedding', True),
        head=config.get('head', 'projection'),
        head_config=config.get('head_config', ml_collections.ConfigDict()),
        rope_config=config.get('rope_config', ml_collections.ConfigDict()),
        dtype=dtype,
    )

  def setup(self):
    self.dataset_mean, self.dataset_std = {
        'imagenet': (
            jnp.array([0.485, 0.456, 0.406]),
            jnp.array([0.229, 0.224, 0.225]),
        ),
        'sat493m': (
            jnp.array([0.430, 0.411, 0.296]),
            jnp.array([0.213, 0.156, 0.143]),
        ),
        'none': (jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 1.0, 1.0])),
    }[self.normalization]

  def compute_attention_mask(
      self,
      mask: MaskArray | None,
      mask_mode: str,
      add_class_token: bool,
  ) -> jt.Bool[jt.Array, 'B 1 L L'] | None:
    """Computes the attention mask based on the input mask and mode."""
    if mask is not None and mask.dtype == bool and mask_mode == 'ignore':
      if add_class_token:
        # Append cls token at start of mask.
        mask = jnp.concatenate([jnp.ones_like(mask[:, 0:1, :]), mask], axis=1)
      # mask true means remove --> attention_mask true means keep.
      # broadcast and invert the mask ('False' means 'ignore'). Shape: B 1 L L.
      attention_mask = ((~mask) & jnp.swapaxes(~mask, 1, 2))[:, None]
      return attention_mask

    return None

  def forward_backbone(
      self,
      x: jt.Float[jt.Array, 'B H W C'],
      *,
      train: bool,
      seqlen: int,
      mask: MaskArray | None,
      mask_mode: str,
      sow_attention: bool,
      extra_tokens: jt.Float[jt.Array, 'B M C'] | None,
  ) -> jt.Float[jt.Array, 'B L D']:
    """Run backbone."""

    if self.normalization != 'none':
      x = x.at[..., :3].set(
          jnp.apply_along_axis(
              lambda c: (c - self.dataset_mean) / self.dataset_std,
              -1,
              x[..., :3],
          )
      )

    x, rope_embeds = ToTokenSequence(
        patches=self.patches,
        positional_embedding=self.positional_embedding,
        pos_interpolation_method=self.pos_interpolation_method,
        pos_interpolation_antialias=self.pos_interpolation_antialias,
        pos_interpolation_offset=self.pos_interpolation_offset,
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        add_class_token=self.add_class_token,
        num_storage_tokens=self.num_storage_tokens,
        posembs=self.posembs,
        conv_embedding=self.conv_embedding,
        rope_config=self.rope_config,
    )(x, train=train, seqlen=seqlen, mask=mask, mask_mode=mask_mode)
    x = nn_layers.IdentityLayer(name='posembed_input')(x)

    attention_mask = self.compute_attention_mask(
        mask, mask_mode, self.add_class_token
    )

    if extra_tokens is not None:
      extra_tokens = nn.Dense(self.hidden_size, name='extra_tokens_projection')(
          extra_tokens
      )
      x = jnp.concatenate([x, extra_tokens], axis=1)

    # Input Encoder.
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
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
          sow_attention_weights=(lyr == self.num_layers - 1) and sow_attention,
          attention_mask=attention_mask,
          rope_embeds=rope_embeds,
      )
      if (
          self.train_last_layers >= 0
          and lyr + 1 == self.num_layers - self.train_last_layers
      ):
        x = jax.lax.stop_gradient(x)
    x = nn.LayerNorm(name='encoder_norm')(x)
    backbone_output = nn_layers.IdentityLayer(name='backbone_output')(x)

    if self.freeze_backbone:
      backbone_output = jax.lax.stop_gradient(backbone_output)

    return backbone_output

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
  ) -> tuple[jt.Float[jt.Array, 'B L D'], jt.Float[jt.Array, 'B D']]:

    if not pooling:
      pooling = self.pooling

    backbone_output = self.forward_backbone(
        x,
        train=train,
        seqlen=seqlen,
        mask=mask,
        mask_mode=mask_mode,
        sow_attention=sow_attention,
        extra_tokens=extra_tokens,
    )

    if pooling == 'tok':
      # Keep only the output class token.
      x = backbone_output[:, 0]
    elif pooling == 'gap':
      x = jnp.mean(backbone_output, axis=1)
      backbone_output = jnp.concatenate([x[:, None], backbone_output], axis=1)
    elif pooling == 'none':
      x = backbone_output
    else:
      raise ValueError(f'Unknown pooling {pooling}')

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

  def get_patch_token_indices(self, height: int, width: int) -> tuple[int, int]:
    """Returns the start and end index of patch tokens.

    Sequence of tokens is:
    [class token] + [storage tokens] + [patch tokens] + [extra tokens]

    Args:
      height: The height of the input image.
      width: The width of the input image.

    Returns:
      A tuple of the start (inclusive) and end (exclusive) index of patch
      tokens.
    """
    num_patch_tokens = (height // self.patches.size[0]) * (
        width // self.patches.size[1]
    )
    start_index = (1 if self.add_class_token else 0) + self.num_storage_tokens
    end_index = start_index + num_patch_tokens
    return start_index, end_index

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
    if mask is not None:
      if mask.dtype == bool:
        mask = jax.lax.collapse(mask, 0, -2)
      else:
        mask = jax.lax.collapse(mask, 0, -1)

    if extra_tokens is not None:
      extra_tokens = jax.lax.collapse(extra_tokens, 0, -2)
    logits_flat, backbone_output_flat = self(
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

    logits = logits_flat.reshape(x.shape[:-3] + logits_flat.shape[1:])
    backbone_output = backbone_output_flat.reshape(
        x.shape[:-3] + backbone_output_flat.shape[1:]
    )

    if self.head == 'fervers':
      # Perform softmax over all multi-view embeddings.
      flat_tokens = backbone_output.reshape(
          x.shape[:1] + (-1, backbone_output_flat.shape[-1])
      )
      logits = heads.SpatialSoftmax(
          channels=self.head_output_dim,
          name='output_projection',
      )(flat_tokens)
      if x.ndim == 5:
        # Add a dimension for the number of multi-view images.
        logits = logits[..., None, :].repeat(x.shape[1], axis=-2)
      elif x.ndim > 5:
        raise ValueError(f'Unsupported number of dimensions: {x.ndim}')
      if normalize:
        logits = misc.l2_normalize(logits)
    return logits, backbone_output

  @nn.compact
  def encode_masked(
      self,
      x: jt.Float[jt.Array, '... H W 3'],
      mask: float | jt.Int[jt.Array, '... L'],
      *,
      train: bool,
      debug: bool = False,
      mask_mode: str = 'pad_token',
      pooling: None | str = None,
      sow_attention: bool = False,
      extra_tokens: jt.Float[jt.Array, '... M C'] | None = None,
      normalize: bool = False,  # logits
      normalize_per_head: bool = False,  # logits
  ) -> tuple[
      jt.Float[jt.Array, '... D'],
      jt.Float[jt.Array, '... L+1 D'],
      jt.Bool[jt.Array, '... L 1'],
  ]:
    bs = x.shape[:-3]
    x_flat = jax.lax.collapse(x, 0, -3)
    bf, h, w, _ = x_flat.shape
    ph, pw = self.patches.size
    nh, nw = h // ph, w // pw
    num_tokens = nh * nw
    if isinstance(mask, float):
      num_masked = int(num_tokens * mask)
      to_encode, _ = misc.shuffle_and_partition(
          n_batch=bf,
          n_tokens=num_tokens,
          n_masked=num_masked,
          rng=self.make_rng('masking_rng'),
      )
    else:
      to_encode = mask

    if extra_tokens is not None:
      extra_tokens = jax.lax.collapse(extra_tokens, 0, -2)

    logits, x = self(
        x_flat,
        train=train,
        mask=to_encode,
        mask_mode='ignore',
        debug=debug,
        pooling=pooling,
        sow_attention=sow_attention,
        extra_tokens=extra_tokens,
        normalize=normalize,
        normalize_per_head=normalize_per_head,
    )

    assert self.head != 'fervers'

    if mask_mode == 'ignore':
      ret = (logits, x, to_encode)
    elif mask_mode == 'pad_token':
      is_masked = ~misc.indices_to_mask(to_encode, num_tokens)
      x_cls, x_patch = x[..., :1, :], x[..., 1:, :]

      x_unrolled = jnp.zeros(
          (bf, num_tokens) + x_patch.shape[-1:], dtype=x_patch.dtype
      )
      shared_mask_token = self.param(
          'shared_mask_token',
          nn.initializers.normal(stddev=0.02),
          x_patch.shape[-1:],
      )
      x_unrolled = x_unrolled + shared_mask_token
      all_rows = jnp.arange(bf).reshape(bf, 1)
      x_unrolled = x_unrolled.at[all_rows, to_encode].set(
          jax.lax.collapse(x_patch, 0, -2)
      )
      x_unrolled = jnp.concatenate([x_cls, x_unrolled], axis=-2)
      ret = (logits, x_unrolled, is_masked[..., None])
    else:
      raise ValueError(f'Unknown mask mode {mask_mode}')
    out = [x.reshape(*bs, *x.shape[1:]) for x in ret]
    return tuple(out)

  def output_dim(self) -> int:
    """Returns the output dimension of the logits."""
    if self.head_output_dim:
      return self.head_output_dim
    else:
      return self.mlp_dim

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
      raise NotImplementedError('Loading pretrained weights is not implemented.')

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


class ViTModel(base_model.BaseModel):
  """Vision Transformer model for training."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return ViTBackbone.from_config(self.config.model, dtype=model_dtype)
