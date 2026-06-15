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

"""Code for Attention implementation thaht support RoPe."""

from __future__ import annotations

from collections.abc import Callable
import functools
import inspect
from typing import Any, overload
import warnings

import flax.linen as nn
from flax.linen import attention
from flax.linen import initializers
from flax.linen import linear
from flax.linen import module
import flax.typing as ft
import jax
from jax import lax
import jax.numpy as jnp
import jaxtyping as jt

from unigeoclip.scenic.models.utils import rope


def get_positions(
    batch_size: int, height: int, width: int, n_registers: int = 5
) -> jt.Int[jt.Array, '*B L 2']:
  """Returns the positions given a number of patches."""
  coord_x = jnp.arange(height)
  coord_y = jnp.arange(width)

  coords_x_grid, coords_y_grid = jnp.meshgrid(coord_x, coord_y)
  coords_all = jnp.stack(
      [coords_x_grid.flatten(), coords_y_grid.flatten()], axis=-1
  )
  positions = jnp.concatenate(
      [jnp.zeros((n_registers, 2), dtype=jnp.int32), coords_all]
  )
  positions = jnp.tile(jnp.expand_dims(positions, axis=0), (batch_size, 1, 1))

  return positions


class MultiHeadDotProductAttention(module.Module):
  """Multi-head dot-product attention.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax

    >>> layer = nn.MultiHeadDotProductAttention(num_heads=8, qkv_features=16)
    >>> key1, key2, key3, key4, key5, key6 = jax.random.split(jax.random.key(0),
    6)
    >>> shape = (4, 3, 2, 5)
    >>> q, k, v = jax.random.uniform(key1, shape), jax.random.uniform(key2,
    shape), jax.random.uniform(key3, shape)
    >>> variables = layer.init(jax.random.key(0), q)

    >>> # different inputs for inputs_q, inputs_k and inputs_v
    >>> out = layer.apply(variables, q, k, v)
    >>> # equivalent to layer.apply(variables, inputs_q=q, inputs_k=k,
    inputs_v=k)
    >>> out = layer.apply(variables, q, k)
    >>> # equivalent to layer.apply(variables, inputs_q=q, inputs_k=q) and
    layer.apply(variables, inputs_q=q, inputs_k=q, inputs_v=q)
    >>> out = layer.apply(variables, q)

    >>> attention_kwargs = dict(
    ...     num_heads=8,
    ...     qkv_features=16,
    ...     kernel_init=nn.initializers.ones,
    ...     bias_init=nn.initializers.zeros,
    ...     dropout_rate=0.5,
    ...     deterministic=False,
    ...     )
    >>> class Module(nn.Module):
    ...   attention_kwargs: dict
    ...
    ...   @nn.compact
    ...   def __call__(self, x, dropout_rng=None):
    ...     out1 = nn.MultiHeadDotProductAttention(**self.attention_kwargs)(x,
    dropout_rng=dropout_rng)
    ...     out2 = nn.MultiHeadDotProductAttention(**self.attention_kwargs)(x,
    dropout_rng=dropout_rng)
    ...     return out1, out2
    >>> module = Module(attention_kwargs)
    >>> variables = module.init({'params': key1, 'dropout': key2}, q)

    >>> # out1 and out2 are different.
    >>> out1, out2 = module.apply(variables, q, rngs={'dropout': key3})
    >>> # out3 and out4 are different.
    >>> # out1 and out3 are different. out2 and out4 are different.
    >>> out3, out4 = module.apply(variables, q, rngs={'dropout': key4})
    >>> # out1 and out2 are the same.
    >>> out1, out2 = module.apply(variables, q, dropout_rng=key5)
    >>> # out1 and out2 are the same as out3 and out4.
    >>> # providing a `dropout_rng` arg will take precedence over the `rngs` arg
    in `.apply`
    >>> out3, out4 = module.apply(variables, q, rngs={'dropout': key6},
    dropout_rng=key5)

  Attributes:
    num_heads: Number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    dtype: The dtype of the computation (default: infer from inputs and params)
    param_dtype: The dtype passed to parameter initializers (default: float32)
    qkv_features: Dimension of the key, query, and value.
    out_features: Dimension of the last projection
    broadcast_dropout: Use a broadcasted dropout along batch dims.
    dropout_rate: Dropout rate.
    deterministic: If False, the attention weight is masked randomly using
      dropout, whereas if True, the attention weights are deterministic.
    precision: Numerical precision of the computation see ``jax.lax.Precision``
      for details.
    kernel_init: Initializer for the kernel of the Dense layers.
    out_kernel_init: Optional Initializer for the kernel of the output Dense
      layer, if None, ``kernel_init`` will be used.
    bias_init: Initializer for the bias of the Dense layers.
    out_bias_init: Optional Initializer for the bias of the output Dense layer,
      if None, ``bias_init`` will be used.
    use_bias: Whether pointwise QKVO dense transforms use bias.
    use_rope: Whether to use RoPe.
    n_registers: Number of registers.
    norm_layer: Normalization layer.
    attention_fn: dot_product_attention or compatible function. Accepts query,
      key, value, and returns output of shape ``[bs, dim1, dim2, ..., dimN,,
      num_heads, value_channels]``
    decode: Whether to prepare and use an autoregressive cache.
    normalize_qk: Should QK normalization be applied (arxiv.org/abs/2302.05442).
    qk_attn_weights_einsum_cls: factory function to create the einsum for
      computing the attention weights.
    attn_weights_value_einsum_cls: factory function to create the einsum for
      computing the product of the attention weights and the values.
  """

  num_heads: int
  dtype: ft.Dtype | None = None
  param_dtype: ft.Dtype = jnp.float32
  qkv_features: int | None = None
  out_features: int | None = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.0
  deterministic: bool | None = None
  precision: ft.PrecisionLike = None
  kernel_init: ft.Initializer = linear.default_kernel_init
  out_kernel_init: ft.Initializer | None = None
  bias_init: ft.Initializer = initializers.zeros_init()
  out_bias_init: ft.Initializer | None = None
  use_bias: bool = True
  use_rope: bool = False
  n_registers: int = 5
  norm_layer: type[nn.Module] = nn.LayerNorm
  attention_fn: Callable[..., ft.Array] = attention.dot_product_attention
  decode: bool = False
  normalize_qk: bool = False
  force_fp32_for_softmax: bool = False
  # Deprecated, will be removed.
  qkv_dot_general: ft.DotGeneralT | None = None
  out_dot_general: ft.DotGeneralT | None = None
  qkv_dot_general_cls: Any = None
  out_dot_general_cls: Any = None
  qk_attn_weights_einsum_cls: Callable[..., Callable[..., ft.Array]] | None = (
      None
  )
  attn_weights_value_einsum_cls: (
      Callable[..., Callable[..., ft.Array]] | None
  ) = None

  @overload
  def __call__(
      self,
      inputs_q: ft.Array,
      inputs_k: ft.Array | None = None,
      inputs_v: ft.Array | None = None,
      *,
      mask: ft.Array | None = None,
      height: int | None = None,
      width: int | None = None,
      deterministic: bool | None = None,
      dropout_rng: ft.PRNGKey | None = None,
      sow_weights: bool = False,
  ):
    ...

  @overload
  def __call__(
      self,
      inputs_q: ft.Array,
      *,
      inputs_kv: ft.Array | None = None,
      mask: ft.Array | None = None,
      height: int | None = None,
      width: int | None = None,
      deterministic: bool | None = None,
      dropout_rng: ft.PRNGKey | None = None,
      sow_weights: bool = False,
  ):
    ...

  @module.compact
  def __call__(
      self,
      inputs_q: ft.Array,
      inputs_k: ft.Array | None = None,
      inputs_v: ft.Array | None = None,
      *,
      inputs_kv: ft.Array | None = None,
      height: int | None = None,
      width: int | None = None,
      mask: ft.Array | None = None,
      deterministic: bool | None = None,
      dropout_rng: ft.PRNGKey | None = None,
      sow_weights: bool = False,
  ):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    If both inputs_k and inputs_v are None, they will both copy the value of
    inputs_q (self attention).
    If only inputs_v is None, it will copy the value of inputs_k.

    Args:
      inputs_q: input queries of shape ``[batch_sizes..., length, features]``.
      inputs_k: key of shape ``[batch_sizes..., length, features]``. If None,
        inputs_k will copy the value of inputs_q.
      inputs_v: values of shape ``[batch_sizes..., length, features]``. If None,
        inputs_v will copy the value of inputs_k.
      inputs_kv: key/values of shape ``[batch_sizes..., length, features]``. If
        None, inputs_kv will copy the value of inputs_q. This arg will be
        deprecated soon. Use inputs_k and inputs_v instead.
      height: height of the image.
      width: width of the image.
      mask: attention mask of shape ``[batch_sizes..., num_heads, query_length,
        key/value_length]``. Attention weights are masked out if their
        corresponding mask value is ``False``.
      deterministic: if false, the attention weight is masked randomly using
        dropout, whereas if true, the attention weights are deterministic.
      dropout_rng: optional rng key to pass to the attention layer's dropout
        mask. Otherwise, self.make_rng('dropout') is used instead.
      sow_weights: if ``True``, the attention weights are sowed into the
        'intermediates' collection. Remember to mark 'intermediates' as mutable
        via ``mutable=['intermediates']`` in order to have that collection
        returned.

    Returns:
      output of shape ``[batch_sizes..., length, features]``.
    """
    if inputs_kv is not None:
      if inputs_k is not None or inputs_v is not None:
        raise ValueError(
            'If either `inputs_k` or `inputs_v` is not None, `inputs_kv` must'
            ' be None. If `inputs_kv` is not None, both `inputs_k` and'
            ' `inputs_v` must be None. We recommend using `inputs_k` and'
            ' `inputs_v` args, since `inputs_kv` will be deprecated soon. See'
            ' https://github.com/google/flax/discussions/3389 for more'
            ' information.'
        )
      inputs_k = inputs_v = inputs_kv
      warnings.warn(
          'The inputs_kv arg will be deprecated soon. '
          'Use inputs_k and inputs_v instead. See '
          'https://github.com/google/flax/discussions/3389 '
          'for more information.',
          DeprecationWarning,
      )
    else:
      if inputs_k is None:
        if inputs_v is not None:
          raise ValueError(
              '`inputs_k` cannot be None if `inputs_v` is not None. To have'
              ' both `inputs_k` and `inputs_v` be the same value, pass in the'
              ' value to `inputs_k` and leave `inputs_v` as None.'
          )
        inputs_k = inputs_q
      if inputs_v is None:
        inputs_v = inputs_k
      elif inputs_v.shape[-1] == inputs_v.shape[-2]:
        warnings.warn(
            f'You are passing an array of shape {inputs_v.shape} '
            'to the `inputs_v` arg, when you may have intended '
            'to pass it to the `mask` arg. As of Flax version '
            '0.7.4, the function signature of '
            "MultiHeadDotProductAttention's `__call__` method "
            'has changed to `__call__(inputs_q, inputs_k=None, '
            'inputs_v=None, *, inputs_kv=None, mask=None, '
            'deterministic=None)`. Use the kwarg `mask` instead. '
            'See https://github.com/google/flax/discussions/3389 '
            'and read the docstring for more information.',
            DeprecationWarning,
        )

    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
        f'Memory dimension ({qkv_features}) must be divisible by number of'
        f' heads ({self.num_heads}).'
    )
    head_dim = qkv_features // self.num_heads

    dense = functools.partial(
        linear.DenseGeneral,
        axis=-1,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        features=(self.num_heads, head_dim),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        precision=self.precision,
        dot_general=self.qkv_dot_general,
        dot_general_cls=self.qkv_dot_general_cls,
    )
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query, key, value = (
        dense(name='query')(inputs_q),
        dense(name='key')(inputs_k),
        dense(name='value')(inputs_v),
    )

    if self.normalize_qk:
      # Normalizing query and key projections stabilizes training with higher
      # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
      query = self.norm_layer(
          name='query_ln',
          use_bias=False,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(
          query
      )  # type: ignore[call-arg]
      key = self.norm_layer(
          name='key_ln',
          use_bias=False,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(
          key
      )  # type: ignore[call-arg]

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    if self.decode:
      # detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      cached_key = self.variable(
          'cache', 'cached_key', jnp.zeros, key.shape, key.dtype
      )
      cached_value = self.variable(
          'cache', 'cached_value', jnp.zeros, value.shape, value.dtype
      )
      cache_index = self.variable(
          'cache', 'cache_index', lambda: jnp.array(0, dtype=jnp.int32)
      )
      if is_initialized:
        (
            *batch_dims,
            max_length,
            num_heads,
            depth_per_head,
        ) = cached_key.value.shape
        # shape check of cached keys against query input
        expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
        if expected_shape != query.shape:
          raise ValueError(
              'Autoregressive cache shape error, '
              'expected query shape %s instead got %s.'
              % (expected_shape, query.shape)
          )
        # update key, value caches with our new 1d spatial slices
        cur_index = cache_index.value
        zero = jnp.array(0, dtype=lax.dtype(cur_index.dtype))
        indices: tuple[int | jax.Array, ...] = (zero,) * len(batch_dims) + (
            cur_index,
            zero,
            zero,
        )
        key = lax.dynamic_update_slice(cached_key.value, key, indices)
        value = lax.dynamic_update_slice(cached_value.value, value, indices)
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
        # causal mask for cached decoder self-attention:
        # our single query position should only attend to those key
        # positions that have already been generated and cached,
        # not the remaining zero elements.
        mask = attention.combine_masks(
            mask,
            jnp.broadcast_to(
                jnp.arange(max_length) <= cur_index,
                tuple(batch_dims) + (1, 1, max_length),
            ),
        )

    if (
        self.dropout_rate > 0.0
    ):  # Require `deterministic` only if using dropout.
      m_deterministic = module.merge_param(
          'deterministic', self.deterministic, deterministic
      )
      if not m_deterministic and dropout_rng is None:
        dropout_rng = self.make_rng('dropout')
    else:
      m_deterministic = True

    if self.use_rope:
      batch_size = query.shape[0]

      positions = get_positions(batch_size, height, width, self.n_registers)
      query = rope.apply_rotary_encoding_2d(
          query,
          positions=positions,
          base_frequency=10000.0,
      )

      key = rope.apply_rotary_encoding_2d(
          key,
          positions=positions,
          base_frequency=10000.0,
      )

    # `qk_attn_weights_einsum` and `attn_weights_value_einsum` are optional
    # arguments that can be used to override the default `jnp.einsum`. They
    # exist for quantized einsum support in AQT.
    qk_attn_weights_einsum = (
        self.qk_attn_weights_einsum_cls()
        if self.qk_attn_weights_einsum_cls
        else None
    )
    attn_weights_value_einsum = (
        self.attn_weights_value_einsum_cls()
        if self.attn_weights_value_einsum_cls
        else None
    )
    # apply attention
    attn_args = (query, key, value)
    # This kwargs list match the default nn.dot_product_attention.
    # For custom `attention_fn`s, invalid kwargs will be filtered.
    attn_kwargs = dict(
        mask=mask,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=m_deterministic,
        dtype=self.dtype,
        precision=self.precision,
        force_fp32_for_softmax=self.force_fp32_for_softmax,
        qk_attn_weights_einsum=qk_attn_weights_einsum,
        attn_weights_value_einsum=attn_weights_value_einsum,
    )
    attn_kwargs = {
        k: v
        for k, v in attn_kwargs.items()
        if k in inspect.signature(self.attention_fn).parameters
    }
    if sow_weights:
      x = self.attention_fn(*attn_args, **attn_kwargs, module=self)
    else:
      x = self.attention_fn(*attn_args, **attn_kwargs)
    # back to the original inputs dimensions
    out = linear.DenseGeneral(
        features=features,
        axis=(-2, -1),
        kernel_init=self.out_kernel_init or self.kernel_init,
        bias_init=self.out_bias_init or self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
        dot_general=self.out_dot_general,
        dot_general_cls=self.out_dot_general_cls,
        name='out',  # type: ignore[call-arg]
    )(x)
    return out
