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

"""RoPE positional embedding.

Adapted from the pytorch implementation of dinov3.
https://github.com/facebookresearch/dinov3/blob/main/dinov3/layers/rope_position_encoding.py
"""

import enum
from flax import linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt


class NormalizeCoords(str, enum.Enum):
  MIN = 'min'
  MAX = 'max'
  SEPARATE = 'separate'


class RopePositionEmbedding(nn.Module):
  """Flax implementation of the RoPE positional embedding.

  This module generates sinusoidal embeddings from 2D coordinates.
  """

  embed_dim: int
  num_heads: int
  base: float | None = 100.0
  min_period: float | None = None
  max_period: float | None = None
  normalize_coords: NormalizeCoords = NormalizeCoords.SEPARATE
  shift_coords: float | None = None
  jitter_coords: float | None = None
  rescale_coords: float | None = None
  dtype: jnp.dtype = jnp.bfloat16

  def setup(self):
    """Initializes non-trainable attributes like the rotation periods."""

    if (self.embed_dim % (4 * self.num_heads)) != 0:
      raise ValueError(
          f"embed_dim ({self.embed_dim}) must be divisible by 4 * num_heads"
          f" ({4 * self.num_heads})"
      )

    both_periods = self.min_period is not None and self.max_period is not None
    if (self.base is None and not both_periods) or (
        self.base is not None and both_periods
    ):
      raise ValueError(
          "Either `base` or `min_period`+`max_period` must be provided."
      )

    self.d_head = self.embed_dim // self.num_heads

    if self.base is not None:
      periods = self.base ** (
          2
          * jnp.arange(self.d_head // 4, dtype=self.dtype)
          / (self.d_head // 2)
      )
    else:
      base = self.max_period / self.min_period
      exponents = jnp.linspace(0, 1, self.d_head // 4, dtype=self.dtype)
      periods = self.min_period * (base**exponents)

    self.periods = periods

  def _prepare_coords(self, h: int, w: int) -> jnp.ndarray:
    """Prepares coordinates in the range [-1, +1]."""

    if h == w:
      coords_h = (2.0 * jnp.arange(0.5, h, dtype=self.dtype) / h) - 1.0
      coords_w = coords_h
    else:
      match self.normalize_coords:
        case NormalizeCoords.MAX:
          div_h = div_w = max(h, w)
        case NormalizeCoords.MIN:
          div_h = div_w = min(h, w)
        case NormalizeCoords.SEPARATE:
          div_h, div_w = h, w

      coords_h = (2.0 * jnp.arange(0.5, h, dtype=self.dtype) / div_h) - 1.0
      coords_w = (2.0 * jnp.arange(0.5, w, dtype=self.dtype) / div_w) - 1.0

    grid_h, grid_w = jnp.meshgrid(coords_h, coords_w, indexing="ij")
    coords = jnp.stack([grid_h, grid_w], axis=-1).reshape(-1, 2)

    return coords

  def _augment_coords(self, coords: jnp.ndarray) -> jnp.ndarray:
    """Applies random augmentations to the coordinates."""

    rope_rng = self.make_rng("rope")
    shift_rng, jitter_rng, rescale_rng = jax.random.split(rope_rng, 3)

    if self.shift_coords is not None:
      shift_hw = jax.random.uniform(
          shift_rng,
          (2,),
          dtype=self.dtype,
          minval=-self.shift_coords,
          maxval=self.shift_coords,
      )
      coords += shift_hw[None, :]

    if self.jitter_coords is not None:
      jitter_max = jnp.log(self.jitter_coords)
      jitter_hw = jnp.exp(
          jax.random.uniform(
              jitter_rng,
              (2,),
              dtype=self.dtype,
              minval=-jitter_max,
              maxval=jitter_max,
          )
      )
      coords *= jitter_hw[None, :]

    if self.rescale_coords is not None:
      rescale_max = jnp.log(self.rescale_coords)
      rescale_hw = jnp.exp(
          jax.random.uniform(
              rescale_rng,
              (1,),
              dtype=self.dtype,
              minval=-rescale_max,
              maxval=rescale_max,
          )
      )
      coords *= rescale_hw

    return coords

  @nn.compact
  def __call__(
      self, h: int, w: int, deterministic: bool = True
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generates the sin/cos positional embeddings.

    Args:
        h: Height of the input grid.
        w: Width of the input grid.
        deterministic: If False, applies random shifting, jittering, and
          rescaling for data augmentation during training.

    Returns:
        A tuple containing (sin, cos) embeddings.
    """

    coords = self._prepare_coords(h, w)

    if not deterministic:
      coords = self._augment_coords(coords)

    angles = 2 * jnp.pi * coords[:, :, None] / self.periods[None, None, :]
    angles = jnp.tile(angles.reshape(h * w, -1), (1, 2))
    return (jnp.sin(angles), jnp.cos(angles))


def apply_rope(
    tensor: jt.Float[jt.Array, "B S H D"],
    sin: jt.Float[jt.Array, "S_ROPE D"],
    cos: jt.Float[jt.Array, "S_ROPE D"],
) -> jt.Float[jt.Array, "B S H D"]:
  """Applies Rotary Position Embeddings (RoPE) to the input tensor.

  All operations are computed in the same dtype as the RoPE embeddings and then
  casted back to the original input dtype.

  Args:
    tensor: The input tensor, expected in shape [B, S, H, D] (batch,
      sequence_length, num_heads, head_dim).
    sin: The sine component of the rotary embeddings.
    cos: The cosine component of the rotary embeddings.

  Returns:
    The tensor with RoPE applied, in the same shape and dtype as the input.
  """

  original_dtype = tensor.dtype
  tensor = tensor.astype(sin.dtype)
  tensor_transposed = jnp.transpose(tensor, (0, 2, 1, 3))  # [B, H, S, D]

  prefix_len = tensor_transposed.shape[-2] - sin.shape[-2]
  assert prefix_len >= 0, (
      "Input sequence length must not be less than the embedding sequence"
      " length."
  )

  unrotated_prefix = tensor_transposed[:, :, :prefix_len, :]
  tokens_to_rotate = tensor_transposed[:, :, prefix_len:, :]
  part1, part2 = jnp.split(tokens_to_rotate, 2, axis=-1)

  rotated_counterpart = jnp.concatenate([-part2, part1], axis=-1)
  rotated_part = (tokens_to_rotate * cos) + (rotated_counterpart * sin)
  tensor = jnp.concatenate([unrotated_prefix, rotated_part], axis=-2)

  return jnp.transpose(tensor, (0, 2, 1, 3)).astype(original_dtype)


def dot_product_attention(
    query: jt.Float[jt.Array, "B S H D"],
    key: jt.Float[jt.Array, "B S H D"],
    value: jt.Float[jt.Array, "B S H D"],
    rope_embeds: (
        tuple[jt.Float[jt.Array, "S_ROPE D"], jt.Float[jt.Array, "S_ROPE D"]]
        | None
    ) = None,
    **kwargs,
) -> jt.Float[jt.Array, "B S H D"]:
  """Dot-product attention with optional Rotary Positional Embeddings.

  Args:
    query: Query tensor.
    key: Key tensor.
    value: Value tensor.
    rope_embeds: Optional tuple of (sin, cos) rope embeddings.
    **kwargs: Keyword arguments to pass to the dot_product_attention function.

  Returns:
    The dot-product attention output.
  """

  if rope_embeds is not None:
    sin, cos = rope_embeds

    query = apply_rope(query, sin, cos)
    key = apply_rope(key, sin, cos)

  return nn.dot_product_attention(query, key, value, **kwargs)
