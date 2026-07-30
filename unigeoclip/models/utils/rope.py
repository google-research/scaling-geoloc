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

"""Utils for Rotary Positional Encodings (RoPE).

See the blogpost https://blog.eleuther.ai/rotary-embeddings/ and the paper
https://arxiv.org/abs/2104.09864.
"""

import jax.numpy as jnp
import jaxtyping as jt


def apply_rotary_encoding(
    x: jt.Float[jt.Array, '*B L H D'],
    *,
    positions: jt.Int[jt.Array, '*B L'],
    base_frequency: float,
    scale_factor: float = 1.0,
) -> jt.Float[jt.Array, '*B L H D']:
  """Returns the rotary positional encodings.

  Computes R*x, the multiplication between the rotation matrix R, and input x.
  If the embedding dimensionality is odd-numbered, a dummy dimension is added to
  the end of the rotation matrix.

  NOTE: Compared to the Chipmunk rope implementation, this uses the faster
  version and rotated all the channels.

  Args:
    x: Array of embeddings
    positions: Vectors holding the position of each token in the sequence.
    base_frequency: Base frequency used to compute rotations. This value also
      roughly specifies the maximum sequence length for identifying unique
      positions.
    scale_factor: The scale factor to use for Position Interpolation (PI), which
      we set as L' / L, where L is the original sequence length and L' is the
      new context window size beyond the pre-trained, where L' > L.

  Returns:
    Array of rotary encoded input
  """
  freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2)
  timescale = base_frequency**freq_exponents
  radians = positions[..., None] / timescale[None, None, :]
  radians: jt.Float[jt.Array, '*B L 1 D/2'] = radians[..., None, :]

  # Applies position interpolation (PI).
  if scale_factor < 1.0:
    raise ValueError(f'scale_factor must be >= 1.0, got {scale_factor}')
  radians = radians / scale_factor

  sin, cos = jnp.sin(radians), jnp.cos(radians)
  x1, x2 = jnp.split(x, 2, axis=-1)
  res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
  return res


def get_cos_sin_for_pos(
    positions: jt.Int[jt.Array, '*B L'], base: float, dim: int
) -> tuple[jt.Float[jt.Array, '*B L D/2'], jt.Float[jt.Array, '*B L D/2']]:
  """Returns the cosine and sine grid values for a given base and positions."""
  inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2) / dim))
  # freqs: (*B, L, D/2)
  freqs = jnp.einsum('...l,d->...ld', positions, inv_freq)

  # Concatenate the frequencies along the last dimension
  freqs = jnp.concatenate([freqs, freqs], axis=-1)

  cos = jnp.cos(freqs)  # (*B, L, D)
  sin = jnp.sin(freqs)

  return cos, sin


def rotate_half(
    x: jt.Float[jt.Array, '*B L D'],
) -> jt.Float[jt.Array, '*B L D']:
  """Rotates the last half of the input."""
  # x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  x1, x2 = jnp.split(x, 2, axis=-1)
  return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope1d(
    tokens: jt.Float[jt.Array, '*B L D'],
    pos1d: jt.Int[jt.Array, '*B L'],
    base_frequency: float,
) -> jt.Float[jt.Array, '*B L D']:
  """Applies rotary positional encodings to a 1D sequence."""
  assert pos1d.ndim == 2
  dim = tokens.shape[-1]
  cos, sin = get_cos_sin_for_pos(pos1d, base_frequency, dim)

  # Apply the RoPE operation
  rotated_tokens = rotate_half(tokens)
  return (tokens * cos[:, None, :, :]) + (rotated_tokens * sin[:, None, :, :])


def apply_rotary_encoding_2d(
    tokens: jt.Float[jt.Array, '*B L H D'],
    positions: jt.Int[jt.Array, '*B L 2'],
    base_frequency: float,
) -> jt.Float[jt.Array, '*B L H D']:
  """Takes as input a batch of tokens and their positions and applies rotary positional encodings.

  Args:
      tokens: batch_size x nheads x ntokens x dim
      positions: batch_size x ntokens x 2 (y and x position of each token)
      base_frequency: base frequency used to compute rotations

  Returns:
      tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim).
  """
  tokens = jnp.transpose(tokens, axes=(0, 2, 1, 3))
  assert (
      tokens.shape[3] % 2 == 0
  ), 'number of dimensions should be a multiple of two'
  assert positions.ndim == 3 and positions.shape[-1] == 2  # Batch, Seq, 2

  y_tokens, x_tokens = jnp.split(tokens, 2, axis=-1)

  y_pos = positions[:, :, 0]
  x_pos = positions[:, :, 1]

  y_tokens = apply_rope1d(y_tokens, y_pos, base_frequency)

  x_tokens = apply_rope1d(x_tokens, x_pos, base_frequency)

  tokens = jnp.concatenate((y_tokens, x_tokens), axis=-1)
  return jnp.transpose(tokens, axes=(0, 2, 1, 3))
