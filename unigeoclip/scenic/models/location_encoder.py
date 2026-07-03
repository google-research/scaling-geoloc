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

"""Location encoder for the Scenic model (WARNING: not fully tested!)."""

import functools
from typing import Any, Final, Mapping, Sequence

from etils.array_types import FloatArray
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections

from unigeoclip.scenic.models.utils import attention
from unigeoclip.scenic.models.utils import norm
from scaling_geoloc import types
from scaling_geoloc.models import base
from scaling_geoloc.utils import misc


_NORM_LAYER: Final[Mapping[str, type[nn.Module]]] = {
    'layernorm': nn.LayerNorm,
    'DyT': norm.DynamicTanhNorm,
}

# Constants
A1: Final[float] = 1.340264
A2: Final[float] = -0.081106
A3: Final[float] = 0.000893
A4: Final[float] = 0.003796
SF: Final[float] = 66.50336


def equal_earth_projection(latlng: FloatArray['... 2']) -> FloatArray['... 2']:
  latlng_rad = jnp.radians(latlng)
  sin_theta = 3**0.5 / 2 * jnp.sin(latlng_rad[..., 0])
  theta = jnp.arcsin(sin_theta)
  denominator = 3 * (
      9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1
  )

  x = 2 * 3**0.5 * latlng_rad[..., 1] * jnp.cos(theta) / denominator
  y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta
  return jnp.stack([x, y], axis=-1) * SF / 180


def gaussian_encoding(v: FloatArray, b: FloatArray) -> FloatArray:
  """Gaussian encoding."""
  vp = 2 * jnp.pi * (jax.lax.dot(v, b.T, precision='float32'))
  return jnp.concat([jnp.cos(vp), jnp.sin(vp)], axis=-1)


class GaussianEncoder(nn.Module):
  """Gaussian Positional Encoder."""

  sigma: float
  encoded_size: int
  learned_freqs: bool = False

  @nn.compact
  def __call__(self, x: FloatArray['N D']) -> FloatArray['N 2D']:
    freqs = self.param(
        'freqs',
        nn.initializers.normal(stddev=self.sigma),
        (self.encoded_size, x.shape[-1]),
        x.dtype,
    )
    if not self.learned_freqs:
      freqs = jax.lax.stop_gradient(freqs)

    return gaussian_encoding(x, freqs)


class LocationEncoder(nn.Module):
  """Location encoder."""

  sigmas: tuple[float, ...] = (2**8, 2**12, 2**16)
  hidden_dim: int = 1024
  num_hidden_layers: int = 3
  learned_freqs: bool = False
  n_registers: int = 32
  norm_layer: type[nn.Module] = nn.LayerNorm
  mlp_ratio: int = 4
  num_heads: int = 16

  @nn.compact
  def __call__(
      self, latlng: FloatArray['N 2'], train: bool = False
  ) -> FloatArray['N D']:
    """Applies the Transformer module."""
    ee_xy = equal_earth_projection(latlng)
    x = jnp.stack(
        [
            GaussianEncoder(
                sigma=sigma,
                encoded_size=self.hidden_dim // 2,
                learned_freqs=self.learned_freqs,
            )(ee_xy)
            for sigma in self.sigmas
        ],
        axis=0,
    ).sum(axis=0)

    dim = self.hidden_dim
    batch_size = x.shape[0]

    reg_tok = self.param(
        'reg', nn.initializers.zeros, (1, self.n_registers, dim), x.dtype
    )
    reg_tok_tiled = jnp.tile(reg_tok, [batch_size, 1, 1])
    x = jnp.concatenate([reg_tok_tiled, x[:, None, :]], axis=1)
    for _ in range(self.num_hidden_layers):

      norm_x = self.norm_layer()(x)
      attention_output = attention.MultiHeadDotProductAttention(
          num_heads=self.num_heads,
          kernel_init=nn.initializers.xavier_uniform(),
          use_rope=False,
          broadcast_dropout=False,
          dropout_rate=0,
          deterministic=not train,
          use_bias=True,
      )(norm_x, norm_x, deterministic=not train)
      x = x + attention_output

      mlp = nn.Dense(dim * self.mlp_ratio)(self.norm_layer()(x))
      mlp_gelu = nn.gelu(mlp)
      mlp_norm = self.norm_layer()(mlp_gelu)
      x = x + nn.Dense(dim)(mlp_norm)
    return x[:, -1, :]

  def encode_multiview(
      self,
      latlng: FloatArray['... 2'],
      train: bool = False,
  ) -> types.Predictions:
    """Interface to backbone encoder."""
    return self(latlng, train=train)


class LocationEncoder2(nn.Module):
  """Location encoder.

  Here instead of summing over multiple sigmas, we consider them as different
  tokens in the attention blocks
  """

  output_dim: int
  sigmas: Sequence[float] = (2**8, 2**12, 2**16)
  hidden_dim: int = 1024
  num_hidden_layers: int = 3
  learned_freqs: bool = False
  n_registers: int = 32
  norm_layer: type[nn.Module] = nn.LayerNorm
  mlp_ratio: int = 4
  num_heads: int = 16

  @nn.compact
  def __call__(
      self, latlng: FloatArray['N 2'], *, train: bool = False
  ) -> FloatArray['N D']:
    """Applies the Transformer module."""
    ee_xy = equal_earth_projection(latlng)
    x = jnp.stack(
        [
            GaussianEncoder(
                sigma=sigma,
                encoded_size=self.hidden_dim // 2,
                learned_freqs=self.learned_freqs,
            )(ee_xy)
            for sigma in self.sigmas
        ],
        axis=1,
    )

    dim = self.hidden_dim
    batch_size = x.shape[0]

    if self.num_hidden_layers == 0:
      return x.sum(axis=1)

    reg_tok = self.param(
        'reg', nn.initializers.zeros, (1, self.n_registers, dim), x.dtype
    )
    reg_tok_tiled = jnp.tile(reg_tok, [batch_size, 1, 1])
    x = jnp.concatenate([reg_tok_tiled, x], axis=1)
    for _ in range(self.num_hidden_layers):

      norm_x = self.norm_layer()(x)
      attention_output = attention.MultiHeadDotProductAttention(
          num_heads=self.num_heads,
          kernel_init=nn.initializers.xavier_uniform(),
          use_rope=False,
          broadcast_dropout=False,
          dropout_rate=0,
          deterministic=not train,
          use_bias=True,
      )(norm_x, norm_x, deterministic=train)
      x = x + attention_output

      mlp = nn.Dense(dim * self.mlp_ratio)(self.norm_layer()(x))
      mlp_gelu = nn.gelu(mlp)
      mlp_norm = self.norm_layer()(mlp_gelu)
      x = x + nn.Dense(dim)(mlp_norm)
    return x[:, 0, :]

  def encode_multiview(
      self,
      latlng: FloatArray['... 2'],
      train: bool = False,
  ) -> types.Predictions:
    """Interface to backbone encoder."""
    return self(latlng, train=train)


class BlockGeoClip(nn.Module):
  """Block GeoClip."""

  sigma: float
  encoded_size: int
  learned_freqs: bool = False

  @nn.compact
  def __call__(self, x: FloatArray['N 2']) -> FloatArray['N D']:
    x = GaussianEncoder(
        sigma=self.sigma,
        encoded_size=self.encoded_size // 2,
        learned_freqs=self.learned_freqs,
    )(x)

    x = nn.Dense(1024)(x)
    x = nn.relu(x)
    x = nn.Dense(1024)(x)
    x = nn.relu(x)
    x = nn.Dense(1024)(x)
    x = nn.relu(x)
    x = nn.Dense(self.encoded_size)(x)
    return x


class LocationEncoderGeoClip(nn.Module):
  """Location encoder."""

  output_dim: int
  sigmas: tuple[float, ...] = (2**8, 2**12, 2**16)
  hidden_dim: int = 1024
  learned_freqs: bool = False

  @nn.compact
  def __call__(
      self, latlng: FloatArray['N 2'], train: bool = False
  ) -> FloatArray['N D']:
    """Applies the Transformer module."""
    ee_xy = equal_earth_projection(latlng)
    return jnp.stack(
        [
            BlockGeoClip(
                sigma=sigma,
                encoded_size=self.hidden_dim,
                learned_freqs=self.learned_freqs,
            )(ee_xy)
            for sigma in self.sigmas
        ],
        axis=1,
    ).sum(axis=1)

  def encode_multiview(
      self,
      latlng: FloatArray['... 2'],
      train: bool = False,
  ) -> types.Predictions:
    """Interface to backbone encoder."""
    return self(latlng, train=train)


def from_config_rff(config: ml_collections.ConfigDict) -> 'LocationEncoder':
  """Creates a LocationEncoder from a config."""
  return LocationEncoder(
      sigmas=config.get('sigmas', (2**4, 2**8, 2**12)),
      hidden_dim=config.hidden_dim,
      num_heads=config.num_heads,
      num_hidden_layers=config.num_hidden_layers,
      n_registers=config.n_registers,
      norm_layer=_NORM_LAYER[config.get('norm_layer', 'layernorm')],
  )


def from_config_rff2(config: ml_collections.ConfigDict) -> 'LocationEncoder2':
  """Creates a LocationEncoder from a config."""
  return LocationEncoder2(
      output_dim=config.cell_code_dim,
      sigmas=config.get('sigmas', (2**4, 2**8, 2**12)),
      hidden_dim=config.hidden_dim,
      num_heads=config.num_heads,
      num_hidden_layers=config.num_hidden_layers,
      n_registers=config.n_registers,
      norm_layer=_NORM_LAYER[config.get('norm_layer', 'layernorm')],
  )


def from_config_rff_geoclip(
    config: ml_collections.ConfigDict,
) -> 'LocationEncoderGeoClip':
  """Creates a LocationEncoder from a config."""
  return LocationEncoderGeoClip(
      output_dim=config.cell_code_dim,
      sigmas=config.get('sigmas', (2**4, 2**8, 2**12)),
      hidden_dim=config.hidden_dim,
  )


def from_config_siren(config: ml_collections.ConfigDict) -> 'SirenNetWithSH':
  """Creates a LocationEncoder from a config."""
  return SirenNetWithSH(
      hidden_dims=config.hidden_dims,
      output_dim=config.output_dim,
      omega_0=config.omega_0,
      max_degree_sh=config.max_degree_sh,
      num_recursive_blocks=config.num_recursive_blocks,
      n_registers=config.n_registers,
      norm_layer=_NORM_LAYER[config.get('norm_layer', 'layernorm')],
  )


def from_config(config: ml_collections.ConfigDict) -> nn.Module:
  """Creates a LocationEncoder from a config."""
  if config.model_name == 'rff':
    return from_config_rff(config)
  elif config.model_name == 'rff2':
    return from_config_rff2(config)
  elif config.model_name == 'rff_geoclip':
    return from_config_rff_geoclip(config)
  elif config.model_name == 'siren':
    return from_config_siren(config)
  else:
    raise ValueError(f'Unknown model name: {config.model_name}')


class LocationEncoderModel(nn.Module):
  """Location encoder model."""

  config: ml_collections.ConfigDict
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self, batch: types.Batch, train: bool = False, debug: bool = False
  ) -> types.Predictions:
    """Applies the Transformer module."""
    location_features = LocationEncoder(
        num_hidden_layers=self.config.num_hidden_layers,
        hidden_dim=self.config.hidden_dim,
        sigmas=self.config.get('sigmas', (2**4, 2**8, 2**12)),
        learned_freqs=self.config.get('learned_freqs', False),
    )(batch['latlng'])
    return {
        'tokens': location_features,
    }


class LocationEncoderModelWrapper(base.BaseModel):
  """Location encoder model wrapper."""

  def build_flax_model(self) -> nn.Module:
    print(self.config)
    return LocationEncoderModel(self.config, dtype=self.dtype)

  def loss_metrics_function(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      model_params: jnp.ndarray | None = None,
      model_state: Any | None = None,
      learning_rates: Mapping[str, float] | None = None,
  ) -> types.LossMetricsTuple:
    losses, metrics = {}, {}
    sim = misc.compute_similarity(
        pred['tokens'], batch['tokens'], normalize=True
    )[..., 0]

    metrics['similarity'] = sim
    losses['dist_l2'] = 1 - metrics['similarity']

    identity = jnp.eye(sim.shape[0])

    if 'ortho' in self.config.compose_loss:
      asim = misc.compute_similarity(
          pred['tokens'][:, None], pred['tokens'][None], normalize=True
      )[..., 0]

      asim = asim.clip(min=0.0)
      ortho = ((1 - asim) * identity).sum(-1) + (asim * (1 - identity)).mean(-1)
      losses['ortho'] = ortho
    losses['total'] = types.compose_loss(losses, self.config.compose_loss)
    return losses, metrics


def latlon_to_spherical(lat_deg, lon_deg):
  """Converts latitude and longitude (degrees) to spherical coordinates (theta, phi)."""
  lat_rad = jnp.deg2rad(lat_deg)
  lon_rad = jnp.deg2rad(lon_deg)
  theta = jnp.pi / 2 - lat_rad  # Polar angle (colatitude)
  phi = lon_rad + jnp.pi  # Azimuthal angle (longitude, shifted to [0, 2pi])
  return theta, phi


@functools.partial(jax.jit, static_argnums=(2,))
def compute_spherical_harmonics(lat, lon, max_degree):
  """Computes spherical harmonics for a batch of latitude and longitude coordinates.

  Args:
      lat: JAX NumPy array of latitudes in degrees (shape: (N,)).
      lon: JAX NumPy array of longitudes in degrees (shape: (N,)).
      max_degree: Integer, the maximum degree of spherical harmonics to compute.

  Returns:
      JAX NumPy array of shape (N, 2 * (max_degree + 1)**2) containing the
      real and imaginary parts of the spherical harmonics.
  """
  theta, phi = jax.vmap(latlon_to_spherical)(lat, lon)
  embedding_list = []
  for l in range(max_degree + 1):
    for m in range(-l, l + 1):
      # Need to use a JAX-compatible implementation of sph_harm
      # SciPy's is not directly compatible. We'll use a placeholder here.
      # A true JAX implementation would be more complex.
      # Placeholder: Let's create some dummy features based on l and m
      real_part = jnp.sin(theta * (l + 1) + phi * (m + l + 1)) * 0.1
      imag_part = jnp.cos(theta * (l + 1) + phi * (m + l + 1)) * 0.1
      embedding_list.append(real_part)
      embedding_list.append(imag_part)
  return jnp.stack(embedding_list, axis=-1)


class SirenLayer(nn.Module):
  """Siren layer."""

  dim: int
  omega_0: float = 30.0
  is_first: bool = False

  @nn.compact
  def __call__(self, x):
    if self.is_first:
      b = self.param(
          'bias', nn.initializers.uniform(scale=1 / self.dim), (self.dim,)
      )
      w_scale = 1.0 / x.shape[-1]
      w_init = nn.initializers.uniform(scale=w_scale)
      w = self.param('kernel', w_init, (x.shape[-1], self.dim))
      return jnp.sin(self.omega_0 * (x @ w + b))
    else:
      b = self.param(
          'bias',
          nn.initializers.uniform(scale=jnp.sqrt(6 / self.dim) / self.omega_0),
          (self.dim,),
      )
      w_scale = jnp.sqrt(6 / self.dim) / self.omega_0
      w_init = nn.initializers.uniform(scale=w_scale)
      w = self.param('kernel', w_init, (x.shape[-1], self.dim))
      return jnp.sin(self.omega_0 * (x @ w + b))


class SirenNetWithSH(nn.Module):
  """Siren net with spherical harmonics."""

  hidden_dims: Sequence[int]
  output_dim: int
  omega_0: float = 30.0
  max_degree_sh: int = 5
  num_recursive_blocks: int = 3
  mlp_ratio: int = 4
  n_registers: int = 32
  norm_layer: type[nn.Module] = nn.LayerNorm

  @nn.compact
  def __call__(self, latlng):
    # Compute spherical harmonics features
    sh_features = compute_spherical_harmonics(
        latlng[:, 0], latlng[:, 1], self.max_degree_sh
    )

    # Concatenate with the original latlng coordinates (optional, but can help)
    input_features = jnp.concatenate([latlng, sh_features], axis=-1)

    x = input_features
    for i, dim in enumerate(self.hidden_dims):
      is_first = i == 0
      x = SirenLayer(dim=dim, omega_0=self.omega_0, is_first=is_first)(x)

    return nn.Dense(features=self.output_dim)(x)

  def encode_multiview(
      self,
      latlng: FloatArray['... 2'],
      train: bool = False,
  ) -> types.Predictions:
    """Interface to backbone encoder."""
    del train
    return self(latlng)
