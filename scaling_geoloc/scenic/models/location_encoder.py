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

"""Location encoder for the Scenic model (WARNING: not fully tested!)."""

import functools
from typing import Any, Final, Mapping, Sequence

from etils.array_types import FloatArray
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections

from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.models import base
from scaling_geoloc.utils import misc


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
  vp = 2 * jnp.pi * (v @ b.T)
  return jnp.concat([jnp.cos(vp), jnp.sin(vp)], axis=-1)


class GaussianEncoder(nn.Module):
  """Gaussian Positional Encoder."""

  sigma: float
  encoded_size: float
  learned_bias: bool = False

  @nn.compact
  def __call__(self, x: FloatArray['N D']) -> FloatArray['N 2D']:
    """Applies the Transformer module."""

    bias = self.param(
        'bias',
        nn.initializers.normal(stddev=self.sigma),
        (self.encoded_size, x.shape[-1]),
        x.dtype,
    )
    if not self.learned_bias:
      bias = jax.lax.stop_gradient(bias)

    return gaussian_encoding(x, bias)


class LocationEncoderBlock(nn.Module):
  """Location encoder block."""

  sigma: float
  output_dim: int
  hidden_dim: int
  num_layers: int
  learned_bias: bool = False

  @nn.compact
  def __call__(self, x: FloatArray['N 2']) -> FloatArray['N 2']:
    """Applies the Transformer module."""
    layers = [
        GaussianEncoder(
            sigma=self.sigma,
            encoded_size=self.hidden_dim // 4,
            learned_bias=self.learned_bias,
        ),
    ]
    for _ in range(self.num_layers):
      layers.extend([nn.Dense(self.hidden_dim), nn.relu])
    layers = nn.Sequential(layers)
    head = nn.Dense(self.output_dim)
    return head(layers(x))


class LocationEncoder(nn.Module):
  """Location encoder."""

  output_dim: int
  sigmas: tuple[float, ...] = (2**8, 2**12, 2**16)  # (2**0, 2**4, 2**8)
  hidden_dim: int = 1024
  num_hidden_layers: int = 3
  learned_bias: bool = False

  @nn.compact
  def __call__(self, latlng: FloatArray['N 2']) -> FloatArray['N D']:
    """Applies the Transformer module."""
    location_features = 0.0
    ee_xy = equal_earth_projection(latlng)
    for sigma in self.sigmas:
      location_features = location_features + LocationEncoderBlock(
          sigma=sigma,
          output_dim=self.output_dim,
          hidden_dim=self.hidden_dim,
          num_layers=self.num_hidden_layers,
          learned_bias=self.learned_bias,
      )(ee_xy)
    return location_features


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
        output_dim=batch['tokens'].shape[-1],
        num_hidden_layers=self.config.num_hidden_layers,
        hidden_dim=self.config.hidden_dim,
        sigmas=self.config.get('sigmas', (2**4, 2**8, 2**12)),
        learned_bias=self.config.get('learned_bias', False),
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
