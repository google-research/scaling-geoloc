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

"""Utility functions to work with 3D gaussians."""

from typing import TypeAlias
from etils.array_types import FloatArray
import jax
import jax.numpy as jnp
from scaling_geoloc.utils import misc

Gaussians: TypeAlias = tuple[
    FloatArray['... 3'], FloatArray['... 3 3'], FloatArray['...']
]


def gaussians_from_points(
    w_points: FloatArray['... N 3'],
    weights: FloatArray['... N 1'],
    min_observations: int = 30,
) -> Gaussians:
  center = misc.masked_mean(w_points, weights, axis=-2)
  cov = misc.batched_cov(w_points, weights=weights[..., 0], bias=True)
  cov = jnp.where(weights[..., 0].sum(-1)[..., None, None] > 0, cov, jnp.inf)
  occ = weights.sum((-1, -2))
  occ = jnp.where(occ > min_observations, occ, 0)
  return center, cov, occ


def kde_gaussian(
    x: FloatArray['3'],
    mu: FloatArray['N 3'],
    cov: FloatArray['N 3 3'],
    normalize: bool = False,
) -> FloatArray['... 1']:
  """Evaluate a 3D gaussian at location x."""
  d = x - mu
  val_e = ((jnp.einsum('...j,...ji->...i', d, jnp.linalg.pinv(cov))) * d).sum(
      -1
  )
  if normalize:
    det = jnp.linalg.det(cov)
    dim = x.shape[-1]
    probs = (
        1.0
        / (2 * jnp.pi) ** (dim / 2)
        / jnp.linalg.det(cov) ** 0.5
        * jnp.exp(-0.5 * val_e)
    )

    return (probs * (~jnp.isnan(det)))[..., None]
  else:
    return jnp.exp(-0.5 * val_e)


def probabilities_between_gaussians(
    gauss_query: Gaussians,
    gauss_map: Gaussians,
    normalize: bool = False,
) -> FloatArray:
  """Pool features via guassians."""
  center_m, cov_m, occ_m = gauss_map
  center_q, cov_q, occ_q = gauss_query
  gauss_query = jnp.concatenate(
      [
          center_q,
          jax.lax.collapse(cov_q, -2),
          occ_q[..., None],
      ],
      -1,
  )

  def gather(gaussian_q: FloatArray['13']):
    occ_q = gaussian_q[-1:]
    center_q = gaussian_q[:3]
    cov_q = gaussian_q[3:-1].reshape(3, 3)

    prob_q_in_m = kde_gaussian(
        center_q, center_m, cov_m * 2, normalize=normalize
    )
    prob_m_in_q = kde_gaussian(
        center_m, center_q, cov_q * 2, normalize=normalize
    )

    mask = (occ_q > 0) & (occ_m > 0)
    weights = (prob_q_in_m * prob_m_in_q) ** 0.5 * mask
    return weights
  weights = jnp.apply_along_axis(gather, -1, gauss_query)
  return weights


batched_probabilities_between_gaussians = jax.vmap(
    probabilities_between_gaussians, in_axes=(0, 0), out_axes=0
)
