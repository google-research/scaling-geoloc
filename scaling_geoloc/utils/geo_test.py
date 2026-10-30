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

"""Checks that interpolation weights follow the supplied neighbor order."""

from unittest import mock

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from scaling_geoloc.utils import geo

_ORIGINS = [np.zeros(3), np.array([7.0, -4.0, 2.0])]
_BASES = [
    np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
    np.array([[2.0, 1.0, -1.0], [-1.0, 3.0, 2.0]]),
]


def neighbors(origin, basis):
  return np.array(
      [
          origin,
          origin + basis[0],
          origin + basis[1],
          origin + basis[0] + basis[1],
      ]
  )


@pytest.mark.parametrize("basis", _BASES)
@pytest.mark.parametrize("corner", range(4))
def test_weight_one_at_the_corresponding_neighbor(basis, corner):
  coords = neighbors(_ORIGINS[1], basis)
  weights = geo.bilinear_interpolation_weights(coords[corner], coords)
  np.testing.assert_allclose(weights, np.eye(4)[corner], atol=1e-14)


@pytest.mark.parametrize(
    "st", [(0.2, 0.7), (0.8, 0.1), (0.0, 0.4), (0.5, 0.5), (-0.2, 1.3)]
)
@pytest.mark.parametrize("basis", _BASES)
def test_weights_reconstruct_the_query(st, basis):
  coords = neighbors(_ORIGINS[1], basis)
  point = coords[0] + st[0] * basis[0] + st[1] * basis[1]
  original = coords.copy()
  weights = geo.bilinear_interpolation_weights(point, coords)
  np.testing.assert_allclose(weights @ coords, point, atol=1e-13)
  np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-14)
  np.testing.assert_array_equal(coords, original)


@pytest.mark.parametrize("st", [(0.1, 0.6), (0.7, 0.2), (0.5, 0.5)])
def test_values_match_independent_regular_grid_interpolation(st):
  origin, basis = _ORIGINS[1], _BASES[1]
  coords = neighbors(origin, basis)
  # Grid axes are (s, t); coordinate indices are (0,0), (1,0), (0,1), (1,1).
  values = np.array(
      [[2.0, 4.0, 8.0], [5.0, -3.0, 7.0], [-4.0, 6.0, 1.0], [9.0, 2.0, -1.0]]
  )
  grid = np.stack([values[[0, 2]], values[[1, 3]]])
  interpolate = RegularGridInterpolator(([0.0, 1.0], [0.0, 1.0]), grid)
  point = origin + st[0] * basis[0] + st[1] * basis[1]
  weights = geo.bilinear_interpolation_weights(point, coords)
  np.testing.assert_allclose(weights @ values, interpolate([st])[0], atol=1e-13)


@pytest.mark.parametrize(
    "valid",
    [
        [True, False, True, True],
        [True, True, False, True],
        [False, True, False, False],
        [False, False, False, False],
    ],
)
def test_valid_mask_uses_the_matching_neighbor(valid):
  coords = neighbors(_ORIGINS[0], _BASES[0])
  mask = np.array(valid)
  weights = geo.bilinear_interpolation_weights(
      np.array([0.4, 2.1, 0.0]), coords, mask
  )
  expected = np.array([0.24, 0.06, 0.56, 0.14]) * mask
  expected /= max(expected.sum(), 1e-6)
  np.testing.assert_allclose(weights, expected, atol=1e-14)


def test_three_neighbor_fallback_is_unchanged():
  coords = neighbors(_ORIGINS[0], _BASES[0])[:3]
  np.testing.assert_array_equal(
      geo.bilinear_interpolation_weights(np.array([0.3, 0.2, 0.0]), coords),
      [1.0, 0.0, 0.0],
  )


def test_failed_plane_fit_fallback_is_unchanged():
  coords = neighbors(_ORIGINS[0], _BASES[0])
  with mock.patch.object(geo, "st_coordinates_on_plane", return_value=None):
    np.testing.assert_array_equal(
        geo.bilinear_interpolation_weights(coords[0], coords),
        [1.0, 0.0, 0.0, 0.0],
    )
