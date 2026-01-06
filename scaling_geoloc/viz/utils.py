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

"""Visualization utils."""

from collections.abc import Sequence
from etils.array_types import FloatArray
from etils.array_types import IntArray
import numpy as np


# Adapted from: https://github.com/empet/Voxel-data-visualization/
def cube_points(
    position3d: FloatArray['3'], cell_size: float
) -> FloatArray['8 3']:
  """Points of a 3d cube."""
  cube = np.array(
      [
          [0, 0, 0],
          [1, 0, 0],
          [1, 1, 0],
          [0, 1, 0],
          [0, 0, 1],
          [1, 0, 1],
          [1, 1, 1],
          [0, 1, 1],
      ],
      dtype=float,
  ) * cell_size + np.asarray(position3d)

  return cube


# Adapted from: https://github.com/empet/Voxel-data-visualization/
def triangulate_cube_faces(
    voxels: FloatArray['N 3'], cell_size: float
) -> tuple[FloatArray['N*8 3'], Sequence[IntArray['N*12']]]:
  """Triangulate the faces of a 3d cube."""

  voxels = np.asarray(voxels)
  all_cubes = [cube_points(pos, cell_size) for pos in voxels]
  p, q, r = np.array(all_cubes).shape
  vertices, ixr = np.unique(
      np.array(all_cubes).reshape(p * q, r), return_inverse=True, axis=0
  )
  indices = [[], [], []]
  # Each triplet (i, j, k) defines a face/triangle
  for k in range(len(all_cubes)):
    indices[0].extend(
        np.take(
            ixr,
            [
                8 * k,
                8 * k + 2,
                8 * k + 4,
                8 * k + 6,
                8 * k + 5,
                8 * k + 2,
                8 * k + 4,
                8 * k + 3,
                8 * k + 6,
                8 * k + 3,
                8 * k + 4,
                8 * k + 1,
            ],
        )
    )
    indices[1].extend(
        np.take(
            ixr,
            [
                8 * k + 1,
                8 * k + 3,
                8 * k + 5,
                8 * k + 7,
                8 * k + 1,
                8 * k + 6,
                8 * k,
                8 * k + 7,
                8 * k + 2,
                8 * k + 7,
                8 * k + 5,
                8 * k,
            ],
        )
    )
    indices[2].extend(
        np.take(
            ixr,
            [
                8 * k + 2,
                8 * k,
                8 * k + 6,
                8 * k + 4,
                8 * k + 2,
                8 * k + 5,
                8 * k + 3,
                8 * k + 4,
                8 * k + 3,
                8 * k + 6,
                8 * k + 1,
                8 * k + 4,
            ],
        )
    )

  return vertices, indices
