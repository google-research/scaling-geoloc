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

"""Utilities for encoding camera intrinsics."""

import jax.numpy as jnp
import jaxtyping as jt
from scaling_geoloc.data import types as data_types
from scaling_geoloc.scenic.configs import types as config_types


INTRINSICS_INDICES = {
    config_types.EnumIntrinsics.FOCAL_LENGTH_X: (0, 0),
    config_types.EnumIntrinsics.FOCAL_LENGTH_Y: (1, 1),
    config_types.EnumIntrinsics.PRINCIPAL_POINT_X: (0, 2),
    config_types.EnumIntrinsics.PRINCIPAL_POINT_Y: (1, 2),
}


def get_intrinsics_tensor(
    encode_intrinsics: list[config_types.EnumIntrinsics],
    camera: data_types.DataDict,
) -> jt.Float[jt.Array, 'B N 1'] | None:
  """Returns a tensor containing the specified camera intrinsics.

  Args:
    encode_intrinsics: A list of EnumIntrinsics specifying which intrinsics to
      extract.
    camera: A dictionary containing camera parameters, including the camera
      matrix 'K'.

  Returns:
    A tensor of shape [B, N, 1] containing the specified camera intrinsics, or
    None if `encode_intrinsics` is empty or `camera` does not contain 'K'.
  """
  if not encode_intrinsics or 'K' not in camera:
    return None

  indices = [INTRINSICS_INDICES[i] for i in encode_intrinsics]
  rows, cols = zip(*indices)

  camera_matrix: jt.Float[jt.Array, 'B 3 3'] = camera['K']
  intrinsics_tensor = camera_matrix[:, rows, cols]
  return intrinsics_tensor[..., jnp.newaxis]
