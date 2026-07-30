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

"""Functions without visibility (usually unchanged copies)."""

import itertools
from typing import Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import jaxcam


# Source: third_party/py/jaxcam/_src/camera.py
def radial_and_tangential_distort(
    x: jnp.ndarray,
    y: jnp.ndarray,
    radial_distortion: jnp.ndarray | None = None,
    tangential_distortion: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes the distorted pixel positions."""
  dx_radial, dy_radial = 0.0, 0.0
  dx_tangential, dy_tangential = 0.0, 0.0
  r2 = x * x + y * y

  if radial_distortion is not None:
    k1, k2, k3, k4 = radial_distortion
    radial_distortion = r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4)))
    dx_radial = x * radial_distortion
    dy_radial = y * radial_distortion

  if tangential_distortion is not None:
    p1, p2 = tangential_distortion
    dx_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    dy_tangential = 2 * p2 * x * y + p1 * (r2 + 2 * y * y)

  return x + dx_radial + dx_tangential, y + dy_radial + dy_tangential


# Source: third_party/py/jaxcam/_src/camera.py
def radial_and_tangential_undistort(
    x_distorted: jnp.ndarray,
    y_distorted: jnp.ndarray,
    radial_distortion: jnp.ndarray,
    tangential_distortion: jnp.ndarray,
    max_iterations: int = 3,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Undistorts a point in image-space using an iterative optimization.

  This uses Newton-Raphson (related to Gauss-Newton) to find the undistorted
  pixel positiong by finding the roots of the equation `distort(x) - x_d = 0`.

  References:
    https://en.wikipedia.org/wiki/Newton%27s_method#k_variables,_k_functions

  Args:
    x_distorted: The distorted x pixel position.
    y_distorted: The distorted y pixel position.
    radial_distortion: (4,) The radial distortion parameters.
    tangential_distortion: (2,) The tangential distortion parameters.
    max_iterations: The number of iterations to run Newton's method.

  Returns:
    A tuple containing the undistorted x and y positions.
  """

  def f(xy, xy_distorted):
    x_distorted_pred, y_distorted_pred = radial_and_tangential_distort(
        xy[..., 0],
        xy[..., 1],
        radial_distortion=radial_distortion,
        tangential_distortion=tangential_distortion,
    )
    xy_distorted_pred = jnp.stack([x_distorted_pred, y_distorted_pred], axis=-1)
    return xy_distorted_pred - xy_distorted

  batch_shape = x_distorted.shape
  xy_distorted = jnp.stack([x_distorted, y_distorted], axis=-1)
  xy_distorted = xy_distorted.reshape(-1, 2)

  # Run Newton's method.
  jac_fn = jax.jacfwd(f)
  xy = xy_distorted
  for _ in range(max_iterations):
    jac = jax.vmap(jac_fn)(xy, xy_distorted)
    # Solve the linear system: J(x_n)(x{n+1}-x_n) = -f(x_n).
    # residual = x_{n+1} - x_n, so x_{n+1} = residual + x_n.
    residual = jax.vmap(jnp.linalg.solve)(jac, -f(xy, xy_distorted))
    xy = residual + xy

  # Reshape back to the original shape.
  xy = xy.reshape(*batch_shape, 2)
  return xy[..., 0], xy[..., 1]


# Source: third_party/py/jaxcam/_src/camera.py
def distort_local_pixels(
    camera: jaxcam.Camera, x: jnp.ndarray, y: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Distorts normalized image pixels."""
  if camera.use_inverted_distortion:
    x, y = radial_and_tangential_undistort(
        x,
        y,
        radial_distortion=camera.radial_distortion,
        tangential_distortion=camera.tangential_distortion,
    )
  elif camera.has_distortion:
    x, y = radial_and_tangential_distort(
        x,
        y,
        radial_distortion=camera.radial_distortion,
        tangential_distortion=camera.tangential_distortion,
    )

  return x, y


# Source: third_party/py/dm_pix/_src/interpolation.py
def make_linear_interpolation_indices_nd(
    coordinates: chex.Array,
    shape: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
  """Creates linear interpolation indices and weights for ND coordinates.

  Args:
    coordinates: An array of shape (N, M_coordinates).
    shape: The shape of the ND volume, e.g. if N=3 shape=(dim_z, dim_y, dim_x).

  Returns:
    The lower and upper indices of `coordinates` and their weights.
  """
  lower = jnp.floor(coordinates).astype(jnp.int32)
  upper = jnp.ceil(coordinates).astype(jnp.int32)
  weights = coordinates - lower

  # Expand dimensions for `shape` to allow broadcasting it to every coordinate.
  # Expansion size is equal to the number of dimensions of `coordinates` - 1.
  shape = shape.reshape(shape.shape + (1,) * (coordinates.ndim - 1))

  lower = jnp.clip(lower, 0, shape - 1)
  upper = jnp.clip(upper, 0, shape - 1)

  return lower, upper, weights


# Source: third_party/py/dm_pix/_src/interpolation.py
def make_linear_interpolation_indices_flat_nd(
    coordinates: chex.Array,
    shape: Sequence[int],
) -> Tuple[chex.Array, chex.Array]:
  """Creates flat linear interpolation indices and weights for ND coordinates.

  Args:
    coordinates: An array of shape (N, M_coordinates).
    shape: The shape of the ND volume, e.g. if N=3 shape=(dim_z, dim_y, dim_x).

  Returns:
    The indices into the flattened input and their weights.
  """
  coordinates = jnp.asarray(coordinates)
  shape = jnp.asarray(shape)

  if shape.shape[0] != coordinates.shape[0]:
    raise ValueError((
        f"{coordinates.shape[0]}-dimensional coordinates provided for "
        f"{shape.shape[0]}-dimensional input"
    ))

  lower_nd, upper_nd, weights_nd = make_linear_interpolation_indices_nd(
      coordinates, shape
  )

  # Here we want to translate e.g. a 3D-disposed indices to linear ones, since
  # we have to index on the flattened source, so:
  # flat_idx = shape[1] * shape[2] * z_idx + shape[2] * y_idx + x_idx

  # The `strides` of a `shape`-sized array tell us how many elements we have to
  # skip to move to the next position along a certain axis in that array.
  # For example, for a shape=(5,4,2) we have to skip 1 value to move to the next
  # column (3rd axis), 2 values to move to get to the same position in the next
  # row (2nd axis) and 4*2=8 values to move to get to the same position on the
  # 1st axis.
  strides = jnp.concatenate([jnp.cumprod(shape[:0:-1])[::-1], jnp.array([1])])

  # Array of 2^n rows where the ith row is the binary representation of i.
  binary_array = jnp.array(
      list(itertools.product([0, 1], repeat=shape.shape[0]))
  )

  # Expand dimensions to allow broadcasting `strides` and `binary_array` to
  # every coordinate.
  # Expansion size is equal to the number of dimensions of `coordinates` - 1.
  strides = strides.reshape(strides.shape + (1,) * (coordinates.ndim - 1))
  binary_array = binary_array.reshape(
      binary_array.shape + (1,) * (coordinates.ndim - 1)
  )

  lower_1d = lower_nd * strides
  upper_1d = upper_nd * strides

  point_weights = []
  point_indices = []

  for r in binary_array:
    # `point_indices` is defined as:
    # `jnp.matmul(binary_array, upper) + jnp.matmul(1-binary_array, lower)`
    # however, to date, that implementation turns out to be slower than the
    # equivalent following one.
    point_indices.append(jnp.sum(upper_1d * r + lower_1d * (1 - r), axis=0))
    point_weights.append(
        jnp.prod(r * weights_nd + (1 - r) * (1 - weights_nd), axis=0)
    )
  return jnp.stack(point_indices, axis=0), jnp.stack(point_weights, axis=0)
