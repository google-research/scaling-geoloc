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

"""Geometry utils."""

from etils.array_types import BoolArray
from etils.array_types import FloatArray
import jax.numpy as jnp
import jaxcam
from scaling_geoloc.utils import hidden_helpers

MaskedPoints2D = tuple[FloatArray["... 2"], BoolArray["... 1"]]
MaskedPoints2DAndDepth = tuple[
    FloatArray["... 2"], BoolArray["... 1"], FloatArray["... 1"]
]


def project_points(
    camera: jaxcam.Camera, w_points: FloatArray["... 3"]
) -> MaskedPoints2DAndDepth:
  """Projects world-space 3D points (x,y,z) to pixel positions (x,y).

  Also returns a mask for valid points and the z-depth of each point.

  Source: project in third_party/py/jaxcam/_src/camera.py
  Adaptation:
    - Add valid mask and depth to source output.

  Args:
    camera: The camera to project with.
    w_points: (..., 3) The 3D world-space points to project.

  Returns:
    A (..., 2) array containing the projected pixel locations.
    A (..., 1) boolean mask whether the projection is valid.
    A (..., 1) array containing the z-depth of each point.
  """
  eps = jnp.finfo(w_points.dtype).eps
  batch_shape = w_points.shape[:-1]
  local_points = jaxcam.world_points_to_local_points(camera, w_points)

  local_x = local_points[..., 0]
  local_y = local_points[..., 1]
  local_z = local_points[..., 2]  # depth

  # Get normalized local pixel positions.
  x = local_x / local_z
  y = local_y / local_z
  if camera.projection_type.value == "fisheye":
    r = jnp.sqrt(local_x**2 + local_y**2)
    theta = jnp.arctan2(r, local_z)
    fisheye_x = theta / r * local_x
    fisheye_y = theta / r * local_y
    # When theta is small it is approximately the same as perspective.
    x = jnp.where(theta < eps, x, fisheye_x)
    y = jnp.where(theta < eps, y, fisheye_y)

  x, y = hidden_helpers.distort_local_pixels(camera, x, y)

  # Map the distorted ray to the image plane and return the depth.
  pixel_x = camera.focal_length * x + camera.skew * y + camera.principal_point_x
  pixel_y = (
      camera.focal_length * camera.pixel_aspect_ratio * y
      + camera.principal_point_y
  )

  pixels = jnp.stack([pixel_x, pixel_y], axis=-1)
  pixels = pixels.reshape((*batch_shape, 2))

  # Check positive depth
  valid = local_z > eps
  # Check pixel in image.
  valid = valid & (pixels > 0).all(-1)
  valid = valid & (pixel_x < camera.image_size[0])
  valid = valid & (pixel_y < camera.image_size[1])
  return pixels, valid[..., None], local_z[..., None]
