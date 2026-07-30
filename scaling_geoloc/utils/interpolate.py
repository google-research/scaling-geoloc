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

"""Image interpolation and sampling utilities."""

import functools
from typing import Callable

import dm_pix
from etils.array_types import FloatArray
from etils.array_types import IntArray
import jax
import jax.numpy as jnp
import jaxcam
import numpy as np
import scipy.ndimage

from scaling_geoloc.utils import geometry
from scaling_geoloc.utils.image import byte_to_float, float_to_byte


def grid_indices(
    h: int, w: int, batch_dims: tuple[int, ...] | None = None
) -> IntArray['*b h w 2']:
  xy = jnp.stack(
      jnp.meshgrid(jnp.arange(0, w), jnp.arange(0, h), indexing='xy'), axis=-1
  )
  if batch_dims is not None:
    xy = jnp.expand_dims(xy, axis=range(len(batch_dims)))
    xy = jnp.tile(xy, batch_dims + (1, 1, 1))
  return xy


def indices_to_coords(indices: IntArray) -> FloatArray:
  return indices + 0.5


def coords_to_indices(coords: FloatArray) -> IntArray:
  return np.rint(coords - 0.5).astype(np.int32)


# Source: third_party/py/dm_pix/_src/augmentation.py
def get_interpolate_function(
    mode: str,
    order: int,
    cval: float = 0.0,
    numpy: bool = False,
) -> Callable[[FloatArray, FloatArray], FloatArray]:
  """Selects the interpolation function to use based on the given parameters.

  PIX interpolations are preferred given they are faster on accelerators. For
  the cases where such interpolation is not implemented by PIX we really on
  jax.scipy.ndimage.map_coordinates. See specifics below.

  Args:
    mode: the mode parameter determines how the input array is extended beyond
      its boundaries. Modes 'nearest and 'constant' use PIX interpolation, which
      is very fast on accelerators (especially on TPUs). For all other modes,
      'wrap', 'mirror' and 'reflect', we rely on
      `jax.scipy.ndimage.map_coordinates`, which however is slow on
      accelerators, so use it with care.
    order: the order of the spline interpolation. The order has to be in the
      range [0, 1]. Note that PIX interpolation will only be used for order=1,
      for other values we use `jax.scipy.ndimage.map_coordinates`.
    cval: value to fill past edges of input if mode is 'constant'.
    numpy: If true, return a numpy interpolationfunction.

  Returns:
    The selected interpolation function.
  """
  if numpy:
    interpolate_function = functools.partial(
        scipy.ndimage.map_coordinates, mode=mode, order=order, cval=cval
    )
  else:
    if mode == 'nearest' and order == 1:
      interpolate_function = dm_pix.flat_nd_linear_interpolate
    elif mode == 'constant' and order == 1:
      interpolate_function = functools.partial(
          dm_pix.flat_nd_linear_interpolate_constant, cval=cval
      )
    else:
      interpolate_function = functools.partial(
          jax.scipy.ndimage.map_coordinates, mode=mode, order=order, cval=cval
      )
  return interpolate_function


def interpolate_image(
    image: FloatArray['H W C'],
    coords: FloatArray['H W 2'],
    order: int = 1,  # 0 for nearest, 1 for linear
    cval: float = 0.0,
    mode: str = 'constant',
    numpy: bool = False,
) -> FloatArray['H W C']:
  """Interpolates an image at given coordinates."""
  interp_function = get_interpolate_function(mode, order, cval, numpy)
  if numpy:
    image = np.array(image)
    coords = np.array(coords)
    return np.stack(
        [
            interp_function(layer, np.moveaxis(coords, -1, 0))
            for layer in np.permute_dims(image, (2, 0, 1))
        ],
        axis=-1,
    )
  else:
    return jax.vmap(interp_function, in_axes=(-1, None), out_axes=-1)(
        image, jnp.moveaxis(coords, -1, 0)
    )


def spherical_project(
    direction: FloatArray['3'], f_horizontal: float, f_vertical: float
) -> FloatArray['2']:
  """Projects a 3D direction into pixel coordinates of a spherical image."""
  unit_ray = direction / jnp.linalg.norm(direction).clip(min=1.0e-6)
  w_r_cam = jnp.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
  unit_ray = (unit_ray[None] @ w_r_cam.T)[0]
  phi = jnp.acos(unit_ray[2])
  y = phi * f_vertical
  theta = jnp.arctan2(unit_ray[1], unit_ray[0]) + jnp.pi
  x = theta * f_horizontal
  return jnp.stack([x, y])


def img_from_pano(
    pano_img: FloatArray['H W 3'], cam: jaxcam.Camera, **interp_kwargs
) -> FloatArray['Hi Wi 3']:
  """Extracts pixels of a camera from a pano."""
  w, h = cam.image_size
  xy = grid_indices(h, w) + 0.5
  rays = jaxcam.pixels_to_rays(cam, xy)
  ph, pw = pano_img.shape[:2]
  f_v = ph / jnp.pi
  f_h = pw / (2 * jnp.pi)
  pano_xy = jax.vmap(
      jax.vmap(
          functools.partial(spherical_project, f_horizontal=f_h, f_vertical=f_v)
      )
  )(rays)
  pano_ij = jnp.flip(pano_xy, -1)
  return interpolate_image(pano_img, pano_ij - 0.5, **interp_kwargs)


def crop_aerial_image(
    image: FloatArray['H W 3'],  # (x, y, c)
    resolution_meters_source: float,  # meters / pixel
    height_width: tuple[int, int],  # (height, width)
    dy_dx: FloatArray['2'],  # from center, (dy, dx)
    rotation_angle_degrees: float,  # degrees
    resolution_meters_crop: float,  # meters / pixel
    scale_offset_by_extent: bool = True,
    numpy: bool = False,
    order: int = 1,
    make_upright: bool = True,
) -> tuple[FloatArray['4 4'], FloatArray['H W 3'], FloatArray['H W 1']]:
  """Samples a crop from an aerial image."""
  xnp = np if numpy else jnp
  if scale_offset_by_extent:
    offset_meters = (
        dy_dx * xnp.array(image.shape[:-1]) * resolution_meters_source
    )
  else:
    offset_meters = dy_dx
  center_source = resolution_meters_source * xnp.array(image.shape[:-1]) / 2
  center_crop = resolution_meters_crop * xnp.array(height_width[::-1]) / 2
  grid_crop = grid_indices(*height_width) * resolution_meters_crop
  grid_crop = xnp.flip(grid_crop, -1)
  source_t2d_crop = (
      geometry.Transform2D.from_radians(angle=0, t=center_source)
      @ geometry.Transform2D.from_radians(
          angle=xnp.deg2rad(rotation_angle_degrees), t=offset_meters
      )
      @ geometry.Transform2D.from_radians(angle=0, t=-center_crop)
  )
  grid_transformed = (
      source_t2d_crop.transform(grid_crop)
  ) / resolution_meters_source
  valid = (grid_transformed >= 0.0).all(-1) & (
      grid_transformed < xnp.array(image.shape[:-1])[None, None]
  ).all(-1)
  if not numpy:
    image = byte_to_float(image)
  image_crop = interpolate_image(
      image, grid_transformed + 0.5, order=order, numpy=numpy
  )
  if not numpy:
    image_crop = float_to_byte(image_crop)
  source_t_crop = xnp.eye(4)
  if make_upright:
    image_crop = xnp.swapaxes(image_crop, 0, 1)[::-1]
  if numpy:
    source_t_crop[:2, :2] = source_t2d_crop.R
    source_t_crop[:2, -1] = source_t2d_crop.t
  else:
    source_t_crop = source_t_crop.at[:2, :2].set(source_t2d_crop.R)
    source_t_crop = source_t_crop.at[:2, -1].set(source_t2d_crop.t)
  return source_t_crop, image_crop, valid[..., None]
