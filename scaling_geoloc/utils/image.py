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

"""Image utilities: crop, resize, patch, and format conversions."""

from typing import Any

from etils.array_types import Array
from etils.array_types import BoolArray
from etils.array_types import FloatArray
import jax
import jax.numpy as jnp
import numpy as np


def sequence_to_image(
    sequence: FloatArray['... HW C'],
    img_height: int,
    img_width: int,
) -> FloatArray['... H W C']:
  """Converts sequence dimension to height width."""
  return jnp.reshape(
      sequence,
      sequence.shape[:-2] + (img_height, img_width) + (-1,),
  )


def crop_to_multiple(
    images: Array['... H W N'], stride: int
) -> Array['... H2 W2 N']:
  """Crop images so height and width are multiples of stride."""
  h, w = images.shape[-3:-1]
  ch = min((h // stride) * stride, h)
  cw = min((w // stride) * stride, w)
  return images[..., :ch, :cw, :]


def token_mask(
    rng: Any,
    image_shape: tuple[int, ...],
    patch_size: tuple[int, int],
    mask_ratio: float,
    mask_prob: float = 1.0,
) -> BoolArray['... NH NW 1']:
  """Generate a patch-mask for a batch of images."""
  h, w = image_shape[-3:-1]
  rng, prob_rng = jax.random.split(rng, 2)
  nh, nw = h // patch_size[0], w // patch_size[1]
  bs = image_shape[:-3]
  do_mask = jax.random.uniform(prob_rng, shape=bs + (1, 1)) < mask_prob
  is_mask = jax.random.uniform(rng, shape=bs + (nh * nw, 1)) < mask_ratio
  return do_mask & is_mask


def float_to_byte(image: FloatArray['...']) -> FloatArray['...']:
  """Converts a float image to a byte image."""
  return (image * 255.0).clip(0.0, 255.0).astype(jnp.uint8)


def byte_to_float(image: FloatArray['...']) -> FloatArray['...']:
  """Converts a byte image to a float image."""
  return image.astype(jnp.float32) / 255.0


def to_grayscale(image: FloatArray['... H W C']) -> FloatArray['... H W 1']:
  """Converts an RGB image to grayscale."""
  return jnp.sum(
      image * jnp.array([0.2989, 0.5870, 0.1140]), axis=-1, keepdims=True
  ).repeat(axis=-1, repeats=3)


def pad_like(
    inp: FloatArray['... Hi Wi N'], target: FloatArray['... H W N']
) -> FloatArray['... H W N']:
  out = jnp.zeros_like(target)
  out = out.at[..., : inp.shape[-3], : inp.shape[-2], :].set(inp)
  return out


def image_resize_host(
    image: FloatArray['... H W C'],
    shape: tuple[int, ...],
    method: str,
    antialias: bool = True,
    jit: bool = False,
) -> FloatArray['... H W C']:
  """Resize image to a certain shape on host. Disable jit by default."""
  host = jax.devices('cpu')[0]
  with jax.disable_jit(disable=not jit):
    image = jax.device_put(image, host)
    with jax.default_device(host):
      with jax.transfer_guard('allow'):
        return jax.device_get(
            jax.image.resize(
                image,
                shape=shape,
                method=method,
                antialias=antialias,
            )
        )


def pad_to_multiple(
    images: Array['B H W N'], stride: int, fill_value: int = 0
) -> Array['B H2 W2 N']:
  """Pad a batch of images such as their size is a multiple of stride."""
  shape = images.shape[-3:-1]
  pad = stride - np.array(shape) % stride
  return jnp.pad(
      images,
      [(0, 0), (0, pad[0]), (0, pad[1]), (0, 0)],
      constant_values=fill_value,
  )


def resize_to_multiple(
    images: Array['... H W N'],
    stride: int,
    method: str = 'bicubic',
    antialias: bool = True,
) -> Array['... H2 W2 N']:
  """Resize a batch of images such as their size is a multiple of stride."""
  h, w = images.shape[-3:-1]
  ch = (h // stride + 1) * stride
  cw = (w // stride + 1) * stride
  return jax.image.resize(
      images,
      (images.shape[:-3], ch, cw, images.shape[-1]),
      method=method,
      antialias=antialias,
  )


def reshape_image_to_patches(
    array: Array['... H W C'],
    patch_height: int,
    patch_width: int,
    autopad: bool = False,
    image_like: bool = False,
) -> Array['... (NH*NW) PH PW C'] | Array['... NH NW PH PW C']:
  """Reshapes an image into an array of patched tokens."""
  orig_shape = array.shape[:-3]
  array = jax.lax.collapse(array, 0, -3)
  if autopad:
    if patch_height != patch_width:
      raise ValueError('Patch height and width must be equal for padding.')
    array = pad_to_multiple(array, patch_height)
  _, input_height, input_width, input_channels = array.shape
  num_rows, num_cols = input_height // patch_height, input_width // patch_width
  array = array.reshape(
      *orig_shape,
      num_rows,
      patch_height,
      num_cols,
      patch_width,
      input_channels,
  )
  array = array.swapaxes(-4, -3)
  if image_like:
    return array
  return array.reshape(
      *orig_shape,
      num_rows * num_cols,
      patch_height,
      patch_width,
      input_channels,
  )


def reshape_patches_to_image(
    array: Array['... (NH*NW) PH PW C'] | Array['... NH NW PH PW C'],
    num_rows: int,
    num_cols: int,
    image_like: bool = False,
) -> Array['... H W C']:
  """Reshapes an array of patched tokens into an image."""
  orig_shape = array.shape[: (-5 if image_like else -4)]
  patch_height, patch_width, input_channels = array.shape[-3:]
  array = array.reshape(
      *orig_shape,
      num_rows,
      num_cols,
      patch_height,
      patch_width,
      input_channels,
  )
  array = array.swapaxes(-4, -3)
  return array.reshape(
      *orig_shape,
      num_rows * patch_height,
      num_cols * patch_width,
      input_channels,
  )


def patchify(x: Array['... H W C'], ph: int, pw: int) -> Array['... L C']:
  """Converts an image to a sequence of patches."""
  return jax.lax.collapse(
      reshape_image_to_patches(crop_to_multiple(x, ph), ph, pw), -3, -1
  )
