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

"""Image augmentation utils (jittable)."""

from collections.abc import Callable, Sequence
from typing import TypeAlias
import dm_pix
from etils.array_types import FloatArray
from etils.array_types import IntArray
import jax
import jax.numpy as jnp

AugmentationFn: TypeAlias = Callable[
    [FloatArray['H W C'], jnp.ndarray], FloatArray['... H W C']
]


def maybe_apply(
    fn: AugmentationFn,
    probability: float,
) -> AugmentationFn:
  """Applies a function with a given probability."""
  def _maybe_apply(x, rng):
    rng, aug_rng = jax.random.split(rng, 2)
    # return fn(x, aug_rng) if jax.random.uniform(rng) < probability else x
    return jnp.where(jax.random.uniform(rng) < probability, fn(x, aug_rng), x)

  return _maybe_apply


def random_color_jitter(
    probability: float,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
) -> AugmentationFn:
  """Create a function for random color jittering."""
  def _adjust_brightness(x, rng):
    if brightness == 0:
      return x
    else:
      return dm_pix.adjust_brightness(
          x,
          jax.random.uniform(rng, minval=-brightness, maxval=brightness),
      )

  def _adjust_contrast(x, rng):
    if contrast == 0:
      return x
    else:
      return dm_pix.adjust_contrast(
          x,
          1 + jax.random.uniform(rng, minval=-contrast, maxval=contrast),
      )

  def _adjust_saturation(x, rng):
    if saturation == 0:
      return x
    else:
      return dm_pix.adjust_saturation(
          x, 1 + jax.random.uniform(rng, minval=-saturation, maxval=saturation)
      )

  def _adjust_hue(x, rng):
    if hue == 0:
      return x
    else:
      return dm_pix.adjust_hue(
          x, jax.random.uniform(rng, minval=-hue, maxval=hue)
      )

  def _apply_transform(i, x, rng):
    """Apply the i-th transformation."""
    return jax.lax.switch(
        i,
        [
            _adjust_brightness,
            _adjust_contrast,
            _adjust_saturation,
            _adjust_hue,
        ],
        x,
        rng,
    )

  def _random_color_jitter(image, rng):
    _, perm_rng, *aug_rngs = jax.random.split(rng, 6)
    perm = jax.random.permutation(perm_rng, 4)
    for i in range(4):
      image = _apply_transform(perm[i], image, aug_rngs[i]).clip(
          min=0.0, max=1.0
      )
    return image

  return maybe_apply(_random_color_jitter, probability)


def random_flip_left_right(
    probability: float,
) -> AugmentationFn:
  """Create a function for random left-right flipping."""
  return lambda x, rng: dm_pix.random_flip_left_right(
      rng, x, probability=probability
  )


def random_grayscale(
    probability: float,
) -> AugmentationFn:
  """Create a function for random gray scaling."""
  return maybe_apply(
      lambda x, _: dm_pix.rgb_to_grayscale(x, keep_dims=True), probability
  )


def random_gaussian_blur(
    probability: float,
    min_sigma: float,
    max_sigma: float,
    kernel_size: int = 3,
) -> AugmentationFn:
  """Random Gaussain blur.

  Args:
    probability: probability to apply the blur.
    min_sigma: min value for sigma to sample from.
    max_sigma: max value for sigma to sample from.
    kernel_size: kernel size for the blur.

  Returns:
    Augmentation function that applies Gaussian blur.
  """

  def _gaussian_blur(x, rng):
    sigma = jax.random.uniform(rng, minval=min_sigma, maxval=max_sigma)
    return dm_pix.gaussian_blur(x, sigma, kernel_size)

  return maybe_apply(_gaussian_blur, probability)


def random_solarize(
    probability: float,
    threshold: float,
) -> AugmentationFn:
  """Random solarize."""
  return maybe_apply(lambda x, _: dm_pix.solarize(x, threshold), probability)


def _inception_crop(
    rng: jnp.ndarray,
    x: FloatArray['H W 3'],
    crop_range: tuple[int, int],
    crop_size: tuple[int, int],
    keep_aspect_ratio: bool = False,
) -> tuple[FloatArray['Hc Wc 3'], IntArray, IntArray, IntArray, IntArray]:
  """Random distorted crop. Size defined as percent of original shape."""
  _, range_rng, h_offset_rng, w_offset_rng = jax.random.split(rng, 4)
  h, w = x.shape[:2]
  hr, wr = jax.random.randint(
      range_rng, (2,), minval=crop_range[0], maxval=crop_range[1]
  )
  if keep_aspect_ratio:
    wr = hr
  hr, wr = (h * hr / 100).astype(int), (w * wr / 100).astype(int)

  ho = jax.random.randint(h_offset_rng, (), minval=0, maxval=h - hr - 1)
  wo = jax.random.randint(w_offset_rng, (), minval=0, maxval=w - wr - 1)

  h_space = jnp.linspace(ho, ho + hr, crop_size[0])
  w_space = jnp.linspace(wo, wo + wr, crop_size[1])

  coordinates = jnp.stack(jnp.meshgrid(h_space, w_space, indexing='ij'), axis=0)

  interp_function = jax.vmap(
      dm_pix.flat_nd_linear_interpolate, in_axes=(-1, None), out_axes=-1
  )
  x_crop = interp_function(x, coordinates)

  return x_crop, ho, wo, hr, wr


def inception_crop(
    crop_range: tuple[int, int],
    crop_size: tuple[int, int],
    keep_aspect_ratio: bool = False,
) -> AugmentationFn:
  """Random distorted crop. Discards bounding box."""

  def _crop(x, rng):
    return _inception_crop(
        rng, x, crop_range, crop_size, keep_aspect_ratio=keep_aspect_ratio
    )[0]

  return _crop


def _random_stretch(
    rng: jnp.ndarray,
    x: jnp.ndarray,
    max_fraction: float,
) -> jnp.ndarray:
  """Randomly stretches one side of an image."""

  def get_crop_coords(
      size: int,
      start_fraction: float | jax.Array,
      crop_fraction: float | jax.Array,
  ) -> jax.Array:

    keep_fraction = 1.0 - crop_fraction

    start = size * start_fraction
    end = size * (start_fraction + keep_fraction) - 1.0

    return jnp.linspace(start, end, size)

  h, w = x.shape[:2]

  axis_rng, frac_rng, start_rng = jax.random.split(rng, 3)

  # Randomly choose axis (horizontal or vertical).
  axis = jax.random.randint(axis_rng, shape=(), minval=0, maxval=2)

  # Sample the fraction of the image to crop.
  crop_fraction = jax.random.uniform(
      frac_rng, shape=(), minval=0.0, maxval=max_fraction
  )

  # Sample the starting position of the stretch within the cropped range.
  start_fraction = jax.random.uniform(
      start_rng, shape=(), minval=0.0, maxval=crop_fraction
  )

  # Create linear spaces for the original height and width coordinates.
  h_space = jnp.linspace(0.0, h - 1.0, h)
  if h == w:
    w_space = h_space
  else:
    w_space = jnp.linspace(0.0, w - 1.0, w)

  # resize to create the stretching effect.
  h_space = jnp.where(
      axis == 1, get_crop_coords(h, start_fraction, crop_fraction), h_space
  )
  w_space = jnp.where(
      axis == 0, get_crop_coords(w, start_fraction, crop_fraction), w_space
  )

  coordinates = jnp.stack(jnp.meshgrid(h_space, w_space, indexing='ij'), axis=0)

  # Resample each channel of the image using the new coordinates.
  def resample_channel(channel_slice: jnp.ndarray) -> jnp.ndarray:
    return jax.scipy.ndimage.map_coordinates(
        channel_slice, coordinates, order=1, mode='nearest'
    )

  img_chw = jnp.moveaxis(x, -1, 0)
  stretched_img_chw = jax.vmap(resample_channel)(img_chw)

  return jnp.moveaxis(stretched_img_chw, 0, -1)


def random_stretch(
    probability: float,
    max_fraction: float,
) -> AugmentationFn:
  """Randomly stretches the side of an image."""

  return maybe_apply(
      lambda x, rng: _random_stretch(rng, x, max_fraction), probability
  )


def random_padding(
    probability: float,
    max_fraction: float,
) -> AugmentationFn:
  """Randomly overwrites one of the image borders with zeros."""

  def _overwrite_border(x: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
    """Applies the symmetric border overwrite transformation."""

    h, w = x.shape[-3:-1]
    rng_axis, rng_fraction = jax.random.split(rng)

    axis = jax.random.randint(rng_axis, (), 0, 2)
    fraction = jax.random.uniform(rng_fraction) * max_fraction

    if h == w:
      pad_size = (h * fraction).astype(jnp.int32)
      h_coords = jnp.arange(h)[:, None]
      mask_top_bottom = h_coords < pad_size
      vertical_mask = jnp.logical_or(
          mask_top_bottom, h_coords >= (h - pad_size)
      )
      horizontal_mask = jnp.rot90(vertical_mask)
    else:
      pad_h = (h * fraction).astype(jnp.int32)
      pad_w = (w * fraction).astype(jnp.int32)
      h_coords = jnp.arange(h)[:, None]
      w_coords = jnp.arange(w)[None, :]
      mask_top = h_coords < pad_h
      mask_bottom = h_coords >= (h - pad_h)
      mask_left = w_coords < pad_w
      mask_right = w_coords >= (w - pad_w)
      vertical_mask = jnp.logical_or(mask_top, mask_bottom)
      horizontal_mask = jnp.logical_or(mask_left, mask_right)

    final_mask_bool = jax.lax.switch(
        axis,
        [
            lambda: jnp.broadcast_to(vertical_mask, (h, w)),
            lambda: jnp.broadcast_to(horizontal_mask, (h, w)),
        ],
    )

    final_mask_bool = final_mask_bool[..., None]
    return jnp.where(final_mask_bool, 0, x)

  return maybe_apply(_overwrite_border, probability)


def random_crop(
    crop_size: tuple[int, int],
    resize: bool = False,
) -> AugmentationFn:
  """Random crop (no rescaling). Discards bounding box."""

  def _crop(x, rng):
    _, h_offset_rng, w_offset_rng = jax.random.split(rng, 3)
    h, w = x.shape[-3:-1]
    hr, wr = crop_size
    ho = jax.random.randint(h_offset_rng, (), minval=0, maxval=h - hr + 1)
    wo = jax.random.randint(w_offset_rng, (), minval=0, maxval=w - wr + 1)
    x_crop = jax.lax.dynamic_slice(
        x,
        x.shape[:-3] + (ho, wo, 0),
        x.shape[:-3] + (hr, wr, x.shape[-1]),
    )
    if resize:
      return jax.image.resize(
          x_crop,
          x.shape,
          jax.image.ResizeMethod.LINEAR,
      )
    else:
      return x_crop

  return _crop


def per_image(fn: AugmentationFn, vmap: bool = True) -> AugmentationFn:
  """Wrapper that applies the augmentation to each image in the batch."""
  def _per_image(images, rng):
    flat_images = images.reshape(-1, *images.shape[-3:])
    _, *per_image_rngs = jax.random.split(rng, 1 + flat_images.shape[0])
    per_image_rngs = jnp.stack(per_image_rngs)
    if vmap:
      aug_images = jax.vmap(fn, in_axes=(0, 0))(flat_images, per_image_rngs)
    else:
      aug_images = jnp.stack(
          [
              fn(flat_images[i], per_image_rngs[i])
              for i in range(flat_images.shape[0])
          ],
          axis=0,
      )
    return aug_images.reshape(*images.shape[:-3], *aug_images.shape[-3:])

  return _per_image


def sequential(fns: Sequence[AugmentationFn]) -> AugmentationFn:
  """Wrapper that applies the augmentation to each image in the batch."""

  def _sequential(x, rng):
    for fn in fns:
      rng, new_rng = jax.random.split(rng, 2)
      x = fn(x, new_rng)
    return x

  return _sequential


def get_crop_fn(name: str, crop_size: tuple[int, int]) -> AugmentationFn:
  """Returns a crop function."""
  if name == 'inception':
    return per_image(
        inception_crop((60, 100), crop_size, keep_aspect_ratio=True)
    )
  if name == 'inception_allow_aspect':
    return per_image(
        inception_crop((60, 100), crop_size, keep_aspect_ratio=False)
    )
  elif name == 'random':
    return per_image(random_crop(crop_size, resize=False))
  elif name == 'random_resize':
    return per_image(
        random_crop(
            tuple(int(0.5 * x) for x in crop_size),
            resize=True,
        )
    )
  elif name == 'maybe_random_resize':
    return per_image(
        maybe_apply(
            random_crop(tuple(int(0.5 * x) for x in crop_size), resize=True),
            0.7,
        )
    )
  elif name == 'center':
    return lambda x, rng: dm_pix.center_crop(x, *crop_size, channel_axis=-1)
  elif name == 'none' or name is None or name == 'None':
    return lambda x, rng: x
  else:
    raise ValueError(f'Unknown crop name: {name}')
