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

"""Dataset preprocessing transformations."""

import dataclasses

from etils.array_types import FloatArray
import flax
import grain.python as grain
import numpy as np

from scaling_geoloc import types
from scaling_geoloc.utils import image as image_utils
from scaling_geoloc.utils import misc


@dataclasses.dataclass(frozen=True)
class MaybeCollate(grain.MapTransform):
  """Collate lists of parameters to an array."""

  def map(self, data: types.DataDict) -> types.DataDict:

    def maybe_stack(v):
      if isinstance(v, list):
        try:
          return np.stack(v, axis=0)
        except ValueError:
          return v
      else:
        return v

    return misc.map_dict(
        lambda k, v: maybe_stack(v),
        data,
    )


@dataclasses.dataclass(frozen=True)
class NormalizePixels(grain.MapTransform):
  """Normalize pixels to [0, 1]."""

  def map(self, data: types.DataDict) -> types.DataDict:
    return misc.map_dict(
        lambda k, v: image_utils.byte_to_float(v)
        if 'image' in k and v.dtype == np.uint8
        else v,
        data,
    )


@dataclasses.dataclass(frozen=True)
class ResizeImageLike(grain.MapTransform):
  """Resizes image to a final_size, stretching it as necessary."""

  final_size: tuple[int, int]

  def map(self, data: types.DataDict) -> types.DataDict:

    target_h, target_w = self.final_size

    for scene in ['query', 'aerial']:
      if scene not in data:
        continue
      image = data[scene]['image']
      h, w, _ = image.shape
      for k, v in data[scene].items():
        if not isinstance(v, np.ndarray) or np.ndim(v) != 3:
          continue
        if v.shape[:2] == (h, w):
          data[scene][k] = image_utils.image_resize_host(
              v,
              shape=(target_h, target_w, v.shape[-1]),
              method='bilinear' if 'image' in k else 'nearest',
              antialias=True,
              jit=False,  # Disable jit to avoid recompilations.
          )
    return data


@dataclasses.dataclass(frozen=True)
class CenterCropImageLike(grain.MapTransform):
  """Center crop all image like elements."""

  crop_size: tuple[int, int]  # w, h

  def map(
      self,
      data: types.DataDict,
  ) -> types.DataDict:
    """Returns the center crop region from the provided cfg and images sizes."""
    for scene in ['query', 'aerial']:
      if scene not in data:
        continue
      image = data[scene]['image']
      h, w, _ = image.shape

      offset_h = int((h - self.crop_size[1]) // 2)
      offset_w = int((w - self.crop_size[0]) // 2)

      for k, v in data[scene].items():
        if not isinstance(v, np.ndarray) or np.ndim(v) != 3:
          continue
        if v.shape[:2] == (h, w):
          if h >= self.crop_size[1] and w >= self.crop_size[0]:
            data[scene][k] = v[
                offset_h: offset_h + self.crop_size[1],
                offset_w: offset_w + self.crop_size[0],
                :,
            ]
          else:
            data[scene][k] = image_utils.image_resize_host(
                v,
                shape=(
                    self.crop_size[1],
                    self.crop_size[0],
                    v.shape[-1],
                ),
                method='bilinear' if k == 'image_pixels' else 'nearest',
                jit=False,  # Disable jit to avoid recompilations.
            )
    return data


@dataclasses.dataclass(frozen=True)
class ResizeAndPadImageLike(grain.MapTransform):
  """Resizes the longest edge to crop_size and pads the other."""

  crop_size: tuple[int, int]  # w, h

  def _resize_and_pad(
      self,
      target_size: int,
      v: FloatArray['H W C'],
      k: str,
      h: int,
      w: int,
  ) -> FloatArray['H W C']:
    """Resizes and pads the image."""

    # take the longest edge
    if w >= h:
      new_w = target_size
      new_h = int(h * target_size / w)
    else:
      new_h = target_size
      new_w = int(w * target_size / h)

    method = 'bilinear' if 'image' in k else 'nearest'
    resized_v = image_utils.image_resize_host(
        v,
        shape=(new_h, new_w, v.shape[-1]),
        method=method,
        jit=False,
    )

    pad_h = target_size - new_h
    pad_w = target_size - new_w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    pad_width = (
        (pad_top, pad_bottom),
        (pad_left, pad_right),
        (0, 0),
    )
    padded_v = np.pad(resized_v, pad_width, mode='constant', constant_values=0)

    return padded_v

  def map(
      self,
      data: types.DataDict,
  ) -> types.DataDict:
    """Resizes and pads image-like elements."""
    target_h, target_w = self.crop_size
    if target_h != target_w:
      raise NotImplementedError(
          'Crop size must be square for ResizeAndPadImageLike.'
      )
    target_size = target_h

    for scene in ['query', 'aerial']:
      if scene not in data:
        continue

      image = data[scene]['image']
      h, w, _ = image.shape

      for k, v in data[scene].items():
        if not isinstance(v, np.ndarray) or np.ndim(v) != 3:
          continue

        if v.shape[:2] == (h, w):
          data[scene][k] = self._resize_and_pad(
              target_size, v, k, h, w
          )

    return data


@dataclasses.dataclass(frozen=True)
class FilterKeys(grain.MapTransform):
  """Filter (nested) keys by pattern. We simply check 'pattern in key'."""

  keep: tuple[str, ...] | None = None  # None keeps everything
  exclude: tuple[str, ...] | None = None  # None excludes nothing
  sep: str | None = '.'

  def map(self, data: types.DataDict) -> types.DataDict:
    flat_dict = flax.traverse_util.flatten_dict(data, sep=self.sep)
    keep_keys = []  # We could cache this lazily
    for k in flat_dict.keys():
      keep = self.keep is None or any((pattern in k) for pattern in self.keep)
      exclude = self.exclude is not None and any(
          (pattern in k) for pattern in self.exclude
      )
      if keep and not exclude:
        keep_keys.append(k)
    return flax.traverse_util.unflatten_dict(
        {keep_k: flat_dict[keep_k] for keep_k in keep_keys}, sep=self.sep
    )


class FilterMissingAerialData(grain.FilterTransform):
  """Remove examples that are missing aerial data."""

  def filter(self, data: types.DataDict) -> bool:
    return data.get('aerial', None) is not None


class FilterInvalidCells(grain.FilterTransform):
  """Remove examples with invalid cell indices."""

  def filter(self, data: types.DataDict) -> bool:
    cell_idx = data.get('cell_idx', None)
    return cell_idx is not None and cell_idx >= 0
