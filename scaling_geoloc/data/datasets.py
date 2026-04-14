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

"""Dataset for training foundation models."""

import collections
from collections.abc import Callable, Mapping, Sequence
import copy
import dataclasses
import functools
import json
import os
from typing import Any, cast, Final

from absl import logging
import big_vision.pp.builder as pp_builder
import big_vision.pp.ops_text  # pylint: disable=unused-import
import big_vision.pp.proj.paligemma.ops  # pylint: disable=unused-import
from etils.array_types import BoolArray
from etils.array_types import FloatArray
from etils.array_types import IntArray
import flax
import grain.python as grain
import jax
import jax.numpy as jnp
import ml_collections
import nltk
import numpy as np
import pyproj
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib.big_transfer import builder as big_builder
from scenic.projects.dino import ops  # pylint: disable=unused-import
from scenic.projects.dino.datasets import dino_dataset
import scipy.spatial.transform as scipy_transform
import tensorflow as tf
import tensorflow_datasets as tfds

from scaling_geoloc.data import geo
from scaling_geoloc.scenic import combined_data_source
from scaling_geoloc.scenic import indexed_data_source
from scaling_geoloc.scenic import sv_aerial_data_source
from scaling_geoloc.scenic import types
from scaling_geoloc.utils import augmentations
from scaling_geoloc.utils import grids
from scaling_geoloc.utils import io as lama_io
from scaling_geoloc.utils import misc
from google3.pyglib import gfile
from google3.pyglib import resources
from google3.util.geometry.python import s2


@dataclasses.dataclass(frozen=True)
class S2CellTokenToId(grain.MapTransform):
  """Converts S2Cell token to ID or index and adds parent at level 14."""

  s2cell_id_to_idx: Mapping[int, int]

  def map(self, data: types.DataDict) -> types.DataDict:
    data_dict = dict(data)
    s2cell_token = data_dict['s2cell']
    s2cell = s2.S2CellId.from_token(s2cell_token)
    if self.s2cell_id_to_idx is None:
      return data_dict
    s2cell_at_target_level = s2cell.parent_at_level(14)
    data_dict['s2cell_id'] = np.array(
        self.s2cell_id_to_idx[s2cell_at_target_level.id], dtype=np.int32
    )
    data_dict['s2cell_14'] = np.array(
        s2cell_at_target_level.id, dtype=np.uint64
    )
    return data_dict


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
        lambda k, v: misc.byte_to_float(v)
        if ('image_pixels' in k or 'rgb' in k) and v.dtype == np.uint8
        else v,
        data,
    )


@dataclasses.dataclass(frozen=True)
class MapToMap(grain.MapTransform):
  """Create queries from the map views, i.e. no temporal changes."""

  num_views_per_collection: Mapping[str, int]

  def map(self, data: types.DataDict) -> types.DataDict:
    """Does not work with pointcloud data right now."""
    data['query'] = jax.tree.map(
        lambda x: x[: self.num_views_per_collection['query']], data['map']
    )
    data['map'] = jax.tree.map(
        lambda x: x[-self.num_views_per_collection['map'] :], data['map']
    )
    return data


@dataclasses.dataclass(frozen=True)
class SampleViews(grain.RandomMapTransform):
  """Subsample views from the full data dict."""

  num_views_per_collection: Mapping[str, int]

  def random_map(
      self, data: types.DataDict, rng: np.random.Generator
  ) -> types.DataDict:
    for collection_name, num_views in self.num_views_per_collection.items():
      if num_views == 0:
        del data[collection_name]
        continue
      num_views_in_collection = data[collection_name]['views'][
          'image_pixels'
      ].shape[0]
      sampled_indices = rng.choice(
          num_views_in_collection, size=num_views, replace=False
      ).astype(int)
      data[collection_name]['views'] = jax.tree.map(
          lambda x: x[sampled_indices],  # pylint: disable=cell-var-from-loop
          data[collection_name]['views'],
      )
    return data


def get_process_batch_size(batch_size: int) -> int:
  if batch_size % jax.device_count() != 0:
    raise ValueError(
        f'batch_size {batch_size} must be divisible by '
        f' jax.device_count() ({jax.device_count()})'
    )
  return (batch_size // jax.device_count()) * jax.local_device_count()


@dataclasses.dataclass(frozen=True)
class FilterLiDAR(grain.FilterTransform):
  """Filter elements with too small point cloud."""

  min_num_points: int

  def filter(self, data: types.DataDict) -> bool:
    """Filters a single element."""
    all_scenes_have_enough_points = True
    for scene in ['map', 'query']:
      scene_data = data[scene]
      if 'pointcloud' in scene_data:
        all_scenes_have_enough_points &= (
            scene_data['pointcloud']['points'].shape[0] >= self.min_num_points
        )
    return all_scenes_have_enough_points


@dataclasses.dataclass(frozen=True)
class ProcessLiDAR(grain.RandomMapTransform):
  """Process LiDAR point cloud."""

  num_points: int

  def random_map(
      self, data: types.DataDict, rng: np.random.Generator
  ) -> types.DataDict:
    for scene in ['map', 'query']:
      scene_data = data[scene]
      if 'pointcloud' in scene_data:
        num_available_points = scene_data['pointcloud']['points'].shape[0]
        sampled_indices = rng.choice(
            range(num_available_points),
            size=self.num_points,
            replace=False,
        ).astype(int)
        for k in [
            'points',
            'origins',
            'timestamps',
            'reflectivities',
            'semantic_labels',
        ]:
          data[scene]['pointcloud'][k] = scene_data['pointcloud'][k][
              sampled_indices
          ]
    return data


@dataclasses.dataclass(frozen=True)
class FilterPixelsWithSemantics(grain.MapTransform):
  """Update pixel mask with semantics. Also updates lidar mask."""

  ignore_classes: Sequence[str] = ('unknown',)
  mask_classes: Sequence[str] = ('self', 'sky', 'lidar_artifact')

  enforce_2d3d_semantic_consistency: bool = True

  def map(self, data: types.DataDict) -> types.DataDict:
    mask_class_indices = [
        types.SEMANTIC_NAME_TO_INDEX[c] for c in self.mask_classes
    ]
    ignore_class_indices = [
        types.SEMANTIC_NAME_TO_INDEX[c] for c in self.ignore_classes
    ]
    for scene in ['map', 'query']:
      semantic_raster = data[scene]['views']['semantic_raster']

      pixel_mask = np.full_like(
          data[scene]['views']['image_pixels'][..., :1],
          types.MaskType.VALID.value,
          dtype=np.uint8,
      )
      pixel_mask = np.where(
          np.isin(semantic_raster, mask_class_indices),
          types.MaskType.INVALID.value,
          pixel_mask,
      )
      pixel_mask = np.where(
          np.isin(semantic_raster, ignore_class_indices),
          types.MaskType.IGNORE.value,
          pixel_mask,
      )

      data[scene]['views']['pixel_mask'] = pixel_mask

      if 'lidar_mask' in data[scene]['views']:
        lidar_mask = data[scene]['views']['lidar_mask'] & (
            pixel_mask == types.MaskType.VALID.value
        )
        if 'lidar_semantic_labels' in data[scene]['views']:
          lidar_semantic_labels = data[scene]['views']['lidar_semantic_labels']
          lidar_mask &= ~np.isin(
              lidar_semantic_labels, mask_class_indices + ignore_class_indices
          )

          if self.enforce_2d3d_semantic_consistency:
            lidar_mask &= semantic_raster == lidar_semantic_labels
        data[scene]['views']['lidar_mask'] = lidar_mask
    return data


class RemovePointcloud(grain.MapTransform):
  """Removes the pointcloud from the data dict."""

  def map(self, data: types.DataDict) -> types.DataDict:
    for k in data.keys():
      if isinstance(data[k], dict) and 'pointcloud' in data[k]:
        del data[k]['pointcloud']
    return data


@dataclasses.dataclass(frozen=True)
class AddGrid3DTransform(grain.MapTransform):
  """Selects a reference view and adds a 3D grid transform."""

  grid_3d: grids.Grid3D
  z_offset: float | None = None
  center_around_reference: bool = False

  def map(self, data: types.DataDict) -> types.DataDict:
    # We first obtain the coordinate system of the cell (gravity-aligned).
    ecef_t_scene = geo.ecef_t_s2cell(
        data['s2cell'].decode('utf-8'), altitude=0.0
    ).astype(data['ecef_t_w'].dtype)
    scene_t_w = np.linalg.inv(ecef_t_scene) @ data['ecef_t_w']
    # Select the first query as the reference view.
    w_t_ref = np.linalg.inv(data['query']['views']['cam_t_w'][0])
    scene_t_ref = scene_t_w @ w_t_ref

    grid_t_scene = grids.grid_t_w(
        scene_t_ref,
        self.grid_3d,
        z_offset=self.z_offset,
        center_around_reference=self.center_around_reference,
    )

    data['scene_t_w'] = scene_t_w
    data['grid_t_w'] = grid_t_scene @ scene_t_w
    return data


@dataclasses.dataclass(frozen=True)
class VoxelizeLidarProjections(grain.MapTransform):
  """Adds voxel indices for LiDAR points."""

  grid_3d: grids.Grid3D

  def map(self, data: types.DataDict) -> types.DataDict:
    grid_t_w = data['grid_t_w']
    for scene in ['map', 'query']:
      scene_data = data[scene]
      w_xyz = scene_data['views']['w_lidar_points']
      grid_xyz = np.einsum(
          '...ij,...hwj->...hwi',
          grid_t_w[..., :3, :],
          misc.to_homogeneous(w_xyz),
      )
      grid_index = self.grid_3d.xyz_to_index(grid_xyz)
      grid_id = self.grid_3d.index_to_id(grid_index)[..., None]

      valid = (
          scene_data['views']['lidar_mask']
          & self.grid_3d.index_in_grid(grid_index)[..., None]
      )
      grid_id = np.where(valid, grid_id, -1)
      data[scene]['views']['grid_3d_id'] = grid_id
    return data


@dataclasses.dataclass(frozen=True)
class DominantVoxelsPerPatch(grain.MapTransform):
  """Gather dominant grid_3d_id per patch, and collect statistics."""

  patch_size: int = 16
  num_voxels_per_patch: int = 1
  top_k_unique_ids: int = 400

  def map(self, data: types.DataDict) -> types.DataDict:
    for scene in ['map', 'query']:
      scene_data = data[scene]
      grid_id = scene_data['views']['grid_3d_id']
      grid_id_padded = misc.crop_to_multiple(grid_id, self.patch_size)

      grid_id_patched = misc.reshape_image_to_patches(
          grid_id_padded, self.patch_size, self.patch_size, image_like=True
      )

      best_ids, counts = grids.gather_top_k_ids(
          grid_id_patched, self.num_voxels_per_patch, start_dim=-3
      )
      data[scene]['views']['grid_3d_id_patch'] = grid_id_patched
      data[scene]['views']['grid_3d_id_patch_best'] = best_ids
      data[scene]['views']['grid_3d_id_patch_count'] = counts
      data[scene]['views']['grid_3d_id_unique'] = jnp.unique(
          best_ids, size=self.top_k_unique_ids, fill_value=-1
      )
    return data


@dataclasses.dataclass(frozen=True)
class DINOAugment(grain.MapTransform):
  """Create multiview crop from images."""

  process_fn: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]
  keep: bool = False

  def map(self, data: types.DataDict) -> types.DataDict:
    images_dict = {
        'image': tf.convert_to_tensor(
            misc.concat_all('image_pixels', data, axis=-4)
        ),
        'query_image': tf.convert_to_tensor(
            misc.concat_all('image_pixels', data['query'], axis=-4)
        ),
        'map_image': tf.convert_to_tensor(
            misc.concat_all('image_pixels', data['map'], axis=-4)
        ),
    }
    crops = self.process_fn(images_dict)
    crops = {k: v.numpy() for k, v in crops.items()}  # pytype: disable=attribute-error

    if self.keep:
      return {
          **crops,
          **data,
      }
    else:
      return crops


@dataclasses.dataclass
class DINOCrop(grain.RandomMapTransform):
  """Create inception crops from images."""

  crop_size: int = 224
  crop_size_local: int = 96
  num_local_crops: int = 4

  crop_range: tuple[int, int] = (40, 100)
  crop_range_local: tuple[int, int] = (5, 40)
  flip_lr: bool = True
  photo_augment: bool = False

  augment: str | None = None  # Overrides above if specified

  # Auto-filled
  process_fn: (
      Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] | None
  ) = None

  augment_global0: augmentations.AugmentationFn = augmentations.per_image(
      augmentations.sequential([
          augmentations.random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1),
          augmentations.random_grayscale(0.2),
          augmentations.random_gaussian_blur(1.0, 0.1, 2.0),
      ])
  )

  augment_global1: augmentations.AugmentationFn = augmentations.per_image(
      augmentations.sequential([
          augmentations.random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1),
          augmentations.random_grayscale(0.2),
          augmentations.random_gaussian_blur(0.1, 0.1, 2.0),
          augmentations.random_solarize(0.2, 0.5),
      ])
  )

  augment_local: augmentations.AugmentationFn = augmentations.per_image(
      augmentations.sequential([
          augmentations.random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1),
          augmentations.random_grayscale(0.2),
          augmentations.random_gaussian_blur(0.5, 0.1, 2.0),
      ])
  )

  def __post_init__(self):
    augment = self.augment
    if augment is None:
      augment = (
          'copy("image_pixels",'
          f' "global_view0")|inception_crop({self.crop_size},'
          f' {self.crop_range[0]}, {self.crop_range[1]},'
          ' data_key="global_view0")'
          + ('|flip_lr(data_key="global_view0")' if self.flip_lr else '')
          + '|copy("image_pixels",'
          f' "global_view1")|inception_crop({self.crop_size},'
          f' {self.crop_range[0]}, {self.crop_range[1]},'
          ' data_key="global_view1")'
          + ('|flip_lr(data_key="global_view1")' if self.flip_lr else '')
          + ''.join([
              f'|copy("image_pixels", "local_view{i}")'
              for i in range(self.num_local_crops)
          ])
          + ''.join([
              f'|inception_crop({self.crop_size_local},'
              f' {self.crop_range_local[0]}, {self.crop_range_local[1]},'
              f' data_key="local_view{i}")'
              for i in range(self.num_local_crops)
          ])
          + ''.join([
              f'|flip_lr(data_key="local_view{i}")'
              for i in range(self.num_local_crops)
              if self.flip_lr
          ])
          + '|keep("global_view0", "global_view1"'
          + ''.join([f', "local_view{i}"' for i in range(self.num_local_crops)])
          + ')'
      )
    logging.info('DINO Augmentation: %s', augment)
    self.process_fn = big_builder.get_preprocess_fn(augment, log_data=False)

  def crop_and_stack(
      self, image: FloatArray['N H W C'], rng: np.random.Generator
  ) -> dict[str, FloatArray]:
    with tf.device('CPU'):
      tf_image = tf.convert_to_tensor(image)
      crops = self.process_fn({'image_pixels': tf_image})
      crops = {k: dataset_utils.tf_to_numpy(v) for k, v in crops.items()}
    if self.photo_augment:
      seed0, seed1, seed_local = rng.integers(0, 100000, size=3)
      crops['global_view0'] = self.augment_global0(
          crops['global_view0'],
          jax.random.PRNGKey(seed0),
      )
      crops['global_view1'] = self.augment_global1(
          crops['global_view1'], jax.random.PRNGKey(seed1)
      )
      if self.num_local_crops > 0:
        for i in range(self.num_local_crops):
          crops[f'local_view{i}'] = self.augment_local(
              crops[f'local_view{i}'],
              jax.random.PRNGKey(seed_local),
          )
    return crops

  def random_map(
      self, data: types.DataDict, rng: np.random.Generator
  ) -> types.DataDict:
    for scene in ['query', 'map']:
      if scene not in data:
        continue
      crops = self.crop_and_stack(data[scene]['views']['image_pixels'], rng)
      for k, v in crops.items():
        data[scene]['views'][k] = v
    return data


@dataclasses.dataclass(frozen=True)
class ResizeImageLike(grain.MapTransform):
  """Resizes image to a final_size, stretching it as necessary."""

  final_size: tuple[int, int]

  def map(self, data: types.DataDict) -> types.DataDict:

    target_h, target_w = self.final_size

    for scene in ['query', 'map']:
      if scene not in data:
        continue
      image_pixels = data[scene]['views']['image_pixels']
      n, h, w, _ = image_pixels.shape

      for k, v in data[scene]['views'].items():
        if not isinstance(v, np.ndarray) or np.ndim(v) != 4:
          continue
        if v.shape[-4:-1] == (n, h, w):
          data[scene]['views'][k] = misc.image_resize_host(
              v,
              shape=(n, target_h, target_w, v.shape[-1]),
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
    for scene in ['query', 'map']:
      if scene not in data:
        continue
      image_pixels = data[scene]['views']['image_pixels']
      n, h, w, _ = image_pixels.shape

      offset_h = int((h - self.crop_size[1]) // 2)
      offset_w = int((w - self.crop_size[0]) // 2)

      for k, v in data[scene]['views'].items():
        if not isinstance(v, np.ndarray) or np.ndim(v) != 4:
          continue
        if v.shape[-4:-1] == (n, h, w):
          if h >= self.crop_size[1] and w >= self.crop_size[0]:
            data[scene]['views'][k] = v[
                :,
                offset_h : offset_h + self.crop_size[1],
                offset_w : offset_w + self.crop_size[0],
                :,
            ]
          else:
            data[scene]['views'][k] = misc.image_resize_host(
                v,
                shape=(
                    n,
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
      v: FloatArray['N H W C'],
      k: str,
      n: int,
      h: int,
      w: int,
  ) -> FloatArray['N H W C']:
    """Resizes and pads the image."""

    # take the longest edge
    if w >= h:
      new_w = target_size
      new_h = int(h * target_size / w)
    else:
      new_h = target_size
      new_w = int(w * target_size / h)

    method = 'bilinear' if 'image' in k else 'nearest'
    resized_v = misc.image_resize_host(
        v,
        shape=(n, new_h, new_w, v.shape[-1]),
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
        (0, 0),
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

    for scene in ['query', 'map']:
      if scene not in data:
        continue

      original_image_pixels = data[scene]['views']['image_pixels']
      n, h, w, _ = original_image_pixels.shape

      for k, v in data[scene]['views'].items():
        if not isinstance(v, np.ndarray) or np.ndim(v) != 4:
          continue

        if v.shape[-4:-1] == (n, h, w):
          data[scene]['views'][k] = self._resize_and_pad(
              target_size, v, k, n, h, w
          )

    return data


def _sample_block_size(
    rng: np.random.Generator,
    height: int,
    width: int,
    scale: tuple[float, float],
    aspect_ratio_scale: tuple[float, float],
) -> tuple[int, int]:
  """Sample block size."""
  # Sample block scale.
  randv = rng.uniform(size=())
  min_s, max_s = scale
  mask_scale = min_s + randv * (max_s - min_s)
  max_keep = int(height * width * mask_scale)
  # Sample block aspect-ratio.
  min_ar, max_ar = aspect_ratio_scale
  aspect_ratio = min_ar + randv * (max_ar - min_ar)
  # Compute block height and width (given scale and aspect-ratio).
  h = int(round((max_keep * aspect_ratio) ** 0.5))
  w = int(round((max_keep / aspect_ratio) ** 0.5))
  while h > height:
    h -= 1
  while w > width:
    w -= 1

  return (h, w)


def _sample_block_mask(
    rng: np.random.Generator,
    height: int,
    width: int,
    block_size: tuple[int, int],
    min_keep: int = 4,
    acceptable_regions: BoolArray['H W'] | None = None,
) -> tuple[BoolArray['H W'], BoolArray['H W']]:
  """Sample block mask."""
  h, w = block_size

  def constrain_mask(
      mask: BoolArray, acceptable_regions: BoolArray['H W'], tries: int = 0
  ) -> BoolArray['H W']:
    """Helper to restrict given mask to a set of acceptable regions."""
    num = max(
        int(len(acceptable_regions) - tries),
        0,
    )
    for k in range(num):
      mask &= acceptable_regions[k]

  # Loop to sample masks until we find a valid one.
  tries = 0
  timeout = og_timeout = 20
  valid_mask = False
  mask = None
  top, left = 0, 0
  while not valid_mask:
    # Sample block top-left corner.
    top = rng.integers(0, height - h + 1, size=())
    left = rng.integers(0, width - w + 1, size=())
    mask = np.zeros((height, width), dtype=bool)
    mask[top : top + h, left : left + w] = True
    # Constrain mask to a set of acceptable regions.
    if acceptable_regions is not None:
      constrain_mask(mask, acceptable_regions, tries)
    mask = mask.flatten()
    # If mask too small try again.
    valid_mask = len(mask) > min_keep
    if not valid_mask:
      timeout -= 1
      if timeout == 0:
        tries += 1
        timeout = og_timeout
  mask = mask.squeeze()  # pytype: disable=attribute-error
  mask_complement = np.ones((height, width), dtype=bool)
  mask_complement[top : top + h, left : left + w] = False
  return mask, mask_complement


@dataclasses.dataclass(frozen=True)
class JEPAMask(grain.RandomMapTransform):
  """Create masks for JEPA."""

  patch_size: int
  enc_mask_scale: tuple[float, float] = (0.85, 1.0)
  pred_mask_scale: tuple[float, float] = (0.15, 0.2)
  aspect_ratio: tuple[float, float] = (0.75, 1.5)
  num_enc_masks: int = 1
  num_pred_masks: int = 4
  min_keep: int = 10
  allow_overlap: bool = False

  def generate_masks(
      self,
      rng: np.random.Generator,
      p_size: tuple[int, int],
      e_size: tuple[int, int],
      nh: int,
      nw: int,
  ) -> Mapping[str, np.ndarray]:
    """Generate masks for one image."""

    outputs = collections.defaultdict(list)
    for _ in range(self.num_pred_masks):
      mask, mask_c = _sample_block_mask(
          rng, nh, nw, p_size, min_keep=self.min_keep
      )
      outputs['patch_masks_predictor'].append(mask[..., None])
      outputs['patch_masks_complement'].append(mask_c)

    acceptable_regions = outputs['patch_masks_complement']
    if self.allow_overlap:
      acceptable_regions = None

    for _ in range(self.num_enc_masks):
      mask, _ = _sample_block_mask(
          rng, nh, nw, e_size, acceptable_regions=acceptable_regions
      )
      outputs['patch_masks_encoder'].append(mask[..., None])

    return {k: np.stack(v, axis=0) for k, v in outputs.items()}

  def random_map(
      self, data: types.DataDict, rng: np.random.Generator
  ) -> types.DataDict:
    for scene in ['map', 'query']:
      if scene not in data:
        continue
      scene_data = data[scene]
      images = scene_data['views']['image_pixels']
      n, h, w = images.shape[:-1]
      nh, nw = h // self.patch_size, w // self.patch_size
      # Sample block sizes and masks. Equal for all images in a view collection.
      p_size = _sample_block_size(
          rng, nh, nw, self.pred_mask_scale, self.aspect_ratio
      )
      e_size = _sample_block_size(
          rng, nh, nw, self.enc_mask_scale, self.aspect_ratio
      )
      all_masks = collections.defaultdict(list)
      for _ in range(n):
        masks_for_image = self.generate_masks(rng, p_size, e_size, nh, nw)
        for k, v in masks_for_image.items():
          all_masks[k].append(v)

      for k, v in all_masks.items():
        data[scene]['views'][k] = np.stack(v, axis=0)
    return data


latlng_transformer_ecef: Final[pyproj.Transformer] = (
    pyproj.Transformer.from_proj(
        pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84'),
        pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84'),
    )
)


def latlng_from_ecef(ecef_positions: FloatArray['N 3']) -> FloatArray['N 2']:
  """Converts ECEF coordinates to latlng."""
  long, lat, _ = latlng_transformer_ecef.transform(
      ecef_positions[:, 0],
      ecef_positions[:, 1],
      ecef_positions[:, 2],
      radians=False,
  )
  return np.stack([lat, long], 1)


def get_bilinear_interpolation_weights(
    latlng: FloatArray['2'],
    s2cell_id_to_idx: Mapping[int, int],
    target_level: int,
) -> tuple[IntArray['4'], FloatArray['4']]:
  """Returns the interpolation weights for the given latlng."""
  neighbor_cells, weights = geo.bilinear_interpolate_s2cells(
      latlng[0].item(), latlng[1].item(), target_level
  )
  neighbor_cell_idx = np.array(
      [s2cell_id_to_idx.get(cell.id, -1) for cell in neighbor_cells],
      dtype=np.int64,
  )
  if len(neighbor_cells) == 3:
    neighbor_cell_idx = np.concatenate([neighbor_cell_idx, [-1]])
    weights = np.concatenate([weights, [0.0]])
  # Handle invalid cells by masking.
  valid = neighbor_cell_idx > -1
  weights = (weights * valid) / weights[valid].sum().clip(min=1.0e-6)
  return neighbor_cell_idx, weights


@dataclasses.dataclass(frozen=False)
class ConvertS2CellIds(grain.MapTransform):
  """Convert s2cell ids to indices."""

  target_level: int
  s2cell_id_to_idx: Mapping[int, int]
  frustum_depth: float = 50.0
  frustum_hfov_deg: float = 90.0
  max_frustum_cells: int = 16
  cam_p_frustum: FloatArray['3 3'] | None = None  # Auto-computed in post_init.

  def __post_init__(self):
    self.cam_p_frustum = geo.camera_frustum_vertices(
        self.frustum_depth, self.frustum_hfov_deg
    )

  def latlng_to_s2cell(self, latlng: FloatArray['2']) -> s2.S2CellId:
    return s2.S2CellId.from_lat_lng(
        s2.S2LatLng.from_degrees(*latlng)
    ).parent_at_level(self.target_level)

  def latlng_to_idx(
      self, latlng: FloatArray['2'], allow_missing: bool = True
  ) -> int:
    cell_id = self.latlng_to_s2cell(latlng).id
    if allow_missing:
      return self.s2cell_id_to_idx.get(cell_id, -1)
    else:
      return self.s2cell_id_to_idx[cell_id]

  def map(self, data: types.DataDict) -> types.DataDict:
    s2cell = s2.S2CellId.from_token(data['s2cell']).parent_at_level(
        self.target_level
    )

    # Compute latlng and ecef_t_cam for the query views.
    ecef_t_cam = data['ecef_t_w'][None] @ np.linalg.inv(
        data['query']['views']['cam_t_w']
    )

    data['query']['views']['ecef_t_cam'] = ecef_t_cam
    data['query']['views']['latlng'] = latlng_from_ecef(ecef_t_cam[:, :3, -1])

    # Update the cell index from the actual latlng positions.
    # Necessary if we go down to a finer s2cell level.
    center_latlng = data['query']['views']['latlng'].mean(0)
    new_cell = self.latlng_to_s2cell(center_latlng)
    new_cell_idx = self.s2cell_id_to_idx.get(new_cell.id, -1)
    if new_cell_idx == -1:
      data['cell_idx'] = self.s2cell_id_to_idx[s2cell.id]
      logging.error(
          'Failed to find cell %d for latlng (%f, %f) at level %d, original'
          ' cell id: %d (%s).',
          new_cell.id,
          center_latlng[0].item(),
          center_latlng[1].item(),
          self.target_level,
          s2cell.id,
          data['s2cell'].decode('utf-8'),
      )
    else:
      data['cell_idx'] = new_cell_idx

    # Add neighbor cell indices.
    neighbor_cells = new_cell.all_neighbors(level=self.target_level)
    neighbor_cell_idx = [
        self.s2cell_id_to_idx.get(cell.id, -1) for cell in neighbor_cells
    ]
    data['neighbor_cell_idx'] = np.array(
        [data['cell_idx']] + neighbor_cell_idx, dtype=np.int32
    )

    adjacent_idx, adjacent_weights = get_bilinear_interpolation_weights(
        center_latlng, self.s2cell_id_to_idx, self.target_level
    )
    data['adjacent_cell_idx'] = adjacent_idx
    data['adjacent_weights'] = adjacent_weights

    # Compute latlng coordinates of the camera frustums.
    if 'frustum_latlng' not in data['query']['views']:
      frustums_ll = np.stack(
          [
              latlng_from_ecef(
                  geo.transform_points(w_t_cam, self.cam_p_frustum)
              )
              for w_t_cam in ecef_t_cam
          ],
          0,
      )
      data['query']['views']['frustum_latlng'] = frustums_ll
    else:
      frustums_ll = data['query']['views']['frustum_latlng']

    # Compute cells that overlap with the frustum.
    def cells_covering_frustum(frustum_latlng):
      """Returns cells that overlap with the frustum."""
      frustum_cells_, overlap_frustum_, overlap_cell_ = (
          geo.enclosing_s2cell_ids(
              frustum_latlng,
              min_level=self.target_level,
              max_level=self.target_level,
              compute_overlap=True,
          )
      )  # Already sorted by overlap in frustum. Biggest overlap first.
      num_cells = len(frustum_cells_)
      frustum_cell_idx = -np.ones(self.max_frustum_cells, dtype=np.int32)
      overlap_frustum = np.zeros(self.max_frustum_cells, dtype=np.float32)
      overlap_cell = np.zeros(self.max_frustum_cells, dtype=np.float32)
      frustum_cell_idx[:num_cells] = [
          self.s2cell_id_to_idx.get(cell.id, -1) for cell in frustum_cells_
      ]
      overlap_frustum[:num_cells] = overlap_frustum_
      overlap_cell[:num_cells] = overlap_cell_
      overlap_frustum = np.where(frustum_cell_idx > -1, overlap_frustum, 0.0)
      overlap_cell = np.where(frustum_cell_idx > -1, overlap_cell, 0.0)
      return frustum_cell_idx, overlap_frustum, overlap_cell

    frustums_cover = [cells_covering_frustum(fr_ll) for fr_ll in frustums_ll]

    # Compute cell index of the camera frustum vertices.
    data['query']['views']['frustum_cell_idx'] = np.stack(
        [x[0] for x in frustums_cover], axis=0
    )
    data['query']['views']['frustum_overlap'] = np.stack(
        [x[1] for x in frustums_cover], axis=0
    )
    data['query']['views']['frustum_overlap_cell'] = np.stack(
        [x[2] for x in frustums_cover], axis=0
    )

    if 'aerial' in data:
      aerial_data = data['aerial']
      s2cell = s2.S2CellId.from_token(aerial_data['s2cell']).parent_at_level(
          self.target_level
      )
      center_latlng = aerial_data['center_ll']
      new_cell = self.latlng_to_s2cell(center_latlng)
      aerial_data['cell_idx'] = self.s2cell_id_to_idx.get(
          new_cell.id, self.s2cell_id_to_idx.get(s2cell.id, -1)
      )
      adjacent_idx, adjacent_weights = get_bilinear_interpolation_weights(
          center_latlng, self.s2cell_id_to_idx, self.target_level
      )
      aerial_data['adjacent_cell_idx'] = adjacent_idx
      aerial_data['adjacent_weights'] = adjacent_weights
    return data


@dataclasses.dataclass(frozen=False)
class InterpolateS2CellTree(grain.MapTransform):
  """Interpolate s2cell tree."""

  s2cell_ids_tree: Sequence[IntArray['Ni']]
  cell_levels: Sequence[int]
  s2cell_ids_tree_to_idx: Sequence[Mapping[int, int]] = dataclasses.field(
      default_factory=lambda: [{}]
  )

  def __post_init__(self):
    self.s2cell_ids_tree_to_idx = [
        {k.item(): i for i, k in enumerate(v)} for v in self.s2cell_ids_tree
    ]

  def get_tree_interpolation_weights(
      self,
      latlng: FloatArray['2'],
  ) -> tuple[IntArray['H 4'], FloatArray['H 4']]:
    """Returns the interpolation weights for the given latlng."""
    adjacent_idxs, weights = list(
        zip(*[
            get_bilinear_interpolation_weights(
                latlng, s2cell_id_to_idx, cell_level
            )
            for cell_level, s2cell_id_to_idx in zip(
                self.cell_levels, self.s2cell_ids_tree_to_idx
            )
        ])
    )
    return np.stack(adjacent_idxs, 0), np.stack(weights, 0)

  def map(self, data: types.DataDict) -> types.DataDict:
    latlng = data['query']['views']['latlng'].mean(0)
    adjacent_cell_idx_tree, adjacent_weights_tree = (
        self.get_tree_interpolation_weights(latlng)
    )
    data['adjacent_cell_idx_tree'] = adjacent_cell_idx_tree
    data['adjacent_weights_tree'] = adjacent_weights_tree
    if 'aerial' in data:
      aerial_data = data['aerial']
      adjacent_cell_idx_tree, adjacent_weights_tree = (
          self.get_tree_interpolation_weights(aerial_data['center_ll'])
      )
      aerial_data['adjacent_cell_idx_tree'] = adjacent_cell_idx_tree
      aerial_data['adjacent_weights_tree'] = adjacent_weights_tree
    return data


@dataclasses.dataclass(frozen=True)
class FilterS2Cells(grain.FilterTransform):
  """Filter s2cells by count."""

  valid_s2cell_ids: set[int]
  target_level: int

  def filter(self, data: types.DataDict) -> bool:
    return (
        s2.S2CellId.from_token(data['s2cell'])
        .parent_at_level(self.target_level)
        .id
        in self.valid_s2cell_ids
    )


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


@dataclasses.dataclass(frozen=True)
class AddAerialData(grain.RandomMapTransform):
  """Add aerial data to the data dict."""

  data_dir: str
  data_source: sv_aerial_data_source.S2CellDataSource

  def random_map(
      self, data: types.DataDict, rng: np.random.Generator
  ) -> types.DataDict:
    s2cell_token = data['s2cell']
    if self.data_source.exists(s2cell_token):
      aerial_data = self.data_source.random_sample_in_s2cell(s2cell_token, rng)
      aerial_data = {
          k: np.stack(v, 0) if isinstance(v, list) else v
          for k, v in aerial_data.items()
      }
      aerial_data['rgb'] = misc.byte_to_float(aerial_data['rgb'])
      data['aerial'] = aerial_data
    return data


class FilterMissingAerialData(grain.FilterTransform):
  """Remove examples that are missing aerial data."""

  def filter(self, data: types.DataDict) -> bool:
    return data.get('aerial', None) is not None


def _get_tokenize_area_summary_pp_fn():
  return pp_builder.get_preprocess_fn(
      'lower(key="text")|tok(length=64, model="gemma", bos="no", eos="sticky",'
      ' key="text")'
  )


@dataclasses.dataclass(frozen=True)
class TokenizeAreaSummary(grain.RandomMapTransform):
  """Tokenize area summary."""

  _pp_fn_internal: Callable[[dict[str, Any]], dict[str, Any]] | None = (
      dataclasses.field(init=False, repr=False)
  )

  def __post_init__(self):
    nltk.data.path.append(
        os.path.join(
            resources.GetARootDirWithAllResources(),
            'google3/third_party/py/nltk/nltk_data',
        )
    )
    object.__setattr__(self, '_pp_fn_internal', None)

  def get_pp_fn(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
    if getattr(self, '_pp_fn_internal', None) is None:
      object.__setattr__(
          self, '_pp_fn_internal', _get_tokenize_area_summary_pp_fn()
      )
    return cast(
        Callable[[dict[str, Any]], dict[str, Any]], self._pp_fn_internal
    )

  def random_map(
      self, data: types.DataDict, rng: np.random.Generator
  ) -> types.DataDict:
    if 'area_summaries' not in data:
      return data
    text = data['area_summaries']
    sents = nltk.tokenize.sent_tokenize(text.decode('utf-8', errors='ignore'))
    if sents:
      start_index = rng.integers(len(sents))
      sentence = ' '.join(sents[start_index:])
    else:
      sentence = ''
    data['area_summaries'] = self.get_pp_fn()({'text': sentence})['text']
    return data

  def __getstate__(self):
    return {}

  def __setstate__(self, state: dict[str, Any]):
    self.__post_init__()


def read_s2cell_ids_and_counts(
    paths: Sequence[str],
    target_level: int,
    num_neighbors: int = 0,
    filename: str = 'views_per_s2cell_after_sampling.json',
    min_count: int = 1,
    divisible_by: int = 1,
) -> tuple[Sequence[s2.S2CellId], Sequence[int]]:
  """Read valid s2cell tokens with counts, and transform them to the target level."""
  s2cells_with_counts = {}
  for base_path in paths:
    path = os.path.join(base_path, filename)
    if 'json' in path:
      with gfile.Open(path, 'r') as f:
        s2cells_with_counts = s2cells_with_counts | json.load(f)
    elif 'csv' in path:
      index_df = lama_io.read_sharded_csv(
          path, literal_eval=False, max_workers=4
      )
      s2cells_i_to_idxs = index_df.set_index(index_df.columns[0]).to_dict()[
          'indices'
      ]
      s2cells_with_counts = s2cells_with_counts | {
          k: len(v) for k, v in s2cells_i_to_idxs.items()
      }
  logging.info('Read %d s2cells.', len(s2cells_with_counts))
  s2cells_at_target_level_with_counts = collections.defaultdict(list)
  for s2cell, count in s2cells_with_counts.items():
    s2cell_id = s2.S2CellId.from_token(s2cell.encode('utf-8'))
    s2cell_id_at_target_level = s2cell_id.parent_at_level(target_level)
    s2cells_at_target_level_with_counts[s2cell_id_at_target_level].append(count)
  s2cells_at_target_level_with_counts = {
      k: sum(v) for k, v in s2cells_at_target_level_with_counts.items()
  }
  s2cells_at_target_level_with_counts = {
      k: v
      for k, v in s2cells_at_target_level_with_counts.items()
      if v >= min_count
  }

  if num_neighbors > 0:
    for s2cell_id in s2cells_at_target_level_with_counts.keys():
      neighbors = get_s2cell_neighbors(s2cell_id, target_level, num_neighbors)
      for neighbor in neighbors:
        s2cells_at_target_level_with_counts[neighbor] += 0
  logging.info(
      'Found %d s2cells at level %d.',
      len(s2cells_at_target_level_with_counts),
      target_level,
  )
  s2cell_ids, counts = tuple(
      zip(
          *sorted(
              s2cells_at_target_level_with_counts.items(), key=lambda x: x[0]
          )
      )
  )
  s2cell_ids = s2cell_ids[: len(s2cell_ids) // divisible_by * divisible_by]
  counts = counts[: len(counts) // divisible_by * divisible_by]
  return s2cell_ids, counts


def children_at_level(
    s2cell_id: s2.S2CellId,
    level: int,
) -> list[s2.S2CellId]:
  """Returns the children of the s2cell_id at a specific level."""
  cell_level = s2cell_id.level
  delta = level - cell_level
  cell_ids = [s2cell_id]
  for i in range(delta):
    cell_ids = [c.vertex_neighbors(cell_level + 1 + i) for c in cell_ids]
    cell_ids = sum(cell_ids, start=[])
  return cell_ids


def get_s2cell_neighbors(
    s2cell_id: s2.S2CellId,
    level: int,
    num_neighbors: int,
) -> list[s2.S2CellId]:
  """Returns the n-layers of outer s2cell-neighbors at a specific level.

  Arguments:
    s2cell_id: The s2cell id to get the neighbors of.
    level: The level of the neighbors.
    num_neighbors: The number of layers of neighbors to get.

  Returns:
    A sorted list of s2cell ids of the neighbors.
  """
  if num_neighbors == 0:
    neighbors = []
  elif num_neighbors == 1:
    neighbors = s2cell_id.all_neighbors(level)
  else:
    neighbors = s2cell_id.all_neighbors(level)
    for _ in range(num_neighbors - 1):
      new_neighbors = sum([p.all_neighbors(level) for p in neighbors], start=[])
      neighbors = neighbors + list(
          set(new_neighbors)
          - set(neighbors)
          - set(children_at_level(s2cell_id, level))
      )
  return neighbors


def get_tree_levels(
    cell_level: int,
    max_tree_height: int | None = None,
    tree_step: int = 1,
) -> Sequence[int]:
  """Returns the levels of the tree."""
  if max_tree_height is None:
    max_tree_height = (cell_level + 1) // tree_step
  levels = [cell_level - i * tree_step for i in range(max_tree_height)]
  assert levels[-1] >= 0
  return levels[::-1]  # Make ascending.


def build_s2cell_tree(
    s2cell_ids: IntArray['N'],
    cell_level: int,
    max_tree_height: int | None = None,
    num_neighbors: int = 1,  # Currently supports 0, 1 and 2.
    tree_step: int = 1,
    divisible_by: int = 1,
) -> tuple[Sequence[int], Sequence[IntArray['Ni']], Sequence[IntArray['Ni K']]]:
  """Build a tree of s2cells."""
  # leves are in descending order.
  levels = get_tree_levels(cell_level, max_tree_height, tree_step)[::-1]
  num_levels = len(levels)
  s2cell_ids_tree = [s2cell_ids]  # each entry is a list
  leafs = [np.arange(s2cell_ids.shape[0])[:, None]]
  last_s2cells = [s2.S2CellId(x.item()) for x in s2cell_ids]

  idx_to_tree_idx = np.zeros((s2cell_ids.shape[0], num_levels), dtype=int)
  idx_to_tree_idx[:, -1] = np.arange(s2cell_ids.shape[0])

  def s2cell_id_to_array(s2cell_ids):
    return np.array([c.id for c in s2cell_ids], dtype=np.uint64)

  for i, level in enumerate(levels[1:]):
    logging.info(
        'Building cell tree at level %d from %d leafs.',
        level,
        len(last_s2cells),
    )
    last_s2cells = [x.parent_at_level(level) for x in last_s2cells]
    (parent_s2cells, parent_idx) = np.unique(last_s2cells, return_inverse=True)

    idx_to_tree_idx[:, -2 - i] = np.vectorize(lambda x: parent_idx[x])(  # pylint: disable=cell-var-from-loop
        idx_to_tree_idx[:, -1 - i]
    )

    if divisible_by > 1 and parent_s2cells.shape[0] % divisible_by != 0:
      # pad with random neighbor cells.
      target_size = (
          (parent_s2cells.shape[0] // divisible_by) + 1
      ) * divisible_by
      logging.info(
          'Padding parent cells with %d neighbors.',
          target_size - parent_s2cells.shape[0],
      )
      parent_s2cell_list = parent_s2cells.tolist()
      for s2cell_id in parent_s2cells:
        neighbors = s2cell_id.all_neighbors(level)
        for neighbor in neighbors:
          if neighbor not in last_s2cells:
            parent_s2cell_list.append(neighbor)
        if len(parent_s2cell_list) >= target_size:
          parent_s2cell_list = parent_s2cell_list[:target_size]
          break
      parent_s2cells = np.array(parent_s2cell_list)

    parent_s2cell_ids = np.array(
        [x.id for x in parent_s2cells], dtype=np.uint64
    )
    s2cell_id_to_idx = {k: i for i, k in enumerate(s2cell_ids_tree[-1])}

    leaf_s2cell_ids = np.array([
        s2cell_id_to_array(
            children_at_level(p, level + tree_step)
            + get_s2cell_neighbors(
                p,
                level + tree_step,
                num_neighbors * tree_step,
            )
        )  # The first 4 indices are the vertices of the child cells.
        for p in parent_s2cells
    ])
    leaf_s2cell_idx = np.vectorize(lambda x: s2cell_id_to_idx.get(x, -1))(  # pylint: disable=cell-var-from-loop
        leaf_s2cell_ids
    )
    num_children = ((2 ** (tree_step - 1) + tree_step * num_neighbors) * 2) ** 2
    if leaf_s2cell_idx.shape[1] < num_children:
      leaf_s2cell_idx = np.concatenate(
          [
              leaf_s2cell_idx,
              -np.ones(
                  (
                      leaf_s2cell_idx.shape[0],
                      num_children - leaf_s2cell_idx.shape[1],
                  ),
                  dtype=np.int32,
              ),
          ],
          axis=-1,
      )
    last_s2cells = parent_s2cells
    leafs.append(leaf_s2cell_idx)
    s2cell_ids_tree.append(parent_s2cell_ids)
  leafs[0] = idx_to_tree_idx
  s2cell_ids_tree.reverse()
  leafs.reverse()
  return levels[::-1], s2cell_ids_tree, leafs


def transformations_from_config(
    data_config: ml_collections.ConfigDict,
    meta_data: types.DataDict | None = None,
) -> tuple[grain.Transformations, types.DataDict]:
  """Builds the transformations from the dataset config."""
  config = data_config.preprocess
  meta_data = {} if meta_data is None else meta_data
  transformations = [
      MaybeCollate(),
      NormalizePixels(),
  ]

  if config.get('tokenize_area_summary', False):
    transformations.append(TokenizeAreaSummary())

  if config.get('from_aerial'):
    transformations = [AerialToScene()] + transformations

  if config.get('from_arcollect'):
    transformations = [ArCollectToScene()] + transformations

  if config.get('from_osv5m'):
    transformations = [Osv5mToScene()] + transformations

  if config.get('map_to_map'):
    transformations += [
        MapToMap(num_views_per_collection=config.sample_num_views.to_dict())
    ]
  else:
    transformations += [
        SampleViews(num_views_per_collection=config.sample_num_views.to_dict())
    ]

  if config.filter_pixels_with_semantics:
    transformations.append(
        FilterPixelsWithSemantics(
            config.ignore_classes,
            config.mask_classes,
            config.enforce_2d3d_semantic_consistency,
        )
    )

  # Pointcloud processing
  if config.num_lidar_points > 0:
    transformations.append(FilterLiDAR(min_num_points=config.num_lidar_points))
    transformations.append(ProcessLiDAR(num_points=config.num_lidar_points))
  else:
    transformations.append(RemovePointcloud())

  if config.add_grid_3d:
    meta_data['grid_3d'] = grids.Grid3D.from_extent_meters(
        config.grid_3d.extent_meters, config.grid_3d.cell_size
    )
    transformations.append(
        AddGrid3DTransform(
            grid_3d=meta_data['grid_3d'],
            z_offset=config.grid_3d.z_offset,
            center_around_reference=config.grid_3d.center_around_reference,
        )
    )
  if config.voxelize_lidar_projections:
    transformations.append(VoxelizeLidarProjections(meta_data['grid_3d']))

  if config.num_dominant_voxels_per_patch > 0:
    transformations.append(
        DominantVoxelsPerPatch(
            patch_size=config.patch_size,
            num_voxels_per_patch=config.num_dominant_voxels_per_patch,
        )
    )

  if config.get('dino_crop'):
    transformations.append(DINOCrop(**config.dino_crop.to_dict()))

  if config.get('filter_keys'):
    transformations.append(FilterKeys(**config.filter_keys.to_dict()))

  if config.get('center_crop') is not None:
    transformations.append(CenterCropImageLike(config.center_crop))

  if config.get('resize_pad') is not None:
    transformations.append(ResizeAndPadImageLike(config.resize_pad))

  if config.get('stretch_size') is not None:
    transformations.append(ResizeImageLike(config.stretch_size))

  if config.get('jepa_mask'):
    transformations.append(JEPAMask(**config.jepa_mask.to_dict()))

  if config.get('aerial', None):
    unique_locations = list(
        set(data_config.locations.train + data_config.locations.valid)
    )
    aerial_config = sv_aerial_data_source.S2CellDataSource.Config(
        locations=unique_locations, **config.aerial.to_dict()
    )
    transformations.append(
        AddAerialData(
            data_config.data_dir,
            sv_aerial_data_source.S2CellDataSource(
                data_config.data_dir, aerial_config
            ),
        )
    )

  if config.get('aerial') or data_config.get('aerial'):
    transformations.append(FilterMissingAerialData())

  if config.get('build_s2cell_index'):
    if 's2cell_tokens' not in meta_data:
      base_path = os.path.join(
          data_config.data_dir,
          data_config.dirname,
      )
      all_paths = [
          os.path.join(base_path, location)
          for location in set(
              data_config.locations.train + data_config.locations.valid
          )
      ]
      s2cell_ids, _ = read_s2cell_ids_and_counts(
          all_paths,
          14,
          filename=config.get(
              'index_file_name', 'views_per_s2cell_after_sampling.json'
          ),
          num_neighbors=config.get('num_neighbors', 0),
          min_count=1,
          divisible_by=1,
      )
      s2cell_id_to_idx_14 = {
          s2cell_id.id: idx for idx, s2cell_id in enumerate(s2cell_ids)
      }
      meta_data['s2cell_id_to_idx_14'] = s2cell_id_to_idx_14
      target_level = config.get('s2cell_level', 17)
      s2cell_ids, s2cell_counts = read_s2cell_ids_and_counts(
          all_paths,
          target_level,
          filename=config.get(
              'index_file_name', 'views_per_s2cell_after_sampling.json'
          ),
          num_neighbors=config.get('num_neighbors', 0),
          min_count=1,
          divisible_by=1,
      )
      s2cell_tokens = [
          s2cell_id.token().decode('utf-8') for s2cell_id in s2cell_ids
      ]
      s2cell_token_to_idx = {
          s2cell_token: idx for idx, s2cell_token in enumerate(s2cell_tokens)
      }
      s2cell_tokens = np.array(s2cell_tokens)
      s2cell_id_to_idx = {
          s2cell_id.id: idx for idx, s2cell_id in enumerate(s2cell_ids)
      }
      meta_data['s2cell_tokens'] = s2cell_tokens
      meta_data['s2cell_ids'] = np.array(
          [x.id for x in s2cell_ids], dtype=np.uint64
      )
      meta_data['s2cell_level'] = target_level
      meta_data['s2cell_id_to_idx'] = s2cell_id_to_idx
      meta_data['s2cell_token_to_idx'] = s2cell_token_to_idx
      meta_data['s2cell_counts'] = s2cell_counts

      def ll_center(s2cell_id: s2.S2CellId) -> FloatArray['2']:
        s2_ll = s2cell_id.to_lat_lng()
        return np.array([s2_ll.lat.degrees, s2_ll.lng.degrees])

      s2_ll = np.stack(np.vectorize(ll_center, otypes=[np.ndarray])(s2cell_ids))
      s2_ecef = geo.ecef_from_latlng(s2_ll)
      meta_data['s2cell_centers_ecef'] = s2_ecef
      meta_data['s2cell_centers_ll'] = s2_ll

      # Build a hierarchical tree of s2cells.
      if config.get('build_s2cell_tree', True):
        (
            meta_data['tree_levels'],
            meta_data['s2cell_ids_tree'],
            meta_data['cell_tree'],
        ) = build_s2cell_tree(
            meta_data['s2cell_ids'],
            target_level,
            max_tree_height=config.get('cell_tree_max_height', None),
            num_neighbors=config.get('cell_tree_num_neighbors', 1),
            tree_step=config.get('cell_tree_step', 1),
            divisible_by=config.get('divisible_by', jax.device_count()),
        )
    transformations.append(
        FilterS2Cells(
            set(meta_data['s2cell_id_to_idx']), meta_data['s2cell_level']
        )
    )
    transformations.append(
        ConvertS2CellIds(
            meta_data['s2cell_level'], meta_data['s2cell_id_to_idx']
        )
    )

    if 'cell_tree' in meta_data and config.get('interpolate_cell_tree', True):
      transformations.append(
          InterpolateS2CellTree(
              meta_data['s2cell_ids_tree'], meta_data['tree_levels']
          )
      )

  if 's2cell_id_to_idx_14' in meta_data:
    transformations.append(
        S2CellTokenToId(
            s2cell_id_to_idx=meta_data['s2cell_id_to_idx_14'],
        )
    )
  return transformations, meta_data


class ImageNetToScene(grain.MapTransform):

  def map(self, data: types.DataDict) -> types.DataDict:
    rename = {
        'image': 'image_pixels',
    }
    data = {(rename[k] if k in rename else k): [v] for k, v in data.items()}
    return {'query': {'views': data}}


class AerialToScene(grain.MapTransform):
  """Convert aerial data to scene format."""

  def map(self, data: types.DataDict) -> types.DataDict:
    keep_and_rename = {
        'rgb': 'image_pixels',
        'extent': 'extent',
        'resolution': 'resolution',
        'ecef_t_grid': 'ecef_t_grid',
    }
    view_data = {
        keep_and_rename[k]: v for k, v in data.items() if k in keep_and_rename
    }
    view_data['cam_t_w'] = [
        np.eye(4, dtype=np.float32) for _ in view_data['extent']
    ]
    view_data['image_name'] = [
        f'{data["s2cell"]}_{i}' for i in view_data['extent']
    ]
    return {
        'query': {'views': view_data},
        's2cell': data['s2cell'],
        'center_ll': data['center_ll'],
        'dataset_index': data['dataset_index'],
        'ecef_t_w': geo.ecef_t_s2cell(
            data['s2cell'].decode('utf-8'), altitude=0.0
        ),
    }


@dataclasses.dataclass(frozen=True)
class ArCollectToScene(grain.MapTransform):

  def map(self, data):
    ecef_t_w = geo.ecef_t_s2cell(data['s2_cell_token'].decode('utf-8'), 0.0)
    ecef_t_cam = np.eye(4)
    ecef_t_cam[:3, :3] = scipy_transform.Rotation.from_quat(
        data['ecef_q_camera']
    ).as_matrix()
    ecef_t_cam[:3, 3] = data['ecef_p_camera']
    cam_t_w = np.linalg.inv(ecef_t_cam) @ ecef_t_w

    return {
        's2cell': data['s2_cell_token'],
        'ecef_t_w': ecef_t_w,
        'query': {
            'views': {
                'cam_t_w': [cam_t_w],
                'image_pixels': [data['image']],
                'image_name': [data['uuid']],
                'image_index': [data['image_index']],
            }
        },
        'center_ll': data['latlng'],
        'dataset_index': data['dataset_index'],
    }


@dataclasses.dataclass(frozen=True)
class Osv5mToScene(grain.MapTransform):

  def map(self, data: types.DataDict) -> types.DataDict:
    return {
        's2cell': data['s2_cell_token'],
        'latlng': data['latlng'],
        'query': {
            'views': {
                'image_pixels': [data['image']],
                'image_name': [str(data['id'])],
                'image_index': [data['id']],
            }
        },
        'center_ll': data['latlng'],
        'dataset_index': data['dataset_index'],
    }


def get_dummy_batch(
    data_source: grain.RandomAccessDataSource,
    transformations: (
        grain.Transformations | Mapping[str, grain.Transformations]
    ),
    batch_size: int,
    seed: int,
) -> types.Batch:
  """Returns a dummy batch from the data source."""
  dummy_loader = grain.load(
      source=data_source,
      shuffle=True,
      drop_remainder=True,
      seed=seed,
      transformations=transformations,
      batch_size=get_process_batch_size(batch_size),
      worker_count=0,
      read_options=grain.ReadOptions(
          num_threads=1,
          prefetch_buffer_size=1,
      ),
  )
  return next(dummy_loader.__iter__())


class TrueSequentialSampler:
  """Basic sampler implementation that provides records in order."""

  def __init__(
      self,
      batch_size: int,
      num_workers: int,
      num_records: int,
      shard_options: grain.ShardOptions,
      seed: int | None = None,
  ):
    self._batch_size = batch_size
    self._num_workers = max(num_workers, 1)
    self._step_size = self._batch_size * self._num_workers
    if num_records <= 0:
      raise ValueError(
          'Invalid number of records in Sampler. '
          f'Got {num_records} records, but number of records '
          'must be greater than 0.'
      )
    self._num_records = num_records
    self._shard_options = shard_options
    if shard_options.drop_remainder:
      num_records_per_shard = self._num_records // shard_options.shard_count
      self._max_index = num_records_per_shard * shard_options.shard_count
    else:
      self._max_index = self._num_records
    self._seed = seed

  def __repr__(self) -> str:
    return (
        f'TrueSequentialSampler(num_records={self._num_records}, '
        f'shard_options={self._shard_options!r})'
    )

  def __getitem__(self, index: int) -> grain.RecordMetadata:
    if index < 0 or index >= self._max_index:
      raise IndexError(
          f'RecordMetadata object index is out of bounds; Got index {index},'
          f' allowed indices should be in [0, {self._max_index}]'
      )
    rng = None
    if self._seed is not None:
      rng = np.random.Generator(np.random.Philox(key=self._seed + index))

    batch_index, item_index = misc.unravel_global_dataset_index(
        index,
        self._num_workers,
        self._batch_size,
        self._shard_options.shard_count,
    )
    sorted_index = batch_index * self._batch_size + item_index
    return grain.RecordMetadata(index=index, record_key=sorted_index, rng=rng)


def shard_even_split(
    num_examples: int, options: grain.ShardOptions
) -> tuple[int, int]:
  """Returns the interval for the shard when sharding `num_examples` evenly.

  This splits the interval [0, num_examples - 1] into `shard_count` intervals
  and returns the `shard_index`'s interval. If `drop_remainder` is True all
  intervals will have the same size.

  Args:
    num_examples: Number of examples to shard.
    options: Options for sharding the data in this process.

  Returns:
    Tuple with the start and end of the interval. The start is the first
    example that should be included in this interval and end - 1 is the last
    example to be include in the shard.
  """
  examples_per_shard = num_examples // options.shard_count
  shard_start = examples_per_shard * options.shard_index
  shard_end = examples_per_shard * (options.shard_index + 1)

  # Handle remaining examples.
  num_unused_examples = num_examples % options.shard_count

  if num_unused_examples > 0:
    if options.drop_remainder:
      logging.warning(
          'Dropping %d examples of %d examples (shard %d).',
          num_unused_examples,
          num_examples,
          options.shard_count,
      )
    else:
      shard_start += min(options.shard_index, num_unused_examples)
      shard_end += min(options.shard_index + 1, num_unused_examples)
  return shard_start, shard_end


class BatchedIndexSampler(grain.Sampler):
  """Sampler that returns records from a batched index."""

  def __init__(
      self,
      num_workers: int,
      batched_indices: IntArray['N B'],
      shard_options: grain.ShardOptions,
      seed: int | None = None,
      batch_size: int | None = None,  # Set this < B to sample multiple batches.
      shuffle: bool = False,
      num_epochs: int | None = None,
      deterministic_subsampling: bool = False,
      debug: bool = False,
  ):
    self._num_workers = max(num_workers, 1)
    self._shard_options = shard_options
    self._indices = np.asarray(batched_indices)
    self._batch_size = batched_indices.shape[1]
    if batch_size is not None:
      self._batch_size = min(self._batch_size, batch_size)
    self._seed = seed
    self._num_batches_in_shard = self._indices.shape[0]
    self._num_batches = self._num_batches_in_shard * jax.process_count()
    self._num_records = self._num_batches * self._batch_size
    assert self._seed is not None

    self._max_index = (
        None
        if num_epochs is None
        else num_epochs * self._num_records * self._batch_size
    )

    # The batch indices in this shard.
    self._batch_keys = grain.MapDataset.range(self._num_batches_in_shard)
    if shuffle:
      # This enables reshuffling of the batches within the shard.
      self._batch_keys = self._batch_keys.shuffle(seed=seed)

    self._deterministic_subsampling = deterministic_subsampling
    self._debug = debug

  def __repr__(self) -> str:
    return (
        f'BatchedIndexSampler(batched_indices={self._indices.shape},'
        f' shard_options={self._shard_options!r}),'
        f' batch_size={self._batch_size},'
        f' deterministic_subsampling={self._deterministic_subsampling},'
        f' seed={self._seed})'
    )

  def __getitem__(self, index: int) -> grain.RecordMetadata:
    if index < 0 or (self._max_index is not None and index >= self._max_index):
      raise IndexError(
          f'RecordMetadata object index is out of bounds; Got index {index},'
          f' allowed indices should be in [0, {self._max_index}]'
      )
    epoch, index_in_epoch = divmod(index, self._num_records)
    batch_index, item_index = misc.unravel_global_dataset_index(
        index_in_epoch,
        self._num_workers,
        self._batch_size,
        self._shard_options.shard_count,
    )
    # Update with epoch --> This reshuffles the batches if epoch is > 1.
    batch_index = epoch * self._num_batches_in_shard + batch_index
    record_key = self._indices[self._batch_keys[batch_index], item_index]
    # Deterministic sampling in underlying data source (not always supported).
    if self._deterministic_subsampling:
      record_key = record_key + epoch * self._num_batches
    if self._debug:
      logging.info(
          'BATCH SAMPLER: index: %d, usindex: %d, batch_index: %d, item_index:'
          ' %d, process_index: %d, record_key: %d',
          index,
          index_in_epoch % self._shard_options.shard_count,
          batch_index,
          item_index,
          self._shard_options.shard_index,
          record_key,
      )
    rng = None
    if self._seed is not None:
      rng = np.random.Generator(np.random.Philox(key=self._seed + index))
    next_record = grain.RecordMetadata(
        index=index, record_key=record_key, rng=rng
    )
    return next_record


def build_data_loader(
    source: grain.RandomAccessDataSource,
    *,
    num_epochs: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    shard_options: grain.ShardOptions = grain.NoSharding(),
    transformations: grain.Transformations = (),
    batch_size: int | None = None,
    drop_remainder: bool = False,
    worker_count: int | None = 0,
    read_options: grain.ReadOptions | None = None,
    worker_buffer_size: int = 1,
    mining_indices: IntArray['N B'] | None = None,
    mining_similarity: FloatArray['N B'] | None = None,
    mining_batch_size: int | None = None,
) -> grain.DataLoader:
  """Convenient method for simple pipelines on top of a data source.

  Args:
    source: Data source to load from. This can be one of the file data sources
      provided by Grain, a TFDS data source (`tfds.data_source(...)`) or your
      custom data source.
    num_epochs: See IndexSampler.
    shuffle: See IndexSampler.
    seed: See IndexSampler.
    shard_options: See IndexSampler.
    transformations: List of local (stateless) transformations:
    batch_size: Optional batch size. If provided will apply BatchOperation().
    drop_remainder: Whether to drop partial batches.
    worker_count: Number of child processes launched to parallelize the
      transformations among. Zero means processing runs in the same process.
    read_options: Read options for the data loader. See ReadOptions.
    worker_buffer_size: Number of records to buffer in each worker.
    mining_indices: Optional batched indices to sample from.
    mining_similarity: Optional batched similarity scores to sample from.
    mining_batch_size: Optional mini-batch size for the mining indices.

  Returns:
    DataLoader for this dataset.
  """
  if mining_indices is not None:
    del mining_similarity  # Unused.
    mining_batch_size = (
        mining_batch_size
        if mining_batch_size is not None
        else batch_size // jax.local_device_count()
    )
    sampler = BatchedIndexSampler(
        num_workers=worker_count,
        batched_indices=mining_indices,
        shard_options=shard_options,
        seed=seed,
        batch_size=mining_batch_size,
        shuffle=shuffle,
    )
  else:
    sampler = grain.IndexSampler(
        num_records=len(source),
        shuffle=shuffle,
        seed=seed,
        num_epochs=num_epochs,
        shard_options=shard_options,
    )
  if batch_size is not None:
    transformations = list(transformations)
    transformations.append(
        grain.Batch(batch_size, drop_remainder=drop_remainder)
    )
  return grain.DataLoader(
      data_source=source,
      sampler=sampler,
      operations=transformations,
      worker_count=worker_count,
      read_options=read_options,
      worker_buffer_size=worker_buffer_size,
  )


def dataloader_provider_fn(
    *args,
    **kwargs,
) -> Callable[[], grain.DatasetIterator]:
  """Returns a function that provides a DataLoader iterator."""
  return lambda **md: build_data_loader(*args, **{**kwargs, **md}).__iter__()


def create_dataset(
    dataset_config: ml_collections.ConfigDict,
    batch_size: int,
    batch_size_eval: int,
    seed: int,
    transformations: grain.Transformations = (),
    meta_data: types.DataDict | None = None,
) -> types.Dataset:
  """Create a dataset for training."""

  process_batch_size = get_process_batch_size(batch_size)
  process_batch_size_eval = get_process_batch_size(batch_size_eval)

  logging.info(  # pylint: disable=logging-fstring-interpolation
      f'create_dataset: {batch_size=}, {process_batch_size=}, '
      f'{batch_size_eval=}, {process_batch_size_eval=}, '
      f'{jax.local_device_count()=}, {jax.device_count()=}, '
      f'{jax.process_index()=}, {jax.process_count()=}'
  )

  public_dataset_transformations = {
      'imagenet2012': ImageNetToScene(),
  }

  train_transformations = copy.deepcopy(transformations)
  valid_transformations = copy.deepcopy(transformations)

  def build_data_source(
      locations: Sequence[str],
      transformations: grain.Transformations,
      split: str,
      add_index: bool = True,
      make_s2cell_data_source: bool = False,
  ) -> grain.RandomAccessDataSource:
    if locations[0] in public_dataset_transformations:
      # Interface to public datasets is via tfds.
      name = locations[0]
      transformations.insert(  # pytype: disable=attribute-error
          0, public_dataset_transformations[name]
      )
      builder = tfds.builder(name)
      data_source = builder.as_data_source(split)
    elif make_s2cell_data_source:
      s2cell_config = sv_aerial_data_source.S2CellDataSource.Config(
          dirname=dataset_config.dirname,
          split_name=split,
          locations=locations,
          target_level=dataset_config.preprocess.s2cell_level,
          seed=seed,
          deterministic=dataset_config.get('deterministic', False),
          index_file_name=dataset_config.get(
              's2cell_index_file_name', '{split}_grouped_s2cells.csv*'
          ),
      )
      data_source = sv_aerial_data_source.S2CellDataSource(
          dataset_config.data_dir,
          s2cell_config,
      )
    else:
      # Internal datasets from multiple locations.
      base_path = os.path.join(
          dataset_config.data_dir,
          dataset_config.dirname,
      )
      builders = [
          tfds.builder_from_directory(os.path.join(base_path, location))
          for location in locations
      ]
      data_sources = [builder.as_data_source(split) for builder in builders]
      data_source = combined_data_source.CombinedDataSource(data_sources)

    if dataset_config.get('aerial', None):
      aerial_data_source = sv_aerial_data_source.S2CellDataSource(
          dataset_config.data_dir,
          sv_aerial_data_source.S2CellDataSource.Config(
              **{'locations': locations, **dataset_config.aerial.to_dict()}
          ),
      )
      data_source = sv_aerial_data_source.SVAerialDataSource(
          data_source,
          aerial_data_source,
      )

    if add_index and not make_s2cell_data_source:
      data_source = indexed_data_source.IndexedDataSource(data_source)
    return data_source

  train_data_source = build_data_source(
      dataset_config.locations.train,
      train_transformations,
      dataset_config.train_split,
      add_index=dataset_config.get('add_index', True),
      make_s2cell_data_source=dataset_config.get(
          'make_s2cell_data_source', False
      ),
  )
  valid_data_source = build_data_source(
      dataset_config.locations.valid,
      valid_transformations,
      dataset_config.valid_split,
      add_index=dataset_config.get('add_index', True),
      make_s2cell_data_source=dataset_config.get(
          'make_s2cell_data_source', False
      ),
  )

  dummy_batch_fn = functools.partial(
      get_dummy_batch,
      train_data_source,
      train_transformations,
      batch_size,
      seed=999,
  )

  meta_data = {
      'num_train_examples': len(train_data_source),
      'num_valid_examples': len(valid_data_source),
      'dummy_batch_fn': dummy_batch_fn,
      **(meta_data if meta_data is not None else {}),
  }

  train_loader = dataloader_provider_fn(
      source=train_data_source,
      shuffle=dataset_config.shuffle_train,
      seed=seed,
      shard_options=grain.ShardByJaxProcess(drop_remainder=True),
      drop_remainder=dataset_config.get('drop_remainder', True),
      transformations=train_transformations,
      batch_size=process_batch_size,
      worker_count=dataset_config.grain_num_workers,
      read_options=grain.ReadOptions(
          prefetch_buffer_size=dataset_config.grain_prefetch_buffer_size,
          num_threads=dataset_config.grain_num_threads,
      ),
      mining_indices=meta_data.get(
          f'{dataset_config.train_split}_mining_indices', None
      ),
  )

  valid_loader = dataloader_provider_fn(
      source=valid_data_source,
      # Not setting num_epochs=1 to avoid deadlocks or crashes until we fix
      # this properly.
      seed=seed,
      shard_options=grain.ShardByJaxProcess(drop_remainder=True),
      drop_remainder=dataset_config.get('drop_remainder', True),
      shuffle=dataset_config.shuffle_valid,
      transformations=valid_transformations,
      batch_size=process_batch_size_eval,
      worker_count=dataset_config.grain_num_workers_valid,
      read_options=grain.ReadOptions(
          prefetch_buffer_size=dataset_config.grain_prefetch_buffer_size,
          num_threads=dataset_config.grain_num_threads_valid,
      ),
      mining_indices=meta_data.get(
          f'{dataset_config.valid_split}_mining_indices', None
      ),
  )

  return types.Dataset(
      train_iter=train_loader,
      valid_iter=valid_loader,
      meta_data=meta_data,
      train_ds=train_data_source,
      valid_ds=valid_data_source,
  )


def create_dino_dataset(
    dataset_config: ml_collections.ConfigDict,
    batch_size: int,
    batch_size_eval: int,
    seed: int,
    **kwargs,  # pylint: disable=unused-argument
) -> types.Dataset:
  """Create a dataset for training."""
  device_count = jax.device_count()
  local_batch_size = batch_size // jax.process_count()
  eval_local_batch_size = batch_size_eval // jax.process_count()
  device_batch_size = batch_size // device_count
  logging.info('local_batch_size : %d', local_batch_size)
  logging.info('device_batch_size : %d', device_batch_size)
  data_rng = jax.random.PRNGKey(seed)
  scenic_dataset = dino_dataset.get_dataset(
      batch_size=local_batch_size,
      eval_batch_size=eval_local_batch_size,
      num_shards=jax.local_device_count(),
      rng=data_rng,
      dataset_configs=dataset_config,
      dataset_service_address=dataset_config.get('dataset_service_address'),
  )

  dataset = types.Dataset(
      train_iter=scenic_dataset.train_iter,
      meta_data=scenic_dataset.meta_data,
  )

  def get_dino_dummy_batch():
    return next(dataset.train_iter)

  dataset.meta_data['dummy_batch_fn'] = get_dino_dummy_batch
  dataset.meta_data['num_valid_examples'] = 0
  dataset.meta_data['num_local_views'] = dataset_config.num_local_views
  return dataset


def build_dataset(
    dataset_config: ml_collections.ConfigDict,
    batch_size: int,
    batch_size_eval: int,
    seed: int,
    meta_data: types.DataDict | None = None,
) -> types.Dataset:
  """Create a dataset from config."""
  dataset_name = dataset_config.get('dataset_name')
  if dataset_name == 'dino_dataset':
    return create_dino_dataset(
        dataset_config, batch_size, batch_size_eval, seed
    )
  else:
    transformations, meta_data = transformations_from_config(
        dataset_config, meta_data
    )
    return create_dataset(
        dataset_config,
        batch_size,
        batch_size_eval,
        seed,
        transformations=transformations,
        meta_data=meta_data,
    )


def create_overfit_dataset(
    dummy_batch_fn: Callable[[], types.Batch],
    num_train_examples: int,
    num_valid_examples: int,
    meta_data: types.DataDict | None = None,
) -> types.Dataset:
  """Create a dataset that reloads the same batch."""

  def train_iter():
    for _ in range(num_train_examples):
      yield dummy_batch_fn()

  def valid_iter():
    for _ in range(num_valid_examples):
      yield dummy_batch_fn()

  meta_data = {
      **(meta_data or {}),
      'num_train_examples': num_train_examples,
      'num_valid_examples': num_valid_examples,
      'dummy_batch_fn': dummy_batch_fn,
  }
  return types.Dataset(
      train_iter=train_iter,
      valid_iter=valid_iter,
      meta_data=meta_data,
  )
