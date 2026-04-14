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

"""Functions to process training data examples."""

from collections.abc import Sequence
import dataclasses
import datetime
from typing import Any, Mapping

from etils.array_types import FloatArray
import tensorflow as tf
import tensorflow_datasets as tfds


FeatureSpecElem = tfds.features.FeatureConnector | tf.dtypes.DType
FeatureSpecs = tfds.typing.TreeDict[FeatureSpecElem]
AerialTile = tuple[FloatArray['H W 3'], datetime.datetime, float]
Example = Any


@dataclasses.dataclass
class AerialDataConfig:
  """Configuration for aerial data from earth engine."""

  target_level: int | None = None
  image_size: int = 384
  extent_factors: Sequence[float] = (1.0,)
  num_samples: int = 1
  max_rotation_angle_degrees: float = 0.0
  max_offset: float = 0.0
  scale_offset_by_extent: bool = False
  seed: int = 0
  encoding_format: str | None = 'jpeg'
  make_upright: bool = True
  max_num_collections: int = 1
  resolution_meters: float | None = None  # Usually inferred from s2cell.
  min_ratio_valid: float = 0.8
  add_aerial: bool = True
  add_satellite: bool = False

  def data_spec(self) -> FeatureSpecs:
    """Returns the feature specs for the data."""
    num_tiles = (
        len(self.extent_factors) if self.max_num_collections == 1 else None
    )
    return {
        'rgb': tfds.features.Sequence(
            tfds.features.Image(
                shape=(self.image_size, self.image_size, 3),
                dtype=tf.uint8,
                encoding_format=self.encoding_format,
            ),
            length=num_tiles,
        ),
        'extent': tfds.features.Sequence(
            tf.float32,
            length=num_tiles,
        ),
        'resolution': tfds.features.Sequence(
            tf.float32,
            length=num_tiles,
        ),
        'ecef_t_grid': tfds.features.Sequence(
            tfds.features.Tensor(shape=(4, 4), dtype=tf.float64),
            length=num_tiles,
        ),
        'timestamp': tfds.features.Sequence(
            tf.float64,
            length=num_tiles,
        ),
        'capture_resolution': tfds.features.Sequence(
            tf.float64,
            length=num_tiles,
        ),
        'center_ll': tfds.features.Tensor(shape=(2,), dtype=tf.float64),
        's2cell': tfds.features.Text(),
        'rotation_angle_degrees': tf.float64,
        'upright': tf.bool,
        'offset_meters': tfds.features.Tensor(shape=(2,), dtype=tf.float64),
        'num_collections': tf.int32,
    }


@dataclasses.dataclass
class DataConfig:
  """Configuration for data extraction."""

  # Extract LiDAR pointcloud.
  extract_lidar_pointcloud: bool = False
  # Sparsify LiDAR around views AND project the lidar to each image.
  extract_lidar_per_image: bool = True
  # Whether to use the window around the pano or global point cloud for proj.
  project_lidar_per_pano: bool = True
  extract_pixels: bool = True
  extract_semantic_raster: bool = False
  geo_meta_data: Sequence[str] | None = ()
  semantic_raster_model: str = 'MT_SV45_BEST'
  lidar_spacing_sec: float = 0.5
  add_lidar_semantics: bool = False
  image_height: int = 912
  image_width: int | None = None
  pano_stitch_height: int | None = None
  flatten_map: bool = True
  transform_to_local_coordinate_system: bool = False
  # Coordinate system of the projected lidar points (world, camera, None).
  projected_lidar_point_coordinate_system: str | None = None
  lidar_bound_exclude_percent: float = 2.5
  split_into_single_queries: bool = False  # Split scene into single queries.
  add_aerial: bool = False
  aerial_per_query: bool = (
      False  # Add aerial data per query (split_into_single_queries).
  )
  aerial: AerialDataConfig = dataclasses.field(default_factory=AerialDataConfig)
  encoding_format: str | None = None

  def intrinsics_spec(self) -> FeatureSpecs:
    """Returns the feature specs for the intrinsics."""
    return {
        'width': tf.int32,
        'height': tf.int32,
        'K': tfds.features.Tensor(shape=(3, 3), dtype=tf.float64),
        'type': tfds.features.Text(),
        'radial_distortion': tfds.features.Tensor(shape=(3,), dtype=tf.float64),
        'tangential_distortion': tfds.features.Tensor(
            shape=(2,), dtype=tf.float64
        ),
        'fov_deg': tf.float32,
    }

  def view_spec(self) -> FeatureSpecs:
    """Returns the feature specs for a single view."""
    view_spec = {
        'image_name': tfds.features.Text(),
        'pano_name': tfds.features.Text(),
        'camera_name': tfds.features.Text(),
        'cam_t_w': tfds.features.Tensor(shape=(4, 4), dtype=tf.float64),
        'camera': self.intrinsics_spec(),
        'timestamp': tf.float64,
        'run_name': tfds.features.Text(),
        's2cell': tfds.features.Text(),
        'rosette_type': tfds.features.Text(),
        'vehicle_type': tf.int32,
        'scene_type': tf.int32,
    }

    if self.extract_pixels:
      view_spec['image_pixels'] = tfds.features.Image(
          shape=(self.image_height, self.image_width, 3),
          dtype=tf.uint8,
          encoding_format=self.encoding_format,
      )

    if self.extract_semantic_raster:
      view_spec['semantic_raster'] = tfds.features.Tensor(
          shape=(self.image_height, self.image_width, 1), dtype=tf.uint8
      )

    if self.extract_lidar_per_image:
      view_spec['lidar_depth'] = tfds.features.Tensor(
          shape=(self.image_height, self.image_width, 1), dtype=tf.float32
      )
      if self.projected_lidar_point_coordinate_system == 'world':
        view_spec['w_lidar_points'] = tfds.features.Tensor(
            shape=(self.image_height, self.image_width, 3), dtype=tf.float32
        )
      elif self.projected_lidar_point_coordinate_system == 'camera':
        view_spec['cam_lidar_points'] = tfds.features.Tensor(
            shape=(self.image_height, self.image_width, 3), dtype=tf.float32
        )
      else:
        if self.projected_lidar_point_coordinate_system is not None:
          raise ValueError(
              'Unsupported projected_lidar_point_coordinate_system: '
              f'{self.projected_lidar_point_coordinate_system}'
          )
      view_spec['lidar_mask'] = tfds.features.Tensor(
          shape=(self.image_height, self.image_width, 1), dtype=tf.bool
      )
      if self.add_lidar_semantics:
        view_spec['lidar_semantic_labels'] = tfds.features.Tensor(
            shape=(self.image_height, self.image_width, 1), dtype=tf.uint8
        )

    return view_spec

  def lidar_spec(self) -> FeatureSpecs:
    """Returns the feature specs for the lidar data."""
    lidar_spec = {
        'points': tfds.features.Tensor(shape=(None, 3), dtype=tf.float64),
        'origins': tfds.features.Tensor(shape=(None, 3), dtype=tf.float64),
        'timestamps': tfds.features.Tensor(shape=(None, 1), dtype=tf.float64),
        'reflectivities': tfds.features.Tensor(shape=(None, 1), dtype=tf.uint8),
    }

    if self.add_lidar_semantics:
      lidar_spec['semantic_labels'] = tfds.features.Tensor(
          shape=(None, 1), dtype=tf.uint8
      )
    return lidar_spec

  def collection_spec(self, num_views: int | None = None) -> FeatureSpecs:
    """Returns the feature specs for a collection of views."""
    spec = {
        'views': tfds.features.Sequence(self.view_spec(), length=num_views),
        'collection_id': tfds.features.Text(),
    }
    if self.extract_lidar_pointcloud:
      spec['pointcloud'] = self.lidar_spec()
    if self.extract_lidar_pointcloud or self.extract_lidar_per_image:
      spec['lidar_grid_t_w'] = tfds.features.Tensor(
          shape=(4, 4), dtype=tf.float64
      )
      spec['lidar_extent_meters'] = tfds.features.Tensor(
          shape=(3,), dtype=tf.float64
      )
    return spec

  def map_str_collection_spec(
      self, num_views_per_collection: Mapping[str, int]
  ) -> FeatureSpecs:
    return {
        k: self.collection_spec(num_views)
        for k, num_views in num_views_per_collection.items()
    }

  def scene_spec(
      self,
      num_views_query: int | None = None,
      num_views_map: int | None = None,
      num_collections_map: int | None = None,
  ) -> FeatureSpecs:
    """Return feature spec for a scene."""
    spec = {
        's2cell': tfds.features.Text(),
        'query': self.collection_spec(num_views_query),
        'ecef_t_w': tfds.features.Tensor(shape=(4, 4), dtype=tf.float64),
        'overlap': tfds.features.Tensor(
            shape=(num_collections_map,), dtype=tf.float32
        ),
    }

    for geo_type in self.geo_meta_data:
      spec[geo_type] = tf.int32

    if self.add_aerial:
      spec['aerial'] = self.aerial.data_spec()

    if num_views_map != 0 and num_collections_map != 0:
      if self.flatten_map:
        num_flattened_views_map = (
            num_views_map * num_collections_map
            if num_views_map and num_collections_map
            else None
        )
        spec['map'] = self.collection_spec(num_flattened_views_map)
      else:
        spec['map'] = tfds.features.Sequence(
            self.collection_spec(num_views_map), length=num_collections_map
        )
    return spec
