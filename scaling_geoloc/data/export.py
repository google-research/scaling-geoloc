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

import collections
from collections.abc import Sequence
import copy
import dataclasses
import datetime
import functools
import json
import os
import time
from typing import Any, Iterable, Mapping, Tuple

from absl import logging
import apache_beam as beam
import cv2
from etils.array_types import Array
from etils.array_types import FloatArray
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from google3.geo.imagery.streetview.store.client.python import client_request_options
from google3.geo.imagery.streetview.store.client.python import pixel_provider
from google3.geo.imagery.streetview.store.client.python import svms_client_util
from google3.geo.imagery.streetview.store.proto import pixel_options_pb2
from google3.geo.imagery.streetview.store.proto import resources_pb2
from google3.geo.imagery.streetview.store.shared.python import ids as gsv_ids
from google3.location.visualmapping.experimental.fomo.data import data_utils as fomo_data_utils
from scaling_geoloc.data import geo
from scaling_geoloc.data import reverse_geocoder
from scaling_geoloc.data import types
from scaling_geoloc.utils import misc
from google3.location.visualmapping.experimental.lemming.data import earth_engine
from google3.location.visualmapping.experimental.lemming.data import geo as lemming_geo
from google3.location.visualmapping.experimental.lemming.data.loading.python import data_loading_utils
from google3.location.visualmapping.experimental.lemming.utils import geometry
from google3.location.visualmapping.experimental.lemming.utils import grids
from google3.pyglib import gfile
from google3.third_party.array_record.google.flume.py import array_record_io
from google3.util.geometry.python import s2
from google3.util.task import codes_pb2
from google3.util.task.python import error


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


def get_aerial_tile(
    ecef_t_ll: geometry.Transform3D,
    image_size: tuple[int, int],
    resolution_meters: float,
    offset_meters: FloatArray['2'] | None = None,
    rotation_angle_degrees: float | None = None,
    make_upright: bool = True,
    max_num_collections: int = 1,
    **kwargs,  # Forwarded to EarthEngineQuery.get_rgb*.
) -> tuple[FloatArray['4 4'], Sequence[AerialTile]]:
  """Get aerial tile."""
  grid = grids.Grid2D(image_size, resolution_meters)
  if rotation_angle_degrees is not None:
    rotation_angle = np.radians(rotation_angle_degrees)
    rot_z = np.zeros((3, 3))
    rot_z[0, 0] = np.cos(rotation_angle)
    rot_z[0, 1] = np.sin(rotation_angle)
    rot_z[1, 0] = -np.sin(rotation_angle)
    rot_z[1, 1] = np.cos(rotation_angle)
    rot_z[2, 2] = 1
    ll_r_grid = rot_z
  else:
    ll_r_grid = np.eye(3)
  if offset_meters is None:
    offset_meters = np.zeros((2,))

  ll_t_grid = -np.array((*grid.extent_meters, 0)) / 2 + np.array(
      (*offset_meters, 0)
  )
  ll_t_grid = geometry.Transform3D(R=ll_r_grid, t=ll_r_grid @ ll_t_grid)
  ecef_t_grid = ecef_t_ll @ ll_t_grid
  ee_query = earth_engine.EarthEngineQuery(grid, ecef_t_grid)
  if max_num_collections != 1:
    ret = ee_query.get_rgb_collection(
        max_num_assets=max_num_collections, **kwargs
    )
  else:
    ret = ee_query.get_rgb(**kwargs)
    ret = [ret] if ret is not None else []
  if not ret:
    return None, []
  all_pixels = []
  for i, ret_i in enumerate(ret[::-1]):
    if i >= max_num_collections:
      break
    pixels, _, date, pixel_size_meters, _ = ret_i
    if make_upright:
      pixels = np.swapaxes(pixels, 0, 1)[::-1]
    all_pixels.append((pixels, date, pixel_size_meters))
  return ecef_t_grid.to_4x4matrix(), all_pixels


def get_aerial_tile_for_s2cell(
    s2cell_token: str,
    image_size: tuple[int, int],
    resolution_meters: float,
    offset_meters: FloatArray['2'] | None = None,
    rotation_angle_degrees: float | None = None,
    make_upright: bool = True,
    max_num_collections: int = 1,
    **kwargs,  # Forwarded to EarthEngineQuery.get_rgb*.
) -> tuple[FloatArray['4 4'], Sequence[AerialTile]]:
  """Convenience function to get aerial tile for a given s2cell."""
  ecef_t_ll = lemming_geo.get_local_level_frame(
      s2.S2CellId.from_token(s2cell_token.encode('utf-8')).to_lat_lng()
  )

  return get_aerial_tile(
      ecef_t_ll,
      image_size,
      resolution_meters,
      offset_meters=offset_meters,
      rotation_angle_degrees=rotation_angle_degrees,
      make_upright=make_upright,
      max_num_collections=max_num_collections,
      **kwargs,
  )


class FetchAerialImages(beam.DoFn):
  """Fetch aerial imagery."""

  def __init__(self, data_config: AerialDataConfig):
    self.config = data_config

    if self.config.num_samples > 1:
      if (
          self.config.max_rotation_angle_degrees < 1.0e-6
          and self.config.max_offset < 1.0e-6
      ):
        raise ValueError(
            'Either max_rotation_angle_degrees or max_offset must be > 0.0 when'
            ' num_samples > 1.'
        )

  def setup(self):
    self.rng = np.random.default_rng(self.config.seed)
    # Try to initialize EE and, if this fails, try to initialize later
    # or skip the scene if rasters are required.
    earth_engine.initialize_ee_silent()

  def pack_aerial_data_for_location(
      self,
      ecef_t_ll: geometry.Transform3D,
      extent_meters: float | None = None,
      s2cell_token: str | None = None,
  ) -> types.DataDict | None:
    if extent_meters is None:
      if self.config.resolution_meters is None:
        if s2cell_token is None:
          raise ValueError('Either extent_meters or s2cell_token must be set.')
        else:
          s2cell_id = s2.S2CellId.from_token(s2cell_token.encode('utf-8'))
          cell_extent = types.CELL_EXTENTS[s2cell_id.level]
          extent_meters = cell_extent
      else:
        extent_meters = self.config.resolution_meters * self.config.image_size
    data = collections.defaultdict(list)
    extra_data = {}
    rotation_angle_degrees = self.rng.uniform(
        -self.config.max_rotation_angle_degrees,
        self.config.max_rotation_angle_degrees,
    )
    offset_meters = self.rng.uniform(
        -self.config.max_offset, self.config.max_offset, 2
    )
    if self.config.scale_offset_by_extent:
      offset_meters *= extent_meters / 2  # Use factor=1.0 here.
    for i, factor in enumerate(self.config.extent_factors):
      image_extent = extent_meters * factor
      resolution = image_extent / self.config.image_size
      ecef_t_grid, aerial_collections = get_aerial_tile(
          ecef_t_ll,
          (self.config.image_size, self.config.image_size),
          resolution,
          offset_meters=offset_meters,
          rotation_angle_degrees=rotation_angle_degrees,
          make_upright=self.config.make_upright,
          max_num_collections=self.config.max_num_collections,
          min_ratio_valid=self.config.min_ratio_valid,
          add_aerial=self.config.add_aerial,
          add_satellite=self.config.add_satellite,
      )
      if not aerial_collections:
        beam.metrics.Metrics.counter(self.__class__.__name__, 'failed').inc()
        return None
      beam.metrics.Metrics.distribution(
          self.__class__.__name__, 'num_collections'
      ).update(len(aerial_collections))
      if i == 0:
        grid_offset = self.config.image_size * resolution / 2
        grid_t_center = geometry.Transform3D(
            R=np.eye(3), t=np.array([grid_offset] * 2 + [0.0])
        ).to_4x4matrix()
        ecef_t_center = ecef_t_grid @ grid_t_center
        extra_data['center_ll'] = geo.latlng_from_ecef(
            ecef_t_center[:3, -1][None]
        )[0]
        extra_data['num_collections'] = len(aerial_collections)
      for image, date, pixel_size_meters in aerial_collections:
        data['extent'].append(image_extent)
        data['resolution'].append(resolution)
        data['ecef_t_grid'].append(ecef_t_grid)
        data['rgb'].append(image)
        data['timestamp'].append(date.timestamp())
        data['capture_resolution'].append(pixel_size_meters)
    beam.metrics.Metrics.counter(self.__class__.__name__, 'successful').inc()
    if s2cell_token is None:
      e = extent_meters / 2
      corners = np.array([
          [-e, -e, 0],
          [e, -e, 0],
          [e, e, 0],
          [-e, e, 0],
      ])
      ecef_p_corners = ecef_t_ll @ corners
      s2cell_token = geo.enclosing_s2cell_token(
          geo.latlng_from_ecef(ecef_p_corners)
      )
    data = {
        'rotation_angle_degrees': rotation_angle_degrees,
        'offset_meters': offset_meters,
        'upright': self.config.make_upright,
        's2cell': s2cell_token,
        **data,
        **extra_data,
    }
    return data

  def pack_aerial_data(
      self,
      s2cell_token: str,
  ) -> types.DataDict | None:
    ecef_t_ll = lemming_geo.get_local_level_frame(
        s2.S2CellId.from_token(s2cell_token.encode('utf-8')).to_lat_lng()
    )
    return self.pack_aerial_data_for_location(
        ecef_t_ll, s2cell_token=s2cell_token
    )

  def process(self, s2cell_token: str) -> Iterable[tuple[str, types.DataDict]]:
    beam.metrics.Metrics.counter(self.__class__.__name__, 's2cells').inc()

    for _ in range(self.config.num_samples):
      aerial_data = self.pack_aerial_data(s2cell_token)
      if aerial_data is None:
        return
      else:
        yield s2cell_token, aerial_data


class CropsFromAerial(beam.DoFn):
  """Crops aerial images."""

  def __init__(self, data_config: AerialDataConfig, re_center: bool = False):
    self.config = data_config
    self.re_center = re_center

  def setup(self):
    self.rng = np.random.default_rng(self.config.seed)

  def generate_sample(
      self,
      aerial_data: types.DataDict,
      image_idx: int,
      at_ecef: FloatArray['3'] | None = None,  # Sample around this ECEF point.
  ) -> types.DataDict:
    rgb = aerial_data['rgb'][image_idx]
    if aerial_data['upright']:
      rgb = np.swapaxes(rgb[::-1], 0, 1)
    data = collections.defaultdict(list)
    extra_data = {}
    extent_meters = self.config.resolution_meters * self.config.image_size
    rotation_angle_degrees = self.rng.uniform(
        -self.config.max_rotation_angle_degrees,
        self.config.max_rotation_angle_degrees,
    )
    offset_meters = self.rng.uniform(
        -self.config.max_offset, self.config.max_offset, 2
    )
    if self.config.scale_offset_by_extent:
      offset_meters *= extent_meters / 2  # Use factor=1.0 here.

    ecef_t_imagegrid = aerial_data['ecef_t_grid'][image_idx]
    if at_ecef is not None:
      offset_meters += (
          geo.transform_points(np.linalg.inv(ecef_t_imagegrid), at_ecef[None])[
              0, :2
          ]
          - aerial_data['extent'][image_idx] / 2.0
      )
    for i, factor in enumerate(self.config.extent_factors):
      image_extent = extent_meters * factor
      resolution = image_extent / self.config.image_size
      image_t_crop, image_crop, valid_pixels = misc.crop_aerial_image(
          rgb,
          aerial_data['resolution'][image_idx],
          (self.config.image_size, self.config.image_size),
          offset_meters,
          rotation_angle_degrees,
          resolution,
          scale_offset_by_extent=False,
          numpy=True,
      )

      valid_fraction = np.round(valid_pixels.mean() * 100.0).astype(int).item()
      beam.metrics.Metrics.counter(
          self.__class__.__name__, f'{(valid_fraction // 10)*10}%_valid_pixels'
      ).inc()

      ecef_t_grid = ecef_t_imagegrid @ image_t_crop
      if i == 0:
        grid_offset = self.config.image_size * resolution / 2
        grid_t_center = geometry.Transform3D(
            R=np.eye(3), t=np.array([grid_offset] * 2 + [0.0])
        ).to_4x4matrix()
        ecef_t_center = ecef_t_grid @ grid_t_center
        extra_data['center_ll'] = geo.latlng_from_ecef(
            ecef_t_center[:3, -1][None]
        )[0]
        extra_data['num_collections'] = 1
      data['extent'].append(image_extent)
      data['resolution'].append(resolution)
      data['ecef_t_grid'].append(ecef_t_grid)
      data['rgb'].append(image_crop)
      data['timestamp'].append(aerial_data['timestamp'][image_idx])
      data['capture_resolution'].append(aerial_data['timestamp'][image_idx])

    return {
        'rotation_angle_degrees': rotation_angle_degrees,
        'offset_meters': offset_meters,
        'upright': self.config.make_upright,
        's2cell': aerial_data['s2cell'],
        **data,
        **extra_data,
    }

  def process(
      self, aerial_data: types.DataDict
  ) -> Iterable[tuple[str, types.DataDict]]:
    num_images = len(aerial_data['rgb'])

    image_for_sample = self.rng.choice(
        num_images, self.config.num_samples
    ).tolist()
    if self.re_center:
      s2cell_center_ll = s2.S2CellId.from_token(
          aerial_data['s2cell']
      ).to_lat_lng()
      s2cell_center_ecef = geo.ecef_from_latlng(
          np.array(
              [s2cell_center_ll.lat.degrees, s2cell_center_ll.lng.degrees]
          )[None]
      )[0]
    else:
      s2cell_center_ecef = None
    for i, image_idx in enumerate(image_for_sample):
      beam.metrics.Metrics.counter(
          self.__class__.__name__, 'generate_crop'
      ).inc()
      s2cell = aerial_data['s2cell'].decode('utf-8')
      yield f'{s2cell}_sample{i}_{image_idx}', self.generate_sample(
          aerial_data, image_idx, at_ecef=s2cell_center_ecef
      )


class PackViewCollection(beam.DoFn):
  """Extracts all data for a view collection."""

  def __init__(self, config: DataConfig, pose_snapshot_id: str):
    self.config = config
    self.pose_snapshot_id = pose_snapshot_id

  def setup(self):
    self.dlutils = data_loading_utils.DataLoadingUtils.Create()
    self._svms_client = svms_client_util.CreateBatchSvmsClientFromTarget(
        'blade:street-view-metadata-service-staging'
    )
    self._image_request_options = (
        client_request_options.ClientRequestOptions().IncludeMetadata([
            resources_pb2.MetadataPiece.METADATA_IMAGE_ID,
        ])
    ).SetPixelOptions(
        pixel_options_pb2.PixelOptions(
            anonymization=pixel_options_pb2.PixelAnonymizationMode.SAFE
        )
    )
    self._provider = pixel_provider.CreatePixelProvider()

  def load_image_pixels(
      self, image_id: gsv_ids.ImageId, pano_stitch_height: int | None = None
  ) -> Array['H W C']:
    result = self._svms_client.GetCapture(
        image_id.capture_id(), self._image_request_options
    )
    img = result.GetImage(image_id)
    if pano_stitch_height:
      # Force stitching at a specific height. This has no effect on unstitched
      # images.
      stitch_options = (
          pixel_options_pb2.ImagePixelOptions.GsvImageStitchingOptions(
              pano_height=pano_stitch_height
          )
      )
      image_pixel_options = pixel_options_pb2.ImagePixelOptions(
          gsv_image_stitching_options=stitch_options
      )
      jpeg = self._provider.FromImage(img).GetJpeg(image_pixel_options)
    else:
      jpeg = self._provider.FromImage(img).GetJpeg()
    image = cv2.cvtColor(
        cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_ANYCOLOR),
        cv2.COLOR_BGR2RGB,
    )
    return image

  def pack_view(
      self,
      view_key: types.ViewKey,
      view: types.View,
      metadata: types.ViewCollectionMetadata,
  ) -> types.DataDict:
    t_start = time.time()
    scaled_view = view.scale_to_image_size(
        self.config.image_width, self.config.image_height
    )
    if not scaled_view.check_consistency():
      raise ValueError(f'View {view_key.unique_name} has inconsistent shapes.')
    camera_dict = dataclasses.asdict(scaled_view.camera)
    del camera_dict['local_orientation']
    packed_view = {
        'image_name': view_key.unique_name,
        'pano_name': view_key.pano_name,
        'camera_name': view_key.camera_name,
        'cam_t_w': scaled_view.cam_t_w,
        'camera': camera_dict,
        'timestamp': scaled_view.timestamp,
        **{
            attr: getattr(metadata, attr)
            for attr in [
                'run_name',
                's2cell',
                'rosette_type',
                'vehicle_type',
                'scene_type',
            ]
        },
    }

    if self.config.extract_pixels:
      beam.metrics.Metrics.counter(self.__class__.__name__, 'load_pixels').inc()
      packed_view['image_pixels'] = scaled_view.image_pixels

    if self.config.extract_semantic_raster:
      if view_key.is_stitched:
        raise ValueError(
            'Semantic raster extraction is not supported for stitched panos.'
        )
      beam.metrics.Metrics.counter(
          self.__class__.__name__, 'load_semantic_raster'
      ).inc()
      semantic_raster = fomo_data_utils.load_semantic_raster(
          view_key.image_name, self.config.semantic_raster_model
      )
      if semantic_raster is None:
        beam.metrics.Metrics.counter(
            self.__class__.__name__, 'failed_to_load_semantic_raster'
        ).inc()
        # We add the semantics with all pixels 255 (unknown)
        packed_view['semantic_raster'] = np.ones(
            (scaled_view.camera.height, scaled_view.camera.width),
            dtype=np.uint8,
        )[..., None]
      else:
        # Map unlabeled (44) to unknown.
        semantic_raster[semantic_raster == 44] = 255
        packed_view['semantic_raster'] = fomo_data_utils.resize_matrix_nearest(
            semantic_raster,
            shape=(scaled_view.camera.height, scaled_view.camera.width),
        )[..., None]

    logging.info(
        '[%s] Packed view %s: %f s.',
        __class__.__name__,
        view_key.unique_name,
        time.time() - t_start,
    )
    return packed_view

  def pack_views(
      self, view_collection: types.ViewCollection
  ) -> Sequence[types.DataDict]:
    """Packs all views of a view collection into dicts, and (optional) loads image pixels."""
    _, metadata, views = view_collection

    # batch loading of image pixels
    if self.config.extract_pixels:
      t_start = time.time()
      image_ids = list(set([view_key.image_id for view_key in views.keys()]))
      image_pixels = {
          image_id: self.load_image_pixels(
              image_id, pano_stitch_height=self.config.pano_stitch_height
          )
          for image_id in image_ids
      }

      logging.info(
          '[%s] Loaded image pixels for %d views (%d unique): %f s.',
          __class__.__name__,
          len(views.keys()),
          len(image_ids),
          time.time() - t_start,
      )
      for view_key in views.keys():
        if views[view_key].is_pano_sample():
          t_start_sample = time.time()
          views[view_key].image_pixels = views[
              view_key
          ].sample_pixels_from_pano(image_pixels[view_key.image_id])
          logging.info(
              '[%s] Loaded pano sample %s: %f s.',
              __class__.__name__,
              view_key.unique_name,
              time.time() - t_start_sample,
          )
        else:
          views[view_key].image_pixels = image_pixels[view_key.image_id]
        if not views[view_key].check_consistency():
          raise ValueError(
              f'View {view_key.unique_name} has inconsistent shapes.'
          )

    return [
        self.pack_view(view_key, view, metadata)
        for view_key, view in views.items()
    ]

  def _get_lidar(
      self,
      run_name: str,
      min_timestamp: float,
      max_timestamp: float,
      w_t_ecef: FloatArray['4 4'] | None = None,
  ) -> types.DataDict:
    lidar_data = self.dlutils.GetLiDARPoints(
        run_name,
        min_timestamp,
        max_timestamp,
        self.pose_snapshot_id,
        add_semantic_labels=self.config.add_lidar_semantics,
    )

    lidar_results = {
        'points': lidar_data.ecef_points.T,
        'origins': lidar_data.ecef_origins.T,
        'timestamps': np.array(lidar_data.timestamps)[..., None],
        'reflectivities': np.array(lidar_data.reflectivities, dtype=np.uint8)[
            ..., None
        ],
    }

    if w_t_ecef is not None:
      lidar_results['points'] = geo.transform_points(
          w_t_ecef, lidar_results['points']
      )
      lidar_results['origins'] = geo.transform_points(
          w_t_ecef, lidar_results['origins']
      )

    if self.config.add_lidar_semantics:
      if all(np.array(lidar_data.semantic_labels) == 255):
        beam.metrics.Metrics.counter(
            self.__class__.__name__, 'collection_without_lidar_semantic_labels'
        ).inc()
      lidar_results['semantic_labels'] = np.array(
          lidar_data.semantic_labels, dtype=np.uint8
      )[..., None]
    return lidar_results

  def _project_lidar_to_image(
      self,
      view_key: types.ViewKey,  # pylint: disable=unused-argument
      view: types.View,
      lidar_data: types.DataDict,
      max_spacing_sec: float | None = None,
  ) -> Mapping[str, Array['H W C']]:
    """Computes a depth map from lidar data."""
    t_start = time.time()
    scaled_view = view.scale_to_image_size(
        self.config.image_width, self.config.image_height
    )
    image_height = scaled_view.camera.height
    image_width = scaled_view.camera.width
    vsfm_camera = scaled_view.to_visionsfm_camera()
    w_points = lidar_data['points']
    c_points = geo.transform_points(scaled_view.cam_t_w, w_points)
    if max_spacing_sec:
      w_points = w_points[
          np.abs(lidar_data['timestamps'] - view.timestamp) < max_spacing_sec
      ]
    points_2d, depth = vsfm_camera.Project(w_points)
    valid = vsfm_camera.ArePixelsInFrustum(points_2d) & (depth > 0)
    valid = np.where(valid)[0]
    # Sort valid indices by abs. temporal distance in descending order.
    # Hacky solution to select the most recent lidar points per pixel.
    temporal_distance = np.abs(
        lidar_data['timestamps'][valid, 0] - view.timestamp
    )
    valid = valid[np.argsort(temporal_distance)[::-1]]
    pixel_coords = points_2d[valid].astype(int)

    projections = {}
    # Mask
    projections['lidar_mask'] = np.zeros(
        (image_height, image_width, 1), dtype=bool
    )
    projections['lidar_mask'][pixel_coords[:, 1], pixel_coords[:, 0], 0] = True
    # Depth
    projections['lidar_depth'] = np.zeros(
        (image_height, image_width, 1), dtype=np.float32
    )

    projections['lidar_depth'][pixel_coords[:, 1], pixel_coords[:, 0], 0] = (
        depth[valid].astype(np.float32)
    )

    if self.config.projected_lidar_point_coordinate_system == 'world':
      projections['w_lidar_points'] = np.zeros(
          (image_height, image_width, 3),
          dtype=np.float32,
      )

      projections['w_lidar_points'][
          pixel_coords[:, 1], pixel_coords[:, 0], :
      ] = w_points[valid].astype(np.float32)
    elif self.config.projected_lidar_point_coordinate_system == 'camera':
      projections['cam_lidar_points'] = np.zeros(
          (image_height, image_width, 3),
          dtype=np.float32,
      )
      projections['cam_lidar_points'][
          pixel_coords[:, 1], pixel_coords[:, 0], :
      ] = c_points[valid].astype(np.float32)
    else:
      if self.config.projected_lidar_point_coordinate_system is not None:
        raise ValueError(
            'Unsupported projected_lidar_point_coordinate_system: '
            f'{self.config.projected_lidar_point_coordinate_system}'
        )

    # Semantic labels
    if self.config.add_lidar_semantics:
      projections['lidar_semantic_labels'] = (
          np.ones(
              (image_height, image_width, 1),
              dtype=np.uint8,
          )
          * 255
      )
      projections['lidar_semantic_labels'][
          pixel_coords[:, 1], pixel_coords[:, 0]
      ] = lidar_data['semantic_labels'][valid]
    logging.info(
        '[%s] Projected lidar points to image %s: %f s.',
        __class__.__name__,
        view.image_name,
        time.time() - t_start,
    )
    return projections

  def pack_lidar(
      self,
      view_collection: types.ViewCollection,
      w_t_ecef: FloatArray['4 4'] | None = None,
  ) -> types.DataDict:
    """Packs lidar data for a view collection."""
    _, metadata, views = view_collection
    pano_timestamps = {vk.pano_name: vk.pano_timestamp for vk in views.keys()}
    min_timestamp = min(pano_timestamps.values())
    max_timestamp = max(pano_timestamps.values())
    t_start = time.time()
    if self.config.extract_lidar_per_image:
      pano_lidar_data = {
          pano_name: self._get_lidar(
              metadata.run_name,
              timestamp - self.config.lidar_spacing_sec,
              timestamp + self.config.lidar_spacing_sec,
              w_t_ecef=w_t_ecef,
          )
          for pano_name, timestamp in pano_timestamps.items()
      }
      lidar_keys = pano_lidar_data[next(iter(pano_lidar_data))].keys()
      packed_lidar_data = {
          k: np.concatenate([v[k] for v in pano_lidar_data.values()], axis=0)
          for k in lidar_keys
      }
      if self.config.project_lidar_per_pano:
        packed_lidar_data['image_projections'] = {
            view_key.unique_name: self._project_lidar_to_image(
                view_key, view, pano_lidar_data[view_key.pano_name]
            )
            for view_key, view in views.items()
        }
      else:
        packed_lidar_data['image_projections'] = {
            view_key.unique_name: self._project_lidar_to_image(
                view_key, view, packed_lidar_data
            )
            for view_key, view in views.items()
        }
    else:
      packed_lidar_data = self._get_lidar(
          metadata.run_name, min_timestamp, max_timestamp, w_t_ecef=w_t_ecef
      )
      packed_lidar_data['image_projections'] = {
          view_key.unique_name: self._project_lidar_to_image(
              view_key, view, packed_lidar_data
          )
          for view_key, view in views.items()
      }

    logging.info(
        '[%s] Loaded lidar points for %d panos over a duration of %f'
        ' seconds: %f s.',
        __class__.__name__,
        len(pano_timestamps),
        max_timestamp - min_timestamp,
        time.time() - t_start,
    )
    return packed_lidar_data

  def pack_collection(
      self,
      view_collection: types.ViewCollection,
      w_t_ecef: FloatArray['4 4'] | None = None,
  ) -> types.DataDict | None:
    collection_id, _, _ = view_collection

    if w_t_ecef is not None:
      # Transform poses to the local coordinate frame defined by w_t_ecef.
      view_collection = (
          view_collection[0],
          view_collection[1],
          {
              view_key: dataclasses.replace(
                  view, cam_t_w=view.cam_t_w @ np.linalg.inv(w_t_ecef)  # pytype: disable=attribute-error
              )
              for view_key, view in view_collection[2].items()
          },
      )
    try:
      packed_views = self.pack_views(view_collection)
    except error.StatusNotOk as err:
      beam.metrics.Metrics.counter(
          self.__class__.__name__, 'failed_to_pack_views'
      ).inc()
      logging.warning('Failed to pack views: %s', err.message)
      return None
    packed_data = {
        'views': packed_views,  # poses, intrinsics, rgb, ..
        'collection_id': collection_id,
    }

    if (
        self.config.extract_lidar_pointcloud
        or self.config.extract_lidar_per_image
    ):
      try:
        packed_lidar_data = self.pack_lidar(view_collection, w_t_ecef=w_t_ecef)
      except error.StatusNotOk as err:
        if err.code == codes_pb2.INVALID_ARGUMENT:
          beam.metrics.Metrics.counter(
              self.__class__.__name__, 'LidarReader_invalid_argument'
          ).inc()
          logging.warning('Failed to load lidar data for %s', collection_id)
          return None
        elif err.code == codes_pb2.NOT_FOUND:
          beam.metrics.Metrics.counter(
              self.__class__.__name__, 'Lidar_data_not_found.'
          ).inc()
          logging.warning('Lidar data not found for %s', collection_id)
          return None
        elif err.code == codes_pb2.OUT_OF_RANGE:
          beam.metrics.Metrics.counter(
              self.__class__.__name__, 'timestamp_out_of_range.'
          ).inc()
          logging.warning('LiDAR timestamp out of range for %s', collection_id)
          return None
        elif err.code == codes_pb2.INTERNAL:
          if err.message.startswith('Service is overloaded'):
            beam.metrics.Metrics.counter(
                self.__class__.__name__, 'Lidar_service_overloaded.'
            ).inc()
            logging.warning('Beam work unit cancelled for %s', collection_id)
            return None
          else:
            beam.metrics.Metrics.counter(
                self.__class__.__name__, err.message
            ).inc()
            logging.warning(
                'Beam work unit cancelled with unknown error for %s; %s',
                collection_id,
                err.message,
            )
            return None
        raise

      lb = self.config.lidar_bound_exclude_percent
      if packed_lidar_data['points'].shape[0] > 0:
        lower_bound = np.percentile(packed_lidar_data['points'], lb, axis=0)
        upper_bound = np.percentile(
            packed_lidar_data['points'], 100 - lb, axis=0
        )
      else:
        lower_bound = np.zeros(3)
        upper_bound = np.zeros(3)

      grid_t_w = np.eye(4)
      grid_t_w[:3, 3] = -lower_bound
      packed_data['lidar_grid_t_w'] = grid_t_w
      packed_data['lidar_extent_meters'] = upper_bound - lower_bound

      if self.config.extract_lidar_per_image:
        # Add lidar projections (depth, ...) to the respective views
        packed_data['views'] = [
            {
                **view,
                **{
                    map_name: packed_lidar_data['image_projections'][
                        view['image_name']
                    ][map_name]
                    for map_name in packed_lidar_data['image_projections'][
                        view['image_name']
                    ]
                },
            }
            for view in packed_data['views']
        ]
        # Remove image projections from the pointcloud data
        del packed_lidar_data['image_projections']

      if self.config.extract_lidar_pointcloud:
        # Add LiDAR poiintcloud to the collection
        packed_data['pointcloud'] = packed_lidar_data
    return packed_data

  def pack_collection_safe(
      self,
      view_collection: types.ViewCollection,
      w_t_ecef: FloatArray['4 4'] | None = None,
  ) -> types.DataDict | None:
    try:
      return self.pack_collection(view_collection, w_t_ecef=w_t_ecef)
    except error.StatusNotOk as err:
      if err.code == codes_pb2.CANCELLED:
        return  # Exit to propagate flume work-unit cancellations.
      beam.metrics.Metrics.counter(
          self.__class__.__name__, 'data_packing_failed'
      ).inc()
      logging.warning('Failed to load data for %s', view_collection[0])
      raise

  def process(
      self, view_collection: types.ViewCollection
  ) -> Iterable[Tuple[str, types.DataDict]]:
    if self.config.transform_to_local_coordinate_system:
      w_t_ecef = np.linalg.inv(
          geo.ecef_t_s2cell(view_collection[1].s2cell, altitude=0.0)
      )
    else:
      w_t_ecef = None
    packed_data = self.pack_collection_safe(view_collection, w_t_ecef=w_t_ecef)
    if packed_data is not None:
      yield view_collection[0], packed_data


class PackScene(beam.DoFn):
  """Pack data for a scene."""

  def __init__(self, config: DataConfig, pose_snapshot_id: str):
    self.packer = PackViewCollection(config, pose_snapshot_id)
    if self.packer.config.add_aerial:
      self.aerial_packer = FetchAerialImages(self.packer.config.aerial)

  def setup(self):
    self.packer.setup()
    if self.packer.config.add_aerial:
      self.aerial_packer.setup()

    if self.packer.config.geo_meta_data:
      self.reverse_geocoder = reverse_geocoder.CombinedReverseGeocoder(
          self.packer.config.geo_meta_data
      )

  def concatenate(
      self, collection_id: str, packed_data_sequence: Sequence[types.DataDict]
  ) -> types.DataDict:
    # Concatenate the data from all view collections of a scene.
    # They can be grouped again by metadata on the views.
    concatenated_data = {
        'views': sum([data['views'] for data in packed_data_sequence], []),
        'collection_id': collection_id,
    }

    if self.packer.config.extract_lidar_pointcloud:
      concatenated_data['pointcloud'] = {
          feature_name: np.concatenate(
              [
                  packed_collection['pointcloud'][feature_name]
                  for packed_collection in packed_data_sequence
              ],
              axis=0,
          )
          for feature_name in packed_data_sequence[0]['pointcloud'].keys()
      }

    if 'lidar_grid_t_w' in packed_data_sequence[0]:
      for key in ['lidar_grid_t_w', 'lidar_extent_meters']:
        concatenated_data[key] = np.stack(
            [
                packed_collection[key]
                for packed_collection in packed_data_sequence
            ],
            axis=0,
        ).max(axis=0)
    return concatenated_data

  def pack_aerial_data(
      self, scene: types.Scene, ecef_t_ll: FloatArray['4 4'] | None = None
  ) -> types.DataDict | None:
    t_start = time.time()
    if ecef_t_ll is None:
      center_ll = (
          np.stack(
              [view.latlng_position()[0] for view in scene.query[2].values()]
          )
          .mean(0)
          .tolist()
      )
      ecef_t_ll = lemming_geo.get_local_level_frame(
          s2.S2LatLng.from_degrees(center_ll[0], center_ll[1])
      )
    else:
      ecef_t_ll = geometry.Transform3D.from_Rt(
          ecef_t_ll[:3, :3], ecef_t_ll[:3, 3]
      )

    aerial_data = self.aerial_packer.pack_aerial_data_for_location(
        ecef_t_ll, s2cell_token=scene.s2cell
    )
    logging.info('pack_aerial_data took %f seconds', time.time() - t_start)

    return aerial_data

  def process(
      self, scene_tuple: Tuple[str, types.Scene]
  ) -> Iterable[Tuple[str, types.DataDict]]:
    scene_id, scene = scene_tuple
    packed_data: dict[str, Any] = {}
    packed_data['s2cell'] = scene.s2cell
    packed_data['ecef_t_w'] = (
        scene.ecef_t_w
        if self.packer.config.transform_to_local_coordinate_system
        else np.eye(4)
    )

    packed_data['overlap'] = scene.overlap

    if self.packer.config.geo_meta_data:
      lat, lng = (
          geo.latlng_from_ecef(
              np.linalg.inv(
                  np.stack([view.cam_t_w for view in scene.query[2].values()])
              )[:, :3, 3]
          )
          .mean(0)
          .tolist()
      )
      geo_labels = self.reverse_geocoder.get_labels_for_latlng(lat, lng)
      geo_ids = self.reverse_geocoder.labels_to_ids(geo_labels)
      for geo_type, geo_id in geo_ids.items():
        packed_data[geo_type] = geo_id

    w_t_ecef = (
        np.linalg.inv(scene.ecef_t_w)
        if self.packer.config.transform_to_local_coordinate_system
        else None
    )
    packed_data['query'] = self.packer.pack_collection(
        scene.query, w_t_ecef=w_t_ecef
    )
    packed_data['map'] = [
        self.packer.pack_collection_safe(map_i, w_t_ecef=w_t_ecef)
        for map_i in scene.map
    ]

    failed_to_pack = packed_data['query'] is None or any(
        [map_i is None for map_i in packed_data['map']]
    )
    if failed_to_pack:
      beam.metrics.Metrics.counter(
          self.__class__.__name__, 'failed_to_pack_scene'
      ).inc()
      logging.warning('Failed to pack scene %s', scene_id)
      return

    if self.packer.config.add_aerial:
      aerial_data = self.pack_aerial_data(scene)
      if aerial_data is not None:
        packed_data['aerial'] = aerial_data
      else:
        beam.metrics.Metrics.counter(
            self.__class__.__name__, 'failed_to_pack_aerial_data'
        ).inc()
        logging.warning('Failed to pack aerial data for scene %s', scene_id)
        return

    if not packed_data['map']:
      # If there are no map collections -> remove the map key.
      del packed_data['map']
    elif self.packer.config.flatten_map:
      packed_data['map'] = self.concatenate(scene_id, packed_data['map'])
    beam.metrics.Metrics.counter(self.__class__.__name__, 'packed_scenes').inc()

    if self.packer.config.extract_lidar_pointcloud:
      # Add LiDAR pointcloud to the scene
      packed_data['pointcloud'] = packed_data['query']['pointcloud']
    if self.packer.config.split_into_single_queries:
      # Split the scene into multiple scenes, each with a single query view.
      query_views = packed_data['query'].pop('views')
      for query in query_views:
        q_data = copy.deepcopy(packed_data)
        q_data['query']['views'] = [query]
        if (
            self.packer.config.add_aerial
            and self.packer.config.aerial_per_query
        ):
          q_data['aerial'] = self.pack_aerial_data(
              scene, ecef_t_ll=packed_data['ecef_t_w']
          )
          if q_data['aerial'] is None:
            beam.metrics.Metrics.counter(
                self.__class__.__name__, 'failed_to_pack_aerial_data'
            ).inc()
            logging.warning('Failed to pack aerial data for scene %s', scene_id)
            return
        yield f"{scene_id}/{query['image_name']}", q_data
    else:
      yield scene_id, packed_data


@functools.cache
def _get_shard_lengths(
    info: tfds.core.dataset_info.DatasetInfo, split_name: str
) -> list[int]:
  """Returns the shard lengths for a given split."""
  with gfile.Open(f'{info.data_dir}/dataset_info.json') as f:
    d = json.load(f)
  (split,) = [split for split in d['splits'] if split['name'] == split_name]
  return [int(shard_length) for shard_length in split['shardLengths']]


def _get_offset(
    info: tfds.core.dataset_info.DatasetInfo, split: str, shard: int
) -> int:
  """Returns shard offset (the total number of examples in previous shards)."""
  shard_lengths = _get_shard_lengths(info, split)
  assert shard < len(shard_lengths), (shard, len(shard_lengths))
  return sum(shard_lengths[:shard])


def tfds_id_to_data_source_index(
    info: tfds.core.dataset_info.DatasetInfo, tfds_id: bytes
) -> int:
  """Converts a tfds_id to a data source index."""
  # A tfds_id looks like this:
  # b'examples-{split}.array_record-{shard_index}-of-01106__{local_index}'
  tfds_id = tfds_id.decode('utf-8')
  file, shard_info = tfds_id.split('.array_record-')
  split_name = file.split('-')[1]
  shard = int(shard_info.split('-')[0])
  idx_in_shard = int(shard_info.split('__')[1])
  return _get_offset(info, split_name, shard) + idx_in_shard


def frustums_latlng_from_example(
    example: Example, frustum_depth: float = 50
) -> FloatArray['V N 2']:
  """Returns the latlng coordinates of the camera frustums."""
  qv = example['query']['views']
  camera = qv['camera']
  frustums_ll = []
  for i, cam_t_w in enumerate(qv['cam_t_w']):
    cam_p_frustum = geo.camera_frustum_vertices(
        frustum_depth, camera['fov_deg'][i]
    )
    ecef_t_cam = example['ecef_t_w'] @ np.linalg.inv(cam_t_w)
    frustums_ll.append(
        geo.latlng_from_ecef(geo.transform_points(ecef_t_cam, cam_p_frustum))
    )

  return np.stack(frustums_ll)


def center_latlng_from_example(example: Example) -> FloatArray['2']:
  if 'center_ll' in example:
    return example['center_ll']
  qv = example['query']['views']
  centers_ecef = (
      example['ecef_t_w'] @ np.linalg.inv(np.stack(qv['cam_t_w'], 0))
  )[:, :3, -1]
  centers_ll = geo.latlng_from_ecef(np.stack(centers_ecef))
  return np.mean(centers_ll, axis=0)


def center_ecef_from_example(example: Example) -> FloatArray['3']:
  qv = example['query']['views']
  centers_ecef = (
      example['ecef_t_w'][None] @ np.linalg.inv(np.stack(qv['cam_t_w'], 0))
  )[:, :3, -1]
  return np.mean(centers_ecef, axis=0)


def get_nested(example: Example, key: str) -> Any:
  nested_keys = key.split('.')
  values = example
  for k in nested_keys:
    values = values[k]
  return values


def s2cell_token_at_level(example: Example, level: int) -> bytes:
  s2cell_id = s2.S2CellId.from_token(example['s2cell'])
  if s2cell_id.level < level:
    center_ll = center_latlng_from_example(example)
    return (
        s2.S2CellId.from_lat_lng(s2.S2LatLng.from_degrees(*center_ll.tolist()))
        .parent_at_level(level)
        .token()
    )
  else:
    return s2cell_id.parent_at_level(level).token()


class CountExamples(beam.DoFn):
  """Simple DoFn that counts and emits examples."""

  def __init__(self, label: str):
    self.label = label

  def process(self, serialized_example: bytes) -> Iterable[bytes]:
    beam.metrics.Metrics.counter(
        self.__class__.__name__, f'num_{self.label}_examples'
    ).inc()
    yield serialized_example


class DumpToJson(beam.PTransform):
  """Dumps a PCollection of (key, value) to a json file."""

  def __init__(self, output_path: str, num_shards: int | None = 1):
    self.output_path = output_path
    self.num_shards = num_shards

  def expand(
      self, pcoll: beam.PCollection[tuple[Any, Any]]
  ) -> beam.PCollection[tuple[Any, Any]]:
    _ = (
        pcoll
        | beam.combiners.ToDict()
        | beam.Map(json.dumps)
        | beam.io.WriteToText(
            self.output_path,
            num_shards=self.num_shards,
            shard_name_template='',
        )
    )
    return pcoll


class CountPerKeyAndDumpToJson(beam.PTransform):
  """Counts the number of elements per key and writes the counts to a json file."""

  def __init__(self, output_path: str, num_shards: int = 1):
    self.output_path = output_path
    self.num_shards = num_shards

  def expand(self, pcoll: beam.PCollection[Any]) -> beam.PCollection[Any]:
    _ = (
        pcoll
        | beam.combiners.Count.PerKey()
        | DumpToJson(self.output_path, self.num_shards)
    )
    return pcoll


def serialize_examples(
    named_examples: beam.PCollection[Tuple[str, types.DataDict]],
    features_dict: tfds.features.FeaturesDict,
    split_name: str,
) -> beam.PCollection[bytes]:
  """Serializes named examples to a specified format."""
  return (
      named_examples
      | f'RemoveKeys_{split_name}' >> beam.Map(lambda x: x[1])
      | f'SerializeExamples_{split_name}'
      >> beam.Map(features_dict.serialize_example)
      | f'CountAndEmitExamples_{split_name}'
      >> beam.ParDo(CountExamples(split_name))
  )


def serialize_and_write_examples(
    named_examples: beam.PCollection[Tuple[str, types.DataDict]],
    features_dict: tfds.features.FeaturesDict,
    output_dir: str,
    num_shards: int | None = None,
    split_name: str = 'train',
) -> tfds.features.FeaturesDict:
  """Serializes and writes packed scenes."""
  # WriteToArrayRecord: num_shards and @N cannot be both specified.
  serialized = serialize_examples(named_examples, features_dict, split_name)
  serialized = serialized | f'Reshuffle_{split_name}' >> beam.Reshuffle()
  output_suffix = f'examples-{split_name}.array_record'
  if num_shards is None:
    output_suffix = f'{output_suffix}@*'
  _ = (
      serialized
      | f'WriteToArrayRecord_{split_name}'
      >> array_record_io.WriteToArrayRecord(
          os.path.join(output_dir, output_suffix),
          # group_size:1 needed for random access
          # go/grain-g3doc/internal/performance/array_record#verifying-group-size
          record_writer_options='group_size:1',
          num_shards=num_shards,
      )
  )

  return features_dict


def pack_serialize_and_write_scenes(
    scenes: beam.PCollection[Tuple[str, types.Scene]],
    data_config: DataConfig,
    pose_snapshot_id: str,
    output_dir: str,
    num_shards: int | None = None,
    split_name: str = 'train',
    num_views_query: int | None = None,
    num_views_map: int | None = None,
    num_collections_map: int | None = None,
) -> Tuple[
    beam.PCollection[Tuple[str, types.DataDict]],
    tfds.features.FeaturesDict,
]:
  """Convenience method to pack, serialize and write scenes."""

  features_dict = tfds.features.FeaturesDict(
      data_config.scene_spec(
          num_views_query=num_views_query,
          num_views_map=num_views_map,
          num_collections_map=num_collections_map,
      )
  )

  scenes = scenes | f'ReshuffleScenes_{split_name}' >> beam.Reshuffle()
  packed_scenes = scenes | f'PackScenes_{split_name}' >> beam.ParDo(
      PackScene(data_config, pose_snapshot_id)
  )

  return packed_scenes, serialize_and_write_examples(
      packed_scenes,
      features_dict,
      output_dir,
      num_shards=num_shards,
      split_name=split_name,
  )


def pack_serialize_and_write_collections(
    view_collections: beam.PCollection[Tuple[str, types.ViewCollection]],
    data_config: DataConfig,
    pose_snapshot_id: str,
    output_dir: str,
    num_shards: int | None = None,
    split_name: str = 'train',
) -> Tuple[
    beam.PCollection[Tuple[str, types.DataDict]],
    tfds.features.FeaturesDict,
]:
  """Convenience method to pack, serialize and write view collections."""

  features_dict = tfds.features.FeaturesDict(data_config.collection_spec())

  packed_collections = (
      view_collections
      | f'PackCollections_{split_name}'
      >> beam.ParDo(PackViewCollection(data_config, pose_snapshot_id))
  )

  return packed_collections, serialize_and_write_examples(
      packed_collections,
      features_dict,
      output_dir,
      num_shards=num_shards,
      split_name=split_name,
  )
