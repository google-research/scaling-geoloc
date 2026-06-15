# Copyright 2026 Google LLC.
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
import dataclasses
import datetime
from typing import Any, Iterable

from absl import logging
import apache_beam as beam
from etils.array_types import FloatArray
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from scaling_geoloc.data import export
from scaling_geoloc.data import geo
from scaling_geoloc.data import types
from scaling_geoloc.utils import misc
from google3.location.visualmapping.experimental.lemming.data import earth_engine
from google3.location.visualmapping.experimental.lemming.data import geo as lemming_geo
from google3.location.visualmapping.experimental.lemming.utils import geometry
from google3.location.visualmapping.experimental.lemming.utils import grids
from google3.util.geometry.python import s2

serialize_and_write_examples = export.serialize_and_write_examples
center_ecef_from_example = export.center_ecef_from_example
s2cell_token_at_level = export.s2cell_token_at_level
center_latlng_from_example = export.center_latlng_from_example

# Define what is part of the public API
__all__ = [
    'serialize_and_write_examples',
    'DSMDataConfig',
    'center_ecef_from_example',
    'FetchDSMImages',
    's2cell_token_at_level',
    'center_latlng_from_example',
    'CropsFromDSM',
]

FeatureSpecElem = tfds.features.FeatureConnector | tf.dtypes.DType
FeatureSpecs = tfds.typing.TreeDict[FeatureSpecElem]
DSMTile = tuple[FloatArray['H W'], datetime.datetime, float]
Example = Any


@dataclasses.dataclass
class DSMDataConfig:
  """Configuration for dsm data from earth engine."""

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
  resolution_meters: float | None = None  # Usually inferred from s2cell.
  min_ratio_valid: float = 0.8

  def data_spec(self) -> FeatureSpecs:
    """Returns the feature specs for the data."""
    num_tiles = len(self.extent_factors)
    return {
        'dsm': tfds.features.Sequence(
            tfds.features.Tensor(
                shape=(self.image_size, self.image_size),
                dtype=tf.float32,
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


def get_dsm_tile(
    ecef_t_ll: geometry.Transform3D,
    image_size: tuple[int, int],
    resolution_meters: float,
    offset_meters: FloatArray['2'] | None = None,
    rotation_angle_degrees: float | None = None,
    make_upright: bool = True,
    min_ratio_valid: float = 0.8,
) -> tuple[FloatArray['4 4'], Sequence[DSMTile]]:
  """Get dsm tile."""
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
  try:
    ee_query = earth_engine.EarthEngineQuery(grid, ecef_t_grid)
    ret = ee_query.get_dsm(min_ratio_valid=min_ratio_valid)
  except Exception as e:  # pylint: disable=broad-except
    logging.exception('Failed to load EarthEngineQuery: %s', repr(e))
    return None, []
  if not ret:
    return None, []
  all_pixels = []
  pixels, _, date, pixel_size_meters = ret
  if make_upright:
    pixels = np.swapaxes(pixels, 0, 1)[::-1]
  all_pixels.append((pixels, date, pixel_size_meters))
  return ecef_t_grid.to_4x4matrix(), all_pixels


class FetchDSMImages(beam.DoFn):
  """Fetch dsm imagery."""

  def __init__(self, data_config: DSMDataConfig):
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

  def pack_dsm_data_for_location(
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
      ecef_t_grid, dsm_collections = get_dsm_tile(
          ecef_t_ll,
          (self.config.image_size, self.config.image_size),
          resolution,
          offset_meters=offset_meters,
          rotation_angle_degrees=rotation_angle_degrees,
          make_upright=self.config.make_upright,
          min_ratio_valid=self.config.min_ratio_valid,
      )
      if not dsm_collections:
        beam.metrics.Metrics.counter(self.__class__.__name__, 'failed').inc()
        return None
      if i == 0:
        grid_offset = self.config.image_size * resolution / 2
        grid_t_center = geometry.Transform3D(
            R=np.eye(3), t=np.array([grid_offset] * 2 + [0.0])
        ).to_4x4matrix()
        ecef_t_center = ecef_t_grid @ grid_t_center
        extra_data['center_ll'] = geo.latlng_from_ecef(
            ecef_t_center[:3, -1][None]
        )[0]
        extra_data['num_collections'] = len(dsm_collections)
      for image, date, pixel_size_meters in dsm_collections:
        data['extent'].append(image_extent)
        data['resolution'].append(resolution)
        data['ecef_t_grid'].append(ecef_t_grid)
        data['dsm'].append(image)
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

  def pack_dsm_data(
      self,
      s2cell_token: str,
  ) -> types.DataDict | None:
    ecef_t_ll = lemming_geo.get_local_level_frame(
        s2.S2CellId.from_token(s2cell_token.encode('utf-8')).to_lat_lng()
    )
    return self.pack_dsm_data_for_location(ecef_t_ll, s2cell_token=s2cell_token)

  def process(self, s2cell_token: str) -> Iterable[tuple[str, types.DataDict]]:
    beam.metrics.Metrics.counter(self.__class__.__name__, 's2cells').inc()

    for _ in range(self.config.num_samples):
      dsm_data = self.pack_dsm_data(s2cell_token)
      if dsm_data is None:
        return
      else:
        yield s2cell_token, dsm_data


class CropsFromDSM(beam.DoFn):
  """Crops dsm images."""

  def __init__(self, data_config: DSMDataConfig, re_center: bool = False):
    self.config = data_config
    self.re_center = re_center

  def setup(self):
    self.rng = np.random.default_rng(self.config.seed)

  def generate_sample(
      self,
      dsm_data: types.DataDict,
      image_idx: int,
      at_ecef: FloatArray['3'] | None = None,  # Sample around this ECEF point.
  ) -> types.DataDict:
    dsm = dsm_data['dsm'][image_idx]
    dsm = jnp.expand_dims(dsm, 2)
    if dsm_data['upright']:
      dsm = np.swapaxes(dsm[::-1], 0, 1)
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

    ecef_t_imagegrid = dsm_data['ecef_t_grid'][image_idx]
    if at_ecef is not None:
      offset_meters += (
          geo.transform_points(np.linalg.inv(ecef_t_imagegrid), at_ecef[None])[
              0, :2
          ]
          - dsm_data['extent'][image_idx] / 2.0
      )
    for i, factor in enumerate(self.config.extent_factors):
      image_extent = extent_meters * factor
      resolution = image_extent / self.config.image_size
      image_t_crop, image_crop, valid_pixels = misc.crop_aerial_image(
          dsm,
          dsm_data['resolution'][image_idx],
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
      data['dsm'].append(jnp.squeeze(image_crop, axis=2))
      data['timestamp'].append(dsm_data['timestamp'][image_idx])
      data['capture_resolution'].append(
          dsm_data['capture_resolution'][image_idx]
      )

    return {
        'rotation_angle_degrees': rotation_angle_degrees,
        'offset_meters': offset_meters,
        'upright': self.config.make_upright,
        's2cell': dsm_data['s2cell'],
        **data,
        **extra_data,
    }

  def process(
      self, dsm_data: types.DataDict
  ) -> Iterable[tuple[str, types.DataDict]]:
    num_images = len(dsm_data['dsm'])

    image_for_sample = self.rng.choice(
        num_images, self.config.num_samples
    ).tolist()
    if self.re_center:
      s2cell_center_ll = s2.S2CellId.from_token(dsm_data['s2cell']).to_lat_lng()
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
      s2cell = dsm_data['s2cell'].decode('utf-8')
      yield f'{s2cell}_sample{i}_{image_idx}', self.generate_sample(
          dsm_data, image_idx, at_ecef=s2cell_center_ecef
      )
