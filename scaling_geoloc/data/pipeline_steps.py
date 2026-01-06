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

"""Library functions for extracting geo data."""

import collections
from collections.abc import Callable, Sequence
import dataclasses
import functools
import json
import re
from typing import Any, Iterable, Tuple

from absl import logging
import apache_beam as beam
from etils.array_types import BoolArray
from etils.array_types import FloatArray
import jaxcam
import numpy as np

from google3.cityblock.base import run_tag_pb2
from google3.cityblock.base import vehicles_pb2
from google3.cityblock.exposure.python import exposure_model
from google3.geo.imagery.streetview.store.client.python import capture
from google3.geo.imagery.streetview.store.client.python import client_request_options
from google3.geo.imagery.streetview.store.client.python import svms_client
from google3.geo.imagery.streetview.store.client.python import svms_client_util
from google3.geo.imagery.streetview.store.proto import image_type_pb2
from google3.geo.imagery.streetview.store.proto import request_options_pb2
from google3.geo.imagery.streetview.store.proto import resources_pb2 as resources
from google3.geo.imagery.streetview.store.shared.python import ids as gsv_ids
from scaling_geoloc.data import geo
from scaling_geoloc.data import matching
from scaling_geoloc.data import types
from scaling_geoloc.utils import misc
from google3.location.visualmapping.experimental.lemming.data.loading.python import data_loading_utils
from google3.location.visualmapping.experimental.utils.python import cbdb_utils
from google3.pyglib import gfile
from google3.util.geometry.python import s2
from google3.util.task import codes_pb2
from google3.util.task.python import error


def write_metrics(
    pipeline_results: beam.runners.runner.PipelineResult,
    output_dir: str,
    suffix: str = '',
) -> None:
  """Turns *all* beam metrics into a json-encoded text file."""
  metrics = pipeline_results.metrics().query()

  def key_to_string(k: beam.metrics.execution.MetricKey) -> str:
    return ':'.join((k.step, k.metric.namespace, k.metric.name))

  # For distributions, we map a result to a dictionary of its aggregates.
  def distribution_aggregates(
      result: beam.metrics.cells.DistributionResult,
  ):
    return {
        'min': result.min,
        'max': result.max,
        'count': result.count,
        'sum': result.sum,
        'mean': result.mean,
    }

  counter_keys = [key_to_string(c.key) for c in metrics['counters']]
  if len(set(counter_keys)) < len(counter_keys):
    logging.warning(
        'Found duplicate counter keys! n_unique=%d, n_total=%d',
        len(set(counter_keys)),
        len(counter_keys),
    )
  counters = {key_to_string(c.key): c.result for c in metrics['counters']}

  distributions = {  # pylint: disable = g-complex-comprehension
      key_to_string(d.key): distribution_aggregates(d.result)
      for d in metrics['distributions']
  }

  results = {
      'counters': counters,
      'distributions': distributions,
  }

  with gfile.Open(f'{output_dir}/metrics{suffix}.json', 'w') as f:
    f.write(json.dumps(results, indent=2))


def runs_crossing_s2cells(
    s2cell_tokens: Sequence[str],
) -> Sequence[cbdb_utils.CbdbRunMetadata]:
  """Extract run metadata for runs that cross the given S2 cells."""
  s2cell_ids = [
      s2.S2CellId.from_token(token.encode('utf-8')) for token in s2cell_tokens
  ]
  runs = cbdb_utils.GetRunMetadataFromS2CellId(s2cell_ids)
  logging.info('Found %d runs', len(runs))

  return runs


class ExpandS2Cells(beam.DoFn):
  """Expand S2 cells to a given level."""

  def __init__(self, target_level: int):
    self.target_level = target_level

  def process(self, s2cell_token: str) -> Iterable[str]:
    beam.metrics.Metrics.counter(
        self.__class__.__name__, 'num_s2cells_before_expansion'
    ).inc()
    parent = s2.S2CellId.from_token(s2cell_token.encode('utf-8'))
    if parent.level > self.target_level:
      parent = parent.parent_at_level(self.target_level)
    child = parent.child_begin_at_level(self.target_level)
    while child != parent.child_end_at_level(self.target_level):
      beam.metrics.Metrics.counter(
          self.__class__.__name__, 'num_s2cells_after_expansion'
      ).inc()
      yield child.token().decode('utf-8')
      child = child.next()


class FilterRunsByType(beam.DoFn):
  """Discard runs of the wrong type."""

  @dataclasses.dataclass
  class Config:
    scene_types: Sequence[str] = ('OUTDOOR', 'INDOOR')
    vehicle_types: Sequence[str] = ('CAR', 'TREKKER')
    rosette_types: Sequence[str] = ('H1',)

  def __init__(
      self,
      scene_types: Sequence[str],
      vehicle_types: Sequence[str],
      rosette_types: Sequence[str],
  ):
    self.scene_types = set(
        [run_tag_pb2.RunTagProto.Scene.Value(name) for name in scene_types]
    )
    self.vehicle_types = set([
        vehicles_pb2.VehicleIdRange.VehicleType.Value(name)
        for name in vehicle_types
    ])
    self.rosette_types = set(rosette_types)

  def process(self, run: types.RunTuple) -> Iterable[types.RunTuple]:
    run_name, run_data = run
    if run_data.rosette_type not in self.rosette_types:
      beam.metrics.Metrics.counter(
          self.__class__.__name__, 'wrong_rosette'
      ).inc()
      return
    if run_data.scene_type not in self.scene_types:
      beam.metrics.Metrics.counter(self.__class__.__name__, 'wrong_scene').inc()
      return
    if run_data.vehicle_type not in self.vehicle_types:
      beam.metrics.Metrics.counter(
          self.__class__.__name__, 'wrong_vehicle'
      ).inc()
      return
    beam.metrics.Metrics.counter(self.__class__.__name__, 'selected').inc()
    yield (run_name, run_data)


class SegmentsFromRuns(beam.DoFn):
  """Extract segments from runs."""

  def __init__(self, pose_snapshot_id: str):
    self.pose_snapshot_id = pose_snapshot_id

  def setup(self):
    self.dlutils = data_loading_utils.DataLoadingUtils.Create()

  def process(self, run: types.RunTuple) -> Iterable[types.SegmentTuple]:
    run_name, run_data = run
    try:
      run_segment = self.dlutils.GetPanoLocations(
          run_name, self.pose_snapshot_id
      )
    except error.StatusNotOk as err:
      if err.code == codes_pb2.NOT_FOUND:
        beam.metrics.Metrics.counter(
            self.__class__.__name__, 'pose_for_run_not_found'
        ).inc()
        logging.error('Skip run %s because of error: %s', run_name, err.message)
        return  # Exit to propagate flume work-unit cancellations.
      raise

    ecef_positions, timestamps, pano_names = [], [], []

    for pano in run_segment.panos:
      ecef_positions.append(pano.ecef_position)
      timestamps.append(pano.pano_id.capture_timestamp_sec())
      pano_names.append(pano.pano_id.ToString())

    ecef_positions = np.array(ecef_positions)
    segment = types.Segment(
        run_name=run_name,
        s2cell=geo.enclosing_s2cell_token(geo.latlng_from_ecef(ecef_positions)),
        pano_names=np.array(
            [pano.pano_id.ToString() for pano in run_segment.panos]
        ),
        ecef_positions=ecef_positions,
        timestamps=np.array(
            [pano.pano_id.capture_timestamp_sec() for pano in run_segment.panos]
        ),
    )

    yield (run_name, run_data, segment)


def split_into_continuous_subsegments(
    sorted_id_list: list[int],
) -> Sequence[Sequence[int]]:
  """Split a list of sorted ids into continuous subsegments."""
  subsegments = [sorted_id_list[:1]]
  for idx in sorted_id_list[1:]:
    if idx - subsegments[-1][-1] == 1:
      subsegments[-1].append(idx)
    else:
      subsegments.append([idx])
  return subsegments


class SplitSegmentsByS2Cell(beam.DoFn):
  """Split segments at s2cell boundaries on a fixed target level.

  Remove parts of segments that are outside the required s2cells.
  """

  def __init__(
      self,
      s2cell_tokens: Sequence[str],
      target_level: int,
      exact: bool = True,
  ):
    self.s2cell_tokens = s2cell_tokens

    self.target_level = target_level
    self.exact = exact

  def setup(self):
    s2cell_ids = [
        s2.S2CellId.from_token(token.encode('utf-8'))
        for token in self.s2cell_tokens
    ]
    self.s2cell_regions = {
        token: s2.S2Cell(s2cell_id)
        for token, s2cell_id in zip(self.s2cell_tokens, s2cell_ids)
    }

    self.s2cell_union = s2.S2CellUnion(s2cell_ids)

    if not self.exact:
      # we use the rectangular bound instead
      self.s2cell_regions = {
          token: s2cell.rect_bound()
          for token, s2cell in self.s2cell_regions.items()
      }

  def process(
      self, name_rundata_segment: types.SegmentTuple
  ) -> Iterable[types.SegmentTuple]:
    segment_name, run_data, segment = name_rundata_segment

    latlng_positions = geo.latlng_from_ecef(segment.ecef_positions)
    s2_positions = [
        s2.S2LatLng.from_degrees(lat, lng).to_point()
        for (lat, lng) in latlng_positions
    ]

    ids_by_cell_token = collections.defaultdict(list)

    for i, s2_position in enumerate(s2_positions):
      if not self.s2cell_union.contains_point(s2_position):
        continue
      s2_cell_id = s2.S2CellId.from_point(s2_position).parent_at_level(
          self.target_level
      )
      ids_by_cell_token[s2_cell_id.token()].append(i)

    for s2cell_token_bytes, ids_inside_s2cell in ids_by_cell_token.items():
      s2cell_token = s2cell_token_bytes.decode()
      beam.metrics.Metrics.counter(
          self.__class__.__name__, 's2cells_with_crossed_segments'
      ).inc()
      # force continuousness
      subsegments = split_into_continuous_subsegments(ids_inside_s2cell)
      for subsegment in subsegments:
        beam.metrics.Metrics.counter(
            self.__class__.__name__, 'new_segment_for_s2cell'
        ).inc()
        subsegment_name = (
            segment_name + f'/{s2cell_token}/{subsegment[0]}:{subsegment[-1]}'
        )
        yield (
            subsegment_name,
            run_data,
            types.Segment(
                run_name=segment.run_name,
                s2cell=s2cell_token,
                pano_names=segment.pano_names[subsegment],
                ecef_positions=segment.ecef_positions[subsegment],
                timestamps=segment.timestamps[subsegment],
            ),
        )


class SplitSegmentsByDuration(beam.DoFn):
  """Split segments into smaller segments."""

  def __init__(self, duration_sec: float):
    self.duration_sec = duration_sec

  def process(
      self, name_rundata_segment: types.SegmentTuple
  ) -> Iterable[types.SegmentTuple]:
    run_name, run_data, segment = name_rundata_segment
    start = 0

    for i, timestamp in enumerate(segment.timestamps):
      if timestamp - segment.timestamps[start] > self.duration_sec:
        yield run_name + f'/{start}:{i}', run_data, types.Segment(
            run_name=segment.run_name,
            s2cell=segment.s2cell,
            pano_names=segment.pano_names[start:i],
            ecef_positions=segment.ecef_positions[start:i],
            timestamps=segment.timestamps[start:i],
        )
        start = i

    if start + 1 < len(segment.timestamps):
      beam.metrics.Metrics.counter(
          self.__class__.__name__, 'num_segments'
      ).inc()
      yield (
          run_name + f'/{start}:{len(segment.timestamps)}',
          run_data,
          types.Segment(
              run_name=segment.run_name,
              s2cell=segment.s2cell,
              pano_names=segment.pano_names[start:],
              ecef_positions=segment.ecef_positions[start:],
              timestamps=segment.timestamps[start:],
          ),
      )

    return


class SubsampleSegments(beam.DoFn):
  """Subsample segments to N panos with a minimum spacing in time and space."""

  def __init__(
      self,
      max_panos: int,
      min_spacing_sec: float,
      min_spacing_meters: float | None = None,
  ):
    self.max_panos = max_panos
    self.min_spacing_sec = min_spacing_sec
    self.min_spacing_meters = min_spacing_meters

  def process(
      self, name_rundata_segment: types.SegmentTuple
  ) -> Iterable[types.SegmentTuple]:
    segment_name, run_data, segment = name_rundata_segment

    num_panos = len(segment.timestamps)
    if num_panos < self.max_panos:
      beam.metrics.Metrics.counter(
          self.__class__.__name__, 'too_few_panos_in_segment'
      ).inc()
      return
    duration = segment.timestamps[-1] - segment.timestamps[0]
    if duration < self.max_panos * self.min_spacing_sec:
      beam.metrics.Metrics.counter(
          self.__class__.__name__, 'segment_too_short_sec'
      ).inc()
      return

    if self.min_spacing_meters is None:
      # This assumes that the panos are approx. uniformly spaced in time.
      segment_idxs = np.round(
          np.linspace(0, num_panos - 1, self.max_panos)
      ).astype(int)
    else:
      # Subsample segments spatially.
      segment_idxs = [0]
      for i in range(1, num_panos):
        distance_to_last_pano = np.linalg.norm(
            segment.ecef_positions[i] - segment.ecef_positions[segment_idxs[-1]]
        )
        if distance_to_last_pano > self.min_spacing_meters:
          segment_idxs.append(i)
          if len(segment_idxs) == self.max_panos:
            break
      # If we have too few panos, we do not return a sample.
      if len(segment_idxs) < self.max_panos:
        beam.metrics.Metrics.counter(
            self.__class__.__name__, 'too_few_panos_after_spatial_subsampling'
        ).inc()
        return

    beam.metrics.Metrics.counter(
        self.__class__.__name__, 'num_valid_segments'
    ).inc()
    yield (
        segment_name,
        run_data,
        types.Segment(
            run_name=segment.run_name,
            s2cell=segment.s2cell,
            pano_names=segment.pano_names[segment_idxs],
            ecef_positions=segment.ecef_positions[segment_idxs],
            timestamps=segment.timestamps[segment_idxs],
        ),
    )


class SampleKFromList(beam.DoFn):
  """Sample K elements from a list."""

  def __init__(
      self,
      seed: int,
      algorithm: str,
      k: int,
      gather_fn: Callable[[Any], FloatArray['D']] = lambda x: x,
      norm_fn: Callable[
          [str, FloatArray['N D']],
          tuple[FloatArray['N D'], BoolArray['N'] | None],
      ] = lambda name, x: x,
      residuals_fn: Callable[
          [FloatArray['N D'], FloatArray['N D']], FloatArray['N K']
      ] = misc.l2_norm,
  ):
    self.k = k
    self.algorithm = algorithm
    self.seed = seed

    self.gather_fn = gather_fn if gather_fn is not None else lambda x: x
    self.norm_fn = norm_fn if norm_fn is not None else lambda x: x
    self.residuals_fn = residuals_fn

  def setup(self):
    self.rng = np.random.default_rng(self.seed)

  def process(
      self, name_and_elements: tuple[str, Iterable[Any]]
  ) -> Iterable[tuple[str, Iterable[Any]]]:
    name, elements = name_and_elements
    elements = list(elements)
    if len(elements) <= self.k:
      yield name, elements
      return

    if self.algorithm == 'uniform' or self.algorithm == 'naive':
      # Simple uniform sampling. No need to gather and normalize.
      indices = self.rng.choice(len(elements), size=self.k, replace=False)
      yield name, [elements[idx] for idx in indices.tolist()]
      return

    elif self.algorithm.startswith('farthest_point_sampling'):
      assert self.algorithm in [
          'farthest_point_sampling',
          'farthest_point_sampling_rng',
      ]
      points_nd = np.stack(
          [self.gather_fn(element) for element in elements], axis=0
      )
      points_nd, valid_n = self.norm_fn(name, points_nd)
      rng = (
          None
          if self.algorithm == 'farthest_point_sampling'
          else self.rng.integers(low=0, high=2**31 - 1, size=1).item()
      )
      _, valid, indices = misc.farthest_point_sampling(
          points_nd[None],
          k=self.k,
          closest_first=True,
          rng=rng,
          residuals_fn=self.residuals_fn,
          pc_mask=valid_n[None] if valid_n is not None else None,
      )
      indices = np.array(indices[0][valid[0]])
      yield name, [elements[idx] for idx in indices.tolist()]
    else:
      raise ValueError(f'Unknown algorithm: {self.algorithm}')


class SplitSegmentsIntoSinglePanos(beam.DoFn):
  """Split segments into single-panos, optionally spaced by a minimum distance."""

  def __init__(self, min_spacing_meters: float | None = None):
    self.min_spacing_meters = min_spacing_meters

  def process(
      self, name_rundata_segment: types.SegmentTuple
  ) -> Iterable[types.SegmentTuple]:
    run_name, run_data, segment = name_rundata_segment

    num_panos = len(segment.timestamps)
    beam.metrics.Metrics.counter(
        self.__class__.__name__,
        'num_panos_before_subsampling',
    ).inc(num_panos)
    last = 0
    for i in range(num_panos):
      # Compute distance to last pano.
      if self.min_spacing_meters is not None:
        distance_to_last_pano = np.linalg.norm(
            segment.ecef_positions[i] - segment.ecef_positions[last]
        )
        if i > 0 and distance_to_last_pano < self.min_spacing_meters:
          # Skip panos that are too close to the last pano.
          continue
        else:
          # Update last pano index.
          last = i
      yield run_name + f'/{i}:{i+1}', run_data, types.Segment(
          run_name=segment.run_name,
          s2cell=segment.s2cell,
          pano_names=segment.pano_names[i : i + 1],
          ecef_positions=segment.ecef_positions[i : i + 1],
          timestamps=segment.timestamps[i : i + 1],
      )
      beam.metrics.Metrics.counter(
          self.__class__.__name__,
          'num_panos_after_subsampling',
      ).inc()
    return


def views_from_pano(
    pano_metadata: data_loading_utils.PanoMetadata,
) -> dict[types.ViewKey, types.View]:
  """Extract views from pano metadata."""
  exposure = exposure_model.ExposureModel(pano_metadata.exposure)

  view_keys = [
      types.ViewKey(
          image_name=image_id.ToString(),
          rosette_type=pano_metadata.exposure.rosette.type,
      )
      for image_id in pano_metadata.image_ids
  ]

  views = {}
  for view_key in view_keys:
    camera_index = view_key.camera_index
    # We use the center pixel exposure time to obtain the camera pose
    center_pixel = 0.5 * np.array(
        [exposure.width(camera_index), exposure.height(camera_index)]
    )
    exposure_time = exposure.pixel_exposure_time(camera_index, center_pixel)
    cam_t_w = exposure.camera_from_world(camera_index, exposure_time)
    intrinsics = types.Intrinsics.from_exposure_pb2(
        pano_metadata.exposure.rosette.camera[camera_index].intrinsics
    )
    views[view_key] = types.View(
        image_name=view_key.image_name,
        cam_t_w=cam_t_w,
        camera=intrinsics,
        timestamp=exposure_time,
    )
  return views


def stitched_view_from_capture(
    capture_i: capture.Capture,
) -> tuple[types.ViewKey, types.View]:
  """Extract stitched view metadata from a capture."""
  exposure_proto = capture_i.GetExposure()
  exposure = exposure_model.ExposureModel(exposure_proto)
  if len(capture_i.GetImages()) != 1:
    raise ValueError(
        'Stitched view must have exactly one image, but has'
        f' {len(capture_i.GetImages())}.'
    )
  image = capture_i.GetImages()[0]
  if image.GetImageType() != image_type_pb2.STITCHED:
    raise ValueError(
        'Stitched view must have image type STITCHED, but has type'
        f' {image_type_pb2.ImageType.Name(image.GetImageType())}.'
    )
  image_size = image.GetImageDimensions()
  height = image_size.height_pixels
  width = image_size.width_pixels
  capture_time = exposure.canonical_time()
  rosette_t_w = exposure.rosette_from_world(capture_time)
  cam_t_rosette = np.eye(4)
  cam_t_rosette[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).T
  cam_t_w = cam_t_rosette @ rosette_t_w  # make the camera forward-facing.
  intrinsics = types.Intrinsics.spherical(width, height)
  view_key = types.ViewKey(
      image_name=image.id().ToString(),
      rosette_type=exposure_proto.rosette.type,
  )
  view = types.View(
      image_name=view_key.image_name,
      cam_t_w=cam_t_w,
      camera=intrinsics,
      timestamp=capture_time,
  )
  return view_key, view


class ViewsFromSegments(beam.DoFn):
  """Extract metadata for segments."""

  def __init__(self, pose_snapshot_id: str, get_stitched_panos: bool = False):
    self.pose_snapshot_id = pose_snapshot_id
    self.get_stitched_panos = get_stitched_panos

  def setup(self):
    self.dlutils = data_loading_utils.DataLoadingUtils.Create()
    self.svms: svms_client.SvmsClient = (
        svms_client_util.CreateBatchSvmsClientFromTarget(
            'blade:street-view-metadata-service-staging'
        )
    )

  # process individually
  def process(
      self, name_rundata_segment: types.SegmentTuple
  ) -> Iterable[types.ViewCollection]:
    segment_name, run_data, segment = name_rundata_segment
    pano_capture_ids = [
        gsv_ids.CaptureId.FromString(pano) for pano in segment.pano_names
    ]
    if self.get_stitched_panos:
      request_options = (
          client_request_options.ClientRequestOptions()
          .IncludeMetadata([
              resources.MetadataPiece.METADATA_IMAGE_ID,
              resources.MetadataPiece.METADATA_CAPTURE_EXPOSURE,
              resources.MetadataPiece.METADATA_IMAGE_TYPE,
              resources.MetadataPiece.METADATA_IMAGE_DIMENSIONS,
              resources.MetadataPiece.METADATA_IMAGE_IS_SKY_CAMERA,
          ])
          .IncludeImageTypes([image_type_pb2.STITCHED])
      )
      pose_selection = request_options_pb2.PoseSelection()
      pose_selection.pose_snapshot_id = self.pose_snapshot_id
      request_options.SetPoseSelection(pose_selection)

      result = self.svms.BatchGetCapture(pano_capture_ids, request_options)
      view_dict = {}
      for collect in result.GetCollects():
        data_stream = collect.GetMainRosetteDataStream()
        for capture_i in data_stream.GetCaptures():
          view_key, view = stitched_view_from_capture(capture_i)
          view_dict[view_key] = view
      metadata = types.ViewCollectionMetadata(
          run_name=segment.run_name,
          s2cell=segment.s2cell,
          pano_split='all',
          num_panos=len(segment.pano_names),
          num_views=len(segment.pano_names),
          **dataclasses.asdict(run_data),
      )
      yield segment_name, metadata, view_dict
    else:
      pano_metadatas = self.dlutils.GetPanoMetadataBatch(
          pano_capture_ids,
          self.pose_snapshot_id,
      )

      all_views = {}
      for pano_metadata in pano_metadatas:
        all_views.update(views_from_pano(pano_metadata))

      scene_id = segment_name
      metadata = types.ViewCollectionMetadata(
          run_name=segment.run_name,
          s2cell=segment.s2cell,
          pano_split='all',
          num_panos=len(segment.pano_names),
          num_views=len(all_views),
          **dataclasses.asdict(run_data),
      )
      beam.metrics.Metrics.counter(
          self.__class__.__name__, 'num_view_collections'
      ).inc()
      yield scene_id, metadata, all_views


class SampleViewsFromStitchedPano(beam.DoFn):
  """Sample views from a stitched pano."""

  @dataclasses.dataclass
  class Config:
    height: int = 448
    width: int = 448
    roll_range: tuple[float, float] = (-10.0, 10.0)
    pitch_range: tuple[float, float] = (-5.0, 45.0)
    yaw_range: tuple[float, float] = (-180.0, 180.0)
    hfov_range: tuple[float, float] = (60.0, 100.0)
    stratify_yaw: bool = True

  def __init__(self, seed: int, num_views_per_pano: int, config: Config):
    self.seed = seed
    self.config = config
    self.num_views_per_pano = num_views_per_pano

  def setup(self):
    self.rng = np.random.default_rng(self.seed)

  def sample_camera(
      self,
      height: int,
      width: int,
      sample_idx: int,
  ) -> jaxcam.Camera:
    """Sample views from a stitched pano."""
    sample_fn = self.rng.uniform
    roll = sample_fn(
        low=self.config.roll_range[0], high=self.config.roll_range[1]
    )
    pitch = sample_fn(
        low=self.config.pitch_range[0], high=self.config.pitch_range[1]
    )
    if self.config.stratify_yaw:
      bin_size = 1 / self.num_views_per_pano
      offset = (sample_fn(low=-0.5, high=0.5) + sample_idx + 0.5) * bin_size
      yaw = (
          offset * (self.config.yaw_range[1] - self.config.yaw_range[0])
          + self.config.yaw_range[0]
      )
    else:
      yaw = sample_fn(
          low=self.config.yaw_range[0], high=self.config.yaw_range[1]
      )
    hfov = sample_fn(
        low=self.config.hfov_range[0], high=self.config.hfov_range[1]
    )
    return misc.camera_from_rpyf(height, width, roll, pitch, yaw, hfov)

  def process(
      self, view_collection: types.ViewCollection
  ) -> Iterable[types.ViewCollection]:
    collection_name, metadata, views = view_collection

    sampled_views = {}
    for view_key, view in views.items():
      camera = view.camera
      if camera.type != 'EQUIRECTANGULAR':
        raise ValueError(
            f'Camera type {camera.type} is not supported for sampling.'
        )
      num_sampled = 0
      while num_sampled < self.num_views_per_pano:
        sampled_jcamera = self.sample_camera(
            self.config.height, self.config.width, num_sampled
        )
        sampledcam_t_cam = np.eye(4)
        sampledcam_t_cam[:3, :3] = np.array(sampled_jcamera.orientation).astype(
            np.float64
        )
        new_view_key = types.ViewKey(
            image_name=view_key.image_name,
            rosette_type=view_key.rosette_type,
            sample_id=num_sampled,
        )
        sampled_views[new_view_key] = dataclasses.replace(
            view,
            camera=types.Intrinsics.from_jaxcam(sampled_jcamera),
            cam_t_w=sampledcam_t_cam @ view.cam_t_w,
        )
        num_sampled += 1

    yield collection_name, dataclasses.replace(
        metadata, num_views=len(sampled_views)
    ), sampled_views


class SplitViewCollection(beam.DoFn):
  """Split a view collection by camera names."""

  def __init__(self, pano_splits: Tuple[str, ...]):
    self.pano_splits = pano_splits

  def process(
      self, view_collection: types.ViewCollection
  ) -> Iterable[types.ViewCollection]:
    view_collection_id, metadata, views = view_collection
    for split in self.pano_splits:
      if split == 'all':
        yield view_collection_id, metadata, views
        continue

      pattern = re.compile(split)
      split_views = {
          k: v for k, v in views.items() if pattern.search(k.camera_name)
      }
      split_collection_id = f'{view_collection_id}/{split}'
      split_metadata = dataclasses.replace(
          metadata, pano_split=split, num_views=len(split_views)
      )
      beam.metrics.Metrics.counter(
          self.__class__.__name__, 'view_collections'
      ).inc()
      yield split_collection_id, split_metadata, split_views


class SingleViewPerCollection(beam.DoFn):
  """Converts a view collection to multiple single-view collections."""

  def process(
      self, view_collection: types.ViewCollection
  ) -> Iterable[types.ViewCollection]:
    _, metadata, views = view_collection
    for view_key, view in views.items():
      view_metadata = dataclasses.replace(
          metadata,
          num_views=1,
          pano_split=view_key.camera_name,
          num_panos=1,
      )
      beam.metrics.Metrics.counter(
          self.__class__.__name__, 'view_collections'
      ).inc()
      yield view_key.image_name, view_metadata, {view_key: view}


class SinglePanoPerCollection(beam.DoFn):
  """Converts a view collection to multiple single-pano collections."""

  def process(
      self, view_collection: types.ViewCollection
  ) -> Iterable[types.ViewCollection]:
    collection_name, metadata, views = view_collection
    views_by_pano = collections.defaultdict(list)
    for view_key in views.keys():
      views_by_pano[view_key.pano_name].append(view_key)

    for pano_name, view_keys in views_by_pano.items():
      beam.metrics.Metrics.counter(
          self.__class__.__name__, 'new_collection_for_pano'
      ).inc()
      collection_metadata = dataclasses.replace(
          metadata,
          num_views=len(view_keys),
          num_panos=1,
      )
      yield f'{collection_name}/{pano_name}', collection_metadata, {
          view_key: views[view_key] for view_key in view_keys
      }


class SampleViewCollection(beam.DoFn):
  """Sample views from a view collection."""

  def __init__(self, seed: int, num_panos: int, num_views_per_pano: int):
    self.seed = seed
    self.num_panos = num_panos
    self.num_views_per_pano = num_views_per_pano

  def setup(self):
    self.rng = np.random.default_rng(self.seed)

  def process(
      self, view_collection: types.ViewCollection
  ) -> Iterable[types.ViewCollection]:
    collection_name, metadata, views = view_collection
    views_by_pano = collections.defaultdict(list)
    for view_key in views.keys():
      views_by_pano[view_key.pano_name].append(view_key)
    pano_names = list(views_by_pano.keys())
    if len(pano_names) < self.num_panos:
      beam.metrics.Metrics.counter(
          self.__class__.__name__,
          f'collection_has_too_few_panos_for_sampling_{self.num_panos}.',
      ).inc()
      return
    sampled_pano_names = self.rng.choice(
        pano_names, size=self.num_panos, replace=False
    )
    sampled_view_keys = []
    for pano_name in sampled_pano_names:
      if len(views_by_pano[pano_name]) < self.num_views_per_pano:
        beam.metrics.Metrics.counter(
            self.__class__.__name__,
            f'pano_has_to_fow_views_for_sampling_{self.num_views_per_pano}',
        ).inc()
        return

      sampled_indices = self.rng.choice(
          range(len(views_by_pano[pano_name])),
          size=self.num_views_per_pano,
          replace=False,
      )
      sampled_view_keys += [
          views_by_pano[pano_name][idx] for idx in sampled_indices
      ]

    sampled_metadata = dataclasses.replace(
        metadata,
        num_views=len(sampled_view_keys),
        num_panos=self.num_panos,
    )
    beam.metrics.Metrics.counter(
        self.__class__.__name__, 'num_view_collections'
    ).inc()
    yield collection_name, sampled_metadata, {
        view_key: views[view_key] for view_key in sampled_view_keys
    }


class CombineViewCollections(beam.DoFn):
  """Combine multiple view collections into a single one."""

  def process(
      self,
      name_and_view_collections: tuple[str, Iterable[types.ViewCollection]],
  ) -> Iterable[types.ViewCollection]:
    views = {}
    metadatas = []
    name, view_collections = name_and_view_collections
    for view_collection in view_collections:
      _, metadata, view_collection_views = view_collection
      metadatas.append(metadata)
      views.update(view_collection_views)
    pano_names = set([view_key.pano_name for view_key in views.keys()])
    yield name, dataclasses.replace(
        metadatas[0], num_views=len(views), num_panos=len(pano_names)
    ), views


class ExplodeSubsampledSegments(beam.DoFn):

  def process(
      self, segments: Iterable[types.SegmentTuple]
  ) -> Iterable[types.SegmentTuple]:
    for segment in segments:
      beam.metrics.Metrics.counter(
          self.__class__.__name__, 'num_segments'
      ).inc()
      yield segment


class LabelTrainValidScenes(beam.DoFn):
  """Adds a train/valid label to scenes in the same group."""

  def __init__(
      self, seed: int, num_valid_per_group: int, allow_run_overlap: bool
  ):
    self.seed = seed
    self.num_valid_per_group = num_valid_per_group
    self.allow_run_overlap = allow_run_overlap

  def setup(self):
    self.rng = np.random.default_rng(self.seed)

  def process(
      self, scenes_in_group: Tuple[Any, Iterable[Tuple[str, types.Scene]]]
  ) -> Iterable[Tuple[int, Tuple[str, types.Scene]]]:
    scenes = scenes_in_group[1]
    num_scenes = len(list(scenes))
    if self.allow_run_overlap:
      # Randomly sample scenes from all runs.
      valid_indices = self.rng.choice(
          num_scenes, self.num_valid_per_group, replace=False
      )
    else:
      # Find one or more runs with a total of num_valid_per_group scenes.
      scene_idx_per_run = collections.defaultdict(list)
      for scene_idx, (_, scene) in enumerate(scenes):
        scene_idx_per_run[scene.query[1].run_name].append(scene_idx)
      scene_idx_per_run = list(scene_idx_per_run.items())
      scene_idx_per_run.sort(key=lambda x: len(x[1]))
      valid_indices = []
      for _, scene_indices in scene_idx_per_run:
        if len(valid_indices) >= self.num_valid_per_group:
          break
        valid_indices += scene_indices

    beam.metrics.Metrics.distribution(
        self.__class__.__name__, 'num_valid_scenes_per_group'
    ).update(len(valid_indices))
    beam.metrics.Metrics.distribution(
        self.__class__.__name__, 'num_train_scenes_per_group'
    ).update(num_scenes - len(valid_indices))
    for i, scene in enumerate(scenes):
      is_val_scene = i in valid_indices
      if is_val_scene:
        beam.metrics.Metrics.counter(
            self.__class__.__name__, 'num_valid_scenes'
        ).inc()
      else:
        beam.metrics.Metrics.counter(
            self.__class__.__name__, 'num_train_scenes'
        ).inc()
      # 0 is train, 1 is valid
      yield int(is_val_scene), scene


@dataclasses.dataclass
class BuildConfig:
  """Configuration for the data building pipeline."""

  filter_runs: FilterRunsByType.Config = dataclasses.field(
      default_factory=FilterRunsByType.Config
  )
  seed: int = 123
  split_duration_sec: float = 1000.0
  max_panos_per_segment: int = 10
  min_spacing_sec: float = 1.0
  min_spacing_meters: float | None = None
  single_view_per_collection: bool = False
  max_panos_query: int = 4
  max_views_per_pano_query: int = 2
  target_level: int = 17
  pano_splits: Tuple[str, ...] | None = ('left', 'right')
  overlap_metric: str = 'mean'  # one of: min, max, mean, first, second
  min_overlap: float = 0.1
  max_overlap: float = 1.0
  max_num_pairs_per_s2cell: int = 10
  max_num_segments_per_s2cell_and_vehicle_type: int | None = 75
  allowed_vehicle_types_query: Tuple[str, ...] | None = ('CAR', 'TREKKER')
  allowed_vehicle_types_map: Tuple[str, ...] | None = ('CAR', 'TREKKER')


class CountAndEmitItems(beam.DoFn):
  """Simple DoFn that counts and emits examples."""

  def __init__(self, label: str):
    self.label = label

  def process(self, item: Any) -> Iterable[Any]:
    beam.metrics.Metrics.counter(
        self.__class__.__name__, f'num_{self.label}'
    ).inc()
    yield item


def build(
    pipeline: beam.Pipeline,
    s2cell_tokens: Sequence[str],
    pose_snapshot_id: str,
    config: BuildConfig,
) -> beam.PCollection[Tuple[str, types.Scene]]:
  """Builds a pipeline to build and match view segments."""
  runs_metadata = runs_crossing_s2cells(s2cell_tokens)
  runs = {
      run.name: types.RunData.from_run_metadata(run) for run in runs_metadata
  }

  runs = pipeline | 'CreateRuns' >> beam.Create(runs)
  filtered_runs = runs | beam.ParDo(
      FilterRunsByType(**dataclasses.asdict(config.filter_runs))
  )
  segments = filtered_runs | 'CreateSegments' >> beam.ParDo(
      SegmentsFromRuns(pose_snapshot_id)
  )

  s2cell_segments = segments | 'SplitByS2Cell' >> beam.ParDo(
      SplitSegmentsByS2Cell(s2cell_tokens, config.target_level)
  )

  filtered_segments = s2cell_segments | beam.ParDo(
      SplitSegmentsByDuration(config.split_duration_sec)
  )

  sampled_segments = filtered_segments | beam.ParDo(
      SubsampleSegments(
          max_panos=config.max_panos_per_segment,
          min_spacing_sec=config.min_spacing_sec,
          min_spacing_meters=config.min_spacing_meters,
      )
  )

  if (
      config.max_num_segments_per_s2cell_and_vehicle_type is not None
      and config.max_num_segments_per_s2cell_and_vehicle_type > 0
  ):
    subsampled_segments = (
        sampled_segments
        | 'IndexSegmentsByS2Cell'
        # Key segments by (s2cell, vehicle_type).
        >> beam.Map(
            lambda segment: (
                (segment[2].s2cell, segment[1].vehicle_type),
                segment,
            )
        )
        | 'SampleSegmentsPerS2Cell'
        >> beam.combiners.Sample.FixedSizePerKey(
            config.max_num_segments_per_s2cell_and_vehicle_type
        )
        | 'DropKeysAfterSubsampling' >> beam.Values()
        | 'ExplodeSubsampledSegments' >> beam.ParDo(ExplodeSubsampledSegments())
    )
  else:
    subsampled_segments = sampled_segments

  views = subsampled_segments | beam.ParDo(ViewsFromSegments(pose_snapshot_id))

  if config.pano_splits is not None:
    views = views | 'SplitViews' >> beam.ParDo(
        SplitViewCollection(config.pano_splits)
    )

  if config.single_view_per_collection:
    views = views | 'SingleViewPerCollection' >> beam.ParDo(
        SingleViewPerCollection()
    )
    # No query sampling required
    query_views = views
  else:
    query_views = views | 'QueryViews' >> beam.ParDo(
        SampleViewCollection(
            config.seed, config.max_panos_query, config.max_views_per_pano_query
        )
    )

  if config.allowed_vehicle_types_map is not None:
    allowed_vehicle_types_map = set([
        vehicles_pb2.VehicleIdRange.VehicleType.Value(name)
        for name in config.allowed_vehicle_types_map
    ])
    views = views | 'FilterOutMapByVehicleType' >> beam.transforms.Filter(
        lambda view_coll: view_coll[1].vehicle_type in allowed_vehicle_types_map
    )
  if config.allowed_vehicle_types_query is not None:
    allowed_vehicle_types_query = set([
        vehicles_pb2.VehicleIdRange.VehicleType.Value(name)
        for name in config.allowed_vehicle_types_query
    ])
    query_views = (
        query_views
        | 'FilterOutQueryByVehicleType'
        >> beam.transforms.Filter(
            lambda view_coll: view_coll[1].vehicle_type
            in allowed_vehicle_types_query
        )
    )

  views = views | 'CountMapViews' >> beam.ParDo(CountAndEmitItems('map_views'))
  query_views = query_views | 'CountQueryViews' >> beam.ParDo(
      CountAndEmitItems('query_views')
  )

  views_by_s2cell = views | 'GroupMapViewCollectionsByS2Cell' >> beam.Map(
      lambda x: (x[1].s2cell, x)
  )
  query_views_by_s2cell = (
      query_views
      | 'GroupQueryViewCollectionsByS2Cell'
      >> beam.Map(lambda x: (x[1].s2cell, x))
  )

  overlap_fn = functools.partial(
      matching.compute_overlap_matrix,
      symmetric=True,
      filter_same_run=True,
      overlap_metric=config.overlap_metric,
  )

  candidates_per_s2cell = {
      'query': query_views_by_s2cell,
      'map': views_by_s2cell,
  } | beam.CoGroupByKey()

  s2cell_candidates_with_overlap = (
      candidates_per_s2cell
      | 'ComputeBEVOverlap'
      >> beam.MapTuple(
          lambda s2cell, view_collections: (
              s2cell,
              view_collections['query'],
              view_collections['map'],
              overlap_fn(view_collections['query'], view_collections['map']),
          )
      )
  )

  scenes = s2cell_candidates_with_overlap | beam.FlatMapTuple(
      functools.partial(
          matching.build_scenes_from_overlap_matrix,
          min_overlap=config.min_overlap,
          max_overlap=config.max_overlap,
          max_num_pairs=config.max_num_pairs_per_s2cell,
      )
  )

  scenes = scenes | 'CountScenes' >> beam.ParDo(CountAndEmitItems('scenes'))

  return scenes
