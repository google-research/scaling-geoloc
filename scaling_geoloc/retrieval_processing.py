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

"""Script to extract retrieval data for a list of s2cells."""

from collections.abc import Callable, Iterable, Mapping, Sequence
import dataclasses
import functools
import json
from typing import Any

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from etils.array_types import BoolArray
from etils.array_types import FloatArray
import fancyflags as ff
import jax.numpy as jnp
import numpy as np

from google3.location.visualmapping.experimental.fomo.data import data_utils
from scaling_geoloc.data import export
from scaling_geoloc.data import geo
from scaling_geoloc.data import pipeline_steps
from scaling_geoloc.data import types
from scaling_geoloc.utils import misc
from google3.pipeline.flume.py import runner
from google3.pyglib import gfile
from google3.util.geometry.python import s2


@dataclasses.dataclass
class BuildConfig:
  """Configuration for the data building pipeline."""

  seed: int = 123
  target_level: int = 17  # The target level at which we sample panos.
  filter_runs: pipeline_steps.FilterRunsByType.Config = dataclasses.field(
      default_factory=pipeline_steps.FilterRunsByType.Config
  )
  create_test_set_for_year: str | None = (
      None  # Create a test set for runs of this year.
  )
  regenerate_test_set_only: bool = (
      False  # If true, only regenerate the test set.
  )
  min_spacing_meters: float | None = 1.0  # Min spacing between panos in a run.
  max_panos_per_s2cell: int | None = None  # Max number of panos per s2cell.
  spatial_sampling: str = 'uniform'  # Spatial sampling method.
  spatial_margin_fraction: float | None = (
      None  # Range [0, 1]. (In FPS sampling). None is no margin.
  )
  pano_splits: tuple[str, ...] | None = None  # Split panos into multiple views.
  views_per_pano: int = 1  # Sample n views from the pano after splitting.
  sample_views_from_stitched_pano: bool = (
      False  # Sample views from stitched pano.
  )
  efficient_pano_loading: bool = (
      True  # Use efficient data loading for pano sampling.
  )
  pano_sampling: pipeline_steps.SampleViewsFromStitchedPano.Config = (
      dataclasses.field(
          default_factory=pipeline_steps.SampleViewsFromStitchedPano.Config
      )
  )
  min_views_per_s2cell: int = 50  # Min number of views per s2cell.
  max_views_per_s2cell: int = 200  # Max number of views per s2cell.
  num_valid_per_s2cell: int = 10  # Number of validation scenes per s2cell.
  max_num_valid_global: int | None = None  # Max number of validation scenes.
  num_test_per_s2cell: tuple[int, int] = (
      5,
      10,
  )  # Number of test scenes per s2cell.

  export_segments_to_json: bool = False


_S2CELL_LIST = flags.DEFINE_list(
    's2cell_list',
    None,
    'A list of S2 cell tokens. Only one of s2cell_list or '
    's2cell_file can be set.',
)

_S2CELL_FILE = flags.DEFINE_string(
    's2cell_file',
    None,
    'A file with a list of S2 cell tokens. Only one of '
    's2cell_list or s2cell_file can be set.',
)

_BUILD = ff.experimental.DEFINE_auto_recursively('build', BuildConfig)

_DATA = ff.experimental.DEFINE_auto_recursively('data', export.DataConfig)

_POSE_SNAPSHOT_ID = flags.DEFINE_string(
    'pose_snapshot_id',
    'kuiper_aerial_aligned_20250801',
    'The pose snapshot to use.',
)

_OUTPUT_PREFIX = flags.DEFINE_string(
    'output_prefix', None, 'Output prefix.', required=True
)


def filter_by_size(
    group: str,
    min_num_elements: int,
) -> Callable[[tuple[str, Sequence[Any]]], bool]:
  """Returns a filter function that filters cells with too few views."""

  def filter_fn(key_and_elements: tuple[str, Iterable[Any]]) -> bool:
    _, elements = key_and_elements
    num_elements = len(list(elements))
    beam.metrics.Metrics.distribution(
        'FilterBySize', f'{group}_num_elements'
    ).update(num_elements)
    if num_elements < min_num_elements:
      beam.metrics.Metrics.counter(
          'FilterBySize', f'{group}_with_too_few_elements'
      ).inc()
      return False
    else:
      beam.metrics.Metrics.counter(
          'FilterBySize', f'{group}_with_enough_elements'
      ).inc()
      return True

  return filter_fn


def explode(key: str, elements: Iterable[Any]) -> Iterable[tuple[str, Any]]:
  for element in elements:
    yield key, element


def center_of_segment(segment_tuple: types.SegmentTuple) -> np.ndarray:
  """Returns the space-time center of a segment."""
  _, _, segment = segment_tuple
  datetime = int(segment.run_name[:6])
  return np.concatenate([[datetime], segment.ecef_positions.mean(0)])


def bbox_norm_categorical_time(
    s2cell: str,
    points: FloatArray['N d'],
    spatial_margin_fraction: float | None = None,  # Range [0, 1].
) -> tuple[FloatArray['N d'], BoolArray['N'] | None]:
  """Returns bbox norm, but makes time dimension categorical."""
  run_ids = np.unique(points[:, 0], return_inverse=True)[1]
  # Start with a random run id.
  run_ids = 1 + run_ids[..., None]
  # Normalize x, y coordinates (meters) in s2cell by the edge length.
  edge_length = types.CELL_EXTENTS[
      s2.S2CellId.from_token(s2cell.encode('utf-8')).level
  ]
  xy = geo.transform_points(
      np.linalg.inv(geo.ecef_t_s2cell(s2cell, 0.0)), points[..., 1:]
  )[:, :2]
  points = np.concatenate([run_ids, xy], axis=-1)
  if spatial_margin_fraction is not None:
    # Remove points that are close to the edge of the s2cell bounding box.
    mask = (np.abs(xy) < (1 - spatial_margin_fraction) * edge_length / 2).all(
        axis=-1
    )
  else:
    mask = None
  return points, mask


def spatio_temporal_residuals(
    a: FloatArray['... 3'],
    b: FloatArray['... 3'],
    min_distance_meters: float = 0.0,  # Min. spatial distance between points.
) -> FloatArray['... 2']:
  """Returns the spatio-temporal residuals between two point arrays."""
  # Treat temporal axis as categorical.
  spatial_distance = misc.l2_norm(a[..., 1:], b[..., 1:])
  temporal_distance = jnp.abs(a[..., :1] - b[..., :1])
  # Scale temporal residuals for balanced sampling over space-time.
  temporal_distance = temporal_distance.clip(max=1.0) * np.mean(
      spatial_distance
  )
  # We consider valid residuals if the spatial distance is larger than the
  # minimum distance.
  valid = spatial_distance > min_distance_meters**2
  return jnp.concatenate(
      [temporal_distance * valid, spatial_distance * valid], axis=-1
  )


def build(
    pipeline: beam.Pipeline,
    s2cell_tokens: Sequence[str],
    runs: Mapping[str, types.RunData],
    pose_snapshot_id: str,
    config: BuildConfig,
    output_path: str,
    suffix: str = '',
) -> beam.PCollection[tuple[str, types.Scene]]:
  """Builds a pipeline to extract retrieval meta-data for a list of runs."""

  runs = pipeline | 'CreateRuns' >> beam.Create(runs)
  filtered_runs = runs | beam.ParDo(
      pipeline_steps.FilterRunsByType(**dataclasses.asdict(config.filter_runs))
  )
  segments = filtered_runs | 'CreateSegments' >> beam.ParDo(
      pipeline_steps.SegmentsFromRuns(pose_snapshot_id)
  )

  s2cell_segments = segments | 'SplitByS2Cell' >> beam.ParDo(
      pipeline_steps.SplitSegmentsByS2Cell(s2cell_tokens, config.target_level)
  )

  if config.export_segments_to_json:
    # Export segments to a json file.
    _ = (
        s2cell_segments
        | 'SegmentsToDict'
        >> beam.MapTuple(
            lambda name, run_data, segment: (
                name,
                {**dataclasses.asdict(run_data), **segment.asflatdict()},
            )
        )
        | 'DumpSegmentsToJson'
        >> export.DumpToJson(
            f'{output_path}/segments{suffix}.json', num_shards=None
        )  # Allow sharding here as this can be huge.
    )

  # Split segments into many single-pano segments that are spaced by a minimum
  # distance.
  single_pano_segments = s2cell_segments | 'SplitIntoSinglePanos' >> beam.ParDo(
      pipeline_steps.SplitSegmentsIntoSinglePanos(
          min_spacing_meters=config.min_spacing_meters
      )
  )

  if config.max_panos_per_s2cell is not None:
    # Per s2cell, sample a up to a fixed number of single-pano segments.
    sampled_pano_segments = (
        single_pano_segments
        | 'IndexSegmentsByS2Cell'
        >> beam.Map(
            lambda segment: (
                segment[2].s2cell,
                segment,
            )
        )
        | 'CountPanos'
        >> export.CountPerKeyAndDumpToJson(
            f'{output_path}/panos_per_s2cell{suffix}.json'
        )
        | 'GroupSegmentsByS2Cell' >> beam.GroupByKey()
        | 'SampleSegmentsPerS2Cell'
        >> beam.ParDo(
            pipeline_steps.SampleKFromList(
                seed=config.seed,
                algorithm=config.spatial_sampling,
                k=config.max_panos_per_s2cell,
                gather_fn=center_of_segment,
                norm_fn=functools.partial(
                    bbox_norm_categorical_time,
                    spatial_margin_fraction=config.spatial_margin_fraction,
                ),
                residuals_fn=functools.partial(
                    spatio_temporal_residuals,
                    min_distance_meters=config.min_spacing_meters,
                ),
            )
        )
        | 'DropKeysAfterSubsampling' >> beam.Values()
        | 'ExplodeSubsampledSegments'
        >> beam.ParDo(pipeline_steps.ExplodeSubsampledSegments())
    )
  else:
    sampled_pano_segments = single_pano_segments

  if config.sample_views_from_stitched_pano:
    sampled_views = (
        sampled_pano_segments
        | 'ExtractStitchedPanos'
        >> beam.ParDo(
            pipeline_steps.ViewsFromSegments(
                pose_snapshot_id, get_stitched_panos=True
            )
        )
        | 'SampleViewsFromStitchedPanos'
        >> beam.ParDo(
            pipeline_steps.SampleViewsFromStitchedPano(
                config.seed + 1,
                config.views_per_pano,
                config.pano_sampling,
            )
        )
    )
  else:
    # Extract views for each segment.
    views = sampled_pano_segments | beam.ParDo(
        pipeline_steps.ViewsFromSegments(pose_snapshot_id)
    )

    # Split pano collections into multiple view collections.
    if config.pano_splits is not None:
      views = views | 'SplitViews' >> beam.ParDo(
          pipeline_steps.SplitViewCollection(config.pano_splits)
      )
    else:
      views = views | 'SplitPanoIntoSingleViews' >> beam.ParDo(
          pipeline_steps.SingleViewPerCollection()
      )

    # Sample n views from each collection.
    sampled_views = views | 'ExtractViews' >> beam.ParDo(
        pipeline_steps.SampleViewCollection(
            config.seed, 1, config.views_per_pano
        )
    )

  sampled_views = sampled_views | 'CountCollections' >> beam.ParDo(
      pipeline_steps.CountAndEmitItems('collections')
  )

  # Sample fixed number of views per s2cell. Remove cells with too few views.
  sampled_views = (
      sampled_views
      | 'IndexViewsByS2Cell' >> beam.Map(lambda x: (x[1].s2cell, x))
      | 'SampleNumViewsPerS2Cell'
      >> beam.combiners.Sample.FixedSizePerKey(config.max_views_per_s2cell)
      | 'FilterCellsWithTooFewViews'
      >> beam.Filter(filter_by_size('s2cells', config.min_views_per_s2cell))
      | 'ExplodeViews' >> beam.FlatMapTuple(explode)
      | 'CountAfterSampling'
      >> export.CountPerKeyAndDumpToJson(
          f'{output_path}/views_per_s2cell_after_sampling{suffix}.json'
      )
      | 'RemoveKeysAfterViewSampling' >> beam.Values()
  )

  if config.sample_views_from_stitched_pano and config.efficient_pano_loading:
    # Group views by pano (avoid loading large stitched panos multiple times).
    sampled_views = (
        sampled_views
        | 'IndexByPano'
        >> beam.Map(lambda x: (list(x[2].keys())[0].pano_name, x))
        | 'GroupByPano' >> beam.GroupByKey()
        | 'CombineViewsInPano'
        >> beam.ParDo(pipeline_steps.CombineViewCollections())
    )

  # Guide to sample negatives:
  # 1. Index by s2cell at higher level (e.g. 14 or 15).
  # 2. Score by distance to query: sc = 1 / distance * (distance < th).
  # 3. For each query, sample ~ sc.

  # Guide to sample positives:
  # 1. Index by s2cell at same level or one above.
  # 2. Compute BEV overlap to ALL views (not only subsampled).
  # 3. For each query, sample 1 view with at least x overlap.
  def create_scene(views: types.ViewCollection) -> tuple[str, types.Scene]:
    """Creates a scene from a view collection."""
    return views[0], types.Scene(
        s2cell=views[1].s2cell,
        ecef_t_w=geo.ecef_t_s2cell(views[1].s2cell, altitude=0.0),
        overlap=np.array([], dtype=np.float32),
        query=views,
        map=[],  # We could add positives + negatives here.
    )

  scenes = sampled_views | 'CreateScenes' >> beam.Map(create_scene)

  return scenes


def create_train_valid_split(
    scenes: beam.PCollection[tuple[str, types.Scene]],
    num_valid_per_s2cell: int,
    seed: int,
    allow_run_overlap: bool = True,
) -> tuple[
    beam.PCollection[tuple[str, types.Scene]],
    beam.PCollection[tuple[str, types.Scene]],
]:
  """Splits scenes into train and valid splits per s2cell."""
  scenes_by_s2cell = (
      scenes
      | 'IndexByS2Cell' >> beam.Map(lambda scene: (scene[1].s2cell, scene))
      | 'GroupScenesByS2Cell' >> beam.GroupByKey()
  )

  train_scenes, valid_scenes = (
      scenes_by_s2cell
      | 'CreateTrainValidSplit'
      >> beam.ParDo(
          pipeline_steps.LabelTrainValidScenes(
              seed, num_valid_per_s2cell, allow_run_overlap
          )
      )
      | 'PartitionIntoTrainValid' >> beam.Partition(lambda x, _: x[0], 2)
  )
  train_scenes = train_scenes | 'RemoveLabelsFromTrainScenes' >> beam.Values()
  valid_scenes = valid_scenes | 'RemoveLabelsFromValidScenes' >> beam.Values()
  return train_scenes, valid_scenes


def yield_runs_crossing_s2cell(
    s2cell: str,
) -> Iterable[tuple[str, types.RunData]]:
  """Yields runs crossing a given s2cell."""
  runs = pipeline_steps.runs_crossing_s2cells([s2cell])
  for run in runs:
    yield run.name, types.RunData.from_run_metadata(run)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if _S2CELL_LIST.value is None and _S2CELL_FILE.value is None:
    raise ValueError('Either s2cell_list or s2cell_file must be set')
  elif _S2CELL_LIST.value is not None and _S2CELL_FILE.value is not None:
    raise ValueError('Only one of s2cell_list and s2cell_file can be set')

  if _S2CELL_LIST.value is not None:
    s2cell_inputs = _S2CELL_LIST.value
  else:
    with gfile.Open(_S2CELL_FILE.value, 'r') as f:
      s2cell_inputs = [line.strip() for line in f if line.strip()]

  logging.info('s2cell_inputs: %s', ','.join(s2cell_inputs))

  # Setup configuration.
  build_config = _BUILD.value()
  data_config = _DATA.value()

  if (
      build_config.regenerate_test_set_only
      and build_config.create_test_set_for_year is None
  ):
    raise ValueError(
        'regenerate_test_set_only requires create_test_set_for_year to be set.'
    )

  # Config postprocessing.
  if (
      build_config.efficient_pano_loading
      and build_config.sample_views_from_stitched_pano
  ):
    # In this case the builder groups by pano, and we need to split the scenes
    # into single-query scenes after data extraction.
    data_config = dataclasses.replace(
        data_config, split_into_single_queries=True
    )

  runner.program_started()
  if not gfile.Exists(_OUTPUT_PREFIX.value):
    gfile.MakeDirs(
        _OUTPUT_PREFIX.value, mode=gfile.LEGACY_GROUP_WRITABLE_WORLD_READABLE
    )

  # Extract runs crossing the s2cells, and dump them to a json file.
  run_file = f'{_OUTPUT_PREFIX.value}/runs.json'
  with beam.Pipeline(runner.FlumeRunner()) as pipeline:
    initial_s2cells = pipeline | 'InitializeS2Cells' >> beam.Create(
        s2cell_inputs
    )
    _ = (
        initial_s2cells
        | 'RunsCrossingS2Cells' >> beam.FlatMap(yield_runs_crossing_s2cell)
        | 'MakeUnique' >> beam.Distinct()
        | 'RunDataToDict'
        >> beam.MapTuple(
            lambda name, run_data: (name, dataclasses.asdict(run_data))
        )
        | 'DumpRunsToJson' >> export.DumpToJson(run_file)
    )

  # Load runs from file.
  with gfile.Open(run_file, 'r') as file:
    runs = json.load(file)
  runs = {
      run_name: types.RunData(**run_data) for run_name, run_data in runs.items()
  }

  logging.info('Total runs: %d', len(runs))

  # Create temporal test split.
  if build_config.create_test_set_for_year is not None:
    test_runs = {
        run_name: run_data
        for run_name, run_data in runs.items()
        if run_name[:4] == build_config.create_test_set_for_year
    }
    logging.info('Test runs: %d', len(test_runs))
    runs = {
        run_name: run_data
        for run_name, run_data in runs.items()
        if run_name not in test_runs
    }
    logging.info('Train/valid runs: %d', len(runs))
  else:
    test_runs = None

  # Write the input and output s2cell lists to the output directory.
  with gfile.Open(f'{_OUTPUT_PREFIX.value}/s2cell_list_input.txt', 'w') as f:
    for s2_cell_token in s2cell_inputs:
      f.write(s2_cell_token + '\n')

  # Write the config to the output directory.
  with gfile.Open(f'{_OUTPUT_PREFIX.value}/config.json', 'w') as f:
    json.dump(
        {
            'build': dataclasses.asdict(build_config),
            'data': dataclasses.asdict(data_config),
        },
        f,
        indent=2,
    )

  features_dict = None
  if not build_config.regenerate_test_set_only:
    # Build training and validation set.
    with beam.Pipeline(runner.FlumeRunner()) as pipeline:
      # Extract retrieval meta-data for the runs.
      scenes = build(
          pipeline,
          s2cell_inputs,
          runs,
          _POSE_SNAPSHOT_ID.value,
          build_config,
          _OUTPUT_PREFIX.value,
      )

      # Split scenes into train and valid splits.
      train_scenes, valid_scenes = create_train_valid_split(
          scenes, build_config.num_valid_per_s2cell, build_config.seed
      )

      if build_config.max_num_valid_global is not None:
        valid_scenes = (
            valid_scenes
            | 'SampleFixedNumberOfValidScenes'
            >> beam.combiners.Sample.FixedSizeGlobally(
                build_config.max_num_valid_global
            )
            | 'ExplodeSampledValidScenes' >> beam.FlatMap(lambda x: x)
        )

      # Pack and serialize the train and valid scenes.
      for split_name, split_scenes in zip(
          ['train', 'valid'], [train_scenes, valid_scenes]
      ):
        _, features_dict = export.pack_serialize_and_write_scenes(
            split_scenes,
            data_config,
            _POSE_SNAPSHOT_ID.value,
            _OUTPUT_PREFIX.value,
            split_name=split_name,
            num_collections_map=0,
        )

      # Write the beam metrics to the output directory.
      pipeline_steps.write_metrics(pipeline.result, _OUTPUT_PREFIX.value)

  # Optionally, create a test set for runs of a specific year.
  if build_config.create_test_set_for_year is not None:
    # Only create a test set where we have training data.
    with gfile.Open(
        f'{_OUTPUT_PREFIX.value}/views_per_s2cell_after_sampling.json', 'rb'
    ) as f:
      s2cell_list_test = list(json.load(f).keys())
    # Overwrite the min and max views per s2cell to the number of test views.
    build_config.min_views_per_s2cell = build_config.num_test_per_s2cell[0]
    build_config.max_views_per_s2cell = build_config.num_test_per_s2cell[1]
    # Build test set in separate beam pipeline.
    with beam.Pipeline(runner.FlumeRunner()) as test_pipeline:
      # Extract retrieval meta-data for the test runs.
      test_scenes = build(
          test_pipeline,
          s2cell_list_test,
          test_runs,
          _POSE_SNAPSHOT_ID.value,
          build_config,
          _OUTPUT_PREFIX.value,
          suffix='_test',
      )

      _, features_dict = export.pack_serialize_and_write_scenes(
          test_scenes,
          data_config,
          _POSE_SNAPSHOT_ID.value,
          _OUTPUT_PREFIX.value,
          split_name='test',
          num_collections_map=0,
      )
    # Write the test beam metrics to the output directory.
    pipeline_steps.write_metrics(
        test_pipeline.result, _OUTPUT_PREFIX.value, suffix='_test'
    )

  # Write the metadata for train, valid and test splits.
  data_utils.write_tfds_metadata(
      data_dir=_OUTPUT_PREFIX.value, features_dict=features_dict
  )


if __name__ == '__main__':
  app.run(main)
