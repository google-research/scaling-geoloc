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

"""Flume pipeline to extract labels from arrayrecord files."""

from collections.abc import Iterable, Sequence
import dataclasses
import functools
import logging

from absl import app
from absl import flags
import apache_beam as beam
import fancyflags as ff
import numpy as np
import tensorflow_datasets as tfds

from unigeoclip.data import export
from google3.location.visualmapping.experimental.fomo.data import data_utils
from scaling_geoloc.data import pipeline_steps
from google3.pipeline.flume.py import runner
from google3.pyglib import gfile
from google3.util.geometry.python import s2

_INPUT_PREFIX = flags.DEFINE_string(
    'input_prefix', None, 'input prefix.', required=True
)

_INPUT_LOCATIONS = flags.DEFINE_list(
    'input_locations',
    None,
    'The locations to concatenate.',
    required=True,
)

_OUTPUT_PREFIX = flags.DEFINE_string(
    'output_prefix', None, 'Output prefix.', required=True
)


# The target_level defines the 'join' level.
_DATA = ff.DEFINE_auto('data', export.DSMDataConfig)

_DSM_PREFIX = flags.DEFINE_string(
    'dsm_prefix', None, 'DSM prefix.', required=True
)

_DSM_SPLIT_NAME = flags.DEFINE_string(
    'dsm_split_name',
    'dsm',
    'DSM split name.',
)

_DSM_RE_CENTER = flags.DEFINE_bool(
    'dsm_re_center',
    True,
    'If true, re-center the DSM crop to the SV camera center.',
)

_SPLIT_NAMES = flags.DEFINE_list(
    'split_names',
    None,
    'The name of the splits to extract labels from (train, valid, test).',
    required=True,
)

_OVERWRITE = flags.DEFINE_bool(
    'overwrite',
    True,
    'If true, overwrite the output prefix if it already exists.',
)

_EXPORT_DSM_TEST_SET = flags.DEFINE_bool(
    'export_dsm_test_set',
    True,
    'If true, export a test set of DSM crops.',
)


_DSM_TARGET_LEVEL = flags.DEFINE_integer(
    'dsm_target_level',
    None,
    'The target level for the DSM crops in the test set.',
)


class AddDsmCropToScene(beam.DoFn):
  """Adds DSM crop to the example."""

  def __init__(self, dsm_data: export.DSMDataConfig, re_center: bool = True):
    self.dsm_data = dsm_data
    self.re_center = re_center

    if dsm_data.num_samples > 1:
      raise ValueError(
          'num_samples > 1 is not supported in join. Use'
          ' dsm_postprocessing instead.'
      )

  def setup(self):
    self.rng = np.random.default_rng(self.dsm_data.seed + 1)
    self.crops_from_dsm = export.CropsFromDSM(self.dsm_data)
    self.crops_from_dsm.setup()

  def process(
      self,
      s2cell_and_data: tuple[bytes, dict[str, Iterable[export.Example]]],
  ) -> Iterable[tuple[str, export.Example]]:
    s2cell, data = s2cell_and_data
    sv_samples = list(data['sv'])
    dsm_samples = list(data['dsm'])
    if not dsm_samples:
      beam.metrics.Metrics.counter(self.__class__.__name__, 'no_dsm').inc()
      return
    if not sv_samples:
      beam.metrics.Metrics.counter(self.__class__.__name__, 'no_sv').inc()
      return

    beam.metrics.Metrics.distribution(
        self.__class__.__name__, 'dsm_samples_per_s2cell'
    ).update(len(dsm_samples))
    beam.metrics.Metrics.distribution(
        self.__class__.__name__, 'sv_samples_per_s2cell'
    ).update(len(sv_samples))

    for i, sv_sample in enumerate(sv_samples):
      dsm_sample = self.rng.choice(dsm_samples)
      # Maybe broadcast here if we have multiple dsm samples?
      dsm_crop = self.crops_from_dsm.generate_sample(
          dsm_sample,
          image_idx=np.random.choice(dsm_sample['dsm'].shape[0]),
          at_ecef=export.center_ecef_from_example(sv_sample)
          if self.re_center
          else None,
      )
      dsm_crop['s2cell'] = sv_sample['s2cell']
      beam.metrics.Metrics.counter(
          self.__class__.__name__, 'sv_dsm_pairs'
      ).inc()
      yield f'{s2cell.decode("utf-8")}_sample{i}', {
          **sv_sample,
          'dsm': dsm_crop,
      }


def update_s2cell_of_example(
    example: export.Example, target_level: int
) -> export.Example:
  return {
      **example,
      's2cell': export.s2cell_token_at_level(example, target_level),
  }


def expand_s2cells_of_example(
    example: export.Example, target_level: int
) -> Iterable[export.Example]:
  """Expands the s2cell of the example to the target level."""
  parent = s2.S2CellId.from_token(example['s2cell'])
  if parent.level > target_level:
    # Target is coarser than the current level.
    parent = parent.parent_at_level(target_level)
  child = parent.child_begin_at_level(target_level)
  while child != parent.child_end_at_level(target_level):
    yield {
        **example,
        's2cell': child.token(),
    }
    child = child.next()


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if gfile.Exists(_OUTPUT_PREFIX.value) and not _OVERWRITE.value:
    raise ValueError(f'Output prefix already exists: {_OUTPUT_PREFIX.value}')
  else:
    gfile.MakeDirs(_OUTPUT_PREFIX.value)

  input_directories = []
  for location in _INPUT_LOCATIONS.value:
    input_directories.append(f'{_INPUT_PREFIX.value}/{location}')
  logging.info('input_directories: %s', ','.join(input_directories))

  builder = tfds.builder_from_directories(input_directories)
  features_dict = builder.info.features
  # Unfortunately this is broken in tfds, so we have to manually concatenate...
  builders = []
  for location in _INPUT_LOCATIONS.value:
    builders.append(
        tfds.builder_from_directory(f'{_INPUT_PREFIX.value}/{location}')
    )

  data_config = _DATA.value()
  target_level = data_config.target_level

  if _DSM_PREFIX.value is not None:
    dsm_builder = tfds.builder_from_directory(_DSM_PREFIX.value)
    features_dict = tfds.features.FeaturesDict(
        {**features_dict, 'dsm': _DATA.value().data_spec()}
    )
    if _EXPORT_DSM_TEST_SET.value:
      test_dsm_config = dataclasses.replace(
          data_config,
          num_samples=1,
          max_rotation_angle_degrees=0.0,
          max_offset=0.0,
      )
      test_dsm_prefix = f'{_OUTPUT_PREFIX.value}/dsm'
      if not gfile.Exists(test_dsm_prefix):
        gfile.MakeDirs(test_dsm_prefix)
        # Write the metadata for train, valid and test splits.
    else:
      test_dsm_config = None
      test_dsm_prefix = None
  else:
    dsm_builder = None
    test_dsm_config = None
    test_dsm_prefix = None

  runner.program_started()

  # Extract labels from arrayrecord files.
  with beam.Pipeline(runner.FlumeRunner()) as pipeline:
    dsm_data = None
    if dsm_builder is not None:
      # Read in DSM data.
      split_name = _DSM_SPLIT_NAME.value
      dsm_data = (
          pipeline
          | f'ReadDsmFromTFDS_{split_name}'
          >> tfds.beam.ReadFromTFDS(dsm_builder, split=split_name)
          | f'ConvertDsmToNumpy_{split_name}' >> beam.Map(tfds.as_numpy)
      )

      # Extract non-offsetted dsm crops for the test set.
      if test_dsm_config is not None:
        dsm_test_data = (
            dsm_data
            | f'ExpandS2Cells_{split_name}'
            >> beam.FlatMap(
                expand_s2cells_of_example,
                target_level=_DSM_TARGET_LEVEL.value or target_level,
            )
            | f'CropDsm_{split_name}'
            >> beam.ParDo(export.CropsFromDSM(data_config, re_center=True))
        )
        export.serialize_and_write_examples(
            dsm_test_data,
            tfds.features.FeaturesDict(test_dsm_config.data_spec()),
            test_dsm_prefix,
            split_name='test_dsm',
        )

    for split_name in _SPLIT_NAMES.value:
      input_datas = []
      # Read the data from each location.
      for location_i, builder_i in zip(_INPUT_LOCATIONS.value, builders):
        input_datas.append(
            pipeline
            | f'ReadFromTFDS_{location_i}_{split_name}'
            >> tfds.beam.ReadFromTFDS(builder_i, split=split_name)
            | f'ConvertToNumpy_{location_i}_{split_name}'
            >> beam.Map(tfds.as_numpy)
        )
      # Concatenate the data from each location.
      data = input_datas | f'Flatten_{split_name}' >> beam.Flatten()
      data = data | f'UpdateS2Cell_{split_name}' >> beam.Map(
          functools.partial(update_s2cell_of_example, target_level=target_level)
      )

      if dsm_data is not None:
        # Join SV and DSM data.
        svdsm_data = {
            'sv': (
                data
                | f'KeySVByS2Cell_{split_name}'
                >> beam.Map(lambda x: (x['s2cell'], x))
            ),
            'dsm': (
                dsm_data
                | f'KeyDsmByS2Cell_{split_name}'
                >> beam.Map(
                    lambda x: (
                        export.s2cell_token_at_level(x, target_level),
                        x,
                    )
                )
            ),
        } | f'CombineSVAndDsm_{split_name}' >> beam.CoGroupByKey()
        data = svdsm_data | f'AddDsmCropToScene_{split_name}' >> beam.ParDo(
            AddDsmCropToScene(data_config, re_center=_DSM_RE_CENTER.value)
        )

      export.serialize_and_write_examples(
          data,
          features_dict,
          _OUTPUT_PREFIX.value,
          split_name=split_name,
      )

  # Write the beam metrics to the output directory.
  pipeline_steps.write_metrics(pipeline.result, _OUTPUT_PREFIX.value)

  # Write the metadata for train, valid and test splits.
  data_utils.write_tfds_metadata(
      data_dir=_OUTPUT_PREFIX.value, features_dict=features_dict
  )

  if test_dsm_prefix is not None:
    # Write the metadata for the test dsm crops.
    assert test_dsm_config is not None
    data_utils.write_tfds_metadata(
        data_dir=test_dsm_prefix,
        features_dict=tfds.features.FeaturesDict(test_dsm_config.data_spec()),
    )


if __name__ == '__main__':
  app.run(main)
