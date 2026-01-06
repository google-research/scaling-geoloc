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

"""Script to extract map data from a list of s2cells."""

from collections.abc import Sequence
import dataclasses
import json

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import fancyflags as ff

from google3.location.visualmapping.experimental.fomo.data import data_utils
from scaling_geoloc.data import export
from scaling_geoloc.data import pipeline_steps
from google3.pipeline.flume.py import runner
from google3.pyglib import gfile


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

_BUILD = ff.experimental.DEFINE_auto_recursively(
    'build', pipeline_steps.BuildConfig
)

_DATA = ff.experimental.DEFINE_auto_recursively('data', export.DataConfig)

_POSE_SNAPSHOT_ID = flags.DEFINE_string(
    'pose_snapshot_id',
    'kuiper_aerial_aligned_20250101',
    'The pose snapshot to use.',
)

_OUTPUT_PREFIX = flags.DEFINE_string(
    'output_prefix', None, 'Output prefix.', required=True
)

_SPLIT_NAME = flags.DEFINE_string(
    'split_name', 'train', 'The split name to use when writing the data.'
)


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

  # setup configuration
  build_config = _BUILD.value()
  data_config = _DATA.value()

  if not gfile.Exists(_OUTPUT_PREFIX.value):
    gfile.MakeDirs(
        _OUTPUT_PREFIX.value, mode=gfile.LEGACY_GROUP_WRITABLE_WORLD_READABLE
    )

  s2cells_at_target_level = data_utils.get_s2cell_tokens_at_s2_level(
      s2cell_inputs, target_level=build_config.target_level
  )

  logging.info('s2cells_at_target_level: %s', ','.join(s2cells_at_target_level))

  runner.program_started()
  with beam.Pipeline(runner.FlumeRunner()) as pipeline:
    with gfile.Open(f'{_OUTPUT_PREFIX.value}/s2cell_list_input.txt', 'w') as f:
      for s2_cell_token in s2cell_inputs:
        f.write(s2_cell_token + '\n')

    with gfile.Open(f'{_OUTPUT_PREFIX.value}/s2cell_list_output.txt', 'w') as f:
      for s2_cell_token in s2cells_at_target_level:
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

    scenes = pipeline_steps.build(
        pipeline, s2cell_inputs, _POSE_SNAPSHOT_ID.value, build_config
    )

    _, features_dict = export.pack_serialize_and_write_scenes(
        scenes,
        data_config,
        _POSE_SNAPSHOT_ID.value,
        _OUTPUT_PREFIX.value,
        split_name=_SPLIT_NAME.value,
    )

  data_utils.write_tfds_metadata(
      data_dir=_OUTPUT_PREFIX.value, features_dict=features_dict
  )

  pipeline_steps.write_metrics(pipeline.result, _OUTPUT_PREFIX.value)


if __name__ == '__main__':
  app.run(main)
