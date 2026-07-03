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

"""Pipeline to fetch aerial imagery for an s2cell list."""

from collections.abc import Sequence
import dataclasses
import json

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import fancyflags as ff
import tensorflow_datasets as tfds

from unigeoclip.internal import export
from google3.location.visualmapping.experimental.fomo.data import data_utils
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

_OUTPUT_PREFIX = flags.DEFINE_string(
    'output_prefix', None, 'Output prefix.', required=True
)

_DATA = ff.DEFINE_auto('data', export.DSMDataConfig)


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
    s2cell_file = str(_S2CELL_FILE.value)
    if s2cell_file.endswith('.txt'):
      with gfile.Open(_S2CELL_FILE.value, 'r') as f:
        s2cell_inputs = [line.strip() for line in f if line.strip()]
    elif s2cell_file.endswith('.json'):
      with gfile.Open(_S2CELL_FILE.value, 'r') as f:
        s2cell_inputs = list(json.load(f).keys())
    else:
      raise ValueError('Unsupported file type for s2cell_file')

  logging.info('s2cell_inputs: %s', ','.join(s2cell_inputs))

  # Setup configuration.
  data_config = _DATA.value()

  runner.program_started()
  if not gfile.Exists(_OUTPUT_PREFIX.value):
    gfile.MakeDirs(
        _OUTPUT_PREFIX.value, mode=gfile.LEGACY_GROUP_WRITABLE_WORLD_READABLE
    )

  # Write the input and output s2cell lists to the output directory.
  with gfile.Open(f'{_OUTPUT_PREFIX.value}/s2cell_list_input.txt', 'w') as f:
    for s2_cell_token in s2cell_inputs:
      f.write(s2_cell_token + '\n')

  # Write the config to the output directory.
  with gfile.Open(f'{_OUTPUT_PREFIX.value}/satellite_config.json', 'w') as f:
    json.dump(
        {
            'data': dataclasses.asdict(data_config),
        },
        f,
        indent=2,
    )

  # Build training and validation set.
  with beam.Pipeline(runner.FlumeRunner()) as pipeline:
    # Extract retrieval meta-data for the runs.
    s2cells = pipeline | 'InitializeS2Cells' >> beam.Create(s2cell_inputs)
    if data_config.target_level is not None:
      s2cells = (
          s2cells
          | 'ExpandS2Cells'
          >> beam.ParDo(pipeline_steps.ExpandS2Cells(data_config.target_level))
          | 'FilterDuplicates' >> beam.Distinct()
      )
    aerial_scenes = s2cells | 'FetchDSMImages' >> beam.ParDo(
        export.FetchDSMImages(data_config)
    )

    features_dict = tfds.features.FeaturesDict(data_config.data_spec())

    # Serialize the aerial data.
    _ = export.serialize_and_write_examples(
        aerial_scenes,
        features_dict,
        _OUTPUT_PREFIX.value,
        split_name='dsm',
    )

  # Write the beam metrics to the output directory.
  pipeline_steps.write_metrics(pipeline.result, _OUTPUT_PREFIX.value)

  # Write the metadata for train, valid and test splits.
  data_utils.write_tfds_metadata(
      data_dir=_OUTPUT_PREFIX.value, features_dict=features_dict
  )


if __name__ == '__main__':
  app.run(main)
