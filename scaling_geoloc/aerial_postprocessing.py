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

"""Flume pipeline to extract labels from arrayrecord files."""

from collections.abc import Sequence
import json
from typing import Any, TypeAlias

from absl import app
from absl import flags
import apache_beam as beam
import fancyflags as ff
import tensorflow_datasets as tfds

from scaling_geoloc.data import export
from google3.pipeline.flume.py import runner
from google3.pyglib import gfile
from google3.util.geometry.python import s2

Example: TypeAlias = Any

_INPUT_PREFIX = flags.DEFINE_string(
    'input_prefix', None, 'input prefix.', required=True
)

_OUTPUT_PREFIX = flags.DEFINE_string(
    'output_prefix', None, 'Output prefix.', required=True
)

_SPLIT_NAME = flags.DEFINE_string(
    'split_name',
    'aerial',
    'The name of the split.',
)

_OUTPUT_SPLIT_NAME = flags.DEFINE_string(
    'output_split_name',
    None,
    'The name of the output split.',
)

_DATA = ff.DEFINE_auto('data', export.AerialDataConfig)

_REQUIRED_S2CELL_FILE = flags.DEFINE_string(
    'required_s2cell_file',
    None,
    'If not None, only process the given s2cells.',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  data_config = _DATA.value()
  builder = tfds.builder_from_directory(_INPUT_PREFIX.value)

  with gfile.Open(f'{_INPUT_PREFIX.value}/satellite_config.json', 'r') as f:
    input_config = export.AerialDataConfig(**json.load(f)['data'])

  assert input_config.target_level == data_config.target_level

  if _REQUIRED_S2CELL_FILE.value is not None:
    s2cell_file = str(_REQUIRED_S2CELL_FILE.value)
    if s2cell_file.endswith('.txt'):
      with gfile.Open(s2cell_file, 'r') as f:
        required_s2cells = [line.strip() for line in f if line.strip()]
    elif s2cell_file.endswith('.json'):
      with gfile.Open(s2cell_file, 'r') as f:
        required_s2cells = list(json.load(f).keys())
    else:
      raise ValueError('Unsupported file type for s2cell_file')
    # Cast to parent level and make unique.
    required_s2cells = set([
        s2.S2CellId.from_token(token.encode('utf-8'))
        .parent_at_level(data_config.target_level)
        .token()
        for token in required_s2cells
    ])
  else:
    required_s2cells = None

  runner.program_started()

  # Extract labels from arrayrecord files.
  with beam.Pipeline(runner.FlumeRunner()) as pipeline:
    split_name = _SPLIT_NAME.value
    aerial_inputs = (
        pipeline
        | f'ReadFromTFDS_{split_name}'
        >> tfds.beam.ReadFromTFDS(builder, split=split_name)
        | f'ConvertToNumpy_{split_name}' >> beam.Map(tfds.as_numpy)
    )

    if required_s2cells is not None:
      aerial_inputs = (
          aerial_inputs
          | f'FilterByS2Cell_{split_name}'
          >> beam.Filter(lambda x: x['s2cell'] in required_s2cells)
      )

    aerial_scenes = aerial_inputs | f'CropAerials_{split_name}' >> beam.ParDo(
        export.CropsFromAerial(data_config)
    )

    features_dict = tfds.features.FeaturesDict(data_config.data_spec())

    export.serialize_and_write_examples(
        aerial_scenes,
        features_dict,
        _OUTPUT_PREFIX.value,
        split_name=_OUTPUT_SPLIT_NAME.value or split_name,
    )


if __name__ == '__main__':
  app.run(main)
