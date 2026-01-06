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

from absl import app
from absl import flags
import apache_beam as beam

from scaling_geoloc.data import labels
from google3.pipeline.flume.py import runner


_OUTPUT_PREFIX = flags.DEFINE_string(
    'output_prefix', None, 'Output prefix.', required=True
)

_SPLIT_NAMES = flags.DEFINE_list(
    'split_names',
    None,
    'The name of the splits to extract labels from (train, valid, test).',
    required=True,
)

_OUTPUT_NAME = flags.DEFINE_string(
    'output_name',
    None,
    'The file name to write the labels to (should include .csv).',
    required=True,
)

_REQUIRED_KEYS = flags.DEFINE_string(
    'required_keys',
    None,
    'A comma-separated list of keys (nested via a.b.c) to extract from the'
    ' arrayrecord files.',
    required=True,
)

_DELIMITER = flags.DEFINE_string(
    'delimiter',
    ';',
    'The delimiter to use when writing the labels to a file.',
)

_NUM_SHARDS = flags.DEFINE_integer(
    'num_shards',
    5_000,  # Use something reasonable.
    'The number of shards to write.',
)


_WRITE_GROUPED_BY_LABELS = flags.DEFINE_bool(
    'write_grouped_by_labels',
    True,
    'Whether to write the grouped labels to a file.',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  runner.program_started()
  required_keys = [k.strip() for k in _REQUIRED_KEYS.value.split(',')]

  # Extract labels from arrayrecord files.
  with beam.Pipeline(runner.FlumeRunner()) as pipeline:

    for split_name in _SPLIT_NAMES.value:
      # Extract the labels for each split.
      split_labels = labels.extract_labels(
          pipeline,
          _OUTPUT_PREFIX.value,
          split_name,
          required_keys,
      )

      # Write flat labels per-row to a file (csv-format).
      labels.write_labels_to_csv(
          split_labels,
          f'{_OUTPUT_PREFIX.value}/{split_name}_{_OUTPUT_NAME.value}',
          required_keys,
          num_shards=_NUM_SHARDS.value,
          delimiter=_DELIMITER.value,
          tag=split_name,
      )

      if _WRITE_GROUPED_BY_LABELS.value:
        labels.write_grouped_labels_to_csv(
            split_labels,
            f'{_OUTPUT_PREFIX.value}/{split_name}_grouped_{_OUTPUT_NAME.value}',
            required_keys,
            delimiter=_DELIMITER.value,
            tag=split_name,
        )


if __name__ == '__main__':
  app.run(main)
