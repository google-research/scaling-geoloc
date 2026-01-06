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

"""Flume pipeline steps to extract labels from arrayrecord files."""

from collections.abc import Iterable, Sequence
import functools
from typing import Any, Callable, Final
import apache_beam as beam
import numpy as np
import tensorflow_datasets as tfds
from scaling_geoloc.data import export

SPECIAL_FUNCTIONS: Final[dict[str, Callable[[export.Example], Any]]] = {
    'frustums_ll': export.frustums_latlng_from_example,
    'center_ll': export.center_latlng_from_example,
    'center_ecef': export.center_ecef_from_example,
    's2cell11': lambda ex: export.s2cell_token_at_level(ex, level=11),
    's2cell12': lambda ex: export.s2cell_token_at_level(ex, level=12),
    's2cell13': lambda ex: export.s2cell_token_at_level(ex, level=13),
    's2cell14': lambda ex: export.s2cell_token_at_level(ex, level=14),
    's2cell15': lambda ex: export.s2cell_token_at_level(ex, level=15),
    's2cell16': lambda ex: export.s2cell_token_at_level(ex, level=16),
    's2cell17': lambda ex: export.s2cell_token_at_level(ex, level=17),
    's2cell18': lambda ex: export.s2cell_token_at_level(ex, level=18),
}


class ExtractLabels(beam.DoFn):
  """Extract labels from examples."""

  def __init__(self, required_keys: Sequence[str]):
    self.required_keys = required_keys

  def process(
      self, idx_and_example: tuple[int, export.Example]
  ) -> Iterable[tuple[int, Sequence[str]]]:
    """Extracts labels from examples."""
    idx, example = idx_and_example
    all_values = []
    for key in self.required_keys:
      extract_fn = SPECIAL_FUNCTIONS.get(
          key, functools.partial(export.get_nested, key=key)
      )
      values = extract_fn(example)
      if isinstance(values, list):
        values = np.array(values)
      if isinstance(values, np.ndarray):
        values = values.tolist()
      if isinstance(values, bytes):
        values = values.decode('utf-8')
      all_values.append(str(values))
    yield idx, all_values


def extract_labels(
    pipeline: beam.Pipeline,
    input_prefix: str,
    split_name: str,
    required_keys: Sequence[str],
    tag: str | None = None,
) -> beam.PCollection[tuple[int, Sequence[str]]]:
  """Extracts labels from examples."""
  builder = tfds.builder_from_directory(input_prefix)
  read_config = tfds.ReadConfig(add_tfds_id=True)
  if tag is None:
    tag = split_name
  labels = (
      pipeline
      | f'ReadFromTFDS_{tag}'
      >> tfds.beam.ReadFromTFDS(
          builder, split=split_name, read_config=read_config
      )
      | f'ConvertToNumpy_{tag}' >> beam.Map(tfds.as_numpy)
      | f'AddDataSourceIndex_{tag}'
      >> beam.Map(
          lambda x: (
              export.tfds_id_to_data_source_index(builder.info, x['tfds_id']),
              x,
          )
      )
      | f'ExtractLabels_{tag}' >> beam.ParDo(ExtractLabels(required_keys))
  )
  return labels


def write_labels_to_csv(
    labels: beam.PCollection[tuple[int, Sequence[str]]],
    output_file: str,
    required_keys: Sequence[str],
    num_shards: int = 5000,  # Use something reasonable.
    delimiter: str = ';',
    tag: str = '',
) -> None:
  """Writes labels to a csv file."""
  _ = (
      labels
      | f'JoinIndexAndLabels_{tag}'
      >> beam.Map(lambda x: delimiter.join([str(x[0]), *x[1]]))
      | f'WriteToFile_{output_file}'
      >> beam.io.WriteToText(
          output_file,
          header=delimiter.join(['index'] + list(required_keys)),
          num_shards=num_shards,
      )
  )


def write_grouped_labels_to_csv(
    labels: beam.PCollection[tuple[int, Sequence[str]]],
    output_file: str,
    required_keys: Sequence[str],
    delimiter: str = ';',
    tag: str = '',
) -> None:
  """Writes grouped labels to a csv file."""
  _ = (
      labels
      | f'SwapIndexAndLabels_{tag}'
      >> beam.Map(lambda x: (delimiter.join(x[1]), x[0]))
      | f'GroupLabels_{tag}' >> beam.GroupByKey()
      | f'MakeIndexSequence_{tag}'
      >> beam.MapTuple(
          lambda labels, indices: delimiter.join([labels, str(list(indices))])
      )
      | f'WriteGroupedToFile_{tag}'
      >> beam.io.WriteToText(
          output_file,
          header=delimiter.join(list(required_keys) + ['indices']),
      )
  )
