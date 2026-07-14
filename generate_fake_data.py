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

import json
import os

from absl import app
from absl import flags
from array_record.python import array_record_module as arw
import numpy as np
import tensorflow_datasets as tfds

from scaling_geoloc.data import dummy_data_source as dds
from scaling_geoloc.data import spec as data_spec
from scaling_geoloc.data import joined_data_source as jds

FLAGS = flags.FLAGS
flags.DEFINE_string('output_dir', 'datasets/city', 'Output directory.')
flags.DEFINE_integer('num_examples', 64, 'Number of training examples.')
flags.DEFINE_integer('num_cells', 1000, 'Number of S2 cells in the grid.')
flags.DEFINE_integer('seed', 0, 'Random seed.')


def _query_feature_spec() -> dict:
  return data_spec.QuerySpec(encoding_format=None).spec()


def _aerial_feature_spec() -> dict:
  return data_spec.AerialSpec(encoding_format=None).spec()


def _query_index_feature_spec() -> dict:
  """cell_idx + query adjacent fields, to be deep-merged at top level."""
  adj = {
      'adjacent_cell_idx': tfds.features.Tensor(
          shape=(data_spec.NUM_ADJACENT_CELLS,), dtype=np.int32
      ),
      'adjacent_weights': tfds.features.Tensor(
          shape=(data_spec.NUM_ADJACENT_CELLS,), dtype=np.float32
      ),
  }
  return {
      'cell_idx': tfds.features.Tensor(shape=(), dtype=np.int32),
      'query': adj,
  }


def _aerial_index_feature_spec() -> dict:
  """aerial adjacent fields only, to be deep-merged at top level."""
  return {
      'aerial': {
          'adjacent_cell_idx': tfds.features.Tensor(
              shape=(data_spec.NUM_ADJACENT_CELLS,), dtype=np.int32
          ),
          'adjacent_weights': tfds.features.Tensor(
              shape=(data_spec.NUM_ADJACENT_CELLS,), dtype=np.float32
          ),
      }
  }


def _write_shard(
    output_dir: str,
    feature_spec: dict,
    records: list[dict],
    split: str = 'train',
) -> None:
  """Writes one ArrayRecord shard plus TFDS metadata to output_dir."""
  features = tfds.features.FeaturesDict(feature_spec)
  os.makedirs(output_dir, exist_ok=True)
  shard = os.path.join(output_dir, f'{split}-00000-of-00001.arrayrecord')

  writer = arw.ArrayRecordWriter(shard, 'group_size:1')
  for record in records:
    writer.write(features.serialize_example(record))
  writer.close()

  features.save_config(output_dir)

  dataset_info = {
      'fileFormat': 'array_record',
      'name': 'dummy',
      'splits': [{
          'filepathTemplate': '{SPLIT}-{SHARD_X_OF_Y}.arrayrecord',
          'name': split,
          'numBytes': str(os.path.getsize(shard)),
          'shardLengths': [str(len(records))],
      }],
      'version': '1.0.0',
  }
  with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
    json.dump(dataset_info, f, indent=2)

  print(f'Wrote {len(records)} records to {output_dir}')


def _write_dataset(output_dir: str, num_examples: int, num_cells: int) -> None:
  source = dds.DummyDataSource(num_examples=num_examples, num_cells=num_cells)
  examples = [source[i] for i in range(num_examples)]

  query_records = [{'image': e['query']['image'], 'latlng': e['query']['latlng']} for e in examples]
  aerial_records = [{'image': e['aerial']['image'], 'latlng': e['aerial']['latlng']} for e in examples]
  query_index_records = [
      {
          'cell_idx': e['cell_idx'],
          'query': {
              'adjacent_cell_idx': e['query']['adjacent_cell_idx'],
              'adjacent_weights': e['query']['adjacent_weights'],
          },
      }
      for e in examples
  ]
  aerial_index_records = [
      {
          'aerial': {
              'adjacent_cell_idx': e['aerial']['adjacent_cell_idx'],
              'adjacent_weights': e['aerial']['adjacent_weights'],
          },
      }
      for e in examples
  ]

  index_dir = os.path.join(output_dir, 'index_level17')
  _write_shard(os.path.join(output_dir, 'query'), _query_feature_spec(), query_records)
  _write_shard(os.path.join(output_dir, 'aerial'), _aerial_feature_spec(), aerial_records)
  _write_shard(os.path.join(index_dir, 'query'), _query_index_feature_spec(), query_index_records)
  _write_shard(os.path.join(index_dir, 'aerial'), _aerial_index_feature_spec(), aerial_index_records)


def _write_metadata(output_dir: str, num_cells: int, seed: int) -> None:
  rng = np.random.default_rng(seed + 1)
  cell_centers_latlng = rng.uniform(
      [-90.0, -180.0], [90.0, 180.0], size=(num_cells, 2)
  ).astype(np.float32)
  index_dir = os.path.join(output_dir, 'index_level17')
  os.makedirs(index_dir, exist_ok=True)
  np.savez(
      os.path.join(index_dir, 'metadata.npz'),
      cell_centers_latlng=cell_centers_latlng,
  )
  print(f'Wrote metadata to {index_dir}')


def main(argv) -> None:
  del argv
  _write_dataset(FLAGS.output_dir, FLAGS.num_examples, FLAGS.num_cells)
  _write_metadata(FLAGS.output_dir, FLAGS.num_cells, FLAGS.seed)

  # Sanity check: round-trip via JoinedDataSource.
  index_dir = os.path.join(FLAGS.output_dir, 'index_level17')
  query_source = tfds.builder_from_directory(
      os.path.join(FLAGS.output_dir, 'query')
  ).as_data_source()['train']
  aerial_source = tfds.builder_from_directory(
      os.path.join(FLAGS.output_dir, 'aerial')
  ).as_data_source()['train']
  query_index = tfds.builder_from_directory(
      os.path.join(index_dir, 'query')
  ).as_data_source()['train']
  aerial_index = tfds.builder_from_directory(
      os.path.join(index_dir, 'aerial')
  ).as_data_source()['train']

  train_source = jds.JoinedDataSource([
      (query_source, 'query'),
      (aerial_source, 'aerial'),
      (query_index, None),
      (aerial_index, None),
  ])

  example = train_source[0]
  assert set(example.keys()) == {'query', 'aerial', 'cell_idx'}, example.keys()
  assert set(example['query'].keys()) == {'image', 'latlng', 'adjacent_cell_idx', 'adjacent_weights'}
  assert set(example['aerial'].keys()) == {'image', 'latlng', 'adjacent_cell_idx', 'adjacent_weights'}
  print('Sanity check passed. Example keys:', {k: type(v) for k, v in example['query'].items()})


if __name__ == '__main__':
  app.run(main)
