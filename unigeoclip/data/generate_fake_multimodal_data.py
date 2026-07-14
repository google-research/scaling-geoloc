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

"""Generate fake 5-modality training data on disk (sv/aerial/dsm/text/latlng).

Writes ArrayRecord shards + TFDS metadata so the multimodal contrastive model
can train from files (in addition to the procedural ``MultiModalDummyDataSource``
used by default). Point the training config at the output with
``--config.data.data_dir=<dir> --config.data.location=<location>``.

    python -m unigeoclip.data.generate_fake_multimodal_data \
        --output_dir=datasets --location=area_summaries
"""

import json
import os

from absl import app
from absl import flags
from array_record.python import array_record_module as arw
import tensorflow_datasets as tfds

from unigeoclip.data import spec as mm_spec
from unigeoclip.data.multimodal_dummy_source import MultiModalDummyDataSource

FLAGS = flags.FLAGS

# NB: flags are defined under ``__main__`` (not at import) so importing
# ``write_multimodal_dataset`` from tests doesn't register module-level flags
# (which would collide with the downstream generator's ``--output_dir``).


def _write_split(
    out_dir: str, features: tfds.features.FeaturesDict, records: list[dict],
    split: str, splits_info: list[dict],
) -> None:
  shard = os.path.join(out_dir, f'{split}-00000-of-00001.arrayrecord')
  writer = arw.ArrayRecordWriter(shard, 'group_size:1')
  for record in records:
    writer.write(features.serialize_example(record))
  writer.close()
  splits_info.append({
      'filepathTemplate': '{SPLIT}-{SHARD_X_OF_Y}.arrayrecord',
      'name': split,
      'numBytes': str(os.path.getsize(shard)),
      'shardLengths': [str(len(records))],
  })


def write_multimodal_dataset(
    out_dir: str,
    num_train: int = 64,
    num_valid: int = 16,
    *,
    sv_image_size: int = 224,
    aerial_image_size: int = 256,
    dsm_image_size: int = 256,
    text_dim: int = 768,
) -> str:
  """Writes the 5-modality ArrayRecord shards + TFDS metadata to ``out_dir``."""
  os.makedirs(out_dir, exist_ok=True)
  source_kwargs = dict(
      sv_image_size=sv_image_size,
      aerial_image_size=aerial_image_size,
      dsm_image_size=dsm_image_size,
      text_dim=text_dim,
  )
  features = tfds.features.FeaturesDict(
      mm_spec.multimodal_feature_spec(**source_kwargs)
  )

  splits = {'train': num_train, 'valid': num_valid}
  splits_info = []
  # Disjoint seed ranges per split so train/valid examples don't overlap.
  offset = 0
  for split, n in splits.items():
    source = MultiModalDummyDataSource(num_examples=offset + n, **source_kwargs)
    records = [source[i] for i in range(offset, offset + n)]
    _write_split(out_dir, features, records, split, splits_info)
    offset += n

  features.save_config(out_dir)
  with open(os.path.join(out_dir, 'dataset_info.json'), 'w') as f:
    json.dump(
        {
            'fileFormat': 'array_record',
            'name': 'unigeoclip_multimodal',
            'splits': splits_info,
            'version': '1.0.0',
        },
        f,
        indent=2,
    )
  return out_dir


def main(argv) -> None:
  del argv
  out_dir = write_multimodal_dataset(
      os.path.join(FLAGS.output_dir, FLAGS.location),
      num_train=FLAGS.num_train,
      num_valid=FLAGS.num_valid,
      sv_image_size=FLAGS.sv_image_size,
      aerial_image_size=FLAGS.aerial_image_size,
      dsm_image_size=FLAGS.dsm_image_size,
      text_dim=FLAGS.text_dim,
  )
  print(
      f'Wrote multimodal fake data (train={FLAGS.num_train}, '
      f'valid={FLAGS.num_valid}) to {out_dir}'
  )

  # Sanity check: round-trip one example through TFDS.
  example = tfds.builder_from_directory(out_dir).as_data_source('train')[0]
  assert set(example.keys()) == {
      'latlng', 'query', 'aerial', 'dsm', 'siglip_embeddings'
  }, example.keys()
  assert set(example['query'].keys()) == {'image_pixels'}
  print('Round-trip OK. Example modalities:', sorted(example.keys()))


if __name__ == '__main__':
  flags.DEFINE_string('output_dir', 'datasets', 'Root output directory.')
  flags.DEFINE_string('location', 'area_summaries', 'Location subdirectory name.')
  flags.DEFINE_integer('num_train', 64, 'Number of train examples.')
  flags.DEFINE_integer('num_valid', 16, 'Number of valid examples.')
  flags.DEFINE_integer('sv_image_size', 224, 'Street-view image size.')
  flags.DEFINE_integer('aerial_image_size', 256, 'Aerial tile size.')
  flags.DEFINE_integer('dsm_image_size', 256, 'DSM tile size.')
  flags.DEFINE_integer('text_dim', 768, 'Precomputed SigLIP text embedding dim.')
  app.run(main)
