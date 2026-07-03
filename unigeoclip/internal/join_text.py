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

from collections.abc import Iterable, Mapping, Sequence
import logging
import os
from typing import Any

from absl import app
from absl import flags
import apache_beam as beam
from big_vision.models.proj.image_text import text_transformer
from big_vision.models.proj.image_text import two_towers
import big_vision.pp.builder as pp_builder
import big_vision.pp.ops_text  # pylint: disable=unused-import
import big_vision.pp.proj.paligemma.ops  # pylint: disable=unused-import
import jax
import nltk
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from unigeoclip.scenic.models import siglip_text
from google3.location.visualmapping.experimental.fomo.data import data_utils
from scaling_geoloc.data import export
from scaling_geoloc.data import pipeline_steps
from google3.pipeline.flume.py import runner
from google3.pyglib import gfile
from google3.pyglib import resources
from google3.storage.f1.client.f1io.python import f1io
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

_EMBEDDING_DIM = 1152
_SIGLIP_EMBEDDING_KEY = 'siglip_So400m_embeddings'
_AREA_SUMMARIES_KEY = 'area_summaries'


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


class SigLipTextEmbedder:
  """A wrapper for SigLIP text model for easy sharing."""

  def __init__(self):
    """Initializes and loads the SigLIP model."""
    nltk.data.path.append(
        os.path.join(
            resources.GetARootDirWithAllResources(),
            'google3/third_party/py/nltk/nltk_data',
        )
    )
    self.pp_fn = pp_builder.get_preprocess_fn(
        f'lower(key="text")|tok(length={64}, model="gemma",'
        ' bos="no", eos="sticky", key="text")'
    )
    model = siglip_text.PreTrainedSigLip2TextModel(
        variant='So400m',
        output_dim=1152,
        vocab_size=256_000,
        model_name='SigLIP2 So400m/16 256',
    )
    dummy_inputs = np.zeros((1, 64), np.int32)
    variables = model.init(jax.random.PRNGKey(0), dummy_inputs)

    # Get checkpoint path for text tower
    init_files = two_towers.VANITY_NAMES.get(model.model_name, model.model_name)
    init_files = {k: f'{init_files}:{k}' for k in ('img', 'txt', 't')}
    txt_init = init_files.pop('text', init_files.pop('txt', None))

    loaded_model_params = text_transformer.load(
        variables['params']['model'], txt_init, {}
    )
    params = {'params': {'model': loaded_model_params}}

    def predict_fn(texts_tokenized):
      return model.apply(params, texts_tokenized, train=False)

    self.jitted_predict = jax.jit(predict_fn)

  def embed(self, texts: Sequence[str]) -> np.ndarray:
    """Embeds a list of texts."""
    if not texts:
      return np.empty((0, _EMBEDDING_DIM), dtype=np.float32)

    nested_sentences = [nltk.tokenize.sent_tokenize(t) for t in texts]
    sentences = []
    text_indices = []
    for i, sents in enumerate(nested_sentences):
      for s in sents:
        sentences.append(s)
        text_indices.append(i)

    if not sentences:
      return np.zeros((len(texts), _EMBEDDING_DIM), dtype=np.float32)

    tokenized_batch = [self.pp_fn({'text': t})['text'] for t in sentences]
    tokenized_batch = np.stack(tokenized_batch)
    sentence_embeddings = self.jitted_predict(tokenized_batch)

    # Group embeddings by text index and max pool.
    text_embeddings_list = [[] for _ in range(len(texts))]
    for i, embedding in zip(text_indices, sentence_embeddings):
      text_embeddings_list[i].append(embedding)

    pooled_embeddings = []
    for i in range(len(texts)):
      if text_embeddings_list[i]:
        pooled_embeddings.append(
            np.max(np.stack(text_embeddings_list[i]), axis=0)
        )
      else:
        pooled_embeddings.append(np.zeros((_EMBEDDING_DIM,), dtype=np.float32))

    return np.stack(pooled_embeddings)


class BatchEmbedText(beam.DoFn):
  """Embeds text in batches using a shared SigLIP model instance."""

  def __init__(self):
    self._shared_handle = beam.utils.shared.Shared()

  def setup(self):
    # beam.utils.shared.Shared is used to share the model instance across
    # threads in a worker. The model is loaded only once per process.
    self.embedder = self._shared_handle.acquire(SigLipTextEmbedder)

  def process(
      self, batch: Sequence[tuple[str, str]]
  ) -> Iterable[tuple[str, np.ndarray]]:
    batch_with_text = [element for element in batch if element[1]]
    if not batch_with_text:
      for cell_id, _ in batch:
        yield cell_id, np.zeros((_EMBEDDING_DIM,), dtype=np.float32)
      return

    texts = [text for _, text in batch_with_text]
    embeddings = self.embedder.embed(texts)

    embedding_by_cell_id = {
        element[0]: embedding
        for element, embedding in zip(batch_with_text, embeddings)
    }

    for cell_id, text in batch:
      if text:
        yield cell_id, embedding_by_cell_id[cell_id]
      else:
        yield cell_id, np.zeros((_EMBEDDING_DIM,), dtype=np.float32)


class AddEmbeddingFromCoGroupByKeyResult(beam.DoFn):
  """Adds text embeddings to examples after joining via CoGroupByKey."""

  def process(
      self,
      element: tuple[str, Mapping[str, Iterable[Any]]],
  ) -> Iterable[export.Example]:
    _, join_result = element
    embeddings = list(join_result['embeddings'])
    texts = list(join_result['texts'])

    embedding = embeddings[0] if embeddings else None

    if embedding is not None:
      text = texts[0] if texts else ''
      metric = 'examples_with_embedding'
    else:
      embedding = np.zeros((_EMBEDDING_DIM,), dtype=np.float32)
      text = ''
      metric = 'examples_without_embedding'

    for example in join_result['examples']:
      ex = dict(example)
      ex[_SIGLIP_EMBEDDING_KEY] = embedding
      ex[_AREA_SUMMARIES_KEY] = text
      beam.metrics.Metrics.counter(self.__class__.__name__, metric).inc()
      yield ex


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
  features_dict = tfds.features.FeaturesDict({
      **features_dict,
      _SIGLIP_EMBEDDING_KEY: tfds.features.Tensor(
          shape=(_EMBEDDING_DIM,), dtype=tf.float32
      ),
      _AREA_SUMMARIES_KEY: tfds.features.Text(),
  })

  # Unfortunately this is broken in tfds, so we have to manually concatenate...
  builders = []
  for location in _INPUT_LOCATIONS.value:
    builders.append(
        tfds.builder_from_directory(f'{_INPUT_PREFIX.value}/{location}')
    )

  runner.program_started()

  query = """
    SELECT
      cell_id,
      text
    FROM
      geo_ng_local_dev.area_summaries.neighborhood_v2
    WHERE
      cell_id != 0
  """

  # Extract labels from arrayrecord files.
  with beam.Pipeline(runner.FlumeRunner()) as pipeline:
    s2cell_text_pcoll = (
        pipeline
        | 'ReadFromF1' >> f1io.ReadFromF1Sql(sql_body=query)
        | 'ExtractS2CellText'
        >> beam.Map(lambda row: (str(row.cell_id), row.text))
    )
    s2cell_embedding_pcoll = (
        s2cell_text_pcoll
        | 'BatchTexts' >> beam.BatchElements(max_batch_size=128)
        | 'EmbedTexts' >> beam.ParDo(BatchEmbedText())
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

      # Add text from location.
      keyed_examples = data | f'KeyExamples_{split_name}' >> beam.Map(
          lambda ex: (
              str(s2.S2CellId.from_token(ex['s2cell']).parent().parent().id),
              ex,
          )
      )
      data = (
          ({
              'examples': keyed_examples,
              'texts': s2cell_text_pcoll,
              'embeddings': s2cell_embedding_pcoll,
          })
          | f'CoGroupByKey_{split_name}' >> beam.CoGroupByKey()
          | f'AddEmbeddingAfterJoin_{split_name}'
          >> beam.ParDo(AddEmbeddingFromCoGroupByKeyResult())
      )

      data = data | f'AddKeys_{split_name}' >> beam.Map(
          lambda x: (x['s2cell'], x)
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


if __name__ == '__main__':
  app.run(main)
