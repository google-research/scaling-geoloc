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

"""K-NN evaluation for classification models.

This script implements a k-NN evaluation protocol for classification models.
It extracts features for a training set (database) and a validation set
(queries), and then for each query, it finds the k-nearest neighbors in the
database and predicts the class via majority vote.
"""

import collections
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from absl import logging
from clu import metric_writers
import jax
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.train_lib import train_utils
import tensorflow as tf
from tensorflow.keras import optimizers

from unigeoclip.scenic import datasets
from unigeoclip.scenic.models import contrastive_merge
from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.evals import base_eval
from scaling_geoloc.scenic.models import base
from scaling_geoloc.utils import misc


def get_all_representations(
    data_iter: Iterator[types.Batch],
    train_state: train_utils.TrainState,
    model: base.BaseModel,
    log_progress_interval: int = 10,
) -> Mapping[str, Any]:
  """Extracts representations for all samples in the iterator."""

  def _get_reps(train_state, batch):
    variables = {'params': train_state.params, **train_state.model_state}
    # We assume the model's apply function returns embeddings in eval mode.
    return model.flax_model.apply(
        variables,
        jax.image.resize(
            batch['input'],
            (batch['input'].shape[0], 256, 256, 3),
            jax.image.ResizeMethod.LINEAR,
        ),
        method=model.flax_model.get_aerial_embeddings,
    )

  pmapped_get_reps = jax.pmap(_get_reps, axis_name='batch')
  return get_all_representations_pmapped(
      data_iter, train_state, pmapped_get_reps, log_progress_interval
  )


def get_all_representations_pmapped(
    data_iter: Iterator[types.Batch],
    train_state: train_utils.TrainState,
    pmapped_get_reps: Any,
    log_progress_interval: int = 10,
    return_input: bool = False,
) -> Mapping[str, Any]:
  """Extracts representations for all samples in the iterator."""

  results = collections.defaultdict(list)

  for i, batch in enumerate(data_iter):
    if i % log_progress_interval == 0:
      logging.info('Extracting features... [%d]', i)

    batch_jit = misc.filter_batch_for_jit(batch)
    batch_size = batch['input'].shape[0]
    num_devices = jax.device_count()
    pad_amount = -batch_size % num_devices
    if pad_amount > 0:
      batch_jit = jax.tree_util.tree_map(
          lambda x, pad_amount=pad_amount: np.pad(
              x, ((0, pad_amount),) + ((0, 0),) * (x.ndim - 1), 'constant'
          ),
          batch_jit,
      )
    batch_sharded = dataset_utils.shard(batch_jit)
    representations = pmapped_get_reps(train_state, batch_sharded)

    unsharded_reps = dataset_utils.unshard(representations)
    unsharded_reps = jax.tree_util.tree_map(
        lambda x, batch_size=batch_size: x[:batch_size], unsharded_reps
    )

    if isinstance(unsharded_reps, Mapping) and 'embedding' in unsharded_reps:
      results['embedding'].append(np.copy(unsharded_reps['embedding']))
    else:
      results['embedding'].append(np.copy(unsharded_reps))

    results['label'].append(np.copy(batch['label']))
    if return_input:
      results['input'].append(np.copy(batch['input']))

  return {k: np.concatenate(v, axis=0) for k, v in results.items()}


def get_linear_probing_metrics(
    query_reps: np.ndarray,
    query_labels: np.ndarray,
    db_reps: np.ndarray,
    db_labels: np.ndarray,
    batch_size: int = 64,
    learning_rate: float = 0.002,
    num_classes: int = 2,
    n_epochs: int = 500,
) -> float:
  """Computes linear probing metrics."""

  optim = optimizers.AdamW(learning_rate=learning_rate)

  module_list = []
  if num_classes == 2:
    module_list.append(tf.keras.layers.Dense(1, activation='sigmoid'))
  else:
    module_list.append(tf.keras.layers.Dense(num_classes, activation='softmax'))
  model = tf.keras.Sequential(module_list)

  if num_classes == 2:
    loss = 'binary_crossentropy'
  else:
    loss = 'sparse_categorical_crossentropy'

  model.compile(
      optimizer=optim,
      loss=loss,
      metrics=['accuracy'],
  )

  dataset = tf.data.Dataset.from_tensor_slices((db_reps, db_labels))
  dataset = (
      dataset.batch(batch_size)
      .shuffle(buffer_size=1024, reshuffle_each_iteration=True)
      .prefetch(tf.data.AUTOTUNE)
  )

  _ = model.fit(
      dataset,
      epochs=n_epochs,
      batch_size=batch_size,
      validation_split=0.0,
      verbose=False,
  )
  predict_dataset = tf.data.Dataset.from_tensor_slices(query_reps)
  predict_dataset = predict_dataset.batch(batch_size)

  probabilities = model.predict(predict_dataset, verbose=False)
  if num_classes == 2:
    predicted_classes = (probabilities > 0.5).astype(int)
  else:
    predicted_classes = probabilities.argmax(axis=-1)
  return float(np.mean(predicted_classes.squeeze() == query_labels))


def compute_metrics(
    *,
    query_reps: np.ndarray,
    query_labels: np.ndarray,
    db_reps: np.ndarray,
    db_labels: np.ndarray,
    batch_sizes: Sequence[int],
    learning_rates: Sequence[float],
    normalize_output_embeddings: bool = True,
    k_values: Sequence[int] = (1, 5, 10),
    num_classes: int = 2,
) -> dict[str, float]:
  """Computes k-NN classification accuracy."""
  metrics = {}

  accs = []
  mean = np.mean(db_reps, axis=0)
  std = np.std(db_reps, axis=0)
  train_embed = (db_reps - mean) / std
  valid_embed = (query_reps - mean) / std
  for batch_size in batch_sizes:
    for learning_rate in learning_rates:
      acc = get_linear_probing_metrics(
          valid_embed,
          query_labels,
          train_embed,
          db_labels,
          batch_size,
          learning_rate,
          num_classes,
      )
      accs.append(acc)
  metrics['linear_probing'] = np.max(accs)

  if normalize_output_embeddings:
    db_reps = contrastive_merge.normalize_features(db_reps)
    query_reps = contrastive_merge.normalize_features(query_reps)

  similarities = np.einsum('qe,de->qd', query_reps, db_reps)

  for k in k_values:
    top_k_indices = np.argsort(similarities, axis=-1)[..., -k:]
    top_k_labels = db_labels[top_k_indices]

    # Majority vote.
    preds = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), 1, top_k_labels
    )
    accuracy = np.mean(preds == query_labels.squeeze())
    metrics[f'knn_accuracy@{k}'] = accuracy * 100.0

  return metrics


class ClassificationEvaluation(base_eval.BaseEval):
  """Evaluation for classification."""

  def __init__(self, model: base.BaseModel, config: ml_collections.ConfigDict):
    self.config = config
    self.model = model

    self._dataset = datasets.build_dataset(
        config.data,
        batch_size=config.batch_size_eval,
        batch_size_eval=config.batch_size_eval,
        seed=155,
    )

    self.train_iter = self._dataset.train_iter
    self.valid_iter = self._dataset.valid_iter

  def get_database_representations(
      self,
      train_state: train_utils.TrainState,
  ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Extracts representations for all samples in the iterator."""
    logging.info('Extracting features for train set (database).')
    train_reps_data = get_all_representations(
        self.train_iter(),
        train_state,
        self.model,
        log_progress_interval=self.config.get('log_progress_interval', 10),
    )

    logging.info('Extracting features for validation set (queries).')
    valid_reps_data = get_all_representations(
        self.valid_iter(),
        train_state,
        self.model,
        log_progress_interval=self.config.get('log_progress_interval', 10),
    )

    return train_reps_data, valid_reps_data

  def evaluate(
      self,
      train_state: train_utils.TrainState,
      writer: metric_writers.MetricWriter | None = None,
      workdir: str | None = None,
  ) -> types.EvalResults:
    """Evaluates the model."""
    train_reps_data, valid_reps_data = self.get_database_representations(
        train_state
    )

    logging.info(
        'Shapes of train and valid reps: %s, %s',
        train_reps_data['embedding'].shape,
        valid_reps_data['embedding'].shape,
    )

    logging.info('Computing k-NN metrics.')
    eval_metrics = compute_metrics(
        query_reps=valid_reps_data['embedding'],
        query_labels=valid_reps_data['label'],
        db_reps=train_reps_data['embedding'],
        db_labels=train_reps_data['label'],
        normalize_output_embeddings=self.model.config.normalize_output_embeddings,
        k_values=self.config.get('k_values', (1, 5, 10)),
        batch_sizes=self.config.batch_sizes_eval,
        learning_rates=self.config.learning_rates,
        num_classes=self.config.num_classes,
    )

    return {}, eval_metrics, {}
