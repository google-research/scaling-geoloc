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

"""Evaluation for dsm encoder.

This script implements an evaluation protocol for dsm encoder on segmentation.
It extracts features for a training set (database) and a validation set
(queries), and then for each query performs linear segmentation.
"""

from collections.abc import Iterator, Mapping, Sequence
import functools
from typing import Any

from absl import logging
from clu import metric_writers
import jax
import ml_collections
import numpy as np
from scenic.train_lib import train_utils
import tensorflow as tf
from tensorflow.keras import optimizers

from unigeoclip.scenic import datasets
from unigeoclip.scenic.evals import classif_eval
from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.evals import base_eval
from scaling_geoloc.scenic.models import base

partial = functools.partial


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
        batch['input'],
        method=model.flax_model.get_dense_dsm_embeddings,
    )

  pmapped_get_reps = jax.pmap(_get_reps, axis_name='batch')
  return classif_eval.get_all_representations_pmapped(
      data_iter, train_state, pmapped_get_reps, log_progress_interval
  )


def reshape_output(
    y_pred: tf.Tensor, num_classes: int, patch_size: int
) -> tf.Tensor:
  """Applies the custom reshaping and transposing logic."""
  batch_size = tf.shape(y_pred)[0]
  height = tf.shape(y_pred)[1]
  width = tf.shape(y_pred)[2]

  y_pred = tf.reshape(
      y_pred, (batch_size, height, width, patch_size, patch_size, num_classes)
  )
  y_pred = tf.transpose(y_pred, perm=[0, 1, 3, 2, 4, 5])
  y_pred = tf.reshape(
      y_pred, (batch_size, height * patch_size, width, patch_size, num_classes)
  )
  y_pred = tf.reshape(
      y_pred, (batch_size, height * patch_size, width * patch_size, num_classes)
  )
  return y_pred


def get_loss_fn(
    y_true: tf.Tensor, y_pred: tf.Tensor, num_classes: int
) -> tf.Tensor:
  y_pred = tf.reshape(y_pred, (-1, num_classes))
  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, ignore_class=-1
  )
  return loss(y_true, y_pred)


def get_metrics(
    query_reps: np.ndarray,
    query_labels: np.ndarray,
    db_reps: np.ndarray,
    db_labels: np.ndarray,
    *,
    batch_size: int = 32,
    learning_rate: float = 0.002,
    num_classes: int = 10,
    n_epochs: int = 200,
    patch_size: int = 2,
) -> tuple[float, float]:
  """Computes linear probing metrics."""
  optim = optimizers.SGD(learning_rate=learning_rate)

  module_list = [
      tf.keras.layers.Dense(num_classes * patch_size * patch_size),
      tf.keras.layers.Lambda(
          partial(
              reshape_output, num_classes=num_classes, patch_size=patch_size
          )
      ),
  ]
  model = tf.keras.Sequential(module_list)

  model.compile(
      optimizer=optim,
      loss=partial(get_loss_fn, num_classes=num_classes),
      metrics=[
          ArgmaxMetricWrapper(
              tf.keras.metrics.MeanIoU(
                  num_classes=num_classes, ignore_class=-1
              ),
              name='mIoU',
          ),
          Acc(name='acc'),
      ],
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
  predict_dataset = tf.data.Dataset.from_tensor_slices(
      (query_reps, query_labels)
  )
  predict_dataset = predict_dataset.batch(batch_size)

  results = model.evaluate(predict_dataset, verbose=False)
  return results[1], results[2]


def compute_metrics(
    *,
    query_reps: np.ndarray,
    query_labels: np.ndarray,
    num_classes: int,
    db_reps: np.ndarray,
    db_labels: np.ndarray,
    batch_sizes: Sequence[int],
    learning_rates: Sequence[float],
) -> dict[str, float]:
  """Computes segmentation metrics."""
  # Assuming the image is square, which leads to a squared number of tokens.
  if db_reps.shape[1] != int(db_reps.shape[1] ** 0.5) ** 2:
    raise ValueError('Non-square number of tokens in database')
  if query_reps.shape[1] != int(query_reps.shape[1] ** 0.5) ** 2:
    raise ValueError('Non-square number of tokens in query')

  db_reps = np.reshape(
      db_reps,
      [
          db_reps.shape[0],
          int(db_reps.shape[1] ** 0.5),
          int(db_reps.shape[1] ** 0.5),
          -1,
      ],
  )
  query_reps = np.reshape(
      query_reps,
      [
          query_reps.shape[0],
          int(query_reps.shape[1] ** 0.5),
          int(query_reps.shape[1] ** 0.5),
          -1,
      ],
  )

  miou_list = []
  acc_list = []
  for batch_size in batch_sizes:
    for learning_rate in learning_rates:
      miou, acc = get_metrics(
          query_reps,
          query_labels,
          db_reps,
          db_labels,
          batch_size=batch_size,
          learning_rate=learning_rate,
          num_classes=num_classes,
      )
      miou_list.append(miou)
      acc_list.append(acc)
  idx = np.argmax(miou_list)
  miou = miou_list[idx]
  acc = acc_list[idx]
  return {'mIoU': miou, 'acc': acc}


class DsmSegmentationEvaluation(base_eval.BaseEval):
  """Evaluation for segmentation of dsm."""

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

    logging.info('Computing segmentation metrics.')
    eval_metrics = compute_metrics(
        query_reps=valid_reps_data['embedding'],
        query_labels=valid_reps_data['label'],
        db_reps=train_reps_data['embedding'],
        db_labels=train_reps_data['label'],
        batch_sizes=self.config.batch_sizes_eval,
        learning_rates=self.config.learning_rates,
        num_classes=self.config.num_classes,
    )

    return {}, eval_metrics, {}


class ArgmaxMetricWrapper(tf.keras.metrics.Metric):
  """A wrapper that applies argmax to predictions before updating a metric."""

  def __init__(self, metric, name='argmax_metric_wrapper', **kwargs):
    super().__init__(name=name, **kwargs)
    # The metric to wrap (e.g., an instance of MeanIoU)
    self.metric = metric

  def update_state(
      self,
      y_true: tf.Tensor,
      y_pred: tf.Tensor,
      sample_weight: tf.Tensor | None = None,
  ) -> None:
    # Convert model predictions (logits/probabilities) to class indices
    y_pred_argmax = tf.argmax(y_pred, axis=-1)

    # Pass the true labels and the class predictions to the wrapped metric
    self.metric.update_state(y_true, y_pred_argmax, sample_weight)

  def result(self) -> tf.Tensor:
    return self.metric.result()

  def reset_state(self) -> None:
    self.metric.reset_state()


class Acc(tf.keras.metrics.Metric):
  """A custom accuracy metric for semantic segmentation that ignores a specific class."""

  def __init__(self, name='masked_accuracy', ignore_class_id=-1, **kwargs):
    super().__init__(name=name, **kwargs)
    self.ignore_class_id = ignore_class_id

    self.total_correct = self.add_weight(
        name='total_correct', initializer='zeros', dtype=tf.float32
    )
    self.total_counted = self.add_weight(
        name='total_counted', initializer='zeros', dtype=tf.float32
    )

  def update_state(
      self,
      y_true: tf.Tensor,
      y_pred: tf.Tensor,
      sample_weight: tf.Tensor | None = None,
  ) -> None:
    y_pred_labels = tf.argmax(y_pred, axis=-1)
    y_true = tf.cast(y_true, y_pred_labels.dtype)

    mask = tf.not_equal(y_true, self.ignore_class_id)
    correct_predictions = tf.equal(y_true, y_pred_labels)

    masked_correct = tf.logical_and(correct_predictions, mask)

    self.total_correct.assign_add(
        tf.reduce_sum(tf.cast(masked_correct, tf.float32))
    )

    self.total_counted.assign_add(tf.reduce_sum(tf.cast(mask, tf.float32)))

  def result(self) -> tf.Tensor:
    return tf.math.divide_no_nan(self.total_correct, self.total_counted)

  def reset_state(self) -> None:
    self.total_correct.assign(0.0)
    self.total_counted.assign(0.0)
