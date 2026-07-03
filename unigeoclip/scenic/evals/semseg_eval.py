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

"""Evaluation for segmentation models.

This script implements an evaluation protocol for segmentation models.
It extracts features for a training set (database) and a validation set
(queries), and then performs segmentation with a DPT model.
"""

from collections.abc import Iterator, Mapping, Sequence
import functools
from typing import Any

from absl import logging
from clu import metric_writers
import flax.jax_utils
from flax.training import train_state as ts
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from scenic.train_lib import train_utils
import tensorflow as tf
import tensorflow_datasets as tfds

from unigeoclip.scenic import datasets
from unigeoclip.scenic.evals import classif_eval
from unigeoclip.scenic.models import dpt
from scaling_geoloc import types
from scaling_geoloc.evals import base_eval
from scaling_geoloc.models import base

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
    return model.flax_model.apply(
        variables,
        batch['input'],
        get_depths=(9, 6, 3),
        method=model.flax_model.get_dense_aerial_embeddings,
    )

  pmapped_get_reps = jax.pmap(_get_reps, axis_name='batch')
  return classif_eval.get_all_representations_pmapped(
      data_iter,
      train_state,
      pmapped_get_reps,
      log_progress_interval,
  )


def normalize_features(
    db_features: np.ndarray,
    query_features: np.ndarray,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
  """Normalizes features across the batch dimension."""
  mean = np.mean(db_features, axis=0, keepdims=True)
  norm = np.std(db_features, axis=0, keepdims=True)
  norm[norm < eps] = eps
  db_features = db_features - mean
  query_features = query_features - mean
  return db_features / norm, query_features / norm


def get_dtp_metrics(
    query_reps: np.ndarray,
    query_labels: np.ndarray,
    db_reps: np.ndarray,
    db_labels: np.ndarray,
    batch_size: int = 32,
    learning_rate: float = 0.002,
    num_classes: int = 10,
    n_epochs: int = 50,
) -> float:
  """Trains a DPT segmentation head and returns its mean IoU."""
  dim = query_reps.shape[-1]
  vit_channels = [dim for _ in range(4)]
  post_process_channels = [128, 256, 512, dim]

  num_devices = jax.local_device_count()

  steps_per_epoch = db_reps.shape[0] // batch_size

  train_db_split = [db_reps[:, i, ...] for i in range(db_reps.shape[1])][::-1]
  test_db_split = [query_reps[:, i, ...] for i in range(query_reps.shape[1])][
      ::-1
  ]

  train_slicing_components = (*train_db_split, db_labels)
  train_ds = tf.data.Dataset.from_tensor_slices(train_slicing_components)

  train_ds = (
      train_ds.shuffle(1000)
      .batch(batch_size, drop_remainder=True)
      .prefetch(tf.data.AUTOTUNE)
  )

  # Create the test dataset using the same simplified logic
  test_slicing_components = (*test_db_split, query_labels)
  test_ds_full = tf.data.Dataset.from_tensor_slices(test_slicing_components)
  test_ds_full = test_ds_full.batch(batch_size).prefetch(tf.data.AUTOTUNE)

  train_iter = tfds.as_numpy(train_ds)

  # --- Initialize Model and Replicate State ---
  key = jax.random.PRNGKey(0)
  # Split the main key
  _, model_key, dropout_init_key = jax.random.split(key, 3)
  dropout_keys = jax.random.split(dropout_init_key, num_devices)

  model = dpt.DPTHead(
      in_channels=vit_channels,
      post_process_channels=post_process_channels,
      n_output_channels=num_classes,
  )

  init_dummy_inputs = [pf[0:1] for pf in train_db_split]
  state = create_train_state(
      model,
      model_key,
      learning_rate,
      init_dummy_inputs,
      n_epochs,
      steps_per_epoch,
      warmup_epochs=1,
  )
  replicated_state = flax.jax_utils.replicate(state)

  for _ in range(1, n_epochs + 1):
    train_loss = 0.0
    num_train_steps = 0
    for batch_tuple in train_iter:
      # --- Explicitly Structure and Shard the Batch ---
      *input_maps, labels = batch_tuple

      # Shard the structured data
      sharded_images = [
          fm.reshape((num_devices, -1) + fm.shape[1:]) for fm in input_maps
      ]
      sharded_labels = labels.reshape((num_devices, -1) + labels.shape[1:])
      sharded_batch = (sharded_images, sharded_labels)

      # --- Run the Training Step ---
      replicated_state, loss = train_step(
          replicated_state, sharded_batch, dropout_keys
      )
      train_loss += loss[0]
      num_train_steps += 1

    train_loss /= num_train_steps

  final_state = flax.jax_utils.unreplicate(replicated_state)

  test_iter_full = tfds.as_numpy(test_ds_full)
  iou_metric = tf.keras.metrics.MeanIoU(
      num_classes=num_classes, name='mean_iou'
  )

  for batch_tuple in test_iter_full:
    *inputs, targets = batch_tuple
    batch = (inputs, targets)

    predictions, true_labels = predict_step(final_state, batch)
    iou_metric.update_state(true_labels, predictions)

  return iou_metric.result().numpy()


def predict_step(
    state: 'DPTTrainState', batch: tuple[Sequence[np.ndarray], np.ndarray]
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Generates predictions for a single batch on one device."""
  inputs, targets = batch
  logits = state.apply_fn(
      {'params': state.params, 'batch_stats': state.batch_stats},
      inputs,
      train=False,
  )
  # Get predicted class by finding the index with the highest score
  predictions = jnp.argmax(logits, axis=-1)

  # Ensure labels have the correct shape and type
  if targets.shape[-1] == 1:
    targets = jnp.squeeze(targets, axis=-1)
  targets = targets.astype(jnp.int32)

  return predictions, targets


@partial(jax.pmap, axis_name='batch')
def train_step(
    state: 'DPTTrainState',
    batch: tuple[Sequence[jnp.ndarray], jnp.ndarray],
    dropout_key: jnp.ndarray,
) -> tuple['DPTTrainState', jnp.ndarray]:
  """Train for a single step, parallelized across devices."""
  inputs, targets = batch
  dropout_train_key = jax.random.fold_in(dropout_key, state.step)

  def loss_fn(params):
    logits, new_model_state = state.apply_fn(
        {'params': params, 'batch_stats': state.batch_stats},
        inputs,
        train=True,
        mutable=['batch_stats'],
        rngs={'dropout': dropout_train_key},
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, targets
    ).mean()
    return loss, new_model_state

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, new_model_state), grads = grad_fn(state.params)

  # Average gradients and loss across all devices
  grads = jax.lax.pmean(grads, axis_name='batch')
  loss = jax.lax.pmean(loss, axis_name='batch')

  # Update state
  new_state = state.apply_gradients(
      grads=grads, batch_stats=new_model_state['batch_stats']
  )
  return new_state, loss


class DPTTrainState(ts.TrainState):
  batch_stats: Any


def create_train_state(
    model: dpt.DPTHead,
    key: jnp.ndarray,
    learning_rate: float,
    dummy_inputs: Sequence[jnp.ndarray],
    num_epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int,
) -> DPTTrainState:
  """Creates initial DPTTrainState."""
  variables = model.init(key, dummy_inputs, train=False)
  params = variables['params']
  batch_stats = variables.get('batch_stats', {})

  warmup_steps = warmup_epochs * steps_per_epoch
  total_steps = num_epochs * steps_per_epoch

  lr_schedule = optax.join_schedules(
      schedules=[
          optax.linear_schedule(
              init_value=0.0,
              end_value=learning_rate,
              transition_steps=warmup_steps,
          ),
          optax.cosine_decay_schedule(
              init_value=learning_rate, decay_steps=total_steps - warmup_steps
          ),
      ],
      boundaries=[warmup_steps],
  )

  tx = optax.adamw(learning_rate=lr_schedule)

  return DPTTrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=tx,
      batch_stats=batch_stats,
  )


def compute_metrics(
    *,
    query_reps: np.ndarray,
    query_labels: np.ndarray,
    num_classes: int,
    db_reps: np.ndarray,
    db_labels: np.ndarray,
    learning_rates: Sequence[float],
    batch_sizes: Sequence[int],
    normalize_output_embeddings: bool = True,
    n_epochs: int = 50,
) -> dict[str, float]:
  """Computes segmentation metrics."""
  metrics = {}
  # Assuming the image is square, it will have a squared number of tokens
  db_reps = np.reshape(
      db_reps,
      [
          db_reps.shape[0],
          db_reps.shape[1],
          int(db_reps.shape[2] ** 0.5),
          int(db_reps.shape[2] ** 0.5),
          -1,
      ],
  )
  query_reps = np.reshape(
      query_reps,
      [
          query_reps.shape[0],
          query_reps.shape[1],
          int(query_reps.shape[2] ** 0.5),
          int(query_reps.shape[2] ** 0.5),
          -1,
      ],
  )

  if normalize_output_embeddings:
    db_reps, query_reps = normalize_features(db_reps, query_reps)

  accs = []
  for batch_size in batch_sizes:
    for learning_rate in learning_rates:
      acc = get_dtp_metrics(
          query_reps,
          query_labels,
          db_reps,
          db_labels,
          batch_size=batch_size,
          learning_rate=learning_rate,
          num_classes=num_classes,
          n_epochs=n_epochs,
      )
      accs.append(acc)
  metrics['DTP'] = np.max(accs)

  return metrics


class SegmentationEvaluation(base_eval.BaseEval):
  """Evaluation for segmentation."""

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
        num_classes=self.config.num_classes,
        normalize_output_embeddings=True,
        batch_sizes=self.config.batch_sizes_eval,
        learning_rates=self.config.learning_rates,
        n_epochs=self.config.get('n_epochs', 50),
    )

    return {}, eval_metrics, {}
