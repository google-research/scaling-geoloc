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
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from scenic.train_lib import train_utils

from unigeoclip import datasets
from unigeoclip.evals import classif_eval
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


def _depth_to_space(
    logits: jnp.ndarray, num_classes: int, patch_size: int
) -> jnp.ndarray:
  """Upsamples per-token logits to per-pixel logits (a pixel shuffle).

  ``[B, h, w, num_classes * patch_size**2]`` ->
  ``[B, h * patch_size, w * patch_size, num_classes]``, so each token expands
  into a ``patch_size x patch_size`` block and the prediction lines up with the
  label resolution.
  """
  b, h, w, _ = logits.shape
  x = logits.reshape(b, h, w, patch_size, patch_size, num_classes)
  x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
  return x.reshape(b, h * patch_size, w * patch_size, num_classes)


def _seg_logits(
    params: Mapping[str, jnp.ndarray],
    reps: jnp.ndarray,
    num_classes: int,
    patch_size: int,
) -> jnp.ndarray:
  """Linear head + pixel shuffle: ``[B, h, w, D]`` -> ``[B, H, W, num_classes]``."""
  tokens = reps @ params['kernel'] + params['bias']
  return _depth_to_space(tokens, num_classes, patch_size)


def _seg_loss(
    params: Mapping[str, jnp.ndarray],
    reps: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
    patch_size: int,
) -> jnp.ndarray:
  """Per-pixel softmax cross-entropy, masking the ignored class (label -1)."""
  logits = _seg_logits(params, reps, num_classes, patch_size).reshape(
      -1, num_classes
  )
  labels = labels.reshape(-1)
  mask = labels >= 0
  ce = optax.softmax_cross_entropy_with_integer_labels(
      logits, jnp.where(mask, labels, 0)
  )
  return jnp.sum(ce * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def _confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
  """Confusion matrix (rows = true, cols = pred) over the given pixels."""
  k = y_true.astype(np.int64) * num_classes + y_pred.astype(np.int64)
  return np.bincount(k, minlength=num_classes**2).reshape(
      num_classes, num_classes
  )


def _mean_iou(cm: np.ndarray) -> float:
  """Mean IoU, averaged over classes that appear in labels or predictions."""
  tp = np.diag(cm).astype(np.float64)
  denom = cm.sum(axis=0) + cm.sum(axis=1) - tp
  valid = denom > 0
  iou = np.where(valid, tp / np.maximum(denom, 1.0), 0.0)
  return float(iou[valid].mean()) if valid.any() else 0.0


def _pixel_accuracy(cm: np.ndarray) -> float:
  """Fraction of correctly classified (non-ignored) pixels."""
  total = cm.sum()
  return float(np.diag(cm).sum() / total) if total > 0 else 0.0


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
    seed: int = 0,
) -> tuple[float, float]:
  """Trains a linear DSM segmentation head in JAX and returns (mIoU, accuracy).

  The head is a single linear layer per ViT token whose output is pixel-shuffled
  up to the label resolution, trained with SGD on masked per-pixel cross-entropy.
  """
  out_dim = num_classes * patch_size * patch_size
  key = jax.random.PRNGKey(seed)
  key, init_key = jax.random.split(key)
  params = {
      'kernel': jax.nn.initializers.glorot_uniform()(
          init_key, (db_reps.shape[-1], out_dim)
      ),
      'bias': jnp.zeros((out_dim,)),
  }

  optimizer = optax.sgd(learning_rate)
  opt_state = optimizer.init(params)
  loss_fn = partial(_seg_loss, num_classes=num_classes, patch_size=patch_size)

  @jax.jit
  def train_step(params, opt_state, reps, labels):
    grads = jax.grad(loss_fn)(params, reps, labels)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state

  predict = jax.jit(
      partial(_seg_logits, num_classes=num_classes, patch_size=patch_size)
  )

  db_reps = jnp.asarray(db_reps)
  db_labels = jnp.asarray(db_labels)
  num_examples = db_reps.shape[0]
  for _ in range(n_epochs):
    key, perm_key = jax.random.split(key)
    perm = jax.random.permutation(perm_key, num_examples)
    for start in range(0, num_examples, batch_size):
      idx = perm[start : start + batch_size]
      params, opt_state = train_step(
          params, opt_state, db_reps[idx], db_labels[idx]
      )

  # Predict on the query set (batched) and accumulate a confusion matrix.
  query_reps = jnp.asarray(query_reps)
  cm = np.zeros((num_classes, num_classes), dtype=np.int64)
  for start in range(0, query_reps.shape[0], batch_size):
    preds = np.asarray(
        jnp.argmax(predict(params, query_reps[start : start + batch_size]), -1)
    )
    labels = np.asarray(query_labels[start : start + batch_size])
    valid = labels >= 0
    cm += _confusion_matrix(labels[valid], preds[valid], num_classes)

  return _mean_iou(cm), _pixel_accuracy(cm)


def compute_metrics(
    *,
    query_reps: np.ndarray,
    query_labels: np.ndarray,
    num_classes: int,
    db_reps: np.ndarray,
    db_labels: np.ndarray,
    batch_sizes: Sequence[int],
    learning_rates: Sequence[float],
    n_epochs: int = 200,
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
          n_epochs=n_epochs,
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
        n_epochs=self.config.get('n_epochs', 200),
    )

    return {}, eval_metrics, {}
