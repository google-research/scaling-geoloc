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

"""Metrics for full set evaluation."""

import collections
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, TypeAlias

from absl import logging
from clu import metric_writers
from etils.array_types import FloatArray
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.train_lib import train_utils

from unigeoclip import datasets as multimodal_datasets
from unigeoclip.evals import utils
from unigeoclip.models import contrastive_merge
from scaling_geoloc.data import datasets
from scaling_geoloc import types
from scaling_geoloc.evals import base_eval
from scaling_geoloc.models import base
from scaling_geoloc.utils import geo
from scaling_geoloc.utils import misc


Batch: TypeAlias = dict[str, Any]
Predictions: TypeAlias = FloatArray['... D']


def get_all_representations(
    data_iter: Iterator[types.Batch],
    num_steps: int,
    train_state: train_utils.TrainState,
    model: base.BaseModel,
    log_progress_interval: int = 10,
) -> dict[str, Any]:
  """Extracts representations for all samples in the iterator."""

  def _get_reps(train_state, batch):
    variables = {'params': train_state.params, **train_state.model_state}
    return model.flax_model.apply(variables, batch, train=False)

  pmapped_get_reps = jax.pmap(_get_reps, axis_name='batch')
  results = collections.defaultdict(list)

  for i in range(num_steps):
    if i % log_progress_interval == 0:
      logging.info('Extracting features... [%d/%d]', i, num_steps)

    batch = next(data_iter)
    # The model expects the original batch structure.
    batch = contrastive_merge.flatten_batch(batch)
    batch_jit = misc.filter_batch_for_jit(batch)
    batch_sharded = dataset_utils.shard(batch_jit)
    representations = pmapped_get_reps(train_state, batch_sharded)

    # Filter out non-array or scalar entries before unsharding.
    sharded_preds = {
        k: v
        for k, v in representations.items()
        if 'preds' in k or 'latlngproj' in k
    }
    unsharded_preds = dataset_utils.unshard(sharded_preds)

    for key, value in unsharded_preds.items():
      results[key].append(np.asarray(value))

    results['latlng'].append(np.asarray(batch['latlng']))
    if 'center_ll' in batch['aerial']:
      results['aerial_latlng'].append(np.asarray(batch['aerial']['center_ll']))
    else:
      raise ValueError(f'No center_ll in aerial: {batch["aerial"].keys()}')

    if 'center_ll' in batch['dsm']:
      results['dsm_latlng'].append(np.asarray(batch['dsm']['center_ll']))
    else:
      raise ValueError(f'No center_ll in dsm: {batch["dsm"].keys()}')

  return {k: np.concatenate(v, axis=0) for k, v in results.items()}


def get_ensemble_distance_meters(
    query_ll: FloatArray['... 2'],
    database_ll: FloatArray['... 2'],
    max_k: int,
    num_queries: int,
    batch_size: int,
    sv_pairs_for_ensemble: Sequence[tuple[Predictions, Predictions]],
) -> FloatArray['N D']:
  """Returns the distance between the query and prediction latlngs."""
  all_distance_meters = []
  for start_idx in range(0, num_queries, batch_size):
    end_idx = min(start_idx + batch_size, num_queries)

    batch_sim_stack = []
    for pred_a_ens, pred_b_ens in sv_pairs_for_ensemble:
      batch_pred_a = pred_a_ens[start_idx:end_idx]
      sim = jnp.einsum('bd,cd->bc', batch_pred_a, pred_b_ens)
      batch_sim_stack.append(sim)

    similarities_batch = jnp.stack(batch_sim_stack, axis=0).mean(axis=0)

    log_probs = nn.log_softmax(similarities_batch, axis=-1)
    probs = jnp.exp(log_probs)

    pred_assignment_batch = jax.lax.top_k(probs, k=max_k)[1]

    query_ll_batch = query_ll[start_idx:end_idx]
    pred_ll_batch = database_ll[pred_assignment_batch]
    distance_meters_batch = geo.haversine_distance(
        query_ll_batch[:, None, :], pred_ll_batch
    ).squeeze(-1)
    all_distance_meters.append(distance_meters_batch)
  return all_distance_meters


def normalize_features(
    features: jt.Float[jt.Array, 'B C'],
    eps: float = 1e-5,
) -> jt.Float[jt.Array, 'B C']:
  """Normalizes features across the feature dimension."""
  invalid = np.linalg.norm(features, axis=-1, keepdims=True) < eps
  y = np.where(invalid, eps, features)
  z = features / np.linalg.norm(y, axis=-1, keepdims=True)
  return np.where(invalid, 0, z.astype(features.dtype))


def compute_metrics(
    query_reps: Mapping[str, Any],
    database_reps: Mapping[str, Any],
    modalities: Sequence[str],
    config: ml_collections.ConfigDict,
) -> dict[str, Any]:
  """Computes retrieval metrics across the full set."""
  metrics = {}
  modalities = sorted(modalities)
  pairs = collections.defaultdict(list)

  for mod_a in modalities:
    for mod_b in modalities:
      if (
          mod_a == mod_b
          or f'{mod_a}_preds' not in query_reps
          or f'{mod_b}_preds' not in database_reps
      ):
        continue
      pairs[tuple(sorted((mod_a, mod_b)))].append((mod_a, mod_b))

  sv_pairs_for_ensemble = []
  for pair, _ in pairs.items():
    mod_a, mod_b = pair
    if mod_b == 'sv':
      mod_a, mod_b = mod_b, mod_a

    if config.all2all_proj:
      if 'cell_code' in mod_b:
        pred_a = query_reps[f'{mod_a}-latlng_preds']
        pred_b = database_reps[f'{mod_b}_preds']
      else:
        pred_a = query_reps[f'{mod_a}-{mod_b}_preds']
        pred_b = database_reps[f'{mod_b}-{mod_a}_preds']
    else:
      pred_a = query_reps[f'{mod_a}_preds']
      pred_b = database_reps[f'{mod_b}_preds']

      if config.latlng_proj and 'latlng' in pair:
        if mod_a == 'latlng' or 'cell_code' in mod_a:
          pred_b = database_reps[f'{mod_b}_latlngproj']
        elif mod_b == 'latlng' or 'cell_code' in mod_b:
          pred_a = query_reps[f'{mod_a}_latlngproj']

    batch_size = config.get('retrieval_batch_size', config.batch_size_eval)
    num_queries = pred_a.shape[0]
    max_k = min(pred_b.shape[0], 100)
    all_pred_assignment = []
    all_is_correct = []

    for start_idx in range(0, num_queries, batch_size):
      end_idx = min(start_idx + batch_size, num_queries)
      is_correct_batch, pred_assignment_batch = utils.get_is_correct_batch(
          pred_a, pred_b, max_k, start_idx, end_idx
      )
      all_is_correct.append(is_correct_batch)
      all_pred_assignment.append(pred_assignment_batch)

    pred_assignment = jnp.concatenate(all_pred_assignment, axis=0)
    is_correct = jnp.concatenate(all_is_correct, axis=0)

    if mod_a == 'sv' and mod_b != 'text':
      sv_pairs_for_ensemble.append((pred_a, pred_b))

    for k in [1, 5, 20, 100]:
      metrics[f'{mod_a}-{mod_b}_recall@{k:03d}'] = np.any(
          is_correct[:, :k], axis=-1
      ).mean()

    if mod_a == 'sv':
      if mod_b == 'latlng' or 'cell_code' in mod_b:
        database_ll = database_reps['latlng']
      elif mod_b == 'aerial':
        database_ll = database_reps['aerial_latlng']
      elif mod_b == 'dsm':
        database_ll = database_reps['dsm_latlng']
      else:
        continue

      query_ll = query_reps['latlng']
      pred_ll = database_ll[pred_assignment]

      distance_meters = geo.haversine_distance(query_ll[:, None, :], pred_ll)[
          ..., 0
      ]

      metrics.update(
          utils.get_distance_metrics(distance_meters, mod_b, max_k=max_k)
      )

  if len(sv_pairs_for_ensemble) > 1:
    batch_size = config.get('retrieval_batch_size', config.batch_size_eval)
    num_queries = sv_pairs_for_ensemble[0][0].shape[0]
    max_k = min(sv_pairs_for_ensemble[0][1].shape[0], 100)

    all_distance_meters = get_ensemble_distance_meters(
        query_ll=query_reps['latlng'],
        database_ll=database_reps['latlng'],
        max_k=max_k,
        num_queries=num_queries,
        batch_size=batch_size,
        sv_pairs_for_ensemble=sv_pairs_for_ensemble,
    )

    distance_meters = jnp.concatenate(all_distance_meters, axis=0)

    metrics.update(
        utils.get_distance_metrics(distance_meters, 'Ensemble', max_k=max_k)
    )
  return metrics


class RetrievalEvaluation(base_eval.BaseEval):
  """Evaluation for full set retrieval."""

  def __init__(self, model: base.BaseModel, config: ml_collections.ConfigDict):
    self.config = config
    self.model = model
    if config.get('multimodal_dummy', False):
      # Reproducible CPU path: sv/aerial/dsm/latlng from the procedural
      # MultiModalDummyDataSource, with query and database the same set.
      self._dataset = multimodal_datasets.create_multimodal_dataset(
          config.data,
          batch_size=config.batch_size_eval,
          batch_size_eval=config.batch_size_eval,
          seed=155,
      )
    else:
      transformations, metadata = datasets.transformations_from_config(
          config.data
      )
      self._dataset = datasets.create_dataset(
          config.data,
          batch_size=config.batch_size_eval,
          batch_size_eval=config.batch_size_eval,
          transformations=transformations,
          meta_data=metadata,
          seed=155,
      )

    self.eval_iter = self._dataset.valid_iter
    self.num_steps = (
        config.get('num_samples')
        or self._dataset.meta_data['num_valid_examples']
    ) // config.batch_size_eval

  def get_representations(
      self,
      train_state: train_utils.TrainState,
  ) -> Mapping[str, Any]:
    """Extracts representations for all samples in the iterator."""
    logging.info('Extracting features for test set.')
    eval_iter = self.eval_iter()
    representations = get_all_representations(
        eval_iter,
        self.num_steps,
        train_state,
        self.model,
        log_progress_interval=self.config.log_progress_interval,
    )
    return representations

  def evaluate(
      self,
      train_state: train_utils.TrainState,
      writer: metric_writers.MetricWriter | None = None,
      workdir: str | None = None,
  ) -> types.EvalResults:
    """Evaluates the model."""
    logging.info('Extracting features for test set.')
    eval_iter = self.eval_iter()
    representations = get_all_representations(
        eval_iter,
        self.num_steps,
        train_state,
        self.model,
        log_progress_interval=self.config.log_progress_interval,
    )
    del eval_iter

    representations = {
        k: normalize_features(v) if 'preds' in k else v
        for k, v in representations.items()
    }

    # For full set evaluation, query and database are the same.
    query_reps = representations
    database_reps = representations

    logging.info('Computing metrics.')

    self.config.all2all_proj = self.model.config.all2all_proj
    self.config.latlng_proj = (
        self.model.config.latlng_proj and not self.config.all2all_proj
    )
    self.config.normalize_output_embeddings = (
        self.model.config.normalize_output_embeddings
    )

    eval_metrics = compute_metrics(
        query_reps, database_reps, self.model.config.modalities, self.config
    )

    logging.info('Computing cell code metrics.')
    if self.config.cell_codes:
      # Build a gallery of candidate locations by binning the evaluation-set
      # locations onto a regular lat/lng grid. Each ``cell_code`` is a grid cell
      # size in degrees (coarser cells -> a smaller, coarser gallery).
      gallery_latlngs = query_reps['latlng']
      for cell_code in self.config.cell_codes:
        modality = f'cell_codes_{cell_code}'
        latlngs = utils.latlng_grid_gallery(gallery_latlngs, cell_code)
        representations = utils.get_latlng_representations(
            train_state,
            latlngs,
            self.model,
            batch_size=self.config.batch_size_eval,
        )

        # encode latlngs
        latlngs_encoded = {
            f'{modality}_preds': jnp.array(representations),
            'latlng': jnp.array(latlngs),
        }
        metrics = compute_metrics(
            query_reps, latlngs_encoded, ['sv', modality], self.config
        )
        eval_metrics.update(metrics)
    return {}, eval_metrics, {}
