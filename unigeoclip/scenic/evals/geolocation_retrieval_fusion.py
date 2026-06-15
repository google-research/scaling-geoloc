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
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.train_lib import train_utils
import tqdm

from unigeoclip.scenic.evals import utils
from unigeoclip.scenic.models import contrastive_merge
from scaling_geoloc.scenic import datasets
from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.evals import base_eval
from scaling_geoloc.scenic.models import base
from scaling_geoloc.utils import io as lama_io
from scaling_geoloc.utils import misc
from google3.storage.f1.client import pywrap_client
from google3.util.geometry.python import s2

Batch: TypeAlias = dict[str, Any]
Predictions: TypeAlias = FloatArray['... D']


def get_all_representations(
    data_iter: Iterator[types.Batch],
    num_steps: int,
    train_state: train_utils.TrainState,
    model: base.BaseModel,
    log_progress_interval: int = 10,
    modalities: Sequence[str] = (),
) -> dict[str, Any]:
  """Extracts representations for all samples in the iterator."""

  def _get_reps(train_state, batch, modalities):
    variables = {'params': train_state.params, **train_state.model_state}
    pred = {}
    modalities_list = list(modalities)
    if 'area_embeddings' not in batch and 'text' in modalities_list:
      modalities_list.remove('text')
    for modality in modalities_list:
      pred[f'{modality}_preds'] = model.flax_model.apply(
          variables,
          batch,
          modalities=[modality],
          method=model.flax_model.get_fused_embeddings,
      )
    modalities_list.remove('sv')
    pred['fusion_preds'] = model.flax_model.apply(
        variables,
        batch,
        modalities=modalities_list,
        method=model.flax_model.get_fused_embeddings,
    )
    return pred

  pmapped_get_reps = jax.pmap(
      _get_reps, axis_name='batch', static_broadcasted_argnums=2
  )
  results = collections.defaultdict(list)

  for i in range(num_steps):
    if i % log_progress_interval == 0:
      logging.info('Extracting features... [%d/%d]', i, num_steps)

    batch = next(data_iter)
    # The model expects the original batch structure.
    batch = contrastive_merge.flatten_batch(batch)
    batch_jit = misc.filter_batch_for_jit(batch)
    batch_sharded = dataset_utils.shard(batch_jit)
    representations = pmapped_get_reps(
        train_state, batch_sharded, tuple(modalities)
    )

    unsharded_preds = dataset_utils.unshard(representations)

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
    pred_a: Predictions,
    pred_b: Predictions,
    max_k: int,
    num_queries: int,
    batch_size: int,
) -> FloatArray['N D']:
  """Returns the distance between the query and prediction latlngs."""
  all_distance_meters = []
  for start_idx in range(0, num_queries, batch_size):
    end_idx = min(start_idx + batch_size, num_queries)

    sim_batch = jnp.einsum('bd,cd->bc', pred_a[start_idx:end_idx], pred_b)

    pred_assignment_batch = jax.lax.top_k(sim_batch, k=max_k)[1]

    query_ll_batch = query_ll[start_idx:end_idx]
    pred_ll_batch = database_ll[pred_assignment_batch]
    distance_meters_batch = misc.haversine_distance(
        query_ll_batch[:, None, :], pred_ll_batch
    ).squeeze(-1)
    all_distance_meters.append(distance_meters_batch)
  return all_distance_meters


def compute_metrics(
    query_reps: Mapping[str, Any],
    database_reps: Mapping[str, Any],
    modalities: list[str],
    config: ml_collections.ConfigDict,
) -> dict[str, Any]:
  """Computes retrieval metrics across the full set."""
  metrics = {}
  modalities_filtered = []
  for modality in modalities:
    if (
        f'{modality}_preds' in query_reps
        or f'{modality}_preds' in database_reps
    ):
      modalities_filtered.append(modality)
  modalities = sorted(modalities_filtered)
  pairs = collections.defaultdict(list)

  for mod_a in modalities:
    for mod_b in modalities:
      if mod_a == mod_b:
        continue
      pairs[tuple(sorted((mod_a, mod_b)))].append((mod_a, mod_b))

  for pair, _ in pairs.items():
    mod_a, mod_b = pair
    if mod_b == 'sv':
      mod_a, mod_b = mod_b, mod_a

    pred_a = query_reps[f'{mod_a}_preds']
    pred_b = database_reps[f'{mod_b}_preds']

    batch_size = config.get('retrieval_batch_size', config.retrieval_batch_size)
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

      distance_meters = misc.haversine_distance(query_ll[:, None, :], pred_ll)[
          ..., 0
      ]

      metrics.update(
          utils.get_distance_metrics(distance_meters, mod_b, max_k=max_k)
      )

  if 'fusion_preds' not in database_reps.keys():
    return metrics

  pred_a = query_reps['sv_preds']
  pred_b = database_reps['fusion_preds']

  batch_size = config.get('retrieval_batch_size', config.batch_size_eval)
  num_queries = pred_a.shape[0]
  max_k = min(pred_a.shape[0], 100)

  all_distance_meters = get_ensemble_distance_meters(
      query_ll=query_reps['latlng'],
      database_ll=database_reps['latlng'],
      pred_a=pred_a,
      pred_b=pred_b,
      max_k=max_k,
      num_queries=num_queries,
      batch_size=batch_size,
  )

  distance_meters = jnp.concatenate(all_distance_meters, axis=0)

  metrics.update(
      utils.get_distance_metrics(distance_meters, 'Fusion', max_k=max_k)
  )
  return metrics


def get_urban_classification_s2_cells(
    tokens: list[bytes],
    *,
    batch_size: int = 1_000,
    f1_client: pywrap_client.Client | None = None,
) -> np.ndarray:
  """Returns the urban classification for each s2 cell token."""
  # go/sme/datasets/maui
  # Assume that all given s2 cells have level >= 13
  maui_level = 13
  parent_ids = np.array(
      [
          s2.S2CellId.from_token(token).parent_at_level(maui_level).id
          for token in tokens
      ],
      dtype=np.uint64,
  )
  parent_ids_unique, index_in_unique = np.unique(
      parent_ids, return_inverse=True
  )
  if f1_client is None:
    f1_client = pywrap_client.Client('/f1/query/prod')
  maui_results_for_unique_ids = {}
  urban_classification_map = {
      0: 'UNSPECIFIED',
      1: 'URBAN_HIGH_DENSITY',
      2: 'URBAN_MEDIUM_DENSITY',
      3: 'URBAN_LOW_DENSITY',
      4: 'TOWN_OR_SETTLEMENT',
      5: 'RURAL',
      6: 'SPARSE',
  }
  for i in tqdm.trange(0, len(parent_ids_unique), batch_size):
    batch_ids = parent_ids_unique[i : i + batch_size].tolist()
    sql_builder = pywrap_client.SqlBuilder("""
    SELECT
      cell_id, urban_classification
    FROM
      geo_data_scalable_metrics.population
      WHERE cell_id in UNNEST(@ids)
    """)
    sql_builder.AddUInt64List('ids', batch_ids)
    query = pywrap_client.QueryBuilder(sql_builder)
    response = f1_client.Query(query)
    for cell_id, urban_classification in response:
      maui_results_for_unique_ids[cell_id] = urban_classification_map.get(
          urban_classification, 'UNSPECIFIED'
      )
  maui_unique_ids = [
      maui_results_for_unique_ids[id_] for id_ in parent_ids_unique
  ]
  return np.array(maui_unique_ids)[index_in_unique]


class RetrievalEvaluation(base_eval.BaseEval):
  """Evaluation for full set retrieval."""

  def __init__(self, model: base.BaseModel, config: ml_collections.ConfigDict):
    self.config = config
    self.model = model
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
        modalities=self.model.config.modalities,
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
        modalities=self.model.config.modalities,
    )
    del eval_iter

    representations = {
        k: contrastive_merge.normalize_features(v) if 'preds' in k else v
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

    logging.info('Computing cell codes metrics.')

    if self.config.cell_codes:
      s2cells = lama_io.read_sharded_metadata_csv(
          self.config.data.dirname,
          self.config.data.locations.valid,
          data_dir='datasets',
          split='*',  # Also load train and valid cells.
          file_name='grouped_s2cells.csv*',
          max_workers=32,
      )['s2cell'].to_list()
      s2cells = np.unique(s2cells)
      s2cells = np.array([x.encode('utf-8') for x in s2cells])
      query_s2cell_tokens = utils.latlng_to_s2cell_token(
          query_reps['latlng'], level=13
      )
      query_urban_classification = get_urban_classification_s2_cells(
          list(query_s2cell_tokens)
      )
      classes = np.unique(query_urban_classification)
      for cell_code in self.config.cell_codes:
        modality = f'cell_codes_{cell_code}'
        s2cells_encoded = utils.convert_s2cell_level(s2cells, cell_code)
        latlngs = utils.s2cell_token_to_latlng(s2cells_encoded)
        representations = utils.get_latlng_representations(
            train_state,
            latlngs,
            self.model,
            batch_size=self.config.batch_size_eval,
        )

        logging.info('Normalizing representations.')
        if self.model.config.normalize_output_embeddings:
          representations = contrastive_merge.normalize_features(
              representations
          )

        # encode latlngs
        database_reps_cell_code = {
            f'{modality}_preds': jnp.array(representations),
            'latlng': jnp.array(latlngs),
        }
        logging.info('Computing metrics for cell codes level %s.', cell_code)
        metrics = compute_metrics(
            query_reps, database_reps_cell_code, ['sv', modality], self.config
        )
        eval_metrics.update(metrics)

        for c in classes:
          idx = query_urban_classification == c
          if not np.any(idx):
            continue
          query_reps_c = {k: v[idx] for k, v in query_reps.items()}

          modality_c = f'cell_codes_{cell_code}_{c}'
          database_reps_c = {
              f'{modality_c}_preds': database_reps_cell_code[
                  f'{modality}_preds'
              ],
              'latlng': database_reps_cell_code['latlng'],
          }

          logging.info(
              'Computing metrics for cell codes level %s and class %s queries.',
              cell_code,
              c,
          )
          metrics = compute_metrics(
              query_reps_c,
              database_reps_c,
              ['sv', modality_c],
              self.config,
          )
          eval_metrics.update(metrics)

    return {}, eval_metrics, {}
