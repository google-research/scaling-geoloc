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

"""Retrieval evaluation."""

import collections
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
import functools
import gc
import os
from typing import Any, TypeAlias

from absl import logging
from clu import metric_writers
import cv2
from etils.array_types import FloatArray
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.train_lib import train_utils

from scaling_geoloc.data import reverse_geocoder
from scaling_geoloc.scenic import datasets
from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.evals import base_eval
from scaling_geoloc.scenic.models import base
from scaling_geoloc.utils import io as lama_io
from scaling_geoloc.utils import misc
from scaling_geoloc.utils import plot
from google3.pyglib import gfile
from google3.util.geometry.python import s2


Batch: TypeAlias = dict[str, Any]
Predictions: TypeAlias = FloatArray['... D']


def get_representation(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    method: str,
    model_rngs: tuple[str, ...] = (),
    debug: bool | None = False,
    capture_intermediates: bool = True,
) -> Predictions:
  """Encodes a set of images."""
  new_rng, _ = jax.random.split(train_state.rng)
  rng_dict = dict(zip(model_rngs, jax.random.split(new_rng, len(model_rngs))))
  for k in rng_dict:
    rng_dict[k] = train_utils.bind_rng_to_host_device(
        rng_dict[k], axis_name='batch', bind_to='device'
    )
  variables = {'params': train_state.params, **train_state.model_state}
  extra_mutable = ['intermediates'] if capture_intermediates else []
  pred, _ = flax_model.apply(
      variables,
      batch,
      train=False,
      method=method,
      mutable=['batch_stats'] + extra_mutable,
      debug=debug,
      rngs=rng_dict,
  )
  return pred[0]


def get_representation_fn(
    representation_fn: Callable[..., Predictions],
    model: base.BaseModel,
    train_state: train_utils.TrainState,
    method: str,
    shard: bool = True,
    unshard: bool = True,
    to_cpu: bool = True,
) -> Callable[[Batch], Predictions]:
  """Prepare simple inference step on a single image."""
  inference_step_pmapped = jax.pmap(
      functools.partial(
          representation_fn,
          flax_model=model.flax_model,
          method=method,
          capture_intermediates=False,
          model_rngs=(),
      ),
      axis_name='batch',
      # We can donate the eval_batch's buffer.
      donate_argnums=(1,),
  )

  def _representation_fn(batch: Batch) -> Predictions:
    if shard:
      batch = dataset_utils.shard(batch)
    pred = inference_step_pmapped(
        train_state,
        batch,
    )
    if to_cpu:
      pred = jax.device_get(pred)
    if unshard:
      pred = dataset_utils.unshard(pred)
    return pred

  return _representation_fn


def flatten_batch(batch: types.Batch) -> types.Batch:
  """Simplifies a batch by removing unnecessary dimensions."""
  batch = misc.filter_batch_for_jit(batch)
  batch_views = batch['query']['views']

  return batch | batch_views


def join_databases_via_s2cell(
    database: Mapping[str, Any],
    other_database: Mapping[str, Any],
    factor: float = 1.0,
    normalize: bool = False,
) -> MutableMapping[str, Any]:
  """Joins the database with another database. Averages the tokens."""
  conversion = {k.item(): i for i, k in enumerate(other_database['s2cell'])}
  other_database_s2cells = set(other_database['s2cell'].tolist())
  database_s2cell_level = s2.S2CellId.from_token(
      database['s2cell'][0].item()
  ).level
  other_database_s2cell_level = s2.S2CellId.from_token(
      other_database['s2cell'][0].item()
  ).level
  if database_s2cell_level != other_database_s2cell_level:
    logging.info(
        'Database s2cell levels are different. database: %d, '
        'other_database: %d',
        database_s2cell_level,
        other_database_s2cell_level,
    )
    if other_database_s2cell_level > database_s2cell_level:
      raise ValueError(
          'Database s2cell level is smaller than other database s2cell level.'
      )
    database_s2cells = np.array([
        s2.S2CellId.from_token(x.item())
        .parent_at_level(other_database_s2cell_level)
        .token()
        for x in database['s2cell']
    ])
  else:
    database_s2cells = database['s2cell']
  valid = np.array(
      [x.item() in other_database_s2cells for x in database_s2cells]
  )

  logging.info(
      'Joint databases:  Total: %d, valid: %d', valid.shape[0], valid.sum()
  )
  other_database_indices = np.array(
      [conversion[x.item()] for x in database_s2cells if x in conversion]
  )

  other_database_tokens = np.zeros_like(database['tokens'])
  other_database_tokens[valid] = other_database['tokens'][
      other_database_indices
  ]

  logging.info('Join factor: %f', factor)
  norm_fn = (
      functools.partial(misc.l2_normalize, numpy=True)
      if normalize
      else lambda x: x * 0.5
  )
  if normalize:
    logging.info('Normalizing tokens after joining.')
  merged_tokens = np.where(
      valid[:, None],
      norm_fn((other_database_tokens * factor + database['tokens'])),
      database['tokens'],
  )
  joint_database = {
      'tokens': merged_tokens,
      'is_joined': valid,
      **{k: v for k, v in database.items() if k != 'tokens'},
  }
  return joint_database


def filter_index_by_labels(
    entity: str,
    database: MutableMapping[str, Any],
    allowed_labels: Sequence[str],
) -> None:
  """Filters the database by country."""
  labels = reverse_geocoder.ReverseGeocoder(entity).get_label_for_latlngs(
      database['ll_positions']
  )
  valid = np.isin(labels, np.array(allowed_labels))
  logging.info(
      '%s filter:  Total: %d, valid: %d', entity, valid.shape[0], valid.sum()
  )
  database['valid'] = database.pop('valid', True) & valid


def filter_index_by_s2cells(
    database: MutableMapping[str, Any],
    allowed_s2cells: Sequence[bytes],
) -> None:
  """Filters the database by s2cell."""
  s2cell_ids = np.array([s2.S2CellId.from_token(x) for x in allowed_s2cells])
  s2cell_union = s2.S2CellUnion(s2cell_ids)
  is_contained = np.array([
      s2cell_union.intersects_cell(s2.S2CellId.from_token(x))
      for x in database['s2cell'].tolist()
  ])
  del s2cell_union
  logging.info(
      'S2cell filter:  Total: %d, valid: %d',
      len(is_contained),
      sum(is_contained),
  )
  database['valid'] = database.pop('valid', True) & is_contained


def crop_index_to_multiple(
    index: Mapping[str, Any], divisor: int
) -> Mapping[str, Any]:
  """Returns a new index with size divisible by the divisor."""
  return {k: v[: v.shape[0] // divisor * divisor] for k, v in index.items()}


def _pmap_shard(x: Any) -> Any:
  x = dataset_utils.shard(x)
  return jax.device_put(x, misc.default_pmap_sharding(x.shape))


class RetrievalEvaluation(base_eval.BaseEval):
  """Evaluation for retrieval."""

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

    self.db_iter = self._dataset.train_iter
    self.eval_iter = self._dataset.valid_iter

    self.db_from_model = self.config.get('db_from_model', None)  # Method

    if self.config.get('db_data', None) is not None:
      db_data_config = self.config.data.copy_and_resolve_references()
      db_data_config.update(self.config.db_data)
      self.db_dataset = datasets.build_dataset(
          db_data_config,
          batch_size=self.config.batch_size_eval,
          batch_size_eval=self.config.batch_size_eval,
          seed=155,
      )
      self.db_iter = self.db_dataset.train_iter
      self._dataset.meta_data['num_train_examples'] = self.db_dataset.meta_data[
          'num_train_examples'
      ]

    self.descriptor_size = self.config.get('descriptor_size', 1024)

    self.num_steps = (
        config.get('num_samples')
        or self._dataset.meta_data['num_valid_examples']
    ) // config.batch_size_eval

    self.num_train_steps = (
        config.get('num_train_samples')
        or self._dataset.meta_data['num_train_examples']
    ) // config.batch_size_eval

    db_size = self.num_train_steps * self.config.batch_size_eval
    query_size = self.num_steps * self.config.batch_size_eval
    db_size_bytes = db_size * self.descriptor_size * 4
    query_size_bytes = query_size * self.descriptor_size * 4

    self.save_topk = self.config.get('save_topk', 5)

    if self.db_from_model is None:
      logging.info('Database size: %d / %.2f GB', db_size, db_size_bytes / 1e9)
    else:
      logging.info('Database provided by model.')
    logging.info('Query size: %d / %.2f GB', query_size, query_size_bytes / 1e9)

    logging.info('num_train_steps: %d', self.num_train_steps)
    logging.info('num_steps: %d', self.num_steps)

  def autotune_retrieval_batch_size(
      self,
      db_size: int,
      feature_dim: int,
      max_utilization: float = 0.2,
  ) -> int:
    """Autotune the retrieval batch size.

    Assumes memory limit is (#db+#query)'fdim*4.

    Args:
      db_size: The number of database elements.
      feature_dim: The dimension of the features.
      max_utilization: The maximum memory utilization in [0, 1].

    Returns:
      The retrieval batch size.
    """
    db_size_per_device = db_size // jax.local_device_count()
    max_memory = jax.local_devices()[0].memory_stats()['bytes_limit']
    # https://docs.jax.dev/en/latest/pallas/tpu/matmul.html
    # Total memory usage is m * d + n * d + n * m = m * d + n * (m+d)
    # n = (MM - m * d) / (m + d)
    batch_size_retrieval = int(
        (max_memory / 4 * max_utilization - feature_dim * db_size_per_device)
        / (feature_dim + db_size_per_device)
    )
    if batch_size_retrieval < 1:
      raise ValueError(
          'Cannot find a batch size that satisfies the memory limit'
          f' {max_memory} with utilization {max_utilization} for feature'
          f' dimension {feature_dim} and db size {db_size_per_device/1e9}GB per'
          ' device.'
      )
    batch_size_retrieval = min(
        self.num_steps * self.config.batch_size_eval,
        batch_size_retrieval,
    )
    batch_size_retrieval = (
        batch_size_retrieval
        // jax.local_device_count()
        * jax.local_device_count()
    )

    batch_size_retrieval = min(
        batch_size_retrieval, self.config.batch_size_eval // jax.device_count()
    )

    logging.info('Retrieval batch size: %d', batch_size_retrieval)
    return batch_size_retrieval

  def build_index(
      self,
      data_iter: Iterator[types.Batch],
      num_steps: int,
      train_state: train_utils.TrainState,
      writer: metric_writers.MetricWriter | None = None,
      workdir: str | None = None,
      method: str | None = None,
      add_pixels: bool = False,
  ) -> MutableMapping[str, Any]:
    del writer, workdir
    method = method or self.config.method
    representation_fn = get_representation_fn(
        get_representation, self.model, train_state, method
    )
    results: dict[str, np.ndarray] = {}
    batch_size = None

    for i in range(num_steps):
      if i % self.config.log_progress_interval == 0:
        self.write_note(f'Extracting index... [{i}/{num_steps}]')

      batch = next(data_iter)
      fbatch = flatten_batch(batch)

      ecef_positions = None
      if 'ecef_t_w' in batch:
        ecef_t_cam = batch['ecef_t_w'] @ np.linalg.inv(fbatch['cam_t_w'][:, 0])
        ecef_positions = ecef_t_cam[:, :3, -1]
        latlng_positions = misc.ecef_to_latlngalt(ecef_positions)[..., :2]
      else:
        latlng_positions = fbatch['latlng']

      query_batch = {
          'image_pixels': fbatch['image_pixels'],
          'latlng_center': latlng_positions,
      }
      if ecef_positions is not None:
        query_batch['ecef_center'] = ecef_positions

      tokens = representation_fn(query_batch)
      if tokens.ndim == 3:
        tokens = tokens.mean(axis=-2)

      step_results = {'tokens': np.asarray(tokens)}
      if ecef_positions is not None:
        step_results['ecef_positions'] = np.asarray(ecef_positions)
      if 'cell_idx' in fbatch:
        step_results['cell_idx'] = np.asarray(fbatch['cell_idx'])
      if add_pixels:
        step_results['image_pixels'] = np.asarray(fbatch['image_pixels'])
      step_results['image_name'] = np.asarray(
          batch['query']['views']['image_name']
      )
      step_results['s2cell'] = np.asarray(batch['s2cell'])
      step_results['dataset_index'] = np.asarray(batch['dataset_index'])
      step_results['ll_positions'] = np.asarray(latlng_positions)

      if i == 0:
        # Pre-allocate the results for all steps and batches.
        batch_size = tokens.shape[0]
        for k, v in step_results.items():
          results[k] = np.empty(
              (num_steps * batch_size, *v.shape[1:]), dtype=v.dtype
          )

      for k, v in step_results.items():
        results[k][i * batch_size : (i + 1) * batch_size] = v
    return results

  def compute_retrieval_results(
      self,
      query: Mapping[str, Any],
      database: Mapping[str, Any],
      gather_to_host: bool = True,
  ) -> tuple[dict[str, Any], dict[str, jnp.ndarray]]:
    """Computes retrieval results and metrics."""
    query_tokens = query['tokens']
    if query_tokens.shape[-1] > database['tokens'].shape[-1]:
      query_tokens = query_tokens[..., : database['tokens'].shape[-1]]
    pred_similarity, pred_assignment = misc.distributed_topk(
        query_tokens,
        database['tokens'],
        k=min(database['tokens'].shape[0], 100),
        similarity_fn='dot',
        where=database.get('valid'),
        recall_target=self.config.get('topk_recall_target', None),
    )

    db_ll_positions = jax.lax.all_gather(
        database['ll_positions'], axis_name='batch', tiled=True
    )
    pred_ll_positions = jnp.take_along_axis(
        db_ll_positions[None], pred_assignment[..., None], axis=-2
    )
    db_dataset_index = jax.lax.all_gather(
        database['dataset_index'], axis_name='batch', tiled=True
    )
    pred_db_dataset_index = jnp.take_along_axis(
        db_dataset_index[None], pred_assignment, axis=-1
    )
    pred_distance = jnp.abs(
        misc.haversine_distance(
            query['ll_positions'][:, None], pred_ll_positions
        )[..., 0]
    ).clip(max=1e6)

    gt_distance, gt_assignment = misc.distributed_topk(
        query['ll_positions'],
        database['ll_positions'],
        k=min(database['ll_positions'].shape[0], 100),
        similarity_fn='haversine',
        ascending=True,
        where=database.get('valid'),
        recall_target=self.config.get('topk_recall_target', None),
    )

    results = {}
    results['distance_to_closest'] = gt_distance[..., 0]
    results['ll_positions'] = query['ll_positions']
    results['dataset_index'] = query['dataset_index']

    for i in [1]:
      results[f'gt_top{i}'] = gt_assignment[:, :i]
      results[f'gt_top{i}_ll_positions'] = jnp.take_along_axis(
          db_ll_positions[None], gt_assignment[:, :i, None], axis=-2
      )
      results[f'gt_top{i}_db_dataset_index'] = jnp.take_along_axis(
          db_dataset_index[None], gt_assignment[:, :i], axis=-1
      )

    for i in [self.save_topk]:
      results['pred_topk'] = pred_assignment[:, :i]
      results['pred_topk_ll_positions'] = pred_ll_positions[:, :i]
      results['pred_topk_db_dataset_index'] = pred_db_dataset_index[:, :i]
      results['pred_topk_similarity'] = pred_similarity[:, :i]

    def recall_topk(k: int) -> jnp.ndarray:
      topk = pred_assignment[:, :k]
      return jnp.any(topk == results['gt_top1'], axis=-1)

    def overlap_topk(k: int) -> jnp.ndarray:
      topk = pred_assignment[:, :k]
      gt_topk = gt_assignment[:, :k]
      return jax.vmap(lambda x, y: jnp.isin(x, y).sum())(topk, gt_topk)

    def min_distance_topk(k: int) -> jnp.ndarray:
      return jnp.min(pred_distance[:, :k], axis=-1)

    metrics = {}

    for k in [1, 5, 20, 100]:
      metrics[f'recall_top{k:03d}'] = recall_topk(k)
      metrics[f'overlap_top{k:03d}'] = overlap_topk(k)
      metrics[f'intersection_top{k:03d}'] = metrics[f'overlap_top{k:03d}'] / k
      min_distance = min_distance_topk(k)
      metrics[f'min_distance_top{k:03d}'] = min_distance
      metrics[f'min_distance_top{k:03d}<1e1'] = min_distance < 1e1
      metrics[f'min_distance_top{k:03d}<2e1'] = min_distance < 2e1
      metrics[f'min_distance_top{k:03d}<5e1'] = min_distance < 5e1
      metrics[f'min_distance_top{k:03d}<1e2'] = min_distance < 1e2
      metrics[f'min_distance_top{k:03d}<2e2'] = min_distance < 2e2
      metrics[f'min_distance_top{k:03d}<5e2'] = min_distance < 5e2
      metrics[f'min_distance_top{k:03d}<1e3'] = min_distance < 1e3
      metrics[f'min_distance_top{k:03d}<1e4'] = min_distance < 1e4
      metrics[f'min_distance_top{k:03d}<1e5'] = min_distance < 1e5

    valid = results['distance_to_closest'] < self.config.get(
        'valid_distance_threshold', 100
    )
    metrics['false_positives@1e2'] = jnp.cumprod(
        1.0 - (pred_distance < 1e2), axis=-1
    ).sum(-1)
    metrics['false_positives@5e2'] = jnp.cumprod(
        1.0 - (pred_distance < 5e2), axis=-1
    ).sum(-1)
    metrics['false_positives@1e3'] = jnp.cumprod(
        1.0 - (pred_distance < 1e3), axis=-1
    ).sum(-1)
    metrics['valid'] = valid
    metrics['distance_to_closest'] = results['distance_to_closest']
    metrics['topk_similarity'] = results['pred_topk_similarity'][..., 0]

    geo_entities = self.config.get('geo_entities', ())
    for geo_entity in geo_entities:
      db_region_ids = jax.lax.all_gather(
          database[geo_entity], axis_name='batch', tiled=True
      )
      pred_region_ids = jnp.take_along_axis(
          db_region_ids[None], pred_assignment, axis=-1
      )
      gt_region_ids = query[geo_entity]
      metrics[f'geo_{geo_entity}/valid'] = gt_region_ids > -1
      for k in [1, 5, 20, 100]:
        same_entity = pred_region_ids[:, :k] == gt_region_ids[:, None]
        metrics[f'geo_{geo_entity}/recall_top{k:03d}'] = jnp.any(
            same_entity, axis=-1
        )
      metrics[f'geo_{geo_entity}/false_positives'] = jnp.cumprod(
          1.0 - same_entity, axis=-1
      ).sum(-1)

    if gather_to_host:
      # Concatenate results from ALL devices along first dimension.
      results = jax.lax.all_gather(results, axis_name='batch', tiled=True)
      metrics = jax.lax.all_gather(metrics, axis_name='batch', tiled=True)
    return results, metrics

  def evaluate_retrieval(
      self,
      query: MutableMapping[str, Any],
      database: MutableMapping[str, Any],
      batch_size: int | None = None,
  ) -> types.EvalResults:
    """Evaluates the model."""

    if self.config.get('restrict_to_countries', []):
      filter_index_by_labels(
          'country', query, self.config.restrict_to_countries
      )
      filter_index_by_labels(
          'country', database, self.config.restrict_to_countries
      )

    if self.config.get('restrict_to_s2cells'):
      if self.config.restrict_to_s2cells == 'data':
        s2cells = lama_io.read_sharded_metadata_csv(
            self.config.data.dirname,
            self.config.data.locations.valid,
            split='*',  # Also load train and valid cells.
            file_name='grouped_s2cells.csv*',
            max_workers=32,
        )['s2cell'].to_list()
        s2cells = [x.encode('utf-8') for x in s2cells]
      elif self.config.restrict_to_s2cells == 'model':
        s2cells = database['s2cell'].to_list()
      else:
        s2cells = lama_io.read_sharded_csv(
            os.path.join(
                self.config.data.data_dir, self.config.restrict_to_s2cells
            ),
            max_workers=32,
        )['s2cell'].to_list()
        s2cells = [x.encode('utf-8') for x in s2cells]
      filter_index_by_s2cells(database, s2cells)

    if self.config.get('geo_entities', None):
      geo_entities = self.config.geo_entities
      logging.info(
          'Adding region labels to query and database: {%s}.', str(geo_entities)
      )
      geocoder = reverse_geocoder.CombinedReverseGeocoder(geo_entities)
      query.update(
          geocoder.labels_to_ids(
              geocoder.get_labels_for_latlngs(query['ll_positions'])
          )
      )
      database.update(
          geocoder.labels_to_ids(
              geocoder.get_labels_for_latlngs(database['ll_positions'])
          )
      )

    if batch_size is None:
      db_size, *_, feature_dim = database['tokens'].shape
      batch_size = self.autotune_retrieval_batch_size(db_size, feature_dim)
    logging.info('Retrieval batch size per device: %d', batch_size)
    batch_size_total = batch_size * jax.local_device_count()
    logging.info('Retrieval batch size per process: %d', batch_size)

    for k in list(database):
      if database[k].dtype.type in [np.str_, np.bytes_]:
        database.pop(k)
    database_sharded = crop_index_to_multiple(
        database, jax.local_device_count()
    )
    # Distribute the database across devices -> can fit more in memory.
    # On dd=2x2, can fit ~85m database tokens with dim=1024.
    database_sharded = jax.tree.map(_pmap_shard, database_sharded)
    logging.info('Sharded database to devices.')
    # Flush the CPU databases from memory.
    for k in list(database):
      del database[k]
    gc.collect()

    for k in list(query):
      if query[k].dtype.type in [np.str_, np.bytes_]:
        query.pop(k)
    num_queries = query['tokens'].shape[0]
    num_steps = num_queries // batch_size_total
    gather_to_host = True
    pmapped_compute_retrieval_results = jax.pmap(
        functools.partial(
            self.compute_retrieval_results, gather_to_host=gather_to_host
        ),
        axis_name='batch',
        donate_argnums=(0,),
    )

    results = {
        'metrics': collections.defaultdict(list),
        'results': collections.defaultdict(list),
    }
    for i in range(num_steps):
      if i % self.config.log_progress_interval == 0:
        logging.info('Computing retrieval results... [%d/%d]', i, num_steps)
        self.write_note(f'Computing retrieval results... [{i}/{num_steps}]')
      start = i * batch_size_total
      end = min((i + 1) * batch_size_total, num_queries)
      if start >= end:
        break
      query_batch = jax.tree.map(
          lambda x: x[start:end], query  # pylint: disable=cell-var-from-loop
      )
      query_batch = dataset_utils.shard(query_batch)
      logging.info('Sharded query batch to devices (%d)', i)
      results_batch, metrics_batch = pmapped_compute_retrieval_results(
          query_batch, database_sharded
      )
      logging.info('Computed retrieval results (%d)', i)

      for name, batch_elements in [
          ('results', results_batch),
          ('metrics', metrics_batch),
      ]:
        if gather_to_host:
          batch_elements = jax.tree.map(lambda x: x[0], batch_elements)
        else:
          batch_elements = dataset_utils.unshard(batch_elements)
        batch_elements = jax.device_get(batch_elements)
        for k, v in batch_elements.items():
          results[name][k].append(np.asarray(v))

    metrics = {
        k: np.concatenate(v, axis=0) for k, v in results['metrics'].items()
    }
    results = {
        k: np.concatenate(v, axis=0) for k, v in results['results'].items()
    }
    valid = metrics['valid']

    summaries = jax.tree.map(lambda x: x[valid].mean().item(), metrics)
    summaries['valid'] = valid.mean().item()
    summaries['num_valid'] = valid.sum().item()
    logging.info('Summaries:\n%s', summaries)

    # Only evaluate geo entity metrics where labels are available.
    geo_entities = self.config.get('geo_entities', ())
    for metric_name, metric in metrics.items():
      if 'valid' in metric_name:
        continue
      for geo_entity in geo_entities:
        if metric_name.startswith(f'{geo_entity}/'):
          summaries[metric_name] = (
              metric[valid & metrics[f'{geo_entity}/valid']].mean().item()
          )

    results = {**metrics, **results}
    if self.config.get('do_plot', False):
      visualizations = self.visualize(results)
      visualizations = {
          k: np.concatenate(v, axis=-3) for k, v in visualizations.items()
      }
    else:
      visualizations = {}
    return results, summaries, visualizations

  def load_database_from_model(
      self, train_state: train_utils.TrainState, method: str
  ) -> Mapping[str, Any]:
    """Returns the database from the model."""
    process_id = jax.process_index()
    with jax.default_device(jax.devices('cpu')[process_id]):
      database = getattr(self.model, method)(train_state)
    return {k: np.asarray(v) for k, v in database.items()}

  def slice_per_process(
      self, database: Mapping[str, Any]
  ) -> MutableMapping[str, Any]:
    """Slices the database per process."""
    process_id = jax.process_index()
    with jax.default_device(jax.devices('cpu')[process_id]):
      db_size = database['tokens'].shape[0]
      db_size = db_size - db_size % max(128, jax.device_count())
      db_process_size = db_size // jax.process_count()
      # Drop remainder for now.
      database = jax.tree.map(lambda x: x[:db_size], database)
      # Get part of the index for this process.
      database = jax.tree.map(
          lambda x: x[
              db_process_size * process_id : db_process_size * (process_id + 1)
          ],
          database,
      )
    return {k: np.asarray(v) for k, v in database.items()}

  @classmethod
  def write_cache(
      cls,
      workdir: str,
      *,
      eval_results: types.EvalResults | None = None,
      queries: Mapping[str, Any] | None = None,
      database: Mapping[str, Any] | None = None,
      allow_pickle: bool = True,
  ) -> None:
    """Writes the cache to workdir."""
    if not gfile.Exists(workdir):
      gfile.MakeDirs(workdir)
    if eval_results is not None:
      lama_io.npsave(
          os.path.join(workdir, f'eval_results_{jax.process_index()}.npz'),
          eval_results,
          allow_pickle=allow_pickle,
      )
    if queries is not None:
      lama_io.npsave(
          os.path.join(workdir, f'queries_{jax.process_index()}.npz'),
          queries,
          allow_pickle=allow_pickle,
      )
    if database is not None:
      lama_io.npsave(
          os.path.join(workdir, f'database_{jax.process_index()}.npz'),
          database,
          allow_pickle=allow_pickle,
      )

  @classmethod
  def load_cache(
      cls,
      workdir: str,
      pattern: str,
      allow_pickle: bool = True,
      max_num_shards: int = 100,
  ) -> Any | None:
    """Loads the cache from workdir."""
    if not gfile.Exists(workdir):
      raise ValueError(f'Cache not found at {workdir}')

    def read_all(pattern: str):
      cache = []
      for i in range(max_num_shards):
        cache_path = os.path.join(workdir, f'{pattern}_{i}.npz')
        if not gfile.Exists(cache_path):
          break
        else:
          logging.info('Loading cache from %s', cache_path)
        cache.append(
            lama_io.npload(
                cache_path,
                allow_pickle=allow_pickle,
            )
        )
        if pattern == 'eval_results':
          break
      if not cache:
        return None

      def _concat(x: list[Any]) -> Any:
        return {
            k: (
                np.concatenate([sample[k] for sample in x], axis=0)
                if isinstance(v, (Sequence, np.ndarray, list, tuple))
                else v
            )
            for k, v in x[0].items()
        }

      if isinstance(cache[0], dict):
        return _concat(cache)
      else:
        return tuple(map(_concat, zip(*cache)))

    return read_all(pattern)

  def evaluate(
      self,
      train_state: train_utils.TrainState,
      writer: metric_writers.MetricWriter | None = None,
      workdir: str | None = None,
  ) -> types.EvalResults:
    """Evaluates the model."""
    if self.db_from_model is None or self.config.get('join_databases', False):
      logging.info('Extracting features for training set.')
      db_iter = self.db_iter()
      database = self.build_index(
          db_iter,
          self.num_train_steps,
          train_state,
          writer,
          workdir,
          method=self.config.get('method_db', self.config.method),
      )
      del db_iter
    else:
      database = None

    if self.db_from_model is not None:
      if not hasattr(self.model, self.db_from_model):
        logging.info(
            'Model does not have method %s, skipping db_from_model.',
            self.db_from_model,
        )
        if database is None:
          return {}, {}, {}
      else:
        logging.info('Extracting database index from model.')
        model_database = self.load_database_from_model(
            train_state, self.db_from_model
        )  # replicated over processes

        if database is not None:
          logging.info('Joining databases via S2 cell.')
          database = join_databases_via_s2cell(
              database,
              model_database,
              factor=self.config.get('join_factor', 1.0),
              normalize=self.config.get('join_normalize', False),
          )
          del model_database
        else:
          database = self.slice_per_process(model_database)

    assert database is not None

    for k, v in database.items():
      logging.info('Database: %s: %s', k, v.shape)

    logging.info('Resetting eval iterator.')
    eval_iter = self.eval_iter()

    logging.info('Extracting features for test set.')
    query = self.build_index(
        eval_iter,
        self.num_steps,
        train_state,
        writer,
        workdir,
        method=self.config.method,
    )
    del eval_iter

    if self.config.get('cache_results', False) and workdir is not None:
      self.write_cache(workdir, queries=query, database=database)

    for k, v in query.items():
      logging.info('Query: %s: %s', k, v.shape)
    eval_results = self.evaluate_retrieval(
        query,
        database,
        self.config.batch_size_retrieval,
    )
    if workdir is not None:
      self.write_cache(workdir, eval_results=eval_results)
    return eval_results

  def get_element_from_dataset(
      self, index: int, is_query: bool
  ) -> types.DataDict:
    if is_query:
      return self._dataset.valid_ds[index]
    elif self.config.get('db_data', None) is not None:
      return self.db_dataset.train_ds[index]
    else:
      return self._dataset.train_ds[index]

  def _get_image(self, element: types.DataDict) -> FloatArray['H W 3']:
    if 'query' in element:
      images = element['query']['views']['image_pixels']
    elif 'rgb' in element:
      images = element['rgb']
    elif 'image' in element:
      images = [element['image']]
    else:
      raise ValueError(f'No image found in element: {element}')
    return images[0].astype(np.float32) / 255.0

  def get_image_from_dataset(
      self, index: int, is_query: bool
  ) -> FloatArray['H W 3']:
    element = self.get_element_from_dataset(index, is_query)
    return self._get_image(element)

  def visualize_retrieval(
      self,
      result: Mapping[str, Any],
      topk: int | None = 5,
  ) -> FloatArray['N H W 3']:
    if topk is None:
      topk = self.save_topk
    query_ds_index = result['dataset_index']
    db_ds_indices = np.concatenate(
        [
            result['gt_top1_db_dataset_index'],
            result['pred_topk_db_dataset_index'][:topk],
        ],
        axis=0,
    )
    db_ll_positions = np.concatenate(
        [
            result['gt_top1_ll_positions'],
            result['pred_topk_ll_positions'][:topk],
        ],
        axis=0,
    )
    db_images = np.stack(
        [self.get_image_from_dataset(i, is_query=False) for i in db_ds_indices]
    )
    query_image = self.get_image_from_dataset(query_ds_index, is_query=True)
    query_image = jax.image.resize(
        query_image, db_images.shape[-3:], method='cubic'
    )
    distances = misc.haversine_distance(
        result['ll_positions'][None], db_ll_positions
    )[..., 0]
    colors = plot.apply_colormap(
        np.log10(distances), cmap='RdYlGn_r', vmin=2, vmax=4
    )
    db_images = misc.draw_image_border(
        db_images, colors[:, None, None], border_width=8
    )
    retrieval_images = np.concatenate(
        [db_images[:1], query_image[None], db_images[1:]], axis=0
    )
    return retrieval_images

  def visualize_queries(
      self,
      plot_indices: Sequence[int],
      metrics: FloatArray['N'],
      columns: int | None = None,
      vmin: float | None = None,
      vmax: float | None = None,
      log10: bool = False,
      cmap: str = 'RdYlGn_r',
  ) -> FloatArray['N H W 3']:

    metrics = metrics[plot_indices]

    query_images = [
        self.get_image_from_dataset(i, is_query=True) for i in plot_indices
    ]

    query_images = np.stack(query_images)
    if log10:
      metrics = np.log10(metrics)
      if vmin is not None:
        vmin = np.log10(vmin)
      if vmax is not None:
        vmax = np.log10(vmax)
    colors = plot.apply_colormap(metrics, cmap=cmap, vmin=vmin, vmax=vmax)
    query_images = misc.draw_image_border(
        query_images, colors[:, None, None], border_width=8
    )

    query_images = np.array(query_images)

    def draw_text(
        img,
        text,
        font=cv2.FONT_HERSHEY_PLAIN,
        pos=(0, 0),
        font_scale=3,
        font_thickness=2,
        text_color=(0, 255, 0),
        text_color_bg=(255, 255, 255),
        pad=2,
    ):
      x, y = pos
      text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
      text_w, text_h = text_size
      cv2.rectangle(
          img,
          (x - pad, y - pad),
          (x + text_w + pad, y + text_h + pad),
          text_color_bg,
          -1,
      )
      return cv2.putText(
          img,
          text,
          (x, y + text_h + font_scale - 1),
          font,
          font_scale,
          text_color,
          font_thickness,
          cv2.FILLED,
      )

    for i, img in enumerate(query_images):
      font = cv2.FONT_HERSHEY_COMPLEX
      val = int(metrics[i].item()) + 1
      query_images[i] = (
          draw_text(
              np.array(img * 255.0, dtype=np.uint8).copy(),
              f'rank={val}' if val < 100 else 'rank>=100',
              pos=(10, 10),
              font=font,
              font_scale=1,
              text_color=(0, 0, 0),
              font_thickness=2,
          )
          / 255.0
      )
    query_images = query_images[np.argsort(metrics)]

    if columns is not None:
      return jnp.concatenate(
          jnp.concatenate(
              query_images.reshape(-1, columns, *query_images.shape[-3:]),
              axis=-3,
          ),
          axis=-2,
      )[None]

    return query_images

  def visualize_samples(
      self,
      plot_indices: Sequence[int],
      results: Mapping[str, Any],
  ) -> types.VisualizationDict:
    """Visualize retrieval results."""
    if self.db_from_model is not None and not self.config.get('db_data', None):
      return {}

    def _get_result(idx: int):
      return {k: v[idx] for k, v in results.items()}

    visualizations = {}
    visualizations['retrieval'] = np.stack([
        np.concatenate(self.visualize_retrieval(_get_result(idx)), axis=-2)
        for idx in plot_indices
    ]).astype(np.float32)
    return visualizations

  def visualize(
      self,
      results: Mapping[str, Any],
      max_num_plots: int = 8,
      false_positive_key: str = 'false_positives@1e2',
  ) -> types.VisualizationDict:
    false_positives = results[false_positive_key]
    to_plot = {
        'consistent': range(max_num_plots),
        'good': np.where(false_positives == 0)[0],
        'close': np.where((false_positives > 0) & (false_positives < 5))[0],
        'bad': np.where(false_positives >= 20)[0],
    }
    visualizations = {}
    for name, indices in to_plot.items():
      if not len(indices):  # pylint: disable=g-explicit-length-test
        continue
      indices = indices[:max_num_plots]
      visualizations_i = self.visualize_samples(indices, results)
      for k, v in visualizations_i.items():
        visualizations[f'{name}/{k}'] = v
    return visualizations
