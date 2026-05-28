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

"""Retrieval evaluation for cross-view, classification, and hybrid tasks."""

import collections
import os
from typing import Callable

from absl import logging
from clu import metric_writers
import grain.python as grain
import jax
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.train_lib import train_utils
import tensorflow_datasets as tfds

from scaling_geoloc import types
from scaling_geoloc.data import datasets
from scaling_geoloc.data.indexed_data_source import IndexedDataSource
from scaling_geoloc.data.joined_data_source import JoinedDataSource
from scaling_geoloc.evals import base_eval
from scaling_geoloc.utils import distributed as distributed_utils
from scaling_geoloc.utils import geo
from scaling_geoloc.utils import image as image_utils


def build_index(
    representation_fn: Callable[[dict], np.ndarray],
    data_source,
    batch_size: int,
) -> dict[str, np.ndarray]:
  """Encodes this process's shard of a data source.

  In multi-process runs the data source is sharded across processes via
  ShardByJaxProcess so each process encodes a disjoint subset.  The returned
  arrays are the local shard only — callers that need the full global array
  (e.g. for replicated databases) must gather separately.

  Args:
    representation_fn: Callable that takes a raw batch dict (with uint8
      'image' and float32 'latlng') and returns float32 embeddings [B, D].
    data_source: grain.RandomAccessDataSource yielding dicts with at least
      'image' (uint8 [H,W,3]) and 'latlng' (float32 [2]).
    batch_size: Number of items per batch.

  Returns:
    Dict with 'embs' [N_local, D], 'latlng' [N_local, 2], and
    'dataset_index' [N_local] (integer index into the original data source).
  """
  n_processes = jax.process_count()
  shard_options = (
      grain.ShardByJaxProcess(drop_remainder=False)
      if n_processes > 1
      else grain.NoSharding()
  )
  loader = datasets.build_data_loader(
      IndexedDataSource(data_source),
      batch_size=batch_size,
      num_epochs=1,
      shuffle=False,
      drop_remainder=False,
      shard_options=shard_options,
  )
  accumulated = collections.defaultdict(list)
  for batch in loader:
    accumulated['embs'].append(np.asarray(representation_fn(batch)))
    for key in ('latlng', 'dataset_index', 'cell_idx'):
      accumulated[key].append(np.asarray(batch[key]))
  return {k: np.concatenate(v, axis=0) for k, v in accumulated.items()}


def evaluate_retrieval(
    query: dict[str, np.ndarray],
    db: dict[str, np.ndarray],
    ks: tuple[int, ...] = (1, 5, 20, 100),
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
  """Exact distributed retrieval metrics using a globally-sharded database.

  The database stays sharded on-device across all processes/devices.  Inside
  pmap, distributed_topk all-gathers queries to every device, each device
  searches its db shard, and all_to_all reassembles the global top-k.
  Results are all_gathered inside pmap so every process returns identical
  per-query results and global summaries.

  Args:
    query: Dict from build_index — must contain 'embs' [Q_local, D] and
      'latlng' [Q_local, 2]. Any extra keys (e.g. 'dataset_index') are
      passed through into the per-query results.
    db: Dict from build_index (or prototype db) — must contain 'embs'
      [N_local, D] and 'latlng' [N_local, 2].
    ks: Top-k values to evaluate.

  Returns:
    Tuple of (per_query, summaries):
      per_query: per-query arrays — 'latlng', 'dataset_index' (if present),
        'topk_idx' [Q, max_k], 'topk_dist' [Q, max_k], and
        per-threshold boolean arrays 'recall@{k}@{threshold}'.
      summaries: scalar means over all queries.
  """
  n_local_devices = jax.local_device_count()

  def _pad_to_multiple(d):
    n = d['embs'].shape[0]
    pad = (-n) % n_local_devices
    if pad:
      logging.info('Padding %d items to align to %d devices.', pad, n_local_devices)
      d = {k: np.concatenate([v, np.zeros_like(v[:pad])], axis=0) for k, v in d.items()}
    valid = np.array([True] * n + [False] * pad)
    return d, valid, n

  query, _, q_size = _pad_to_multiple(query)
  db, db_valid, _ = _pad_to_multiple(db)
  db = {**db, 'valid': db_valid}

  # k must not exceed the per-device db shard size (jax.lax.top_k constraint).
  db_per_device = db['embs'].shape[0] // n_local_devices
  max_k = min(max(ks), db_per_device)
  ks = tuple(min(k, max_k) for k in ks)

  def _pmap_shard(x):
    sharded = dataset_utils.shard(x)
    return jax.device_put(sharded, distributed_utils.default_pmap_sharding(sharded.shape))

  def retrieve_step(q, db):
    _, topk_idx = distributed_utils.distributed_topk(
        q['embs'], db['embs'], k=max_k, where=db['valid']
    )
    all_db = {
        k: jax.lax.all_gather(db[k], axis_name='batch', tiled=True)
        for k in ('latlng', 'dataset_index')
    }
    topk_l = all_db['latlng'][topk_idx]  # [Q_local, max_k, 2]
    topk_dist = geo.haversine_distance(q['latlng'][:, None], topk_l)[..., 0]
    topk_db_idx = all_db['dataset_index'][topk_idx]  # [Q_local, max_k]
    return jax.tree.map(
        lambda x: jax.lax.all_gather(x, axis_name='batch', tiled=True),
        (topk_idx, topk_dist, topk_db_idx),
    )

  pmapped = jax.pmap(retrieve_step, axis_name='batch', donate_argnums=(0,))
  topk_idx, topk_dist, topk_db_dataset_index = jax.tree.map(
      lambda x: jax.device_get(x)[0],  # all devices hold the same; take [0]
      pmapped(
          jax.tree.map(dataset_utils.shard, query),
          jax.tree.map(_pmap_shard, db),
      ),
  )

  # Slice back to original query size, discarding padded rows.
  topk_idx = topk_idx[:q_size]
  topk_dist = topk_dist[:q_size]
  topk_db_dataset_index = topk_db_dataset_index[:q_size]

  per_query: dict[str, np.ndarray] = {
      **{k: query[k][:q_size] for k in ('latlng', 'dataset_index', 'cell_idx') if k in query},
      'topk_idx': topk_idx,
      'topk_dist': topk_dist,
      'topk_dataset_index': topk_db_dataset_index,
  }

  summaries: dict[str, float] = {}
  for k in ks:
    min_dist = topk_dist[:, :k].min(axis=1)
    for key, val in geo.distance_metrics(min_dist).items():
      if key == 'distance_to_gt':
        summaries[f'distance@{k}'] = float(np.mean(val))
        per_query[f'distance@{k}'] = np.asarray(val)
      else:
        threshold = key.removeprefix('within_')
        summaries[f'recall@{k}@{threshold}'] = float(np.mean(val))
        per_query[f'recall@{k}@{threshold}'] = np.asarray(val)

  return per_query, summaries


def _make_encode_fn(
    flax_model,
    train_state: train_utils.TrainState,
    method_name: str,
    mutable_collections: tuple[str, ...] = ('batch_stats',),
) -> Callable[[dict], np.ndarray]:
  """Creates a pmapped, host-side encode function for a given model method.

  The returned function accepts a raw batch dict with uint8 'image' [B,H,W,3]
  and returns float32 embeddings [B, D] on the host.
  """
  n_devices = jax.local_device_count()

  def encode_step(train_state, batch):
    variables = {'params': train_state.params, **train_state.model_state}
    result, _ = flax_model.apply(
        variables,
        batch,
        train=False,
        method=getattr(flax_model, method_name),
        mutable=list(mutable_collections),
    )
    emb, _ = result  # encode_* methods return (emb, tokens)
    return emb

  pmapped = jax.pmap(encode_step, axis_name='batch', donate_argnums=(1,))

  def fn(batch: dict) -> np.ndarray:
    b = batch['image'].shape[0]
    model_batch = {**batch, 'image': image_utils.byte_to_float(batch['image'])}
    pad = (-b) % n_devices
    if pad:
      model_batch = jax.tree.map(
          lambda x: np.concatenate([x, x[:pad]], axis=0), model_batch
      )
    embs = pmapped(train_state, dataset_utils.shard(model_batch))
    return jax.device_get(dataset_utils.unshard(embs))[:b]

  return fn


def join_databases(
    aerial_db: dict[str, np.ndarray],
    proto_db: dict[str, np.ndarray],
    kappa: float = 1.0,
) -> dict[str, np.ndarray]:
  """Blends aerial embeddings with their per-cell prototype embeddings.

  For each aerial entry, looks up the prototype embedding via cell_idx and
  computes a weighted average.  The output has the same size as aerial_db.

  Args:
    aerial_db: Per-process aerial index from build_index — must contain
      'embs' [N, D] and 'cell_idx' [N].
    proto_db: Full (globally-gathered) prototype db from
      build_prototype_database — must contain 'embs' [N_cells, D].
    kappa: Weight for the prototype contribution.

  Returns:
    aerial_db with 'embs' replaced by the blended embeddings.
  """
  cell_idx = aerial_db['cell_idx']
  valid = cell_idx >= 0
  safe_idx = np.where(valid, cell_idx, 0)
  proto_embs = proto_db['embs'][safe_idx]
  blended = np.where(
      valid[:, None],
      (aerial_db['embs'] + proto_embs * kappa) / (1 + kappa),
      aerial_db['embs'],
  )
  return {**aerial_db, 'embs': blended}


def _make_data_source(location_dir: str, index_dir: str, split: str, modality: str):
  """Builds a JoinedDataSource for query or aerial eval data."""
  image_source = tfds.builder_from_directory(
      os.path.join(location_dir, modality)
  ).as_data_source(split)
  index_source = tfds.builder_from_directory(
      os.path.join(index_dir, modality)
  ).as_data_source(split)
  return JoinedDataSource([(image_source, None), (index_source, None)])


class RetrievalEval(base_eval.BaseEval):
  """Retrieval evaluation supporting cross_view, classification, and hybrid tasks."""

  def _build_db(
      self,
      train_state: train_utils.TrainState,
      split: str,
      batch_size: int,
  ) -> dict[str, np.ndarray] | None:
    """Builds the per-process database shard for the configured task."""
    cfg = self.config
    index_dir = os.path.join(cfg.data_dir, cfg.index_subdir)
    aerial_db = None

    if cfg.task in ('cross_view', 'hybrid'):
      encode_aerial = _make_encode_fn(
          self.model.flax_model, train_state, cfg.encode_aerial_method,
          tuple(cfg.mutable_collections),
      )
      aerial_db = build_index(
          encode_aerial,
          _make_data_source(cfg.data_dir, index_dir, split, 'aerial'),
          batch_size,
      )
      logging.info('Aerial DB: %d items, dim %d', *aerial_db['embs'].shape)

    if cfg.task in ('classification', 'hybrid'):
      cpu = jax.devices('cpu')[jax.process_index()]
      with jax.default_device(cpu):
        full_proto = getattr(self.model, cfg.build_prototype_db_method)(train_state)
      full_proto = {k: np.asarray(v) for k, v in full_proto.items()}
      logging.info('Prototype DB: %d cells, dim %d', *full_proto['embs'].shape)

    if cfg.task == 'cross_view':
      return aerial_db
    if cfg.task == 'classification':
      return jax.tree.map(distributed_utils.get_process_slice, full_proto)
    # hybrid: blend each aerial entry with its corresponding prototype.
    if aerial_db is None or full_proto is None:
      return None
    return join_databases(aerial_db, full_proto, kappa=cfg.kappa)

  def evaluate(
      self,
      train_state: train_utils.TrainState,
      writer: metric_writers.MetricWriter | None = None,
      workdir: str | None = None,
  ) -> types.EvalResults:
    del workdir
    cfg = self.config
    split = cfg.split
    ks = tuple(cfg.k_values)
    batch_size = cfg.batch_size
    index_dir = os.path.join(cfg.data_dir, cfg.index_subdir)

    self.write_note('Building query index...')
    encode_query = _make_encode_fn(
        self.model.flax_model, train_state, cfg.encode_query_method,
        tuple(cfg.mutable_collections),
    )
    query_index = build_index(
        encode_query,
        _make_data_source(cfg.data_dir, index_dir, split, 'query'),
        batch_size,
    )
    logging.info('Query index: %d items, dim %d', *query_index['embs'].shape)

    self.write_note(f'Building {cfg.task} database...')
    db = self._build_db(train_state, split, batch_size)
    if db is None:
      logging.error('No database available for task %s.', cfg.task)
      return {}, {}, {}

    self.write_note('Computing retrieval metrics...')
    per_query, summaries = evaluate_retrieval(query_index, db, ks=ks)
    logging.info('Retrieval summaries:\n%s', summaries)

    if writer is not None:
      writer.write_scalars(0, summaries)
    return per_query, summaries, {}

  @classmethod
  def default_config(cls, batch_size: int) -> ml_collections.ConfigDict:
    config = super().default_config(batch_size)
    config.task = 'cross_view'  # 'cross_view' | 'classification' | 'hybrid'
    config.split = 'test'
    config.k_values = (1, 5, 20, 100)
    config.data_dir = ml_collections.config_dict.placeholder(str)
    config.index_subdir = 'index_level17'
    config.encode_query_method = 'encode_query'
    config.encode_aerial_method = 'encode_aerial'
    config.build_prototype_db_method = 'build_prototype_database'
    config.mutable_collections = ('batch_stats',)
    config.kappa = 1.0
    return config
