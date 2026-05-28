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

"""Dataset for geolocation tasks."""

from collections.abc import Callable, Mapping
import copy
import functools
import os

from absl import logging
from etils.array_types import FloatArray
from etils.array_types import IntArray
import grain.python as grain
import jax
import ml_collections
import numpy as np
import tensorflow_datasets as tfds

from scaling_geoloc.data import indexed_data_source
from scaling_geoloc.data import joined_data_source
from scaling_geoloc import types
from scaling_geoloc.data import transforms as dtr
from scaling_geoloc.data import samplers
from scaling_geoloc.utils import distributed


def transformations_from_config(
    pp: ml_collections.ConfigDict,
    meta_data: types.DataDict | None = None,
) -> tuple[grain.Transformations, types.DataDict]:
  """Builds the transformations from the preprocess config."""
  transformations = [
      dtr.MaybeCollate(),
      dtr.NormalizePixels(),
  ]
  if pp.size is not None:
    size = tuple(pp.size)
    if pp.method == 'stretch':
      transformations.append(dtr.ResizeImageLike(final_size=size))
    elif pp.method == 'crop':
      transformations.append(dtr.CenterCropImageLike(crop_size=size))
    elif pp.method == 'pad':
      transformations.append(dtr.ResizeAndPadImageLike(crop_size=size))
  if pp.filter_invalid_cells:
    transformations.append(dtr.FilterInvalidCells())
  return transformations, meta_data


def get_dummy_batch(
    data_source: grain.RandomAccessDataSource,
    transformations: (
        grain.Transformations | Mapping[str, grain.Transformations]
    ),
    batch_size: int,
    seed: int,
) -> types.Batch:
  """Returns a dummy batch from the data source."""
  dummy_loader = grain.load(
      source=data_source,
      shuffle=True,
      drop_remainder=True,
      seed=seed,
      transformations=transformations,
      batch_size=distributed.get_process_batch_size(batch_size),
      worker_count=0,
      read_options=grain.ReadOptions(
          num_threads=1,
          prefetch_buffer_size=1,
      ),
  )
  return next(dummy_loader.__iter__())


def build_data_loader(
    source: grain.RandomAccessDataSource,
    *,
    num_epochs: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    shard_options: grain.ShardOptions = grain.NoSharding(),
    transformations: grain.Transformations = (),
    batch_size: int | None = None,
    drop_remainder: bool = False,
    worker_count: int | None = 0,
    read_options: grain.ReadOptions | None = None,
    worker_buffer_size: int = 1,
    mining_indices: IntArray['N B'] | None = None,
    mining_similarity: FloatArray['N B'] | None = None,
    mining_batch_size: int | None = None,
) -> grain.DataLoader:
  """Convenient method for simple pipelines on top of a data source.

  Args:
    source: Data source to load from. This can be one of the file data sources
      provided by Grain, a TFDS data source (`tfds.data_source(...)`) or your
      custom data source.
    num_epochs: See IndexSampler.
    shuffle: See IndexSampler.
    seed: See IndexSampler.
    shard_options: See IndexSampler.
    transformations: List of local (stateless) transformations:
    batch_size: Optional batch size. If provided will apply BatchOperation().
    drop_remainder: Whether to drop partial batches.
    worker_count: Number of child processes launched to parallelize the
      transformations among. Zero means processing runs in the same process.
    read_options: Read options for the data loader. See ReadOptions.
    worker_buffer_size: Number of records to buffer in each worker.
    mining_indices: Optional batched indices to sample from.
    mining_similarity: Optional batched similarity scores to sample from.
    mining_batch_size: Optional mini-batch size for the mining indices.

  Returns:
    DataLoader for this dataset.
  """
  if mining_indices is not None:
    del mining_similarity  # Unused.
    mining_batch_size = (
        mining_batch_size
        if mining_batch_size is not None
        else batch_size // jax.local_device_count()
    )
    sampler = samplers.BatchedIndexSampler(
        num_workers=worker_count,
        batched_indices=mining_indices,
        shard_options=shard_options,
        seed=seed,
        batch_size=mining_batch_size,
        shuffle=shuffle,
    )
  else:
    sampler = grain.IndexSampler(
        num_records=len(source),
        shuffle=shuffle,
        seed=seed,
        num_epochs=num_epochs,
        shard_options=shard_options,
    )
  if batch_size is not None:
    transformations = list(transformations)
    transformations.append(
        grain.Batch(batch_size, drop_remainder=drop_remainder)
    )
  return grain.DataLoader(
      data_source=source,
      sampler=sampler,
      operations=transformations,
      worker_count=worker_count,
      read_options=read_options,
      worker_buffer_size=worker_buffer_size,
  )


def build_eval_data_loader(
    source: grain.RandomAccessDataSource,
    data_config: ml_collections.ConfigDict,
    batch_size: int,
    transformations: grain.Transformations = (),
    shard_options: grain.ShardOptions = grain.NoSharding(),
    drop_remainder: bool = False,
) -> grain.DataLoader:
  """Builds an eval data loader using grain settings from data_config."""
  return build_data_loader(
      source,
      num_epochs=1,
      shuffle=False,
      drop_remainder=drop_remainder,
      shard_options=shard_options,
      transformations=transformations,
      batch_size=batch_size,
      worker_count=data_config.grain_num_workers_valid,
      read_options=grain.ReadOptions(
          prefetch_buffer_size=data_config.grain_prefetch_buffer_size,
          num_threads=data_config.grain_num_threads_valid,
      ),
  )


def dataloader_provider_fn(
    *args,
    **kwargs,
) -> Callable[[], grain.DatasetIterator]:
  """Returns a function that provides a DataLoader iterator."""
  return lambda **md: build_data_loader(*args, **{**kwargs, **md}).__iter__()


def create_dataset(
    dataset_config: ml_collections.ConfigDict,
    batch_size: int,
    batch_size_eval: int,
    seed: int,
    transformations: grain.Transformations = (),
    meta_data: types.DataDict | None = None,
) -> types.Dataset:
  """Create a dataset for training."""

  process_batch_size = distributed.get_process_batch_size(batch_size)
  process_batch_size_eval = distributed.get_process_batch_size(batch_size_eval)

  logging.info(  # pylint: disable=logging-fstring-interpolation
      f'create_dataset: {batch_size=}, {process_batch_size=}, '
      f'{batch_size_eval=}, {process_batch_size_eval=}, '
      f'{jax.local_device_count()=}, {jax.device_count()=}, '
      f'{jax.process_index()=}, {jax.process_count()=}'
  )

  train_transformations = copy.deepcopy(transformations)
  valid_transformations = copy.deepcopy(transformations)

  def build_data_source(
      split: str,
  ) -> grain.RandomAccessDataSource:
    location_dir = os.path.join(dataset_config.data_dir, dataset_config.location)
    index_dir = os.path.join(location_dir, dataset_config.index_subdir)
    query_source = tfds.builder_from_directory(
        os.path.join(location_dir, 'query')
    ).as_data_source(split)
    aerial_source = tfds.builder_from_directory(
        os.path.join(location_dir, 'aerial')
    ).as_data_source(split)
    query_index = tfds.builder_from_directory(
        os.path.join(index_dir, 'query')
    ).as_data_source(split)
    aerial_index = tfds.builder_from_directory(
        os.path.join(index_dir, 'aerial')
    ).as_data_source(split)
    source = joined_data_source.JoinedDataSource([
        (query_source, 'query'),
        (aerial_source, 'aerial'),
        (query_index, None),
        (aerial_index, None),
    ])
    return indexed_data_source.IndexedDataSource(source)

  train_data_source = build_data_source(
      dataset_config.train_split,
  )
  valid_data_source = build_data_source(
      dataset_config.valid_split,
  )

  dummy_batch_fn = functools.partial(
      get_dummy_batch,
      train_data_source,
      train_transformations,
      batch_size,
      seed=999,
  )

  meta_path = os.path.join(
      dataset_config.data_dir,
      dataset_config.location,
      dataset_config.index_subdir,
      'metadata.npz',
  )
  with np.load(meta_path) as f:
    cell_centers_ll = f['cell_centers_latlng']

  meta_data = {
      'num_train_examples': len(train_data_source),
      'num_valid_examples': len(valid_data_source),
      'dummy_batch_fn': dummy_batch_fn,
      'cell_centers_ll': cell_centers_ll,
      **(meta_data if meta_data is not None else {}),
  }

  train_loader = dataloader_provider_fn(
      source=train_data_source,
      shuffle=dataset_config.shuffle_train,
      seed=seed,
      shard_options=grain.ShardByJaxProcess(drop_remainder=True),
      drop_remainder=dataset_config.get('drop_remainder', True),
      transformations=train_transformations,
      batch_size=process_batch_size,
      worker_count=dataset_config.grain_num_workers,
      read_options=grain.ReadOptions(
          prefetch_buffer_size=dataset_config.grain_prefetch_buffer_size,
          num_threads=dataset_config.grain_num_threads,
      ),
      mining_indices=meta_data.get(
          f'{dataset_config.train_split}_mining_indices', None
      ),
  )

  valid_loader = dataloader_provider_fn(
      source=valid_data_source,
      # Not setting num_epochs=1 to avoid deadlocks or crashes until we fix
      # this properly.
      seed=seed,
      shard_options=grain.ShardByJaxProcess(drop_remainder=True),
      drop_remainder=dataset_config.get('drop_remainder', True),
      shuffle=dataset_config.shuffle_valid,
      transformations=valid_transformations,
      batch_size=process_batch_size_eval,
      worker_count=dataset_config.grain_num_workers_valid,
      read_options=grain.ReadOptions(
          prefetch_buffer_size=dataset_config.grain_prefetch_buffer_size,
          num_threads=dataset_config.grain_num_threads_valid,
      ),
      mining_indices=meta_data.get(
          f'{dataset_config.valid_split}_mining_indices', None
      ),
  )

  return types.Dataset(
      train_iter=train_loader,
      valid_iter=valid_loader,
      meta_data=meta_data,
      train_ds=train_data_source,
      valid_ds=valid_data_source,
  )


def build_dataset(
    dataset_config: ml_collections.ConfigDict,
    batch_size: int,
    batch_size_eval: int,
    seed: int,
    meta_data: types.DataDict | None = None,
) -> types.Dataset:
  """Create a dataset from config."""
  transformations, meta_data = transformations_from_config(
      dataset_config.preprocess, meta_data
  )
  return create_dataset(
      dataset_config,
      batch_size,
      batch_size_eval,
      seed,
      transformations=transformations,
      meta_data=meta_data,
  )


def create_overfit_dataset(
    dummy_batch_fn: Callable[[], types.Batch],
    num_train_examples: int,
    num_valid_examples: int,
    meta_data: types.DataDict | None = None,
) -> types.Dataset:
  """Create a dataset that reloads the same batch."""

  def train_iter():
    for _ in range(num_train_examples):
      yield dummy_batch_fn()

  def valid_iter():
    for _ in range(num_valid_examples):
      yield dummy_batch_fn()

  meta_data = {
      **(meta_data or {}),
      'num_train_examples': num_train_examples,
      'num_valid_examples': num_valid_examples,
      'dummy_batch_fn': dummy_batch_fn,
  }
  return types.Dataset(
      train_iter=train_iter,
      valid_iter=valid_iter,
      meta_data=meta_data,
  )
