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

"""Custom samplers."""

from absl import logging
import grain.python as grain
import numpy as np
from etils.array_types import IntArray
import jax
from scaling_geoloc.utils import misc


class TrueSequentialSampler:
  """Basic sampler implementation that provides records in order."""

  def __init__(
      self,
      batch_size: int,
      num_workers: int,
      num_records: int,
      shard_options: grain.ShardOptions,
      seed: int | None = None,
  ):
    self._batch_size = batch_size
    self._num_workers = max(num_workers, 1)
    self._step_size = self._batch_size * self._num_workers
    if num_records <= 0:
      raise ValueError(
          'Invalid number of records in Sampler. '
          f'Got {num_records} records, but number of records '
          'must be greater than 0.'
      )
    self._num_records = num_records
    self._shard_options = shard_options
    if shard_options.drop_remainder:
      num_records_per_shard = self._num_records // shard_options.shard_count
      self._max_index = num_records_per_shard * shard_options.shard_count
    else:
      self._max_index = self._num_records
    self._seed = seed

  def __repr__(self) -> str:
    return (
        f'TrueSequentialSampler(num_records={self._num_records}, '
        f'shard_options={self._shard_options!r})'
    )

  def __getitem__(self, index: int) -> grain.RecordMetadata:
    if index < 0 or index >= self._max_index:
      raise IndexError(
          f'RecordMetadata object index is out of bounds; Got index {index},'
          f' allowed indices should be in [0, {self._max_index}]'
      )
    rng = None
    if self._seed is not None:
      rng = np.random.Generator(np.random.Philox(key=self._seed + index))

    batch_index, item_index = misc.unravel_global_dataset_index(
        index,
        self._num_workers,
        self._batch_size,
        self._shard_options.shard_count,
    )
    sorted_index = batch_index * self._batch_size + item_index
    return grain.RecordMetadata(index=index, record_key=sorted_index, rng=rng)


def shard_even_split(
    num_examples: int, options: grain.ShardOptions
) -> tuple[int, int]:
  """Returns the interval for the shard when sharding `num_examples` evenly.

  This splits the interval [0, num_examples - 1] into `shard_count` intervals
  and returns the `shard_index`'s interval. If `drop_remainder` is True all
  intervals will have the same size.

  Args:
    num_examples: Number of examples to shard.
    options: Options for sharding the data in this process.

  Returns:
    Tuple with the start and end of the interval. The start is the first
    example that should be included in this interval and end - 1 is the last
    example to be include in the shard.
  """
  examples_per_shard = num_examples // options.shard_count
  shard_start = examples_per_shard * options.shard_index
  shard_end = examples_per_shard * (options.shard_index + 1)

  # Handle remaining examples.
  num_unused_examples = num_examples % options.shard_count

  if num_unused_examples > 0:
    if options.drop_remainder:
      logging.warning(
          'Dropping %d examples of %d examples (shard %d).',
          num_unused_examples,
          num_examples,
          options.shard_count,
      )
    else:
      shard_start += min(options.shard_index, num_unused_examples)
      shard_end += min(options.shard_index + 1, num_unused_examples)
  return shard_start, shard_end


class BatchedIndexSampler(grain.Sampler):
  """Sampler that returns records from a batched index."""

  def __init__(
      self,
      num_workers: int,
      batched_indices: IntArray['N B'],
      shard_options: grain.ShardOptions,
      seed: int | None = None,
      batch_size: int | None = None,  # Set this < B to sample multiple batches.
      shuffle: bool = False,
      num_epochs: int | None = None,
      deterministic_subsampling: bool = False,
      debug: bool = False,
  ):
    self._num_workers = max(num_workers, 1)
    self._shard_options = shard_options
    self._indices = np.asarray(batched_indices)
    self._batch_size = batched_indices.shape[1]
    if batch_size is not None:
      self._batch_size = min(self._batch_size, batch_size)
    self._seed = seed
    self._num_batches_in_shard = self._indices.shape[0]
    self._num_batches = self._num_batches_in_shard * jax.process_count()
    self._num_records = self._num_batches * self._batch_size
    assert self._seed is not None

    self._max_index = (
        None
        if num_epochs is None
        else num_epochs * self._num_records * self._batch_size
    )

    # The batch indices in this shard.
    self._batch_keys = grain.MapDataset.range(self._num_batches_in_shard)
    if shuffle:
      # This enables reshuffling of the batches within the shard.
      self._batch_keys = self._batch_keys.shuffle(seed=seed)

    self._deterministic_subsampling = deterministic_subsampling
    self._debug = debug

  def __repr__(self) -> str:
    return (
        f'BatchedIndexSampler(batched_indices={self._indices.shape},'
        f' shard_options={self._shard_options!r}),'
        f' batch_size={self._batch_size},'
        f' deterministic_subsampling={self._deterministic_subsampling},'
        f' seed={self._seed})'
    )

  def __getitem__(self, index: int) -> grain.RecordMetadata:
    if index < 0 or (self._max_index is not None and index >= self._max_index):
      raise IndexError(
          f'RecordMetadata object index is out of bounds; Got index {index},'
          f' allowed indices should be in [0, {self._max_index}]'
      )
    epoch, index_in_epoch = divmod(index, self._num_records)
    batch_index, item_index = misc.unravel_global_dataset_index(
        index_in_epoch,
        self._num_workers,
        self._batch_size,
        self._shard_options.shard_count,
    )
    # Update with epoch --> This reshuffles the batches if epoch is > 1.
    batch_index = epoch * self._num_batches_in_shard + batch_index
    record_key = self._indices[self._batch_keys[batch_index], item_index]
    # Deterministic sampling in underlying data source (not always supported).
    if self._deterministic_subsampling:
      record_key = record_key + epoch * self._num_batches
    if self._debug:
      logging.info(
          'BATCH SAMPLER: index: %d, usindex: %d, batch_index: %d, item_index:'
          ' %d, process_index: %d, record_key: %d',
          index,
          index_in_epoch % self._shard_options.shard_count,
          batch_index,
          item_index,
          self._shard_options.shard_index,
          record_key,
      )
    rng = None
    if self._seed is not None:
      rng = np.random.Generator(np.random.Philox(key=self._seed + index))
    next_record = grain.RecordMetadata(
        index=index, record_key=record_key, rng=rng
    )
    return next_record
