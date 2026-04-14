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

"""Combined data source for grain pipelines."""

from collections.abc import Sequence
import math

from absl import logging
import grain.python as pygrain
import numpy as np


# From //experimental/users/xiaomingzhao/spin/data/data_sources.py
class CombinedDataSource(pygrain.RandomAccessDataSource):
  """A datasource that combines multiple data sources.

  This data source combines multiple other datasources. A weight can be assigned
  to each data source.

  Since there is no good mechanism in PyGrain to mix data sources yet (it is on
  the roadmap), this achieves this by "smearing" the indices of each data source
  so that the probability of sampling each item matches the assigned weights.
  This is similar to Virtual Memory
  (https://en.wikipedia.org/wiki/Virtual_memory)


  More concretely, we can define:
    `dataset_idx`: An index specifying a dataset.
    `real_per_dataset_idx`: An index specifying a specific value in a dataset.
    `smeared_idx`: An index specifying a value in the combined dataset. This
    treats the data sources as one flattened dataset where the size of each
    dataset is proportional to its weight.

  Attributes:
    data_source: A sequence of data sources to combine.
    weights: A sequence of floating point weights that determine how often a
      data source is sampled. If None, all weights are equal.
    multiplier: A multiplier on the length of the smeared intervals. Increasing
      this reduces bias caused by quantization in the smeared indices.
  """

  def __init__(
      self,
      data_sources: Sequence[pygrain.RandomAccessDataSource],
      weights: Sequence[float] | None = None,
      multiplier: float = 1.0,
  ):
    if weights is None:
      weights = [1.0] * len(data_sources)

    if len(data_sources) != len(weights):
      raise ValueError('data_sources and weights must have the same length.')

    self.data_sources = data_sources
    self.weights = weights

    min_weight = min(weights)
    # Compute the largest possible length of each smeared interval.
    self._smeared_lengths = [
        math.ceil(multiplier * w / min_weight * len(ds))
        for w, ds in zip(weights, data_sources)
    ]

    # Insert 0 to the front so we don't have to deal with a negative index
    # when computing the offset to subtract.
    self._interval_cum_lengths = np.cumsum([0, *self._smeared_lengths])

    logging.info('Created %s', repr(self))

  def dataset_index_from_smeared(self, smeared_idx: int) -> int:
    """Returns the dataset index given a smeared index."""
    return (
        np.searchsorted(self._interval_cum_lengths, smeared_idx, side='right')
        - 1
    )

  def real_indices_from_smeared(self, smeared_idx: int) -> tuple[int, int]:
    """Returns the real index pair given a smeared index.

    Args:
      smeared_idx: The smeared index.

    Returns:
      A tuple containing the data source index and the index of the item within
      the data source.
    """
    data_source_idx = self.dataset_index_from_smeared(smeared_idx)
    idx = smeared_idx - self._interval_cum_lengths[data_source_idx]
    # Rescale the index to an actual index in the dataset.
    real_length = len(self.data_sources[data_source_idx])
    smeared_length = self._smeared_lengths[data_source_idx]
    if real_length != smeared_length:
      idx = math.floor(idx / smeared_length * real_length)
    return data_source_idx, idx

  def smeared_index_from_real(self, data_source_idx: int, idx: int) -> int:
    """Returns the smeared index given a real index pair."""
    offset_idx = self._interval_cum_lengths[data_source_idx]
    real_length = len(self.data_sources[data_source_idx])
    smeared_length = self._smeared_lengths[data_source_idx]
    return offset_idx + round(idx / real_length * smeared_length)

  def __len__(self):
    return sum(self._smeared_lengths)

  def __getitem__(self, index):
    data_source_idx, idx = self.real_indices_from_smeared(index)
    return self.data_sources[data_source_idx][idx]

  def __repr__(self):
    return (
        f'<{self.__class__.__qualname__}('
        f'data_sources = {self.data_sources}'
        f', weights = {self.weights}'
        f', num_records = {len(self)}'
        ')>'
    )
