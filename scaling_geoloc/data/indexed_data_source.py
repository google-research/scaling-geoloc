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

"""Wrapper for a random access data source that adds the index to each element."""

import grain.python as grain


class IndexedDataSource(grain.RandomAccessDataSource):
  """Wrapper for a random access data source that adds the index to each element."""

  def __init__(self, data_source: grain.RandomAccessDataSource):
    self._data_source = data_source

  def __len__(self) -> int:
    return len(self._data_source)

  def __getitem__(self, index: int):
    element = self._data_source[index]
    element['dataset_index'] = index
    return element

  def __repr__(self) -> str:
    return self._data_source.__repr__()
