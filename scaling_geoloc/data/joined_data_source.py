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

"""JoinedDataSource: merges multiple Grain data sources by index.

Typical usage for training (query + aerial images + precomputed cell indices)::

    train_source = JoinedDataSource([
        (query_source,  'query'),   # nested under 'query'
        (aerial_source, 'aerial'),  # nested under 'aerial'
        (query_index,   None),      # deep-merged at top level
        (aerial_index,  None),      # deep-merged at top level
    ])

Each source must expose the same number of examples.  Example ``i`` is
assembled by fetching ``source[i]`` from every constituent source and merging
the results according to the ``key`` rule:

* ``key=None``  — deep-merge into the top-level result dict.  If a key already
  exists and both the existing value and the new value are dicts, the new dict
  is merged into the existing one (shallow update).  Otherwise the new value
  overwrites the existing one.
* ``key='query'`` (or any string) — nest the source's dict under that key,
  merging into any existing sub-dict at that key.
"""

import dataclasses

import grain.python as grain


@dataclasses.dataclass
class JoinedDataSource(grain.RandomAccessDataSource):
  """Combines multiple sources by joining their outputs at the same index.

  Attributes:
    sources: List of ``(source, key)`` pairs.  ``key=None`` deep-merges the
      source's output at the top level; any other string nests the output under
      that key.
  """

  sources: list[tuple[grain.RandomAccessDataSource, str | None]]

  def __len__(self) -> int:
    return len(self.sources[0][0])

  def __getitem__(self, index: int) -> dict:
    result = {}
    for source, key in self.sources:
      data = source[index]
      if key is None:
        for k, v in data.items():
          if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k].update(v)
          else:
            result[k] = v
      else:
        result.setdefault(key, {}).update(data)
    return result
