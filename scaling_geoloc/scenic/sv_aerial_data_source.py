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

import collections
from collections.abc import Sequence
import dataclasses
import json
import os

import grain.python as grain
import jax
import numpy as np
import tensorflow_datasets as tfds
import tqdm

from scaling_geoloc.scenic import combined_data_source
from scaling_geoloc.scenic import types
from scaling_geoloc.utils import io
from google3.pyglib import gfile
from google3.util.geometry.python import s2


class S2CellDataSource(grain.RandomAccessDataSource):
  """A data source for data indexed by S2 cells."""

  @dataclasses.dataclass(frozen=True)
  class Config:
    """Config to build a random access data source indexed by S2 cells."""

    dirname: str
    locations: Sequence[str]
    target_level: int | None = None
    index_file_name: str = 's2cells_indexed.txt'
    split_name: str = 'aerial'
    seed: int = 0
    deterministic: bool = False
    check: bool = True

  def __init__(self, data_dir: str, config: Config):
    self._data_dir = data_dir
    self.config = config
    self.rng = np.random.default_rng(seed=config.seed)
    self.setup()

  def setup(self):
    config = self.config
    data_dir = os.path.join(self._data_dir, config.dirname)
    self.data_sources = []
    data_sources = []
    # Load data sources.
    for location in config.locations:
      location_dir = os.path.join(data_dir, location)
      builder = tfds.builder_from_directory(location_dir)
      data_source = builder.as_data_source(split=config.split_name)
      data_sources.append(data_source)
    # Combine data sources.
    self._data_source = combined_data_source.CombinedDataSource(data_sources)

    def smeared_index(data_source_idx: int, idx: int) -> int:
      return self._data_source._interval_cum_lengths[data_source_idx].item() + idx  # pylint: disable=protected-access

    s2cell_to_idx = collections.defaultdict(list)
    for i, location in enumerate(config.locations):
      # Load index map.
      location_dir = os.path.join(data_dir, location)
      index_map_path = os.path.join(
          location_dir, config.index_file_name.format(split=config.split_name)
      )
      if config.index_file_name.endswith('.csv*'):
        index_df = io.read_sharded_csv(
            index_map_path, literal_eval=False, max_workers=4
        )
        s2cells_i_to_idxs = index_df.set_index(index_df.columns[0]).to_dict()[
            'indices'
        ]
        for s2cell, idxs in tqdm.tqdm(s2cells_i_to_idxs.items()):
          idxs = json.loads(idxs)
          s2cell_to_idx[self.key_for_s2cell(s2cell.encode('utf-8'))].extend(
              [smeared_index(i, j) for j in idxs]
          )
      else:
        with gfile.Open(index_map_path, 'r') as f:
          s2cells = f.readlines()
          for j, s2cell in enumerate(s2cells):
            s2cell = s2cell.strip()
            s2cell_to_idx[self.key_for_s2cell(s2cell.encode('utf-8'))].append(
                smeared_index(i, j)
            )
    self.s2cell_to_idx = {
        s2cell: np.array(indices, dtype=np.int32)
        for s2cell, indices in s2cell_to_idx.items()
    }
    self.total_num_records = max(
        len(indices) for indices in self.s2cell_to_idx.values()
    ) * len(self.s2cell_to_idx)
    self.s2cells = sorted(list(self.s2cell_to_idx.keys()))
    div_by = max(jax.device_count(), 128)
    self.s2cells = self.s2cells[: len(self.s2cells) // div_by * div_by]
    self.s2cell_idx = {s2cell: i for i, s2cell in enumerate(self.s2cells)}

  def __len__(self) -> int:
    return self.total_num_records

  def __getitem__(self, index: int) -> types.DataDict:
    s2cell_idx, offset = index % len(self.s2cells), index // len(self.s2cells)
    candidate_indices = self.s2cell_to_idx[self.s2cells[s2cell_idx]]
    if self.config.deterministic:
      candidate_idx = offset % len(candidate_indices)
    else:
      candidate_idx = self.rng.choice(len(candidate_indices))
    idx = candidate_indices[candidate_idx].item()
    element = self._data_source[idx]
    element['dataset_index'] = idx
    return element

  def key_for_s2cell(self, s2cell: bytes) -> int:
    if self.config.target_level is None:
      return s2.S2CellId.from_token(s2cell).id
    else:
      return (
          s2.S2CellId.from_token(s2cell)
          .parent_at_level(self.config.target_level)
          .id
      )

  def data_for_s2cell(self, s2cell: bytes, idx: int = 0) -> types.DataDict:
    return self._data_source[
        self.s2cell_to_idx[self.key_for_s2cell(s2cell)][idx]
    ]

  def random_sample_in_s2cell(
      self, s2cell: bytes, rng: np.random.Generator
  ) -> types.DataDict:
    s2cell_key = self.key_for_s2cell(s2cell)
    data = self._data_source[
        rng.choice(self.s2cell_to_idx[s2cell_key], 1)[0].item()
    ]
    if self.config.check:
      if self.key_for_s2cell(data['s2cell']) != s2cell_key:
        raise ValueError(f's2cell {s2cell} and {data["s2cell"]} do not match.')

    return data

  def exists(self, s2cell: bytes) -> bool:
    return self.key_for_s2cell(s2cell) in self.s2cell_to_idx

  def sample_in_s2cell(
      self, s2cell: bytes, seed: int | None = None
  ) -> types.DataDict:
    rng = (
        self.rng
        if (seed is None or self.config.deterministic)
        else np.random.default_rng(seed=seed)
    )
    return self.random_sample_in_s2cell(s2cell, rng)

  def all_for_s2cell(self, s2cell: bytes) -> list[types.DataDict]:
    return [
        self._data_source[idx]
        for idx in self.s2cell_to_idx[self.key_for_s2cell(s2cell)]
    ]

  def __repr__(self) -> str:
    return self._data_source.__repr__()


class SVAerialDataSource(grain.RandomAccessDataSource):
  """Wrapper for a random access data source that merges SV and aerial data."""

  def __init__(
      self,
      sv_data_source: grain.RandomAccessDataSource,
      aerial_data_source: S2CellDataSource,
  ):
    self._sv_data_source = sv_data_source
    self._aerial_data_source = aerial_data_source

  def __len__(self) -> int:
    return len(self._sv_data_source)

  def __getitem__(self, index: int):
    element = self._sv_data_source[index]
    if self._aerial_data_source.exists(element['s2cell']):
      element['aerial'] = self._aerial_data_source.sample_in_s2cell(
          element['s2cell'],
          index,
      )
    else:
      element['aerial'] = None
    return element

  def __repr__(self) -> str:
    return self._sv_data_source.__repr__()
