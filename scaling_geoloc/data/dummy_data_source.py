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

"""Dummy dataset for testing, following the spec in data/spec.py.

Each example is a randomly generated SV + aerial pair.  The data is
reproducible: example i is always generated from seed i, so the source can
be used with Grain's random-access interface.

Typical usage::

    source = DummyDataSource(num_examples=256, num_cells=1000)
    example = source[0]   # dict with 'sv' and 'aerial' sub-dicts
"""

import dataclasses

import grain.python as grain
import numpy as np

from scaling_geoloc.data import spec as data_spec


@dataclasses.dataclass
class DummyDataSource(grain.RandomAccessDataSource):
  """Random-access dummy data source matching the ExampleSpec format.

  Attributes:
    num_examples: Number of examples the source exposes.
    num_cells: Size of the cell lookup table (cell indices are drawn from
      [0, num_cells)).
    sv_image_size: Spatial resolution of the Street View images (H = W).
    aerial_image_size: Spatial resolution of the aerial tiles (H = W).
  """

  num_examples: int = 256
  num_cells: int = 1000
  sv_image_size: int = 224
  aerial_image_size: int = 256

  def __len__(self) -> int:
    return self.num_examples

  def __getitem__(self, index: int) -> dict:
    rng = np.random.default_rng(seed=index)

    sv_cell_idx = int(rng.integers(0, self.num_cells))

    sv_adjacent_idx, sv_adjacent_weights = _random_adjacent(
        rng, self.num_cells
    )
    aerial_adjacent_idx, aerial_adjacent_weights = _random_adjacent(
        rng, self.num_cells
    )

    return {
        'cell_idx': np.int32(sv_cell_idx),
        'query': {
            'image': rng.integers(
                0, 256,
                size=(self.sv_image_size, self.sv_image_size, 3),
                dtype=np.uint8,
            ),
            # Random point somewhere on Earth (lat ∈ [−90, 90], lng ∈ [−180, 180]).
            'latlng': rng.uniform(
                [-90.0, -180.0], [90.0, 180.0]
            ).astype(np.float32),
            'adjacent_cell_idx': sv_adjacent_idx,
            'adjacent_weights': sv_adjacent_weights,
        },
        'aerial': {
            'image': rng.integers(
                0, 256,
                size=(self.aerial_image_size, self.aerial_image_size, 3),
                dtype=np.uint8,
            ),
            'latlng': rng.uniform(
                [-90.0, -180.0], [90.0, 180.0]
            ).astype(np.float32),
            'adjacent_cell_idx': aerial_adjacent_idx,
            'adjacent_weights': aerial_adjacent_weights,
        },
    }


def _random_adjacent(
    rng: np.random.Generator, num_cells: int
) -> tuple[np.ndarray, np.ndarray]:
  """Returns random adjacent cell indices and normalised bilinear weights."""
  indices = rng.integers(
      -1, num_cells, size=(data_spec.NUM_ADJACENT_CELLS,), dtype=np.int32
  )
  raw_weights = rng.uniform(
      0.0, 1.0, size=(data_spec.NUM_ADJACENT_CELLS,)
  ).astype(np.float32)
  # Zero out weights for missing cells (index == -1) and normalise.
  raw_weights = np.where(indices >= 0, raw_weights, 0.0)
  total = raw_weights.sum()
  weights = raw_weights / total if total > 0 else raw_weights
  return indices, weights
