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

"""Data source for the PDFM downstream benchmark.

PDFM (Population Dynamics Foundation Model) is a *location*-based regression
benchmark: each region (a ZCTA, identified by its centroid lat/lng) has a set of
continuous socioeconomic targets, predicted from a location embedding.

This source reads a single CSV (default ``<data_dir>/pdfm_benchmark.csv``) with
columns::

    place, lat, lng, <target_0>, <target_1>, ...

i.e. a region id, its centroid, and one column per regression target (target
cells may be empty/NaN). This matches the per-region label tables of the public
PDFM release (https://github.com/google-research/population-dynamics); export
the ZCTA centroids + benchmark labels into that schema to run the eval. Generate
a tiny fake version with
``python -m unigeoclip.data.generate_fake_downstream_data``.
"""

import os

import numpy as np
import pandas as pd

_NON_TARGET_COLS = ('place', 'lat', 'lng')


class PDFMDataSource:
  """Loads region (latlng + socioeconomic targets) data for the PDFM eval."""

  def __init__(self, data_dir: str, csv_name: str = 'pdfm_benchmark.csv'):
    path = os.path.join(data_dir, csv_name)
    if not os.path.exists(path):
      raise FileNotFoundError(
          f'PDFM benchmark CSV not found at {path}. Export the PDFM ZCTA '
          'centroids + benchmark labels to columns [place, lat, lng, '
          '<target>...], or generate a fake one with '
          '`python -m unigeoclip.data.generate_fake_downstream_data`.'
      )
    self.df = pd.read_csv(path)
    self.target_names = [
        c for c in self.df.columns if c not in _NON_TARGET_COLS
    ]

  @property
  def latlngs(self) -> np.ndarray:
    return self.df[['lat', 'lng']].to_numpy(np.float32)

  @property
  def targets(self) -> np.ndarray:
    """``[N, T]`` target matrix (may contain NaNs)."""
    return self.df[self.target_names].to_numpy(np.float32)
