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
benchmark: each region (a US ZIP/ZCTA, identified by its centroid) has a set of
continuous socioeconomic targets, predicted from a location embedding.

The reproduction data is the public PDFM benchmark table ``conus27.csv``
(https://github.com/google-research/population-dynamics/blob/master/data/benchmarks/conus27.csv),
whose columns are::

    place, county, state, latitude, longitude,
    <target_0>, <target_1>, ...,
    imputation_split, superresolution_split, extrapolation_split

This source reads a single CSV (``conus27.csv`` or ``pdfm_benchmark.csv`` under
``data_dir``). It takes the region centroid from ``latitude``/``longitude`` (or
``lat``/``lng``) and treats every remaining *numeric* column as a regression
target, ignoring the id/metadata columns (``place``/``county``/``state``) and the
official ``*_split`` columns (the eval uses its own fixed split, as in
``eval.ipynb``). Target cells may be empty/NaN. Generate a tiny fake version with
``python -m unigeoclip.data.generate_fake_downstream_data``.
"""

import os

import numpy as np
import pandas as pd

# CSVs are looked up in this order when no explicit ``csv_name`` is given.
_CSV_CANDIDATES = ('conus27.csv', 'pdfm_benchmark.csv')
# Non-target columns: region ids/metadata + centroid aliases.
_ID_COLS = ('place', 'county', 'state')
_LAT_ALIASES = ('lat', 'latitude')
_LNG_ALIASES = ('lng', 'lon', 'longitude')


def _resolve_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str:
  for alias in aliases:
    if alias in df.columns:
      return alias
  raise ValueError(
      f'PDFM CSV is missing a centroid column (one of {aliases}); found '
      f'columns: {list(df.columns)}'
  )


class PDFMDataSource:
  """Loads region (latlng + socioeconomic targets) data for the PDFM eval."""

  def __init__(self, data_dir: str, csv_name: str | None = None):
    names = (csv_name,) if csv_name else _CSV_CANDIDATES
    path = next(
        (
            os.path.join(data_dir, n)
            for n in names
            if os.path.exists(os.path.join(data_dir, n))
        ),
        None,
    )
    if path is None:
      raise FileNotFoundError(
          f'PDFM benchmark CSV not found in {data_dir} (looked for '
          f'{", ".join(names)}). Download conus27.csv from '
          'https://github.com/google-research/population-dynamics '
          '(data/benchmarks/conus27.csv), or generate a fake one with '
          '`python -m unigeoclip.data.generate_fake_downstream_data`.'
      )
    self.df = pd.read_csv(path)
    self._lat_col = _resolve_column(self.df, _LAT_ALIASES)
    self._lng_col = _resolve_column(self.df, _LNG_ALIASES)

    non_targets = set(_ID_COLS) | {self._lat_col, self._lng_col}
    # Every remaining numeric column is a regression target; ids/metadata and
    # the official ``*_split`` columns (strings) are dropped.
    self.target_names = [
        c
        for c in self.df.columns
        if c not in non_targets
        and not c.endswith('_split')
        and pd.api.types.is_numeric_dtype(self.df[c])
    ]

  @property
  def latlngs(self) -> np.ndarray:
    return self.df[[self._lat_col, self._lng_col]].to_numpy(np.float32)

  @property
  def targets(self) -> np.ndarray:
    """``[N, T]`` target matrix (may contain NaNs)."""
    return self.df[self.target_names].to_numpy(np.float32)
