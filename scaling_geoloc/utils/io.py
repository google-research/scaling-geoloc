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

"""IO utilities for LaMa."""

import ast
import concurrent.futures
import functools
import io
import os
from typing import Any

import numpy as np
import pandas as pd
import tqdm

from google3.pyglib import gfile


def literal_return(val: str) -> Any:
  try:
    return ast.literal_eval(val)
  except (ValueError, SyntaxError):
    return val


def _read_csv(path: str, literal_eval: bool = False, **kwargs) -> pd.DataFrame:
  with gfile.Open(path, 'r') as f:
    df: pd.DataFrame = pd.read_csv(f, **kwargs)
  if literal_eval:
    df = df.map(literal_return)
  return df


def read_sharded_csv(
    pattern: str,
    index: str | None = None,
    literal_eval: bool = False,
    show_progress: bool = True,
    separator: str = ';',
    max_workers: int = 1,
    **kwargs,
) -> pd.DataFrame:
  """Reads a sharded CSV file into a DataFrame."""
  paths = gfile.Glob(pattern)
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=max_workers
  ) as executor:
    data_frames = list(
        tqdm.tqdm(
            executor.map(
                functools.partial(
                    _read_csv,
                    literal_eval=literal_eval,
                    sep=separator,
                    **kwargs,
                ),
                paths,
            ),
            total=len(paths),
            disable=not show_progress,
        )
    )
  df = pd.concat(data_frames)

  if index is not None:
    df = df.set_index(index)
  return df


def read_sharded_metadata_csv(
    dirname: str,
    locations: list[str],
    split: str,
    file_name: str,
    data_dir: str = 'datasets',
    **kwargs,
) -> pd.DataFrame:
  """Wrapper to read sharded DS from data pipeline output."""
  data_frames = []
  for location in locations:
    pattern = os.path.join(data_dir, dirname, location, f'{split}_{file_name}')
    df = read_sharded_csv(pattern, **kwargs)
    df['location'] = location
    data_frames.append(df)
  return pd.concat(data_frames)


def npsavez(
    file_path: str,
    *args,
    allow_pickle: bool = False,
    **kwargs,
):
  with gfile.Open(file_path, 'wb') as f:
    np.savez(
        f,
        *args,
        allow_pickle=allow_pickle,
        **kwargs,
    )


def npsave(
    file_path: str,
    data: Any,
    allow_pickle: bool = False,
):
  with gfile.Open(file_path, 'wb') as f:
    np.save(
        f,
        data,
        allow_pickle=allow_pickle,
    )


def npload(fname: str, allow_pickle: bool = False) -> Any:
  """Loads `fname` and returns an np.ndarray or dict thereof.

  Adapted from big_vision/utils.py: Add allow_pickle argument and item convert.

  Arguments:
    fname: The file to load.
    allow_pickle: Whether to allow pickle loading.

  Returns:
    The loaded numpy array or dict thereof.
  """
  # Load the data; use local paths directly if possible:
  if os.path.exists(fname):
    loaded = np.load(fname, allow_pickle=allow_pickle)
  else:
    # For other (remote) paths go via gfile+BytesIO as np.load requires seeks.
    with gfile.Open(fname, 'rb') as f:
      data = f.read()
    loaded = np.load(io.BytesIO(data), allow_pickle=allow_pickle)

  # Support loading both single-array files (np.save) and zips (np.savez).
  if isinstance(loaded, np.ndarray) and loaded.ndim == 0:
    return loaded.item()
  if isinstance(loaded, np.ndarray):
    return loaded
  else:
    return dict(loaded)
