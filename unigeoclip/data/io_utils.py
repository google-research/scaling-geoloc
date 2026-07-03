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

"""Tiny local-filesystem replacement for the internal ``gfile`` API.

The downstream data sources were written against Google's ``gfile``. For the
public release we only ever read local files, so ``Open``/``GFile`` map onto the
builtin ``open``. ``GFile`` defaults to binary mode because its callers wrap the
returned handle with ``h5py.File`` / ``rasterio.open``, which require bytes.
"""

import builtins
import os
from typing import Any


def Open(path: str, mode: str = 'r') -> Any:  # pylint: disable=invalid-name
  """Opens a local file, passing ``mode`` through unchanged (text or binary)."""
  return builtins.open(path, mode)


def GFile(path: str, mode: str = 'rb') -> Any:  # pylint: disable=invalid-name
  """Opens a local file, defaulting to binary for h5py/rasterio consumers."""
  if 'b' not in mode:
    mode = mode + 'b'
  return builtins.open(path, mode)


def Exists(path: str) -> bool:  # pylint: disable=invalid-name
  """Returns whether ``path`` exists on the local filesystem."""
  return os.path.exists(path)
