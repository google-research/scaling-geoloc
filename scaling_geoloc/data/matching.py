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

"""Tools to match scenes."""

from collections.abc import Sequence
from typing import Iterable, Tuple

from etils.array_types import FloatArray
from etils.array_types import IntArray
import numpy as np

from scaling_geoloc.data import geo
from scaling_geoloc.data import types
from google3.util.geometry.python import s2


def compute_overlap_matrix(
    view_collections_i: Sequence[types.ViewCollection],  # map
    view_collections_j: Sequence[types.ViewCollection],  # query
    filter_same_run: bool,
    symmetric: bool | None = None,  # None auto-determines symmetry
    overlap_metric: str = 'mean',
) -> FloatArray['M N']:
  """Computes the BEV overlap matrix between two sets of view collections.

  Arguments:
    view_collections_i: First set of view collections (length M).
    view_collections_j: Second set of view collections (length N).
    filter_same_run: If true, filter out pairs of view collections from the same
      run.
    symmetric: Whether the two sets of view collections are identical, and we
      thus need to avoid duplicates. If None, it will be auto-determined based
      on the metadata.
    overlap_metric: Which overlap to use. One of 'min', 'max', 'mean', 'first',
      'second'.

  Returns:
    The overlap matrix between the two sets of view collections (size MxN).
  """
  m = len(view_collections_i)
  n = len(view_collections_j)
  if symmetric is None:
    if m == n:
      # If all view collections have the same metadata, they are symmetric
      symmetric = all([
          vc_i[1] == vc_j[1]
          for vc_i, vc_j in zip(view_collections_i, view_collections_j)
      ])
    else:
      symmetric = False
  overlap = np.zeros((m, n))

  polys_i = [
      geo.s2polygon_union_from_latlng(
          view.latlng_frustum() for view in views_i.values()
      )
      for _, _, views_i in view_collections_i
  ]

  polys_j = [
      geo.s2polygon_union_from_latlng(
          view.latlng_frustum() for view in views_j.values()
      )
      for _, _, views_j in view_collections_j
  ]
  for i, vc_i in enumerate(view_collections_i):
    _, metadata_i, _ = vc_i
    for j in range(i + 1 if symmetric else 0, n):
      vc_j = view_collections_j[j]
      _, metadata_j, _ = vc_j
      if filter_same_run and metadata_i.run_name == metadata_j.run_name:
        continue
      overlaps = s2.S2Polygon.get_overlap_fractions(polys_i[i], polys_j[j])
      overlap[i, j] = {
          'min': min,
          'max': max,
          'mean': np.mean,
          'first': lambda x: x[0],
          'second': lambda x: x[1],
      }[overlap_metric](overlaps)
  return overlap


def pairs_from_overlap_matrix(
    overlap_matrix: FloatArray['M N'],
    min_overlap: float,
    max_overlap: float,
    max_num_pairs: int = -1,
) -> IntArray['L 2']:
  """Sample pairs from an overlap matrix."""
  pairs = np.stack(
      np.where((overlap_matrix > min_overlap) & (overlap_matrix < max_overlap)),
      axis=-1,
  )
  overlap = overlap_matrix[pairs[:, 0], pairs[:, 1]]

  if max_num_pairs > 0 and max_num_pairs < pairs.shape[0]:
    pairs = pairs[
        np.random.choice(
            pairs.shape[0],
            size=max_num_pairs,
            replace=False,
            p=overlap / overlap.sum(),
        )
    ]

  return pairs


def build_scenes_from_overlap_matrix(
    s2cell_token: str,
    query_views: types.ViewCollection,
    map_views: Sequence[types.ViewCollection],
    overlap_matrix: FloatArray['M N'],
    min_overlap: float,
    max_overlap: float,
    max_num_pairs: int = -1,
) -> Iterable[Tuple[str, types.Scene]]:
  """Compute query-map pairs."""

  pair_ids = pairs_from_overlap_matrix(
      overlap_matrix,
      min_overlap=min_overlap,
      max_overlap=max_overlap,
      max_num_pairs=max_num_pairs,
  )

  scenes = {
      f'{s2cell_token}/{i}_{j}': types.Scene(
          s2cell=s2cell_token,
          ecef_t_w=geo.ecef_t_s2cell(s2cell_token, 0.0),
          query=query_views[i],
          map=[map_views[j]],
          overlap=np.array([overlap_matrix[i, j]]).astype(np.float32),
      )
      for i, j in pair_ids
  }

  return scenes.items()
