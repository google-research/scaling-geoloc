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

"""3D grid representations. Adapted from lemming."""

from collections.abc import Mapping
import dataclasses
import functools
from typing import Type, TypeVar
import chex
from etils.array_types import Array
from etils.array_types import BoolArray
from etils.array_types import FloatArray
from etils.array_types import IntArray
import jax
from jax.experimental import sparse as jax_sparse
import jax.numpy as jnp
import numpy as np
from scaling_geoloc.utils import misc

Point = FloatArray['... N']
Index = IntArray['... N']
ID = IntArray['...']
AnyGrid = TypeVar('AnyGrid', bound='GridND')


@dataclasses.dataclass(frozen=True)
class GridND:
  """N-dimensional regular grid.

  Attributes:
    extent: The number of cells along each dimension.
    cell_size: The physical size of each cell, in meters.
    num_cells: The total number of cells.
    extent_meters: The physical size of the grid, in meters.
  """

  extent: tuple[int, ...]
  cell_size: float | FloatArray['N']

  @classmethod
  def from_extent_meters(
      cls: Type[AnyGrid], extent_meters: tuple[float, ...], cell_size: float
  ) -> AnyGrid:
    extent = tuple(i / cell_size for i in extent_meters)
    if not all(e % 1 == 0 for e in extent):
      raise ValueError(
          f'The metric grid extent {extent_meters} is not divisible '
          f'by the cell size {cell_size}.'
      )
    cell_size = jnp.array([cell_size] * len(extent))
    return cls(tuple(map(int, extent)), cell_size)

  @classmethod
  def from_bounding_box(
      cls: Type[AnyGrid],
      extent_meters: FloatArray['N'],
      extent: tuple[int, ...],
  ) -> AnyGrid:
    extent_arr = jnp.array(extent)
    cell_size = jnp.apply_along_axis(
        lambda e: e / extent_arr, -1, extent_meters
    )
    return cls(extent, cell_size)

  def xyz_to_index(self, xyz: Point) -> Index:
    return jnp.floor(xyz / self.cell_size).astype(int)

  def index_to_xyz(self, idx: Index) -> Point:
    return (idx + 0.5) * self.cell_size

  def index_to_id(self, idx: Index) -> ID:
    idx = jnp.moveaxis(idx, -1, 0)
    # We can't use model='raise' or out-of-bounds asserts inside jit.
    return jnp.ravel_multi_index(idx, self.extent, mode='clip')

  def id_to_index(self, ids: ID) -> Index:
    return jnp.stack(jnp.unravel_index(ids, self.extent), -1)

  @property
  def num_cells(self) -> jnp.ndarray:
    return jnp.prod(jnp.array(self.extent))

  @property
  def extent_meters(self) -> FloatArray['N']:
    return jnp.array(self.extent) * self.cell_size

  def index_in_grid(self, idx: Index) -> BoolArray['...']:
    return ((idx >= 0) & (idx < jnp.array(self.extent))).all(-1)

  def xyz_in_grid(self, xyz: Point) -> BoolArray['...']:
    return ((xyz >= 0) & (xyz < self.extent_meters)).all(-1)

  def grid_index(self) -> IntArray['i j k N']:
    grid = jnp.mgrid[tuple(slice(None, e) for e in self.extent)]
    return jnp.moveaxis(grid, 0, -1)

  def flat_grid_index(self) -> IntArray['N']:
    grid_index = self.grid_index()
    return self.grid_index().reshape(-1, grid_index.shape[-1])


@dataclasses.dataclass(frozen=True)
class Grid2D(GridND):
  """2-dimensional regular grid."""

  extent: tuple[int, int]


@dataclasses.dataclass(frozen=True)
class Grid3D(GridND):
  """3-dimensional regular grid."""

  extent: tuple[int, int, int]

  def bev(self) -> Grid2D:
    return Grid2D(self.extent[:2], self.cell_size[..., :2])

# Register Grids to jax for batchability.
jax.tree_util.register_dataclass(
    GridND, data_fields=['cell_size'], meta_fields=['extent']
)

jax.tree_util.register_dataclass(
    Grid2D, data_fields=['cell_size'], meta_fields=['extent']
)

jax.tree_util.register_dataclass(
    Grid3D, data_fields=['cell_size'], meta_fields=['extent']
)


@chex.dataclass
class FeatureGrid:
  """Grid features."""

  grid: Grid3D
  grid_t_w: FloatArray['4 4']
  grid_ids: IntArray['N S']
  features: FloatArray['N C']
  opacities: FloatArray['N 1']


def sparse_volume(
    grid: Grid3D,
    grid_index: Index | ID,
    values: FloatArray['N C'],
    unique_indices: bool = True,
) -> jax_sparse.BCOO:
  """Create a sparse (*N_dims, C) volume of values."""
  if grid_index.ndim == 1:
    grid_index = grid.id_to_index(grid_index)
  shape = grid.extent + (values.shape[-1],)
  return jax_sparse.BCOO(
      (values, grid_index),
      shape=shape,
      unique_indices=unique_indices,
  )


def grid_t_w(
    w_t_ref: FloatArray['4 4'],
    grid: Grid3D,
    z_offset: float | None,
    center_around_reference: bool,
) -> FloatArray['4 4']:
  """Define a grid transform grid_t_w at a reference location."""
  size_meters = grid.extent_meters
  y_ref = (size_meters[1] / 2) if center_around_reference else 0.0
  rotmat = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], w_t_ref.dtype)
  tvec = np.array(
      [
          size_meters[0] / 2,
          y_ref,
          z_offset if z_offset is not None else size_meters[2] / 2,
      ],
      w_t_ref.dtype,
  )

  w_r_ref = w_t_ref[:3, :3]

  grid_t_ref = np.eye(4, dtype=w_t_ref.dtype)
  grid_t_ref[:3, :3] = rotmat
  grid_t_ref[:3, -1] = tvec

  # Keep the grid gravity-aligned by removing any vertical rotation.
  yaw = np.arctan2(*w_r_ref[:2, 0][::-1])
  w_r_ref_upright = np.array(
      [
          [np.cos(yaw), 0, -np.sin(yaw)],
          [np.sin(yaw), 0, np.cos(yaw)],
          [0, -1, 0],
      ],
      w_t_ref.dtype,
  )

  w_t_ref_upright = np.eye(4, dtype=grid_t_ref.dtype)
  w_t_ref_upright[:3, :3] = w_r_ref_upright
  # w_t_ref_upright[:3, :3] = w_r_ref
  w_t_ref_upright[:3, -1] = w_t_ref[:3, -1]
  return grid_t_ref @ np.linalg.inv(w_t_ref_upright)


def gather_top_k_ids(
    grid_ids: IntArray['...(N)'],
    top_k: int,
    start_dim: int = -1,
) -> tuple[IntArray['... k'], IntArray['... k']]:
  """Gathers the top-k most frequent IDs over the last dimension(s)."""

  def most_frequent(ids: IntArray['N']):
    unique_ids, counts = jnp.unique(ids, return_counts=True, size=top_k + 1)
    counts = counts * (unique_ids > -1)
    top_idx = jnp.argsort(counts)[::-1][:top_k]
    return jnp.concatenate([unique_ids[top_idx], counts[top_idx]])

  ids_and_counts = jnp.apply_along_axis(
      most_frequent, -1, jax.lax.collapse(grid_ids, start_dim, None)
  )

  return ids_and_counts[..., :top_k], ids_and_counts[..., top_k:]


def aggregate_features_in_grid(
    grid_ids: IntArray['L'],
    grid_id_map: IntArray['... S'],
    features_map: FloatArray['... C'],
    max_num_observations: int | None = 20,
) -> tuple[FloatArray['L C'], IntArray['L']]:
  """For each id, aggregate the features from the map (mean)."""

  def aggregate_n_by_id(query_id):
    map_has_q = ((grid_id_map == query_id) & (grid_id_map > -1)).any(-1)
    indices = jnp.where(map_has_q, size=max_num_observations, fill_value=-1)
    mask = (jnp.stack(indices, axis=-1) > -1).all(-1)
    count = mask.sum()
    feature_query = (features_map[indices] * mask[..., None]).sum(-2)
    return jnp.concatenate([feature_query, count[..., None]], axis=0)

  features_and_occupancy = jnp.apply_along_axis(
      aggregate_n_by_id, axis=-1, arr=grid_ids[..., None]
  )

  features, occupancy = (
      features_and_occupancy[..., :-1],
      features_and_occupancy[..., -1:],
  )

  return features / occupancy.clip(min=1.0), occupancy.astype(jnp.int32)[..., 0]


aggregate_features_in_grid_batched = jax.vmap(
    aggregate_features_in_grid, in_axes=(0, 0, 0, None), out_axes=0
)


def gather_features_from_grid(
    grid_id_query: IntArray['... S'],
    grid_ids: IntArray['L'],
    features_grid: FloatArray['L C'],
    occupancy_grid: IntArray['L'],
) -> tuple[FloatArray['... C'], IntArray['...']]:
  """Gathers features from a grid by id and take the mean over S features."""

  def gather_by_id(query_ids):
    mask = (
        jnp.isin(grid_ids, query_ids) & (grid_ids > -1) & (occupancy_grid > 0)
    )
    count = mask.sum()
    feature = (features_grid * mask[..., None]).sum(-2)
    return jnp.concatenate([feature, count[None]], axis=0)

  features_and_occupancy_query = jnp.apply_along_axis(
      gather_by_id, axis=-1, arr=grid_id_query
  )
  features, occupancy = (
      features_and_occupancy_query[..., :-1],
      features_and_occupancy_query[..., -1:],
  )
  return features / occupancy.clip(min=1.0), occupancy.astype(jnp.int32)[..., 0]


gather_features_from_grid_batched = jax.vmap(
    gather_features_from_grid, in_axes=(0, 0, 0), out_axes=0
)


def gather_features_via_grid(
    grid_id_query: IntArray['... S'],
    grid_id_map: IntArray['... S'],
    features_map: FloatArray['... C'],
    max_num_cells: int | None = 2000,
    max_num_observations: int | None = 20,
) -> Mapping[str, Array]:
  """Lift feature maps to a grid and collect the features into the query."""
  grid_ids, unique_grid_id_query = jnp.unique(
      grid_id_query, size=max_num_cells, fill_value=-1, return_inverse=True
  )

  unique_grid_id_query = jnp.where(
      (unique_grid_id_query >= max_num_cells) | (grid_id_query < 0),
      -(max_num_cells + 1),  # large negative value that triggers OOB
      unique_grid_id_query,
  )

  features_grid, occupancy_grid = aggregate_features_in_grid(
      grid_ids, grid_id_map, features_map, max_num_observations
  )
  features_query, occupancy_query = gather_features_from_grid(
      grid_id_query, grid_ids, features_grid, occupancy_grid
  )

  return {
      'features_query': features_query,
      'occupancy_query': occupancy_query,
      'features_grid': features_grid,
      'occupancy_grid': occupancy_grid,
      'grid_ids': grid_ids,
      'unique_grid_id_query': unique_grid_id_query,
  }


gather_features_via_grid_batched = jax.vmap(
    gather_features_via_grid, in_axes=(0, 0, 0, None, None), out_axes=0
)


def gather_features_from_map(
    grid_id_query: IntArray['... S'],
    grid_id_map: IntArray['... S'],
    features_map: FloatArray['... C'],
) -> tuple[FloatArray['... C'], IntArray['...']]:
  """Gather features from a map where ids match."""

  valid_map = grid_id_map > -1

  def gather_by_id(query_ids):
    mask = jnp.isin(grid_id_map, query_ids) & valid_map
    mask = jnp.any(mask, axis=-1, keepdims=True)
    count = mask.sum()
    axes = tuple(range(features_map.ndim - 1))
    feature = misc.masked_mean(features_map, mask, axis=axes)
    return jnp.concatenate([feature, count[None]], axis=0)

  features_and_occupancy_query = jnp.apply_along_axis(
      gather_by_id, axis=-1, arr=grid_id_query
  )
  features, occupancy = (
      features_and_occupancy_query[..., :-1],
      features_and_occupancy_query[..., -1:],
  )
  return features, occupancy.astype(jnp.int32)[..., 0]


def grid_matching_loss(
    features_q_grid: FloatArray['B M D'],
    features_m_grid: FloatArray['B M D'],
    weights: FloatArray['B M 1'],
) -> FloatArray['B']:
  """NLL between query and map grid features."""
  similarity = jnp.einsum('bid,bjd->bij', features_q_grid, features_m_grid)
  has_weight = jnp.einsum('bic,bjc->bij', weights, weights) > 0
  # Probs over maps
  dsmax = jax.nn.log_softmax(
      similarity, axis=-1, where=has_weight
  ) + jax.nn.log_softmax(similarity, axis=-2, where=has_weight)

  return -misc.masked_mean(dsmax.diagonal(axis1=-2, axis2=-1), weights[..., 0])


def query_to_grid_matching_loss(
    grid_id_q: IntArray['B N L S'],
    features_q: FloatArray['B N L D'],
    features_m_grid: FloatArray['B G D'],
    do_align: FloatArray['B G 1'],
) -> FloatArray['B']:
  """NLL between query image features and map grid features."""
  similarity = jnp.einsum('bnld,bgd->bnlg', features_q, features_m_grid)

  has_weight = (
      jnp.einsum('bnld,bgd->bnlg', jnp.ones_like(features_q[..., :1]), do_align)
      > 0
  )

  grid_id_q = jnp.where(
      jnp.take_along_axis(has_weight, grid_id_q, axis=-1),
      grid_id_q,
      -do_align.shape[-2] - 1,  # large negative value that triggers OOB
  )

  probs = jax.nn.softmax(similarity, axis=-1, where=has_weight)
  total_probs = jnp.take_along_axis(
      probs, grid_id_q, axis=-1, fill_value=1.0e-10
  ).sum(-1)

  total_log_probs = jnp.log(total_probs)  # B N L

  return -misc.masked_mean(
      total_log_probs, jnp.any(grid_id_q > -1, axis=-1), axis=(-2, -1)
  )


gather_features_from_map_batched = jax.vmap(
    gather_features_from_map, in_axes=(0, 0, 0), out_axes=0
)

map_coordinates_with_channels = jax.vmap(
    jax.scipy.ndimage.map_coordinates,
    in_axes=(-1, None, None, None),
    out_axes=-1,
)


@functools.partial(jax.jit, static_argnames=['mode', 'order', 'cval'])
def interpolate_nd(
    array: FloatArray['... D'],
    points: FloatArray['K N'],
    valid_array: BoolArray['...'] | None = None,
    order: int = 1,
    mode: str = 'nearest',
    cval: float = 0.0,
) -> tuple[FloatArray['K D'], BoolArray['K']]:
  """Interpolate an N-dimensional array at the given points (from SNAP)."""
  size = jnp.asarray(array.shape[:-1])
  valid = jnp.all((points >= 0) & (points < size), -1)
  # The origin of the input is the corner of element (0, 0, 0)
  # but the origin of the indexing is its center.
  points = jnp.moveaxis(points - 0.5, -1, 0)

  # Wrapper to switch between interpolation functions.
  # Improves speed from PIX interpolation.
  interp_function = misc.get_interpolate_function(
      mode=mode, order=order, cval=cval
  )
  batched_interp_function = jax.vmap(
      interp_function, in_axes=(-1, None), out_axes=-1
  )
  values = batched_interp_function(array, points)
  if valid_array is not None:
    nan_mask = jnp.where(valid_array, 0, np.nan)
    nan_points_mask = interp_function(nan_mask, points)
    valid &= ~jnp.isnan(nan_points_mask)
  return values, valid


def extract_topk_unique_ids(
    grid_ids: IntArray['B ...'],
    top_k: int,
) -> IntArray['B K']:
  """Extract unique grid ids."""
  b = grid_ids.shape[0]
  unique_grid_ids, counts = jax.vmap(
      functools.partial(
          jnp.unique,
          size=1_000_000,
          fill_value=-1,
          return_counts=True,
      )
  )(
      grid_ids.reshape(b, -1),
  )
  most_frequent = jnp.argsort(counts, axis=-1, descending=True)[:, :top_k]
  return jnp.take_along_axis(unique_grid_ids, most_frequent, -1)


def fill_missing_ids(
    rng: jnp.ndarray, grid: Grid3D, unique_grid_ids: IntArray['N']
) -> IntArray['N']:
  """Fill missing ids (-1) with random ids that are not in unique_grid_ids."""
  all_grid_ids = grid.index_to_id(grid.flat_grid_index())
  is_contained = jnp.isin(all_grid_ids, unique_grid_ids)
  random_ids = jax.random.choice(
      rng,
      all_grid_ids,
      shape=unique_grid_ids.shape,
      p=~is_contained,
      replace=True,
  )
  return jnp.where(unique_grid_ids > -1, unique_grid_ids, random_ids)


def voxelize_points(
    w_points: FloatArray['... 3'],
    w_mask: BoolArray['... 1'],
    grid: Grid3D,
    grid_tr_w: FloatArray['4 4'] | None = None,
) -> tuple[IntArray['... 1'], FloatArray['... 3']]:
  """Voxelize points."""
  if grid_tr_w is None:
    grid_tr_w = jnp.eye(4, dtype=w_points.dtype)
  grid_points = jnp.einsum(
      'ij,...j->...i', grid_tr_w[:3, :], misc.to_homogeneous(w_points)
  )

  grid_index = grid.xyz_to_index(grid_points)
  grid_id = grid.index_to_id(grid_index)[..., None]
  valid = w_mask & grid.index_in_grid(grid_index)[..., None]
  grid_id = jnp.where(valid, grid_id, -1)
  return grid_id, grid_points
