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

"""Coordinate system conversions."""

from collections.abc import Iterable, Sequence
from typing import Final

from etils.array_types import FloatArray
import jax.numpy as jnp
import numpy as np
import pyproj

from google3.util.geometry.python import s2

latlng_transformer_ecef: Final[pyproj.Transformer] = (
    pyproj.Transformer.from_proj(
        pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84'),
        pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84'),
    )
)

ecef_transformer_latlng: Final[pyproj.Transformer] = (
    pyproj.Transformer.from_proj(
        pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84'),
        pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84'),
        always_xy=True,
    )
)


def transform_points(
    y_t_x: FloatArray['4 4'],
    x_points: FloatArray['N 3'],
) -> FloatArray['N 3']:
  """3D transform points from coordinate system x to y."""
  return x_points @ y_t_x[:3, :3].T + y_t_x[:3, -1]


def np_normalize(x: np.ndarray, axis: int) -> np.ndarray:
  return x / (np.linalg.norm(x, axis=axis, keepdims=True) + np.finfo(float).eps)


def look_at(
    at: np.ndarray,
    w_p_cam: np.ndarray,
    up: tuple[float, float, float] = (0, 0, 1),
) -> FloatArray['4 4']:
  """Computes the SE(3) transformation from a look-at vector and a position."""
  z_axis = np_normalize(at - w_p_cam, axis=-1)
  x_axis = np_normalize(np.cross(z_axis, up, axis=-1), axis=-1)
  y_axis = np_normalize(np.cross(z_axis, x_axis, axis=-1), axis=-1)
  w_r_cam = np.stack([x_axis, y_axis, z_axis], axis=-1)

  w_t_cam = np.eye(4)
  w_t_cam[:3, -1] = w_p_cam
  w_t_cam[:3, :3] = w_r_cam
  return w_t_cam


def latlng_from_ecef(ecef_positions: FloatArray['N 3']) -> FloatArray['N 2']:
  """Converts ECEF coordinates to latlng."""
  long, lat, _ = latlng_transformer_ecef.transform(
      ecef_positions[:, 0],
      ecef_positions[:, 1],
      ecef_positions[:, 2],
      radians=False,
  )
  return np.stack([lat, long], 1)


def ecef_from_latlng(
    latlng_positions: FloatArray['N 2'],
    altitude_meters: FloatArray['N 2'] | float = 0.0,
) -> FloatArray['N 3']:
  """Converts latlng positions with optional altitude to ECEF coordinates."""
  if isinstance(altitude_meters, float):
    altitude_meters = np.ones_like(latlng_positions[:, 0]) * altitude_meters
  x, y, z = ecef_transformer_latlng.transform(
      latlng_positions[:, 1],
      latlng_positions[:, 0],
      altitude_meters,
      radians=False,
      errcheck=True,
  )
  return np.stack([x, y, z], 1)


def ecef_t_s2cell(scell_token: str, altitude: float) -> FloatArray['4 4']:
  """Returns the transformation from ECEF to the given s2cell."""
  center_latlng = s2.S2LatLng.from_point(
      s2.S2Cell(s2.S2CellId.from_token(scell_token.encode('utf-8'))).center()
  )
  return ecef_t_latlng(
      center_latlng.lat.degrees, center_latlng.lng.degrees, altitude
  )


def ecef_t_latlng(
    lat: float, lng: float, altitude: float, delta_lat: float = 1e-3
) -> FloatArray['4 4']:
  """Returns the transformation from a local coordinate system to ECEF.

  The local coordinate system is north-facing and centered at [lat, lng,
  altitude].

  Arguments:
    lat: Latitude in degrees.
    lng: Longitude in degrees.
    altitude: Altitude in meters.
    delta_lat: The latitude offset used to compute the north-facing point.

  Returns:
    The transformation from the local coordinate system to ECEF.
  """
  ecef_center = ecef_from_latlng(np.array([[lat, lng]]), altitude)[0]
  ecef_north = ecef_from_latlng(np.array([[lat + delta_lat, lng]]), altitude)[0]

  ecef_t_local = look_at(ecef_north, ecef_center)
  x_axis, y_axis, z_axis = ecef_t_local[:3, :3].T

  # make z face upwards
  ecef_t_local[:3, :3] = np.stack(
      [np.sign(lat) * x_axis, z_axis, -np.sign(lat) * y_axis], axis=-1
  )

  return ecef_t_local


def s2polyline_from_latlng(
    points: FloatArray['N 2'],
) -> s2.S2Polyline:
  points = [s2.S2LatLng.from_degrees(*p) for p in points]
  return s2.S2Polyline.from_latlngs(points)


def s2polygon_union_from_latlng(
    vertices: Iterable[FloatArray['N 2']],
) -> s2.S2Polygon:
  poly_union = s2.S2Polygon()
  for vertices_i in vertices:
    poly_i = s2polygon_from_latlng([vertices_i])
    poly_union.init_to_union(poly_union, poly_i)
  return poly_union


def s2polygon_from_latlng(
    vertices: Iterable[FloatArray['N 2']],
) -> s2.S2Polygon:
  """From lemming.data.geo."""
  loops = []
  for points in vertices:
    points = [s2.S2LatLng.from_degrees(*p).to_point() for p in points]
    if not s2.S2Loop(points).is_normalized:
      points = points[::-1]
    loops.append(s2.S2Loop(points))
  p = s2.S2Polygon()
  p.init_nested(loops)
  return p


def latlng_from_s2loop(loop: s2.S2Loop) -> FloatArray['N 2']:
  points = []
  for j in range(loop.num_vertices):
    a = s2.S2LatLng.from_point(loop.vertex(j))
    points.append((a.lat.degrees, a.lng.degrees))
  return np.array(points)


def polygon_cell_overlap(
    polygon_i: s2.S2Polygon, cell: s2.S2Cell
) -> tuple[float, float]:
  """Computes the overlap between a polygon and a S2 cell."""
  return s2.S2Polygon.get_overlap_fractions(
      polygon_i, s2.S2Polygon.from_cell(cell)
  )


def enclosing_s2cell_ids(
    latlng: FloatArray['N 2'],
    min_level: int = 5,
    max_level: int = 18,
    max_cells: int = 1,
    close: bool = True,
    compute_overlap: bool = False,
) -> (
    Sequence[s2.S2CellId]
    | tuple[Sequence[s2.S2CellId], Sequence[float], Sequence[float]]
):
  """Get the token of the s2cell that contains the given latlng."""
  if close:
    query = s2polygon_from_latlng([latlng])
  else:
    query = s2polyline_from_latlng(latlng)
  region_coverer_options = s2.S2RegionCoverer.Options()
  region_coverer_options.max_level = max_level
  region_coverer_options.min_level = min_level
  region_coverer_options.max_cells = max_cells
  coverer = s2.S2RegionCoverer(region_coverer_options)
  covering_s2cell_ids = coverer.cover(query)
  if compute_overlap:
    overlap_query = [0 for _ in covering_s2cell_ids]  # Overlap in polygon
    overlap_cell = [0 for _ in covering_s2cell_ids]  # Overlap in cell
    for i, cell_id in enumerate(covering_s2cell_ids):
      overlap_query[i], overlap_cell[i] = polygon_cell_overlap(
          query, s2.S2Cell(cell_id)
      )

    order = np.argsort(overlap_query)[::-1]

    # Sort by overlap in polygon. Biggest overlap first.
    def sort_by_overlap_query(x):
      return [x[i] for i in order]

    return [  # pytype: disable=bad-return-type
        sort_by_overlap_query(x)
        for x in (covering_s2cell_ids, overlap_query, overlap_cell)
    ]
  else:
    return covering_s2cell_ids


def s2cell_token_at_level(s2cell_token: bytes, level: int) -> bytes:
  return s2.S2CellId.from_token(s2cell_token).parent_at_level(level).token()


def enclosing_s2cell_token(
    latlng: FloatArray['N 2'], min_level: int = 5, max_level: int = 18
) -> str:
  """Get the single token of the s2cell that contains the given latlng."""
  covering_s2cell_id = enclosing_s2cell_ids(
      latlng, min_level, max_level, max_cells=1
  )[0]
  # pytype: disable=attribute-error
  return covering_s2cell_id.token().decode()
  # pytype: enable=attribute-error


def camera_frustum_vertices(depth: float, hfov_deg: float) -> FloatArray['3 3']:
  """Extracts the vertices of a camera frustum in camera coordinates.

  The frustum is defined by a horizontal field of view and a depth. It is a
  plane passing through the middle, horizontal scanline of the camera.

  Args:
    depth: The depth of the frustum.
    hfov_deg: The horizontal field of view of the frustum.

  Returns:
    The vertices of the frustum in camera coordinates.
  """
  width = np.tan(np.deg2rad(hfov_deg / 2)) * depth
  return np.array([
      [0, 0, 0],
      [width, 0, depth],
      [-width, 0, depth],
  ])


def st_coordinates_on_plane(
    v1: FloatArray['3'],
    v2: FloatArray['3'],
    p: FloatArray['3'],
    numpy: bool = False,
) -> tuple[FloatArray['2'], FloatArray['2']] | None:
  """Calculates the approximate (s, t) coordinates of a 3D point p on a plane.

  Calculates the approximate (s, t) coordinates of a 3D point p on the plane
  spanned by vectors v1 and v2 originating from the origin.

  Assumes v1 and v2 are linearly independent and p is in the plane of v1 & v2.

  Args:
      v1 (numpy.ndarray): A 3D vector representing one side of the
        parallelogram.
      v2 (numpy.ndarray): A 3D vector representing the other adjacent side.
      p (numpy.ndarray): A 3D vector representing the point for which to find
        the approximate (s, t) coordinates.
      numpy: Whether to use numpy instead of jax (default jax).

  Returns:
      tuple: A tuple containing the approximate (s, t) coordinates, or None.
  """
  xnp = np if numpy else jnp
  v1 = xnp.array(v1)
  v2 = xnp.array(v2)
  p = xnp.array(p)

  # Construct a matrix where the columns are v1 and v2
  system = xnp.column_stack((v1, v2))

  # Use numpy.linalg.lstsq to find the least-squares solution to A * [s, t] = p
  # This will give us the best (s, t) in the plane spanned by v1 and v2
  # that approximates p.
  try:
    st_coordinates, _, _, _ = xnp.linalg.lstsq(system, p, rcond=None)
    s, t = st_coordinates[0], st_coordinates[1]
    return s, t
  except np.linalg.LinAlgError:
    # Only triggers if numpy is used.
    return None


def bilinear_interpolate_s2cells(
    lat: float,
    lng: float,
    level: int,
) -> tuple[Sequence[s2.S2CellId], FloatArray['3|4']]:
  """Returns the neighboring s2cells (at level) with bilinear interpolation weights.

  Assumes that the neighboring s2cells form a parallelogram.

  We do not use the s2 library's s, t coordinates as it breaks on the edges of
  the cube.

  Args:
    lat: Latitude in degrees.
    lng: Longitude in degrees.
    level: The level of the s2cells.

  Returns:
    The neighboring s2cells (at level) with bilinear interpolation weights.
  """
  s2latlng = s2.S2LatLng.from_degrees(lat, lng)
  s2point = s2latlng.to_point()
  s2cell_id = s2.S2CellId.from_point(s2point)
  unit_xyz = np.array([s2point.x, s2point.y, s2point.z])  # Unit vector.

  # The neighbors are ordered by distance and usually of length 4.
  # 1 is the enclosing cell, 2 is the next closest, 3 is 'orthogonal' to 2.
  neighbor_ids = s2cell_id.vertex_neighbors(level)

  neighbor_centers = [s2.S2Cell(x).center() for x in neighbor_ids]
  neighbor_xyz = np.array([[x.x, x.y, x.z] for x in neighbor_centers])

  neighbor_xyz_centered = neighbor_xyz - neighbor_xyz[:1]

  if len(neighbor_ids) == 4:
    # The default case. Applies everywhere except for the corners of the cube.
    v1 = neighbor_xyz_centered[1]
    v2 = neighbor_xyz_centered[2]
    p = unit_xyz - neighbor_xyz[0]
    st = st_coordinates_on_plane(v1, v2, p, numpy=True)
    if st is None:
      # Resort to nearest neighbors.
      return neighbor_ids, np.array([1.0, 0, 0, 0])
    else:
      s, t = st
      return neighbor_ids, np.array(
          [(1 - t) * (1 - s), t * (1 - s), s * (1 - t), t * s]
      )
  else:
    # @ToDo(b/398080276): Implement barycentric interpolation.
    # Here we only have 3 neighbors. This appears at the corners of the cube.
    return neighbor_ids, np.array([1.0, 0, 0])
