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

from typing import Any, Final

from etils.array_types import FloatArray
import jax.numpy as jnp
import jaxcam
import numpy as np
import pyproj

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


def bilinear_interpolation_weights(
    query_coord: FloatArray['3'],
    neighbor_coords: FloatArray['3|4 3'],
    valid: FloatArray['3|4'] | None = None,
) -> FloatArray['3|4']:
  """
  Assumes that the neighboring s2cells form a parallelogram.

  Args:
    query_coord: The 3D coordinates of the query point.
    neighbor_coords: The 3D coordinates of the neighboring points.
    valid: Optional boolean mask indicating which neighbor coordinates are valid.

  Returns:
    The neighboring s2cells (at level) with bilinear interpolation weights.
  """

  # The neighbors are ordered by distance and usually of length 4.
  # 0 is the enclosing cell, 1 is the next closest, 2 is 'orthogonal' to 1.
  neighbor_coords_centered = neighbor_coords - neighbor_coords[:1]
  if neighbor_coords.shape[0] == 4:
    # The default case. Applies everywhere except for the corners of the cube.
    v1 = neighbor_coords_centered[1]
    v2 = neighbor_coords_centered[2]
    p = query_coord - neighbor_coords[0]
    st = st_coordinates_on_plane(v1, v2, p, numpy=True)
    if st is None:
      # Resort to nearest neighbors.
      weights = np.array([1.0, 0, 0, 0])
    else:
      s, t = st
      weights = np.array(
          [(1 - t) * (1 - s), t * (1 - s), s * (1 - t), t * s]
      )
  else:
    # Here we only have 3 neighbors. This appears at the corners of the S2 cube.
    # As this is very rare, we just use NN interpolation.
    weights = np.array([1.0, 0, 0])

  if valid is not None:
    weights = (weights * valid) / weights[valid].sum().clip(min=1.0e-6)
  return weights


def deg2rad(deg: float) -> float:
  return deg / 180 * jnp.pi


def rotation_from_rpy(
    roll: float, pitch: float, yaw: float, degree: bool = False
) -> FloatArray['3 3']:
  """Creates a rotation matrix from roll, pitch, yaw."""
  if degree:
    roll = deg2rad(roll)
    pitch = deg2rad(pitch)
    yaw = deg2rad(yaw)
  rot_x = jnp.zeros((3, 3))
  rot_x = rot_x.at[0, 0].set(1)
  rot_x = rot_x.at[1, 1].set(jnp.cos(pitch))
  rot_x = rot_x.at[1, 2].set(jnp.sin(pitch))
  rot_x = rot_x.at[2, 1].set(-jnp.sin(pitch))
  rot_x = rot_x.at[2, 2].set(jnp.cos(pitch))

  rot_y = jnp.zeros((3, 3))
  rot_y = rot_y.at[0, 0].set(jnp.cos(yaw))
  rot_y = rot_y.at[0, 2].set(-jnp.sin(yaw))
  rot_y = rot_y.at[1, 1].set(1)
  rot_y = rot_y.at[2, 0].set(jnp.sin(yaw))
  rot_y = rot_y.at[2, 2].set(jnp.cos(yaw))

  rot_z = jnp.zeros((3, 3))
  rot_z = rot_z.at[0, 0].set(jnp.cos(roll))
  rot_z = rot_z.at[0, 1].set(jnp.sin(roll))
  rot_z = rot_z.at[1, 0].set(-jnp.sin(roll))
  rot_z = rot_z.at[1, 1].set(jnp.cos(roll))
  rot_z = rot_z.at[2, 2].set(1)
  return rot_z @ rot_x @ rot_y


def camera_from_rpyf(
    height: int, width: int, roll: float, pitch: float, yaw: float, hfov: float
) -> jaxcam.Camera:
  """Creates a camera from roll, pitch, yaw, hfov, height, width."""
  cam_r_identity = rotation_from_rpy(roll, pitch, yaw, degree=True)
  focal_length = width / 2 / jnp.tan(deg2rad(hfov / 2))
  principal_point = jnp.array([width / 2, height / 2]) + 0.5
  image_size = jnp.array([width, height])
  return jaxcam.Camera.create(
      orientation=cam_r_identity,
      position=jnp.zeros(3),
      focal_length=focal_length,
      principal_point=principal_point,
      image_size=image_size,
      is_fisheye=False,
      radial_distortion=np.zeros(3),
      tangential_distortion=np.zeros(2),
  )


def haversine_distance(
    p1_deg: FloatArray['... 2'],
    p2_deg: FloatArray['... 2'],
    numpy: bool = False,
) -> FloatArray['... 1']:
  """Computes the haversine distance between two points (in meters)."""
  xnp = np if numpy else jnp
  d_lat = xnp.radians(p2_deg[..., 0] - p1_deg[..., 0])
  d_lng = xnp.radians(p2_deg[..., 1] - p1_deg[..., 1])
  a = (
      xnp.sin(d_lat / 2) ** 2
      + xnp.cos(xnp.radians(p1_deg[..., 0]))
      * xnp.cos(xnp.radians(p2_deg[..., 0]))
      * xnp.sin(d_lng / 2) ** 2
  )
  c = 2 * xnp.atan2(xnp.sqrt(a), xnp.sqrt(1 - a))
  return c[..., None] * 6371 * 1000


def distance_metrics(
    distance_to_gt: FloatArray['B'], prefix: str = ''
) -> dict[str, Any]:
  """Computes distance metrics."""
  thresholds = [
      50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000,
      25_000, 50_000, 100_000, 200_000, 500_000, 750_000, 1_000_000,
  ]
  key_to_threshold = {
      f'{prefix}within_{t if t < 1000 else t // 1000}{"m" if t < 1000 else "km"}': t
      for t in thresholds
  }
  metrics = {f'{prefix}distance_to_gt': distance_to_gt}
  for key, threshold in key_to_threshold.items():
    metrics[key] = (distance_to_gt < threshold).astype(jnp.float32)
  return metrics


def ecef_to_latlngalt(
    ecef_coords: FloatArray['... 3'],
    max_iter: int = 2,
    tolerance: float | None = None,
) -> FloatArray['... 3']:
  """Converts ECEF coordinates to (lat_deg, lng_deg, altitude_m) via Bowring's method."""
  x = ecef_coords[..., 0]
  y = ecef_coords[..., 1]
  z = ecef_coords[..., 2]

  # WGS 84 ellipsoid parameters
  a = 6378137.0  # Semi-major axis (meters)
  b = 6356752.314245179  # Semi-minor axis (meters)
  e_sq = 1 - (b / a) ** 2  # First eccentricity squared

  p = jnp.sqrt(x**2 + y**2)
  lon_rad = jnp.arctan2(y, x)
  lat_rad = jnp.arctan2(z, p * jnp.sqrt(1 - e_sq))
  alt = jnp.zeros_like(lat_rad)

  for _ in range(max_iter):
    sin_lat = jnp.sin(lat_rad)
    n = a / jnp.sqrt(1 - e_sq * sin_lat**2)
    h_prev = alt
    alt = p / jnp.cos(lat_rad) - n
    lat_rad_prev = lat_rad
    lat_rad = jnp.arctan2(z / (p * (1 - e_sq * n / (n + alt))), 1)
    if tolerance is not None:
      if jnp.all(jnp.abs(lat_rad - lat_rad_prev) < tolerance) and jnp.all(
          jnp.abs(alt - h_prev) < tolerance
      ):
        break

  return jnp.stack([jnp.degrees(lat_rad), jnp.degrees(lon_rad), alt], axis=-1)
