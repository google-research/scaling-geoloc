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

"""Render utils."""

from collections.abc import Mapping
import functools
from etils.array_types import Array
from etils.array_types import FloatArray
import jax
import jax.numpy as jnp
import jaxcam
from scaling_geoloc.utils import grids
from scaling_geoloc.utils import hidden_helpers
from scaling_geoloc.utils import misc


def create_render_grid(
    camera: jaxcam.Camera,
    *,
    depth_range: FloatArray["nd"],
    pixels: FloatArray | tuple[int, int, int] = None,  # (H, W, stride)
    depth_prior: FloatArray["H W 1"] | None = None,
) -> FloatArray["H W nd 3"]:
  """Create sampling grid in view cone."""
  if isinstance(pixels, tuple):
    # Pixels from (h, w, stride).
    h, w, ps = pixels
    pixels = (
        misc.grid_indices(
            h // ps,
            w // ps,
        )
        + 0.5
    ) * ps

  if depth_prior is not None:
    h, w = pixels.shape[-3:-1]
    depth_prior = jax.image.resize(
        depth_prior,
        shape=(h, w, 1),
        method="nearest",
    )

  # Generate rays through pixels
  rays_through_pixels = jaxcam.pixels_to_rays(camera, pixels)

  # Sample depth range.
  depths = jnp.expand_dims(depth_range, range(rays_through_pixels.ndim - 1))
  if depth_prior is not None:
    # If a depth prior is provided, we apply multiplicative offsets.
    depths = depth_prior * depths

  def rays_to_points(camera, rays, depth):
    """Generate points along ray."""
    rays = rays[..., None, :]
    ray_depth = jaxcam.depth_to_ray_depth(camera, rays, depth)
    points = rays * ray_depth[..., None] + camera.position
    pad_shape = [(0, 0)] * len(depth.shape) + [(0, 1)]
    points = jnp.pad(points, pad_shape, constant_values=1.0)
    points = points[..., :3] / points[..., -1:]
    return points

  return rays_to_points(camera, rays_through_pixels, depths)


def integrate_ray_over_voxels(
    samples_xyz: FloatArray["D 3"],  # increasing distance
    f_grid: grids.FeatureGrid,
) -> Mapping[str, Array]:
  """Integrate along ray over (sparse) voxel grid."""
  grid_t_w = f_grid.grid_t_w
  grid = f_grid.grid
  grid_ids = f_grid.grid_ids
  features = f_grid.features
  cell_opacities = f_grid.opacities

  # We transform the samples to the grid frame only for interpolation.
  grid_xyz = jnp.einsum(
      "ij,nj->ni", grid_t_w[:3, :], misc.to_homogeneous(samples_xyz)
  )
  # Generate trilinear interpolation weights and indices.
  interp_ids, weights = (
      hidden_helpers.make_linear_interpolation_indices_flat_nd(
          (grid_xyz / grid.cell_size[None]).T - 0.5, grid.extent
      )
  )  # 8 x D, # 8 X D

  # Create grid index to local index conversion array.
  nx, ny, nz = grid.extent
  sparse_grid_id = -jnp.ones(nx * ny * nz, dtype=jnp.int32)
  sparse_grid_id = sparse_grid_id.at[grid_ids].set(
      jnp.arange(0, grid_ids.shape[0])
  )

  def accumulate_fn(x, weights):
    # return misc.masked_mean(x, weights, axis=0)
    return jnp.sum(x * weights, axis=0)

  def interpolate(ids_i, weights_i):
    """Trilinear weighted interpolation."""
    sparse_grid_ids_i = sparse_grid_id[ids_i]  # local indices.
    features_i = features[sparse_grid_ids_i]
    # cell_opacities_i = cell_opacities[sparse_grid_ids_i, 0]  # 0 to 1
    # A valid interpolation is inside the grid and exists in the sparse grid.
    # is_inside = jnp.isin(ids_i, grid_ids)
    is_inside = sparse_grid_ids_i > -1
    weights_i = weights_i * is_inside  #  * cell_opacities_i
    feature = misc.masked_mean(features_i, weights_i[..., None], axis=0)
    opacity = weights_i.sum()
    return feature, opacity

  interp_features, _ = jax.vmap(interpolate)(interp_ids.T, weights.T)

  # Collect metrics.
  grid_index = grid.xyz_to_index(grid_xyz)
  sample_grid_ids = grid.index_to_id(grid_index)
  sample_local_ids = sparse_grid_id[sample_grid_ids]

  point_in_voxel = (sparse_grid_id[sample_grid_ids] > -1) & grid.index_in_grid(
      grid_index
  )
  raw_opacities = cell_opacities[sample_local_ids, 0]
  raw_opacities = jnp.where(
      raw_opacities > 1.0e-6, raw_opacities, raw_opacities + 1.0e-6
  )
  opacities = raw_opacities * point_in_voxel
  occupancies = opacities > 0

  any_hit = point_in_voxel.any()

  # The weights are (1 - sum(opc, 0, i-1)) * opc[i] = T[i] * opc[i].
  # The transmittance can be computed as the cumprod of (1-opc[i])
  transmittance = jnp.cumprod(
      1.0
      - jnp.concatenate([jnp.zeros_like(opacities[:1]), opacities[:-1]])
      # - 1.0e-6  # Small offset to avoid vanishing gradients.
  )
  # Multiply alpha values in.
  weights = opacities * transmittance
  cdf = jnp.cumsum(weights)
  alpha = jnp.sum(weights)  # Between 0 and 1. 1 means full hit, 0 no hit.

  # The occupancy is the binarized opacity.
  occupancy_cdf = jnp.cumsum(occupancies).clip(max=1.0)
  occupancy_weights = jnp.diff(occupancy_cdf, prepend=0)

  # Depth along the ray (not z-depth).
  ray_depths = (samples_xyz**2).sum(-1) ** 0.5
  # Spacing between sample points.
  sample_dist = jnp.diff(ray_depths, prepend=ray_depths[0])

  # Whether any voxel was hit.
  is_hit = occupancies.any()
  expected_xyz = accumulate_fn(samples_xyz, weights[..., None])

  if len(weights) > 1:
    expected_xyz_cov = jnp.cov(
        samples_xyz, bias=False, aweights=weights, rowvar=False
    )
  else:
    expected_xyz_cov = jnp.zeros_like(expected_xyz)

  # Metrics for first intersection with grid.
  first_hit_idx = jnp.argmax(occupancies)
  first_hit_xyz = samples_xyz[first_hit_idx]
  first_hit_xyz = jnp.where(
      point_in_voxel[first_hit_idx],
      first_hit_xyz,
      -jnp.ones_like(first_hit_xyz) * 1000,
  )
  first_hit_opacity = misc.masked_mean(opacities, occupancy_weights)

  # Blend features from all hits.
  expected_feature = accumulate_fn(interp_features, weights[..., None])

  return {
      "is_hit": is_hit[..., None],
      "any_hit": any_hit[..., None],
      "sample_dist": sample_dist[..., None],
      "occupancies": occupancies[..., None],
      "occupancy_cdf": occupancy_cdf[..., None],
      "occupancy_weights": occupancy_weights[..., None],
      "opacities": opacities[..., None],
      "alpha": alpha[..., None],
      "cdf": cdf[..., None],
      "weights": weights[..., None],
      "raw_opacities": opacities[..., None],
      "point_in_voxel": point_in_voxel[..., None],
      "first_hit_xyz": first_hit_xyz,
      "first_hit_idx": first_hit_idx,
      "first_hit_voxel_idx": sample_grid_ids[first_hit_idx],
      "first_hit_in_voxel": point_in_voxel[first_hit_idx, None],
      "first_hit_opacity": first_hit_opacity[..., None],
      "expected_xyz": expected_xyz,
      "expected_xyz_cov": expected_xyz_cov,
      "expected_feature": expected_feature,
  }


def render_view_from_grid(
    jcam: jaxcam.Camera,
    feature_grid: grids.FeatureGrid,
    pixels: tuple[int, int, int],  # (H, W, stride)
    depth_range: FloatArray["nd"],
    depth_prior: FloatArray["H W 1"] | None = None,
) -> FloatArray["h w c"]:
  """Render features back to images."""
  render_grid = create_render_grid(
      jcam,
      depth_range=depth_range,
      pixels=pixels,
      depth_prior=depth_prior,
  )
  render_fn = functools.partial(integrate_ray_over_voxels, f_grid=feature_grid)
  rendered_views = jax.vmap(jax.vmap(render_fn))(render_grid)
  return rendered_views


def render_views_from_grid(
    jcam: jaxcam.Camera,
    feature_grid: grids.FeatureGrid,
    pixels: tuple[int, int, int],  # (H, W, stride)
    depth_range: FloatArray["nd"],
    depth_prior: FloatArray["N H W 1"] | None = None,
    batch_size: int | None = 1,  # Mini-batch size for reduced memory usage.
) -> FloatArray["N h w c"]:
  """Render feature grid back to N images. Supports mini-batching."""
  render_fn_ = functools.partial(
      render_view_from_grid,
      feature_grid=feature_grid,
      pixels=pixels,
      depth_range=depth_range,
  )

  render_fn_ = jax.checkpoint(render_fn_)

  if batch_size is None:
    map_fn = lambda f, *args, **kwargs: jax.vmap(f)(*args, **kwargs)
  else:
    map_fn = functools.partial(jax.lax.map, batch_size=batch_size)

  if depth_prior is None:
    render_fn = render_fn_
    return map_fn(render_fn, jcam)
  else:

    def render_fn(cam_and_depth_prior):
      return render_fn_(
          cam_and_depth_prior[0],
          depth_prior=cam_and_depth_prior[1],
      )

    return map_fn(render_fn, (jcam, depth_prior))
