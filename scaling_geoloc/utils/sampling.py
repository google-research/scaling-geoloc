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


"""Utilities for sampling geolocated data."""

import jax.numpy as jnp
import numpy as np
from etils.array_types import FloatArray
from typing import Callable
from scaling_geoloc.utils import misc
import jax


def bbox_norm(
    points: FloatArray['N d'],
    axis: int | tuple[int, ...] = 0,
    eps: float = 1.0e-8,
    center: bool = True,
) -> np.ndarray:
  """Returns point coordinates normalized to cube of size [0, 1]."""
  lower_bound = points.min(axis=axis, keepdims=True)
  upper_bound = points.max(axis=axis, keepdims=True)
  points_norm = (points - lower_bound) / (upper_bound - lower_bound).clip(
      min=eps
  )
  if center:
    points_norm = (points_norm - 0.5) * 2.0
  return points_norm


def spatio_temporal_residuals(
    a: FloatArray['... 3'],
    b: FloatArray['... 3'],
    min_distance_meters: float = 0.0,  # Min. spatial distance between points.
) -> FloatArray['... 2']:
  """Returns the spatio-temporal residuals between two point arrays."""
  # Treat temporal axis as categorical.
  spatial_distance = misc.l2_norm(a[..., 1:], b[..., 1:])
  temporal_distance = jnp.abs(a[..., :1] - b[..., :1])
  # Scale temporal residuals for balanced sampling over space-time.
  temporal_distance = temporal_distance.clip(max=1.0) * np.mean(
      spatial_distance
  )
  # We consider valid residuals if the spatial distance is larger than the
  # minimum distance.
  valid = spatial_distance > min_distance_meters**2
  return jnp.concatenate(
      [temporal_distance * valid, spatial_distance * valid], axis=-1
  )


# Source: third_party/py/big_vision/models/xp/fmr3d/pct.py
def farthest_point_sampling(
    pointcloud: jnp.ndarray,
    k: int,
    pc_mask: jnp.ndarray | None = None,
    closest_first: bool = False,
    rng: jnp.ndarray | int | None = None,
    residuals_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = misc.l2_norm,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Gets k farthest points for each example.

  The first point selected is the farthest point from the origin.  The remaining
  points are selected such that they are the farthest from the previously
  selected points.

  Warning: Any points with numerically identical coordinates are considered
  to be the same point and will only be selected once, so please add a small
  amount of noise to the pointcloud if duplicates should be selected
  separately.

  Args:
    pointcloud: [batch, num_points, 3] pointcloud.
    k: number of tokens to sample.
    pc_mask: optional[batch, num_points] mask for valid points in the
      pointcloud, with `1` for valid points and `0` for padding.
    closest_first: if True, the first point selected will be the closest to the
      origin.
    rng: optional jax random key for sampling. If None, the points are selected
      in a deterministic order (farthest next).
    residuals_fn: function to compute the residuals between two points (should
      be positive).

  Returns:
    Tuple of:
      - [batch, k, 3] sampled pointcloud.  Coordinates for padding points have
        0.0 values.
      - [batch, k] mask for valid points in the sampled pointcloud.
      - [batch, k] indices of the sampled points in the original pointcloud.
        Indices for paddings points are -1.
    If k is greater than `num_points`, the original pointcloud and mask
    are returned.
  """
  k = min(k, pointcloud.shape[1])

  if pc_mask is None:
    pc_mask = jnp.ones_like(pointcloud[..., 0], dtype=jnp.bool)

  pc_mask = pc_mask.astype(jnp.bool)

  # Set all padding point coordinates to 0 to remove any NaNs.
  pointcloud = jnp.where(pc_mask[..., jnp.newaxis], pointcloud, 0.0)
  init_point = jnp.zeros_like(pointcloud[:1, :1])
  # Arrange for the first selected point to be the one farthest from the origin.
  residuals = residuals_fn(pointcloud, init_point)
  squared_norm = residuals.sum(-1)
  if closest_first:
    squared_norm = jnp.where(pc_mask, squared_norm, jnp.inf)
    next_selected_index = jnp.argmin(squared_norm, axis=-1)
  else:
    squared_norm = jnp.where(pc_mask, squared_norm, -jnp.inf)
    next_selected_index = jnp.argmax(squared_norm, axis=-1)

  # As "carry" state we maintain the minimum squared distance to the set of
  # selected points so far.  All padding points are initialized to 0.0, and
  # each selected point will have their corresponding distance set to 0.0 also,
  # so we'll know that all valid points have been selected when
  # max(min_squared_dist) is 0.0.
  min_squared_residuals = jnp.where(
      pc_mask[..., None].repeat(residuals.shape[-1], axis=-1), jnp.inf, 0.0
  )

  if isinstance(rng, int):
    rng = jax.random.PRNGKey(rng)

  carry_init = dict(
      next_selected_index=next_selected_index,
      min_squared_residuals=min_squared_residuals,
      rng=rng,
  )

  def inner_fn(carry: dict[str, jnp.ndarray | None], unused_x):
    next_selected_index = carry['next_selected_index']
    min_squared_residuals = carry['min_squared_residuals']
    rng = carry['rng']

    # Gather point for selected index.
    # Shape: [batch, 1, 3]
    current_point = jnp.take_along_axis(
        pointcloud, next_selected_index[:, jnp.newaxis, jnp.newaxis], axis=1
    )

    # Determine if the selected point is valid or if we've already selected all
    # valid points since there are no remaining distances > 0.0.
    current_mask = (
        jnp.max(jnp.sum(min_squared_residuals, axis=-1), axis=-1) > 0.0
    )

    # Set sentinel values if the current point is padding.
    current_point = jnp.where(
        current_mask[:, jnp.newaxis, jnp.newaxis], current_point, 0.0
    )
    current_index = jnp.where(current_mask, next_selected_index, -1)

    # Find the distance from every point to the currently selected point
    # and updated `min_squared_dist` accordingly.
    squared_residuals_to_curr = residuals_fn(pointcloud, current_point)
    min_squared_residuals = jnp.minimum(
        min_squared_residuals, squared_residuals_to_curr
    )
    min_squared_dist = jnp.sum(min_squared_residuals, axis=-1)

    # Find next point to sample.
    if rng is None:
      next_selected_index = jnp.argmax(min_squared_dist, axis=-1)
    else:
      rngs = jax.random.split(rng, num=pointcloud.shape[0] + 1)
      rngs, rng = rngs[:-1], rngs[-1]
      next_selected_index = jax.vmap(
          lambda r, p: jax.random.choice(r, jnp.arange(p.shape[0]), p=p)
      )(rngs, jnp.sqrt(min_squared_dist))

    return dict(
        next_selected_index=next_selected_index,
        min_squared_residuals=min_squared_residuals,
        rng=rng,
    ), dict(
        selected_points=jnp.squeeze(current_point, axis=1),
        selected_masks=current_mask,
        selected_indices=current_index,
    )

  unused_carry, out = jax.lax.scan(inner_fn, carry_init, length=k)

  selected_points = out['selected_points'].swapaxes(0, 1)  # pylint: disable=attribute-error
  selected_masks = out['selected_masks'].swapaxes(0, 1)  # pylint: disable=attribute-error
  selected_indices = out['selected_indices'].swapaxes(0, 1)  # pylint: disable=attribute-error

  return selected_points, selected_masks, selected_indices
