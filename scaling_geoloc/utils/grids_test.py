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

import jax.numpy as jnp
import numpy as np

from scaling_geoloc.utils import grids
from google3.pyglib.contrib.g3_multiprocessing import googletest_launcher
from google3.testing.pybase import googletest


class GridsTest(googletest.TestCase):

  def test_aggregate_features_from_grid(self):
    grid_ids = jnp.array([0, 1, 2, 3, 4, 7, 10])
    w, c, s = 6, 3, 2  # w = num. tokens, c = token dim, s = voxels / patch
    features_map = jnp.ones((w, c))
    grid_id_map = -jnp.ones((w, s), dtype=int)
    max_num_observations = 20

    features_map = features_map.at[2].set(2.0)
    grid_id_map = grid_id_map.at[2].set([1, 3])

    features_map = features_map.at[4].set(5.0)
    grid_id_map = grid_id_map.at[4].set([7, 3])

    # This feature will be ignored, because it is not in the grid.
    features_map = features_map.at[0].set(99.0)
    grid_id_map = grid_id_map.at[0].set([5, -1])

    features_grid, occupancy_grid = grids.aggregate_features_in_grid(
        grid_ids, grid_id_map, features_map, max_num_observations
    )  # L x C

    self.assertEqual(features_grid.shape, (7, c))
    self.assertEqual(occupancy_grid.shape, (7,))

    np.testing.assert_array_equal(
        occupancy_grid, jnp.array([0, 1, 0, 2, 0, 1, 0])
    )

    np.testing.assert_allclose(
        features_grid[1],
        jnp.full(c, 2.0),  # features_map[2]
    )

    np.testing.assert_allclose(
        features_grid[5],
        jnp.ones((c,)) * 5.0,  # features_map[4]
    )

    # Both features 4 and 2 wrote into cell 3, so we expect the mean.
    np.testing.assert_allclose(
        features_grid[3],
        jnp.ones((c,)) * 3.5,  # (features_map[2] + features_map[4]) / 2
    )

    # Nothing wrote into cells[0, 2, 4, 6], so we expect all zero there.
    np.testing.assert_allclose(
        features_grid[jnp.array([0, 2, 4, 6])],
        jnp.zeros((4, c)),
    )

  def test_gather_features_from_grid(self):
    w, c, s = 5, 3, 2

    grid_ids = jnp.array([0, 1, 2, 3, 4, 7, 10])
    features_grid = jnp.ones((7, c))

    for i in range(7):
      features_grid = features_grid.at[i].set(i * 2.0)

    occupancy_grid = jnp.array([0, 1, 0, 0, 2, 1, 0])

    grid_id_query = -jnp.ones((w, s), dtype=int)

    grid_id_query = grid_id_query.at[0].set([4, -1])
    grid_id_query = grid_id_query.at[1].set([1, 7])
    grid_id_query = grid_id_query.at[3].set([1, 2])
    grid_id_query = grid_id_query.at[4].set([8, 8])

    features_query, occupancy_query = grids.gather_features_from_grid(
        grid_id_query, grid_ids, features_grid, occupancy_grid
    )

    self.assertEqual(features_query.shape, (w, c))
    self.assertEqual(occupancy_query.shape, (w,))

    np.testing.assert_array_equal(occupancy_query, jnp.array([1, 2, 0, 1, 0]))

    np.testing.assert_allclose(
        features_query[0],
        jnp.ones((c,)) * 8.0,  # features_grid[4]
    )

    np.testing.assert_allclose(
        features_query[1],
        jnp.ones((c,)) * 6.0,  # (features_grid[1] + features_grid[5]) / 2
    )

    np.testing.assert_allclose(
        features_query[3],
        jnp.ones((c,)) * 2.0,  # features_grid[1]
    )

    # The remaining ones should all be zero.
    np.testing.assert_allclose(
        features_query[jnp.array([2, 4])],
        jnp.zeros((2, c)),
    )

  def test_gather_features_via_grid(self):
    wq, n, wm, c, s = 5, 3, 7, 3, 2

    grid_id_query = -jnp.ones((wq, s), dtype=int)
    grid_id_map = -jnp.ones((n, wm, s), dtype=int)
    features_map = jnp.zeros((n, wm, c))

    max_num_cells = 50
    max_num_observations = 20

    for i in range(n):
      for j in range(wm):
        features_map = features_map.at[i, j].set(i + j)

    grid_id_map = grid_id_map.at[0, 0].set([1, 2])
    grid_id_map = grid_id_map.at[0, 1].set([2, -1])
    grid_id_map = grid_id_map.at[2, 4].set([5, -1])

    grid_id_query = grid_id_query.at[0].set([2, 3])
    grid_id_query = grid_id_query.at[1].set([5, -1])
    grid_id_query = grid_id_query.at[3].set([5, 2])

    results = grids.gather_features_via_grid(
        grid_id_query,
        grid_id_map,
        features_map,
        max_num_cells,
        max_num_observations,
    )

    features_query = results['features_query']
    occupancy_query = results['occupancy_query']

    self.assertEqual(features_query.shape, (wq, c))
    self.assertEqual(occupancy_query.shape, (wq,))

    np.testing.assert_array_equal(occupancy_query, jnp.array([1, 1, 0, 2, 0]))

    np.testing.assert_allclose(
        features_query[0],
        jnp.ones((c,)) * 0.5,  # (features_map[0, 0] + features_map[0, 1]) / 2
    )

    np.testing.assert_allclose(
        features_query[1],
        features_map[2, 4],  # features_map[2, 4]
    )

  def test_gather_features_via_grid_batched(self):
    wq, n, wm, c, s = 5, 3, 7, 3, 2
    b = 2

    grid_id_query = -jnp.ones((b, wq, s), dtype=int)
    grid_id_map = -jnp.ones((b, n, wm, s), dtype=int)
    features_map = jnp.zeros((b, n, wm, c))

    for i in range(b):
      for j in range(n):
        for k in range(wm):
          features_map = features_map.at[i, j, k].set(i * 10 + j + k)

    max_num_cells = 50
    max_num_observations = 20

    grid_id_map = grid_id_map.at[0, 1, 3].set([1, 2])
    grid_id_map = grid_id_map.at[1, 0, 1].set([2, -1])
    grid_id_map = grid_id_map.at[1, 2, 1].set([2, 5])

    grid_id_query = grid_id_query.at[0, 0].set([2, 3])
    grid_id_query = grid_id_query.at[1, 2].set([6, 2])
    grid_id_query = grid_id_query.at[1, 3].set([5, -1])

    results = grids.gather_features_via_grid_batched(
        grid_id_query,
        grid_id_map,
        features_map,
        max_num_cells,
        max_num_observations,
    )

    features_query = results['features_query']
    occupancy_query = results['occupancy_query']

    self.assertEqual(features_query.shape, (b, wq, c))
    self.assertEqual(occupancy_query.shape, (b, wq))

    np.testing.assert_array_equal(
        occupancy_query[0], jnp.array([1, 0, 0, 0, 0])
    )
    np.testing.assert_array_equal(
        occupancy_query[1], jnp.array([0, 0, 1, 1, 0])
    )

    np.testing.assert_allclose(
        features_query[0, 0],
        jnp.ones((c,)) * 4.0,  # features_map[0, 1, 3]
    )

    np.testing.assert_allclose(
        features_query[1, 2],
        jnp.ones((c,))
        * 12.0,  # (features_map[1, 0, 1] + features_map[1, 2, 1]) / 2
    )

    np.testing.assert_allclose(
        features_query[1, 3],
        jnp.ones((c,)) * 13.0,  # features_map[1, 2, 1]
    )


if __name__ == '__main__':
  googletest_launcher.main()
