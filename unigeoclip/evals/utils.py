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

"""Utility functions for evals."""

from typing import Any, TypeAlias

from etils.array_types import FloatArray
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.train_lib import train_utils

from scaling_geoloc.models import base

Predictions: TypeAlias = FloatArray['... D']


def latlng_grid_gallery(
    latlngs: FloatArray['N 2'], cell_size_deg: float
) -> FloatArray['M 2']:
  """Builds a gallery of candidate locations from a regular lat/lng grid.

  Each input lat/lng is binned onto a regular grid with ``cell_size_deg``
  spacing, and the centers of the distinct occupied cells are returned as
  candidate locations to retrieve against. Coarser ``cell_size_deg`` yields a
  smaller, coarser gallery -- the resolution knob for the geolocation metric.

  Args:
    latlngs: Observed ``(latitude, longitude)`` pairs, in degrees.
    cell_size_deg: Grid cell size in degrees.

  Returns:
    The distinct grid-cell centers, in degrees.
  """
  latlngs = np.asarray(latlngs)
  cell_indices = np.floor(latlngs / cell_size_deg)
  centers = (np.unique(cell_indices, axis=0) + 0.5) * cell_size_deg
  return jnp.asarray(centers, dtype=jnp.float32)


def get_distance_metrics(
    distance_meters: FloatArray['...'],
    prefix: str,
    max_k: int,
) -> dict[str, Any]:
  """Computes distance metrics."""
  metrics = {}
  for k in [10, 50, 100, 200, 500, 1000, 5000, 10_000]:
    acc_k = (distance_meters[:, 0] < k).astype(jnp.float32)
    metrics[f'{prefix}_within_{k}m'] = jnp.mean(acc_k)

  for k in [10, 50, 100, 200, 500, 1000]:
    acc_k = (
        distance_meters[
            :,
            :5,
        ].min(axis=-1)
        < k
    ).astype(jnp.float32)
    metrics[f'{prefix}_within_{k}m@5'] = jnp.mean(acc_k)

  for k in [1, 5, 20, max_k]:
    distance_meters_k = distance_meters[:, :k].min(axis=-1)
    metrics[f'{prefix}_Haversine@{k:03d}'] = distance_meters_k.mean()

  return metrics


def get_latlng_representations(
    train_state: train_utils.TrainState,
    latlngs: FloatArray['N 2'],
    model: base.BaseModel,
    batch_size: int,
) -> FloatArray['N D']:
  """Extracts representations for latlngs."""

  num_latlngs_total = latlngs.shape[0]
  num_devices = jax.local_device_count()

  def _get_reps(train_state, latlngs):
    variables = {'params': train_state.params, **train_state.model_state}
    return model.flax_model.apply(
        variables,
        latlngs=latlngs,
        method=model.flax_model.get_location_embeddings,
    )

  pmapped_get_reps = jax.pmap(_get_reps, axis_name='batch')
  all_representations = []
  for i in range(0, num_latlngs_total, batch_size):
    latlngs_batch = latlngs[i : i + batch_size]
    num_latlngs_batch = latlngs_batch.shape[0]
    pad_amount = (num_devices - (num_latlngs_batch % num_devices)) % num_devices
    if pad_amount > 0:
      padding = jnp.zeros(
          (pad_amount, latlngs_batch.shape[1]), dtype=latlngs_batch.dtype
      )
      latlngs_padded = jnp.concatenate([latlngs_batch, padding], axis=0)
    else:
      latlngs_padded = latlngs_batch

    latlngs_sharded = dataset_utils.shard(latlngs_padded)
    representations_padded = pmapped_get_reps(train_state, latlngs_sharded)
    representations = dataset_utils.unshard(representations_padded)
    all_representations.append(representations[:num_latlngs_batch])
  return jnp.concatenate(all_representations, axis=0)


def get_is_correct_batch(
    pred_a: Predictions,
    pred_b: Predictions,
    max_k: int,
    start_idx: int,
    end_idx: int,
) -> tuple[Any, Any]:
  """Returns whether the top k predictions are correct."""
  similarities_batch = jnp.einsum(
      'bd,cd->bc', pred_a[start_idx:end_idx], pred_b
  )
  log_probs = nn.log_softmax(similarities_batch, axis=-1)
  probs = jnp.exp(log_probs)
  pred_assignment_batch = jax.lax.top_k(probs, k=max_k)[1]
  gt_assignment_batch = np.arange(start_idx, end_idx)
  is_correct = pred_assignment_batch == gt_assignment_batch[:, None]
  return is_correct, pred_assignment_batch
