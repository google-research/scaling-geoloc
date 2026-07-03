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

from collections.abc import Sequence
from typing import Any, TypeAlias

from etils.array_types import FloatArray
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.train_lib import train_utils

from scaling_geoloc.models import base
# Public release: the pip ``s2sphere`` package replaces the internal s2 library.
import s2sphere as s2

Predictions: TypeAlias = FloatArray['... D']


def _token_str(token: bytes | str) -> str:
  return token.decode('utf-8') if isinstance(token, bytes) else token


def s2cell_token_to_latlng(
    s2cell_tokens: Sequence[bytes | str],
) -> FloatArray['N 2']:
  """Returns the latlng of the center of the s2cell.

  Args:
    s2cell_tokens: The s2cell token list.

  Returns:
    A tuple containing the latitude and longitude of the center of the s2cell.
  """
  latlngs = []
  for s2cell_token in s2cell_tokens:
    s2cell_id = s2.CellId.from_token(_token_str(s2cell_token))
    s2latlng = s2cell_id.to_lat_lng()
    latlngs.append((s2latlng.lat().degrees, s2latlng.lng().degrees))
  return jnp.array(latlngs)


def latlng_to_s2cell_token(
    latlngs: FloatArray['N 2'], level: int
) -> Sequence[str]:
  """Returns the s2cell token at the given level for the given latlng."""
  tokens = []
  for i in range(latlngs.shape[0]):
    s2latlng = s2.LatLng.from_degrees(
        float(latlngs[i, 0]), float(latlngs[i, 1])
    )
    s2cell_id = s2.CellId.from_lat_lng(s2latlng).parent(level)
    tokens.append(s2cell_id.to_token())
  return tokens


def convert_s2cell_level(
    s2cell_tokens: Sequence[bytes | str], level: int
) -> Sequence[str]:
  """Converts a list of s2cell tokens to a different level.

  Args:
    s2cell_tokens: The s2cell token list.
    level: The target level.

  Returns:
    A list of s2cell tokens at the target level.
  """
  new_tokens = set()
  for token in s2cell_tokens:
    s2cell_id = s2.CellId.from_token(_token_str(token))
    if level < s2cell_id.level():
      new_tokens.add(s2cell_id.parent(level).to_token())
    elif level > s2cell_id.level():
      child = s2cell_id.child_begin(level)
      for _ in range(4 ** (level - s2cell_id.level())):
        new_tokens.add(child.to_token())
        child = child.next()
    else:
      new_tokens.add(s2cell_id.to_token())
  return sorted(list(new_tokens))


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
