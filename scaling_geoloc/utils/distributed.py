
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

"""Various convenience functions."""

from collections.abc import Sequence
import time
from typing import Callable, Any, Mapping
from absl import logging
from etils.array_types import BoolArray
from etils.array_types import FloatArray
from etils.array_types import IntArray
from flax import jax_utils
import jax
import jax.numpy as jnp
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.train_lib import train_utils
import tqdm
from scaling_geoloc.utils import misc


def to_device_safe(data: Any, device: jax.Device) -> Any:
  def maybe_put_to_device(x):
    try:
      return jax.device_put(x, device)
    except TypeError:
      return x

  return jax.tree.map(maybe_put_to_device, data)


def get_process_batch_size(batch_size: int) -> int:
  if batch_size % jax.device_count() != 0:
    raise ValueError(
        f'batch_size {batch_size} must be divisible by '
        f' jax.device_count() ({jax.device_count()})'
    )
  return (batch_size // jax.device_count()) * jax.local_device_count()


def default_pmap_sharding(
    shape: tuple[int, ...],
    sharded_dim: int | None = 0,
    devices: list[jax.Device] | None = None,
) -> jax.sharding.Sharding:
  """Creates a sharding for pmap-style parallelism using only public JAX APIs."""
  if not jax.config.jax_pmap_shmap_merge:
    return jax.sharding.PmapSharding.default(
        shape, sharded_dim=sharded_dim, devices=devices
    )
  if sharded_dim is None:
    if devices is None:
      raise ValueError('One of sharded_dim or devices must be set.')
    mesh = jax.sharding.Mesh(np.array(devices), ('_pmap',))
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  if not shape:
    raise ValueError('shape must be non-empty for sharded_dim != None')
  num_ways_sharded = shape[sharded_dim]
  if devices is None:
    pmap_devices = np.array(jax.local_devices()[:num_ways_sharded])
  else:
    pmap_devices = np.array(devices)
  mesh = jax.sharding.Mesh(pmap_devices, ('_pmap',))
  spec_list = [None] * len(shape)
  spec_list[sharded_dim] = '_pmap'
  return jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec(*spec_list)
  )


def gather_global_fn(
    axis: int = 0,
    tiled: bool = True,
    donate: bool = False,
    start_end: tuple[int, int] | None = None,
) -> Callable[
    [FloatArray['S ...'], tuple[int, int] | None], FloatArray['S ...']
]:
  """Gathers a tensor over all devices."""
  donate_argnums = (0,) if donate else ()
  if axis != 0 and start_end is not None:
    raise ValueError('start_end is not supported for axis != 0.')

  def all_gather_fn(x, start_end):
    if start_end is not None:
      start, end = start_end
      x = x[start:end]
    return jax.lax.all_gather(x, axis_name='batch', axis=axis, tiled=tiled)

  return jax.pmap(
      all_gather_fn,
      axis_name='batch',
      donate_argnums=donate_argnums,
      static_broadcasted_argnums=(1,),
  )


def gather_globally(
    x: FloatArray['S ...'],
    axis: int = 0,
    tiled: bool = True,
    donate: bool = False,
    start_end: tuple[int, int] | None = None,
) -> FloatArray['S ...']:
  """Gathers a tensor over all devices."""
  return gather_global_fn(axis, tiled, donate)(x, start_end)


def gather_locally(x: FloatArray['NL N//NG ...']) -> FloatArray['N//NP ...']:
  """Gathers a tensor over all local devices."""
  return dataset_utils.unshard(x)


def gather_globally_safe(
    x: FloatArray['S ...'],
    tiled: bool = True,
    donate: bool = False,
) -> FloatArray['...']:
  """Gathers a tensor over all devices safely on cpu."""
  n_elements = x.shape[1]
  n_slice = max(n_elements // jax.device_count(), 1)

  host_data = []
  gather_fn = gather_global_fn(tiled=False, donate=donate)
  offset = 0
  while offset < n_elements:
    end = min(offset + n_slice, n_elements)
    data_i = gather_fn(x, (offset, end))  # NG x end-offset
    host_data.append(np.array(jax.device_get(data_i[0])))
    del data_i
    offset = end
  host_data = np.concatenate(host_data, axis=1)
  if tiled:
    return np.concatenate(host_data, axis=0)
  else:
    return host_data


def slice_globally(
    x_replicated: FloatArray['DL ...'],  # replicated
) -> FloatArray['S ...']:
  """Slices a replicated tensor s.t. each device has a different slice."""

  def _slice(x):
    axis_idx = jax.lax.axis_index('batch')
    size_per_device = x.shape[0] // jax.device_count()
    indices = jnp.arange(size_per_device) + size_per_device * axis_idx
    return x[indices]

  return jax.pmap(_slice, axis_name='batch')(x_replicated)


def gather_from_replicated(
    x_replicated: FloatArray['DL ...'],  # replicated
    indices: IntArray['DL K ...'],  # sharded
) -> FloatArray['S ...']:
  """Slices a replicated tensor s.t. each device has a different slice."""
  return jax.pmap(lambda x, i: x[i], axis_name='batch')(x_replicated, indices)


def get_process_slice(
    x: FloatArray['N ...'],  # numpy
) -> FloatArray['N//NP ...']:
  """Returns a slice of a tensor for a single process."""
  size_per_process = x.shape[0] // jax.process_count()
  start = jax.process_index() * size_per_process
  end = (jax.process_index() + 1) * size_per_process
  return x[start:end]


def distribute_globally(
    x: FloatArray['...'], shard_over_processes: bool = True
) -> FloatArray['S ...']:
  """Shards a tensor over all devices."""

  if shard_over_processes:
    x = get_process_slice(x)

  def pmap_shard(x):
    x = dataset_utils.shard(x)
    return jax.device_put(
        x,
        default_pmap_sharding(x.shape),
    )

  return jax.tree.map(pmap_shard, x)


def unravel_global_dataset_index(
    global_dataset_index: int,
    num_workers: int,
    batch_size: int,
    num_shards: int,
) -> tuple[int, int]:
  """Unravels the global dataset index to a batch and item index."""
  unsharded_index = global_dataset_index // num_shards
  step_index = unsharded_index // (batch_size * num_workers)
  local_index = unsharded_index % (batch_size * num_workers)
  worker_index = unsharded_index % num_workers
  batch_index = step_index * num_workers + worker_index
  item_index = (local_index - worker_index) // num_workers
  return batch_index, item_index


def sharded_similarity(
    batch_logits: FloatArray['B ... 1 D'],  # total K dimensions
    sharded_logits: FloatArray['1 ... N//ND D'],  # total K dimensions
    normalize: bool = False,
) -> FloatArray['B ... N']:
  """Computes the similarity between batch logits and sharded logits."""
  # Gather batch logits across devices.
  all_logits = jax.lax.all_gather(
      batch_logits, axis_name='batch', tiled=True  # Concat in axis 0.
  )  # ND*B x ... x D
  # Compute similarity to sharded cell codes per device.
  similarity = misc.compute_similarity(
      all_logits,  # ND*B x ... x 1 x D
      sharded_logits,  # Add device dim: 1 x ... x N // ND x D
      normalize=normalize,
  )[..., 0]
  # Similarity is ND*B x ... x N // ND

  # All-to-all to get similarity across devices.
  return jax.lax.all_to_all(
      similarity,
      axis_name='batch',
      split_axis=0,  # ND*B --> B
      concat_axis=len(similarity.shape) - 1,  # Does not support -1 index.
      tiled=True,  # This chunks the batch dimension: ND*B --> B.
  )


def sharded_batched_gather(
    x: FloatArray['N//DG D'],
    indices: IntArray['B K'],
    axis_name: str = 'batch',
) -> FloatArray['B K D']:
  """Gathers x[indices]."""
  # Still quite memory intensive --> make cyclic.
  size_per_shard = x.shape[0]
  device_offset = jax.lax.axis_index(axis_name) * size_per_shard
  indices = jax.lax.all_gather(indices, axis_name=axis_name) - device_offset
  # DG x B x K
  valid = (indices >= 0) & (indices < size_per_shard)

  # DG x B x K x D
  x_gathered = jnp.take_along_axis(
      x[None, None], indices[..., None], axis=-2, fill_value=0.0
  )
  x_gathered = jnp.where(valid[..., None], x_gathered, 0.0)
  return jax.lax.psum_scatter(x_gathered, axis_name=axis_name)


def unshard_or_unreplicate_from_shape_fn(
    target_shape: tuple[int, ...] | Sequence[tuple[int, ...]],
    half_precision: bool = False,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Unshards a tensor if it has the target shape, otherwise unreplicates."""
  if half_precision:
    logging.info('Unshard or unreplicate with half precision.')
  if isinstance(target_shape[0], int):
    target_shape = [target_shape]

  def unshard_or_unreplicate(x):
    if x.shape in target_shape:
      # Carefully gather sharded cell codes to CPU.
      x = gather_globally_safe(x)
      if half_precision:
        x = x.astype(np.float16)
      return x
    else:
      # All other parameters are just transferred to CPU.
      return jax.device_get(x[0])

  return unshard_or_unreplicate


def shard_or_replicate_from_shape_fn(
    target_shape: tuple[int, ...] | Sequence[tuple[int, ...]],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Shards a tensor if it has the target shape, otherwise replicates."""
  if isinstance(target_shape[0], int):
    target_shape = [target_shape]

  def shard_or_replicate(x):
    if x.shape in target_shape:
      # Cell codes should be sharded over all devices.
      if x.dtype == np.float16:
        x = x.astype(np.float32)
      return distribute_globally(x)
    else:
      # All other parameters should be replicated.
      return jax_utils.replicate(x)

  return shard_or_replicate


def pmean_or_divide_from_shape_fn(
    target_shape: tuple[int, ...] | Sequence[tuple[int, ...]],
    axis_name: str = 'batch',
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Computes the pmean or divide."""
  if isinstance(target_shape[0], int):
    target_shape = [target_shape]

  def pmean_or_divide(x):
    if x.shape in target_shape:
      # pmean divides by the number of devices, so we also do this here.
      return x / jax.device_count()
    else:
      # All other gradients are pmean'ed.
      return jax.lax.pmean(x, axis_name=axis_name)

  return pmean_or_divide


def unshard_or_unreplicate_train_state(
    train_state: train_utils.TrainState,
    unshard_or_unreplicate_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> train_utils.TrainState:
  """Unshard or unreplicate a train state my mapping a function over params and opt_state."""
  logging.info('Starting unreplicate_state....')
  start = time.time()
  # pytype: disable=attribute-error
  train_state = train_state.replace(
      params=jax.tree.map(unshard_or_unreplicate_fn, train_state.params),
      opt_state=jax.tree.map(unshard_or_unreplicate_fn, train_state.opt_state),
      global_step=jax.device_get(
          jax_utils.unreplicate(train_state.global_step)
      ),
      rng=jax.device_get(jax_utils.unreplicate(train_state.rng)),
      model_state=jax.tree.map(
          unshard_or_unreplicate_fn, train_state.model_state
      ),
      metadata=jax.device_get(jax_utils.unreplicate(train_state.metadata)),
  )
  # pytype: enable=attribute-error
  logging.info('unreplicate_state took %f seconds.', time.time() - start)
  return train_state


def shard_or_replicate_train_state(
    train_state: train_utils.TrainState,
    shard_or_replicate_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> train_utils.TrainState:
  """Shard or replicate a train state my mapping a function over params and opt_state."""
  logging.info('Starting replicate_state....')
  start = time.time()
  # pytype: disable=attribute-error
  train_state = train_state.replace(
      params=jax.tree.map(shard_or_replicate_fn, train_state.params),
      opt_state=jax.tree.map(shard_or_replicate_fn, train_state.opt_state),
      global_step=jax_utils.replicate(train_state.global_step),
      rng=jax_utils.replicate(train_state.rng),
      model_state=jax.tree.map(shard_or_replicate_fn, train_state.model_state),
      metadata=jax_utils.replicate(train_state.metadata),
  )
  logging.info('replicate_state took %f seconds.', time.time() - start)
  return train_state
  # pytype: enable=attribute-error#


pmap_topk = jax.pmap(
    misc.find_topk,
    static_broadcasted_argnums=[2, 4, 5, 6],
    out_axes=(1, 1),
    axis_name='batch',
)


def parallel_topk(
    qy: FloatArray['Q D'],  # replicated
    db: FloatArray['N d_size//N D'],
    k: int,
    recall_target: float | None = None,  # None for exact search
    similarity_fn: Callable[[FloatArray, FloatArray], FloatArray] | str = 'dot',
    ascending: bool = False,
) -> tuple[FloatArray['Q K'], IntArray['Q K']]:
  """Parallel nearest neighbor search (top-k)."""
  sim, neighbors = pmap_topk(
      jax.device_put_replicated(qy, jax.local_devices()),
      db,
      k,
      jnp.arange(db.shape[0]) * db.shape[1],  # db_offset
      recall_target,
      similarity_fn,
      ascending,
  )
  # Collapse the device dimension.
  sim, neighbors = jax.tree.map(
      lambda x: jax.lax.collapse(x, start_dimension=1), (sim, neighbors)
  )  # Q x NK
  # Get top-k over all devices.
  sim, resorted_idx = jax.lax.top_k(sim if not ascending else -sim, k=k)
  sim = sim if not ascending else -sim
  neighbors = jnp.take_along_axis(neighbors, resorted_idx, axis=-1)

  return sim, neighbors


def distributed_topk(
    qy: FloatArray['Q D'],  # sharded
    db: FloatArray['d_size//N D'],  # Assumes db is spread across ALL devices.
    k: int,
    recall_target: float | None = None,  # None for exact search
    similarity_fn: Callable[[FloatArray, FloatArray], FloatArray] | str = 'dot',
    ascending: bool = False,
    where: BoolArray['d_size//N'] | None = None,  # Optional mask (True is keep)
    final_topk: bool = True,
) -> tuple[FloatArray['Q K'], IntArray['Q K']]:
  """Distributed nearest neighbor search with pmap. Gather over ALL devices."""
  # Get all queries across devices.
  a_idx = jax.lax.axis_index('batch')
  qy = jax.lax.all_gather(qy, axis_name='batch', tiled=True)
  sim, neighbors = misc.find_topk(
      qy,
      db,
      k,
      db_offset=a_idx * db.shape[0],
      recall_target=recall_target,
      similarity_fn=similarity_fn,
      ascending=ascending,
      where=where,
  )

  sim, neighbors = jax.tree.map(
      lambda x: jax.lax.all_to_all(
          x,
          axis_name='batch',
          split_axis=0,  # ND*B --> B
          concat_axis=len(x.shape) - 1,  # Does not support -1 index.
          tiled=True,  # This chunks the batch dimension: ND*B --> B.
      ),
      (sim, neighbors),
  )

  # Get top-k over all devices.
  if final_topk:
    sim, resorted_idx = jax.lax.top_k(sim if not ascending else -sim, k=k)
    sim = sim if not ascending else -sim
    neighbors = jnp.take_along_axis(neighbors, resorted_idx, axis=-1)

  return sim, neighbors


def pmap_self_topk(
    db: FloatArray['S DB D'],
    k: int,
    batch_size: int,
    show_progress: bool = False,
    where: BoolArray['S DB'] | None = None,
    **kwargs,
) -> tuple[FloatArray['S DB K'], IntArray['S DB K']]:
  """Pmap nearest neighbor search (top-k) with mask."""
  if where is None:
    where = jax_utils.replicate(jnp.ones((db.shape[1],), dtype=jnp.bool_))

  def topk_fn(
      query_shard: FloatArray['Qi D'],
      db: FloatArray['DB D'],
      where: BoolArray['DB'],  # Valid mask.
  ):
    return distributed_topk(
        query_shard,
        db,
        k,
        where=where,
        **kwargs,
    )

  pmapped_topk = jax.pmap(
      topk_fn,
      axis_name='batch',
      donate_argnums=(0,),
  )
  all_sim = []
  all_neighbors = []

  for i in tqdm.tqdm(
      range(db.shape[1] // batch_size + 1), disable=not show_progress
  ):
    start = i * batch_size
    if start >= db.shape[1]:
      break
    end = min((i + 1) * batch_size, db.shape[1])
    sim, neighbors = pmapped_topk(db[:, start:end], db, where)
    all_sim.append(sim)
    all_neighbors.append(neighbors)
  return jnp.concatenate(all_sim, axis=1), jnp.concatenate(
      all_neighbors, axis=1
  )


def sharded_argmax(
    x: FloatArray['ND*B N//ND'],
) -> tuple[FloatArray['B'], IntArray['B']]:
  """Sharded argmax."""

  best_idx_on_device = jnp.argmax(
      jax.lax.stop_gradient(x), axis=-1, keepdims=True
  )
  best_sim_on_device = jnp.take_along_axis(
      jax.lax.stop_gradient(x), best_idx_on_device, axis=-1
  )
  best_sims, best_idxs = [
      jax.lax.all_to_all(
          x,
          axis_name='batch',
          tiled=True,
          split_axis=0,
          concat_axis=len(x.shape) - 1,
      )
      for x in [best_sim_on_device, best_idx_on_device]
  ]
  best_device_idx = jnp.argmax(best_sims, axis=-1, keepdims=True)
  best_idx = jnp.take_along_axis(best_idxs, best_device_idx, axis=-1)
  best_sim = jnp.take_along_axis(best_sims, best_device_idx, axis=-1)
  # Add device offset.
  best_idx = best_idx + best_device_idx * x.shape[-1]
  return best_sim[..., 0], best_idx[..., 0]


def max_and_masked_expsum(
    batch_logits: FloatArray['B D'],  # total K dimensions
    batch_latlng: FloatArray['B 2'],  # total K dimensions
    sharded_logits: FloatArray['N//ND D'],  # total K dimensions
    global_latlng: FloatArray['N 2'],  # total K dimensions
    normalize: bool = False,
    beta: float = 1.0,
    base: float = 0.0,
    distance_threshold: float | None = None,
    required_idx: IntArray['B K'] | None = None,
) -> Mapping[str, Any]:
  """Computes the similarity between batch logits and sharded logits."""
  # Gather batch logits across devices.
  all_logits = jax.lax.all_gather(
      jnp.expand_dims(batch_logits, axis=-2),
      axis_name='batch',
      tiled=True,  # Concat in axis 0.
  )  # ND*B x ... x D

  # Compute similarity to sharded cell codes per device.
  similarity = misc.compute_similarity(
      all_logits,  # ND*B x 1 x D
      sharded_logits[None],  # Add device dim: 1 x N // ND x D
      normalize=normalize,
  )[..., 0]

  best_sim, best_idx = sharded_argmax(similarity)

  pos_sim = None
  if required_idx is not None:
    size_per_shard = similarity.shape[-1]
    device_offset = jax.lax.axis_index('batch') * size_per_shard
    all_required_idx = (
        jax.lax.all_gather(required_idx, axis_name='batch', tiled=True)
        - device_offset
    )  # ND*B x K
    valid = (all_required_idx >= 0) & (all_required_idx < size_per_shard)
    pos_sim = jnp.take_along_axis(
        similarity, all_required_idx, axis=-1, fill_value=0.0
    )
    pos_sim = jnp.where(valid, pos_sim, 0.0)
    pos_sim = jax.lax.psum_scatter(pos_sim, axis_name='batch', tiled=True)

    # Remove positives from sum.
    oob_required_idx = jnp.where(
        all_required_idx < 0, similarity.shape[-1], all_required_idx
    )
    similarity = jnp.put_along_axis(
        similarity,
        oob_required_idx,
        -jnp.inf,
        axis=-1,
        mode='drop',
        inplace=False,
    )

  if distance_threshold is not None:
    all_latlng = jax.lax.all_gather(
        batch_latlng, axis_name='batch', tiled=True  # Concat in axis 0.
    )  # ND*B x ... x 2
    latlng_on_device = global_latlng.reshape((jax.device_count(), -1, 2))
    distances = misc.haversine_distance(  # ND*B x N//ND
        jnp.expand_dims(all_latlng, axis=-2),
        latlng_on_device[jax.lax.axis_index('batch')][None],
    )[..., 0]
    valid_negatives = distances > distance_threshold
    similarity = jnp.where(valid_negatives, similarity, -jnp.inf)

  expsum = jnp.exp((similarity - base) * beta).sum(-1)
  # Similarity is ND*B x ... x N // ND
  expsum = jax.lax.psum_scatter(expsum, axis_name='batch', tiled=True)
  output = {
      'best_sim': best_sim,
      'best_idx': best_idx,
      'expsum': expsum,
  }
  if pos_sim is not None:
    output['pos_sim'] = pos_sim
  return output
