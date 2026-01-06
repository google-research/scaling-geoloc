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

"""SV-Aerial hierarchical code model."""

import functools
from typing import Any, Sequence

from etils.array_types import FloatArray
from etils.array_types import IntArray
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt
import ml_collections
import numpy as np
from scenic.train_lib import train_utils

from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.models import hierarchical_cell_code
from scaling_geoloc.scenic.models import sv_aerial
from scaling_geoloc.scenic.models import sv_aerial_code
from scaling_geoloc.utils import misc


def is_sharded(
    level: int,
    num_levels: int,
    shard_last_n_layers: int,
):
  return level >= (num_levels - shard_last_n_layers)


def concat_tree(
    cell_codes: Sequence[FloatArray['... Ni D']],
    tree_idx: IntArray['N H'],
    xnp: Any = jnp,
    concat: bool = True,
) -> FloatArray['...N D*H|D']:
  """Gather cell codes at each level and concatenate them in the feature dim."""
  if concat:
    join_fn = functools.partial(xnp.concatenate, axis=-1)
  else:
    join_fn = sum
  return join_fn(
      [cell_codes[i][..., tree_idx[..., i], :] for i in range(len(cell_codes))]
  )


def cascade_tree(
    values: Sequence[FloatArray['... Ni D']],
    parents: Sequence[IntArray['Ni']],
    xnp: Any = jnp,
    concat: bool = True,
) -> Sequence[FloatArray['... Ni D*i|D']]:
  """Gather cell codes at each level and concatenate them in the feature dim."""
  outputs = [values[0]]
  for i in range(1, len(parents)):
    parent_values = outputs[i - 1][..., parents[i], :]
    if concat:
      outputs.append(xnp.concatenate([parent_values, values[i]], axis=-1))
    else:
      outputs.append(parent_values + values[i])
  return outputs


def l2_normalize_per_level(
    x: FloatArray['... D'],
    cell_code_dims: Sequence[int],
) -> FloatArray['... D']:
  hierarchical_logits = []
  offset = 0
  for cd in cell_code_dims:
    hierarchical_logits.append(x[..., offset : offset + cd])
    offset += cd
  return jnp.concatenate(hierarchical_logits, axis=-1)


class HierarchicalCellCodes(nn.Module):
  """Cell codes."""

  cell_tree: Sequence[IntArray['N K']]  # Each entry specifies K leaf idx.
  config: ml_collections.ConfigDict
  dtype: jnp.float32

  def setup(self):
    self._cell_code_dims = self.config.cell_code_dims or [
        self.config.cell_code_dim for _ in self.cell_tree
    ]

    def num_cells(level: int):
      if self.is_initializing() or not is_sharded(
          level, len(self.cell_tree), self.config.get('shard_last_n_layers', 0)
      ):
        return self.cell_tree[level].shape[0]
      else:
        return self.cell_tree[level].shape[0] // jax.device_count()

    self.cell_codes = [
        self.param(
            f'cell_codes_{i}',
            nn.initializers.normal(stddev=1 / np.sqrt(self._cell_code_dims[i])),
            (num_cells(i), self._cell_code_dims[i]),
            self.dtype,
        )
        for i in range(len(self.cell_tree))
    ]
    self.total_cell_code_dims = sum([x.shape[1] for x in self.cell_codes])
    self.num_childs = self.cell_tree[0].shape[
        1 if len(self.cell_tree) > 1 else 0
    ]
    self.num_roots = self.cell_tree[0].shape[0]

    self.num_levels = len(self.cell_tree)
    self.idx_to_tree_idx = self.cell_tree[-1]

    parent_idxs = [jnp.zeros_like(self.cell_tree[0])]
    for i in range(1, len(self.cell_tree)):
      parent_and_child = np.unique(
          np.asarray(self.cell_tree[-1])[..., i - 1 : i + 1], axis=0
      )
      if not np.all(
          parent_and_child[..., -1] == np.arange(parent_and_child.shape[0])
      ):
        raise ValueError('Parents not sorted.')
      parent_idxs.append(jnp.array(parent_and_child[..., 0]))
    self.parent_idxs = parent_idxs

  def compute_similarity_to_cell_codes(
      self,
      logits: FloatArray['B ... D'],
      cell_codes: FloatArray['... N|N//ND D'],
      normalize: bool = False,
      sharded: bool = False,
  ) -> FloatArray['B N 1']:
    """Computes similarity to cell codes."""
    logits = jnp.expand_dims(logits, axis=-2)
    cell_codes = cell_codes[None]
    if sharded and not self.is_initializing():
      return misc.sharded_similarity(logits, cell_codes, normalize=normalize)[
          ..., None
      ]
    else:
      return misc.compute_similarity(
          logits,
          cell_codes,
          normalize=normalize,
      )

  @nn.compact
  def hierarchical_logits(
      self, logits: FloatArray['... D']
  ) -> Sequence[FloatArray['... Di']]:
    if self.config.get('shared_projections', True):
      # Directly use the input logits as cell code logits.
      hierarchical_logits_f = logits
      if logits.shape[-1] != self.total_cell_code_dims:
        raise ValueError(
            f'logits.shape[-1] ({logits.shape[-1]}) !='
            f' self.total_cell_code_dims ({self.total_cell_code_dims})'
        )
    else:
      # Add a projection layer to the input logits.
      hierarchical_logits_f = nn.Dense(
          self.total_cell_code_dims, name='hierarchical_logits'
      )(logits)
    hierarchical_logits = []
    offset = 0
    for i in range(self.num_levels):
      hierarchical_logits.append(
          hierarchical_logits_f[..., offset : offset + self._cell_code_dims[i]]
      )
      offset += self._cell_code_dims[i]
    return hierarchical_logits

  def get_cell_codes(self) -> Sequence[FloatArray['Ni D']]:
    cc = self.cell_codes
    if self.config.get('l2_normalize_per_head', False):
      raise NotImplementedError('l2_normalize_per_head not supported.')
    elif self.config.get('l2_normalize', False):
      cc = [
          misc.l2_normalize(x)
          # * (x.shape[-1] / self.total_cell_code_dims) ** 0.5
          for x in cc
      ]
    return cc

  def compute_tree_similarity(
      self,
      logits: FloatArray['B D'],
  ) -> Sequence[FloatArray['B Ni|K']]:
    """Computes similarity to cell codes."""
    hierarchical_logits = self.hierarchical_logits(logits)
    cell_codes = self.get_cell_codes()
    tree_similarity_per_level = []
    for i, h_logits in enumerate(hierarchical_logits):
      tree_similarity_per_level.append(
          self.compute_similarity_to_cell_codes(
              h_logits,
              cell_codes[i],
              normalize=False,
              sharded=is_sharded(
                  i, self.num_levels, self.config.get('shard_last_n_layers', 0)
              ),
          )
      )
    return tree_similarity_per_level

  def compute_similarity(
      self,
      logits: FloatArray['B D'],
  ) -> FloatArray['B Ni|K H']:
    """Computes similarity to cell codes."""
    tree_similarity_per_level = self.compute_tree_similarity(logits)
    if self.config.get('cascade', False):
      tree_similarity_per_level = cascade_tree(
          tree_similarity_per_level, self.parent_idxs, xnp=jnp, concat=False
      )
      tree_similarity = tree_similarity_per_level[-1]
    else:
      tree_similarity = concat_tree(
          tree_similarity_per_level, self.idx_to_tree_idx, xnp=jnp, concat=True
      )
    return tree_similarity.sum(-1)

  def __call__(
      self,
      logits: FloatArray['B D'],
  ) -> dict[str, Any]:
    """Computes similarity to cell codes."""
    tree_similarity = self.compute_similarity(logits)
    return {
        'similarity': tree_similarity[..., 0],
    }


class SVAerialHierarchicalCellCodeRetrieval(sv_aerial.SVAerialRetrieval):
  """SV-Aerial hierarchical code model."""

  cell_tree: Sequence[IntArray['N K']] | None = (
      None  # Each entry specifies K leaf idx.
  )

  def setup(self):
    super().setup()
    if self.cell_tree is None:
      raise ValueError('cell_tree is required.')
    self.sv_temperature = jnp.array(
        self.config.get('cc_temperature', self.config.get('temperature', 0.1))
    )
    self.aerial_temperature = jnp.array(
        self.config.get('cc_temperature', self.config.get('temperature', 0.1))
    )
    if self.config.get('learned_temperature', False):
      self.sv_temperature = self.param(
          'sv_temperature',
          nn.initializers.constant(self.sv_temperature),
          (),
          self.dtype,
      )

      self.aerial_temperature = self.param(
          'aerial_temperature',
          nn.initializers.constant(self.aerial_temperature),
          (),
          self.dtype,
      )
    self.cell_tree_codes = HierarchicalCellCodes(
        self.cell_tree,
        self.config,
        dtype=self.dtype,
    )

  def encode_multiview(
      self,
      batch: types.Batch,
      *args,
      **kwargs,
  ) -> tuple[jt.Float[jt.Array, '... D'], jt.Float[jt.Array, '... L D']]:
    logits, tokens = super().encode_multiview(batch, *args, **kwargs)

    if self.config.get('l2_normalize', False):
      logits = l2_normalize_per_level(logits, self.config.cell_code_dims)
    return logits, tokens

  @nn.compact
  def __call__(
      self,
      batch: types.Batch,
      *,
      train: bool,
      debug: bool = False,
  ) -> types.Predictions:
    pred = super().__call__(batch, train=train, debug=debug)
    batch = sv_aerial.flatten_batch(batch)
    if 'latlng' in batch:
      pred['sv_cc_dist'] = misc.haversine_distance(
          self.cell_centers[None],
          batch['latlng'][:, None],
      )[..., 0]
      pred['sv_cc_mask'] = pred['sv_cc_dist'] > self.config.get(
          'cc_dist_threshold', self.config.get('same_cell_threshold', 200.0)
      )
    all_logits = {
        'sv': pred['logits'],
        'aerial': pred['aerial_logits_p'],
    }
    for name, logits in all_logits.items():
      if self.config.get(f'detach_{name}_embeddings', False):
        logits = jax.lax.stop_gradient(logits)

      sim = self.cell_tree_codes.compute_similarity(logits)
      temperature = getattr(self, f'{name}_temperature')
      pred[f'{name}_sim'] = sim
      pred[f'{name}_cell_idx'] = jnp.argmax(sim, axis=-1)
      pred[f'{name}_best_cell_idx'] = pred[f'{name}_cell_idx']
      pred[f'{name}_cell_prob'] = jnp.exp(
          jnp.take_along_axis(
              sim,
              pred[f'{name}_cell_idx'][:, None],
              axis=-1,
          )[:, 0]
      )
      pred[f'{name}_best_cell_prob'] = jnp.exp(
          jnp.take_along_axis(
              sim,
              pred[f'{name}_cell_idx'][:, None],
              axis=-1,
          )[:, 0]
      )
      pred[f'{name}_temperature'] = temperature[None].repeat(
          sim.shape[0], axis=0
      )
    return pred


class SVAerialHierarchicalCellCodeRetrievalModel(
    sv_aerial_code.SVAerialCellCodeRetrievalModel
):
  """SV-Aerial hierarchical code model."""

  def build_flax_model(self) -> nn.Module:
    self.cell_tree = [
        jnp.asarray(x) for x in self.dataset_meta_data['cell_tree']
    ]
    self.s2cell_centers_ll = jnp.asarray(
        self.dataset_meta_data['s2cell_centers_ll'], dtype=self.dtype
    )
    assert not self.config.get('shard_codes', False)
    return SVAerialHierarchicalCellCodeRetrieval(
        self.config,
        self.dtype,
        cell_tree=self.cell_tree,
        cell_centers=self.s2cell_centers_ll,
    )

  def get_cell_codes_at_level(
      self,
      train_state: train_utils.TrainState,
      level: int,
      do_normalize: bool | None = None,
  ) -> FloatArray['N D']:
    if level < 0:
      level = len(self.cell_tree) + level
    cell_codes = train_state.params['cell_tree_codes'][f'cell_codes_{level}']
    if do_normalize or do_normalize is None:
      if self.config.get('l2_normalize_per_head', False):
        raise NotImplementedError('l2_normalize_per_head not supported yet.')
      if do_normalize or self.config.get('l2_normalize', False):
        cell_codes = cell_codes / (cell_codes**2).sum(-1, keepdims=True) ** 0.5
    return cell_codes

  def get_cell_codes(
      self,
      train_state: train_utils.TrainState,
      do_normalize: bool | None = None,
  ) -> FloatArray['N D']:
    tree_height = len(self.cell_tree)
    tree_codes = [
        self.get_cell_codes_at_level(train_state, i, do_normalize=do_normalize)
        for i in range(tree_height)
    ]
    return hierarchical_cell_code.concat_tree(
        tree_codes,
        self.dataset_meta_data['cell_tree'][-1],
        xnp=np,
    )

  def get_target_shapes(
      self, num_levels: int, is_replicated: bool
  ) -> Sequence[tuple[int, ...]]:
    cells_per_level = [x.shape[0] for x in self.cell_tree][-num_levels:]
    if is_replicated:
      cells_per_level = [x // jax.device_count() for x in cells_per_level]
    target_dims = [
        (num_cells, dim)
        for num_cells, dim in zip(
            cells_per_level, self.config.cell_code_dims[-num_levels:]
        )
    ]
    if is_replicated:
      target_dims = [(jax.local_device_count(), *x) for x in target_dims]
    return target_dims

  def unreplicate_state(
      self, train_state: train_utils.TrainState
  ) -> train_utils.TrainState:
    # pytype: disable=attribute-error
    if self.config.get('shard_last_n_layers', 0):
      train_state = misc.unshard_or_unreplicate_train_state(
          train_state,
          misc.unshard_or_unreplicate_from_shape_fn(
              self.get_target_shapes(
                  num_levels=self.config.shard_last_n_layers,
                  is_replicated=True,
              ),
              half_precision=self.config.get('store_half_precision', True),
          ),
      )
      return train_state
    else:
      return super().unreplicate_state(train_state)
    # pytype: enable=attribute-error

  def replicate_state(
      self, train_state: train_utils.TrainState
  ) -> train_utils.TrainState:
    # pytype: disable=attribute-error
    if self.config.get('shard_last_n_layers', 0):
      # We check for the cell codes by shape here. This is a bit hacky.
      train_state = misc.shard_or_replicate_train_state(
          train_state,
          misc.shard_or_replicate_from_shape_fn(
              self.get_target_shapes(
                  num_levels=self.config.shard_last_n_layers,
                  is_replicated=False,
              ),
          ),
      )
      return train_state
    else:
      return super().replicate_state(train_state)
    # pytype: enable=attribute-error

  def aggregate_grads(self, grad: types.Grad) -> types.Grad:
    if self.config.get('shard_last_n_layers', 0) and 'cell_tree_codes' in grad:
      last_level = len(self.cell_tree) - 1
      return jax.tree.map(
          misc.pmean_or_divide_from_shape_fn(
              grad['cell_tree_codes'][f'cell_codes_{last_level}'].shape
          ),
          grad,
      )
    else:
      return jax.lax.pmean(grad, axis_name='batch')

  def build_s2cell_database(
      self,
      train_state: train_utils.TrainState,  # replicated
  ) -> dict[str, Any]:
    unrep_state = self.unreplicate_state(train_state)
    cell_codes = self.get_cell_codes(unrep_state, True)
    s2cell_database = {
        'tokens': cell_codes,
        'ecef_positions': self.dataset_meta_data['s2cell_centers_ecef'],
        'll_positions': self.dataset_meta_data['s2cell_centers_ll'],
        's2cell': np.array(
            [x.encode('utf-8') for x in self.dataset_meta_data['s2cell_tokens']]
        ),
        'dataset_index': np.arange(
            len(self.dataset_meta_data['s2cell_tokens'])
        ),
    }
    return s2cell_database
