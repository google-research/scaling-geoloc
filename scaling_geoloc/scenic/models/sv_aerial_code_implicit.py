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

"""Code-based implicit model for Street View and Aerial data."""

from etils.array_types import FloatArray
import flax.linen as nn
import jax
import jax.numpy as jnp
from scenic.train_lib import train_utils

from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.models import sv_aerial
from scaling_geoloc.scenic.models import sv_aerial_code
from scaling_geoloc.utils import misc


class SVAerialCodeImplicit(sv_aerial_code.SVAerialCellCodeRetrieval):
  """Code-based implicit model for Street View and Aerial data."""

  def __call__(
      self,
      batch: types.Batch,
      *,
      train: bool,
      debug: bool = False,
  ) -> types.Predictions:
    pred = super().sv_aerial(batch, train=train, debug=debug)
    return pred


class SVAerialCodeImplicitModel(sv_aerial.SVAerialRetrievalModel):
  """SV-Aerial code-based implicit model."""

  required_rngs: tuple[str, ...] = (
      'masking_rng',
      'dropout',
      'augment',
      'augment_aerial',
      'droptok',
      'crop',
      'implicit_cell_codes_dropout',
      'rope',
  )

  def build_flax_model(self) -> nn.Module:
    self.s2cell_centers_ll = jnp.asarray(
        self.dataset_meta_data['s2cell_centers_ll'], dtype=self.dtype
    )

    return SVAerialCodeImplicit(
        self.config,
        self.dtype,
        num_cells=self.s2cell_centers_ll.shape[0],
        cell_centers=self.s2cell_centers_ll,
    )

  def get_cell_codes(
      self,
      train_state: train_utils.TrainState,
      do_normalize: bool | None = None,
  ) -> FloatArray['N D']:
    del do_normalize  # Unused.
    cell_codes = train_state.params['cell_codes_implicit']
    return cell_codes

  def unreplicate_state(
      self, train_state: train_utils.TrainState
  ) -> train_utils.TrainState:
    # pytype: disable=attribute-error
    if self.config.implicit_cell_code_dim > 0 and self.config.get(
        'shard_codes', False
    ):
      train_state = misc.unshard_or_unreplicate_train_state(
          train_state,
          misc.unshard_or_unreplicate_from_shape_fn(
              self.get_cell_codes(train_state, False).shape
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
    if self.config.implicit_cell_code_dim > 0 and self.config.get(
        'shard_codes', False
    ):
      # We check for the cell codes by shape here. This is a bit hacky.
      train_state = misc.shard_or_replicate_train_state(
          train_state,
          misc.shard_or_replicate_from_shape_fn(
              self.get_cell_codes(train_state, False).shape
          ),
      )
      return train_state
    else:
      return super().replicate_state(train_state)
    # pytype: enable=attribute-error

  def aggregate_grads(self, grad: types.Grad) -> types.Grad:
    if (
        self.config.cell_code_dim > 0
        and self.config.get('shard_codes', False)
        and 'cell_codes_implicit' in grad
    ):
      return jax.tree.map(
          misc.pmean_or_divide_from_shape_fn(grad['cell_codes_implicit'].shape),
          grad,
      )
    else:
      return jax.lax.pmean(grad, axis_name='batch')
