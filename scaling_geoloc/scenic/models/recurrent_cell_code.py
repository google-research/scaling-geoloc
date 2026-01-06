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

"""Recurrent cell code retrieval model."""

from collections.abc import Sequence
from typing import Any

from etils.array_types import FloatArray
from etils.array_types import IntArray
import flax.linen as nn
import jax.numpy as jnp
from scenic.model_lib.layers import attention_layers

from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.models import cell_code
from scaling_geoloc.scenic.models import hierarchical_cell_code
from scaling_geoloc.scenic.models import vit


class RecurrentDecoder(vit.Decoder):
  """Recurrent decoder."""

  max_seqlen: int = 100

  def setup(self):
    super().setup()
    self.posemb = attention_layers.Add1DPositionEmbedding(
        name='decoder_pos_emb'
    )

  @nn.compact
  def __call__(
      self,
      inputs: FloatArray['B L D'],
      *,
      train: bool,
      mask: jnp.ndarray | None = None,
  ) -> FloatArray['B L D']:
    """Applies the decoder to the inputs."""
    pos_emb_shape = (1, self.max_seqlen, inputs.shape[-1])
    # Use a fixed (non-learned) sinusoidal position embedding.
    pos_embedding = attention_layers.sinusoidal_init(max_len=self.max_seqlen)(
        None,  # pytype: disable=wrong-arg-types  # jax-ndarray
        pos_emb_shape,
        None,
    )
    x = inputs + pos_embedding[:, : inputs.shape[1], :]
    if train and mask is None:
      mask = nn.make_causal_mask(inputs[..., 0])
    x = super().__call__(x, train=train, mask=mask)
    return x

  def decode(
      self, inputs: FloatArray['B D'], seqlen: int | None = None
  ) -> FloatArray['B L D']:
    """Applies the decoder to the inputs."""
    x = inputs[:, None]
    seqlen = seqlen or self.max_seqlen
    for _ in range(seqlen):
      y = self(x, train=True)
      x = jnp.concatenate([x, y[:, -1:]], axis=-2)
    return x[:, 1:]


class RecurrentCellCodeRetrieval(cell_code.CellCodeRetrieval):
  """Recurrent cell code retrieval model."""

  cell_tree: Sequence[IntArray['N K']]  # Each entry specifies K leaf idx.

  def setup(self):
    super().setup()
    self.cell_tree_codes = hierarchical_cell_code.CellTreeCodes(
        self.cell_tree,
        cell_code_dim=self.config.cell_code_dim,
        temperature=self.config.get('temperature', 0.1),
        global_contrastive=self.config.get('global_contrastive', False),
        max_num_children_inference=self.config.get(
            'max_num_children_inference', 16
        ),
        dtype=self.dtype,
        pooling=self.config.get('pooling', None),
        shared_projections=self.config.get('shared_projections', False),
        cumprod=self.config.get('cumprod', False),
    )

    self.recurrent_decoder = RecurrentDecoder(
        num_layers=self.config.decoder.num_layers,
        mlp_dim=self.config.cell_code_dim,
        num_heads=self.config.decoder.num_heads,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        stochastic_droplayer_rate=0.0,
        dtype=self.dtype,
        max_seqlen=len(self.cell_tree),
    )

  def __call__(
      self, batch: types.Batch, *, train: bool, debug: bool = False
  ) -> types.Predictions:
    image_pixels = self.get_image_pixels(batch, train)
    batch = cell_code.flatten_batch(batch, stride=None)
    logits, tokens = self.encode_multiview(
        {'image_pixels': image_pixels},
        train=train,
        sow_attention=self.config.get('sow_attention', False) and not train,
    )
    if logits.ndim == 3:
      logits, _ = logits[:, 0], logits[:, 1:]
    hierarchical_logits = self.recurrent_decoder.decode(logits)
    hierarchical_logits = jnp.swapaxes(hierarchical_logits, 0, 1)
    cell_pred = self.cell_tree_codes(hierarchical_logits)
    pred = {
        'logits': logits,
        'tokens': tokens,
        'augmented_pixels': image_pixels,
        **cell_pred,
    }

    if 'cell_idx' in batch:
      # GT is available, so compute loss.
      # decode with causal mask.
      gt_tree_idx = self.cell_tree[-1][batch['cell_idx']]
      gt_cell_codes = jnp.stack(
          [
              self.cell_tree_codes.cell_codes[i][gt_tree_idx[:, i]]
              for i in range(len(self.cell_tree) - 1)
          ],
          axis=-2,
      )
      sequence = jnp.concatenate([logits[:, None], gt_cell_codes], axis=-2)
      hierarchical_logits = self.recurrent_decoder(sequence, train=True)
      pred['old_logits'] = pred['hierarchical_logits']
      pred['hierarchical_logits'] = jnp.swapaxes(hierarchical_logits, 0, 1)
      losses, metrics = self.cell_tree_codes.loss_metrics_function(batch, pred)
      pred = {
          'losses': losses,
          'metrics': metrics,
          **pred,
      }
    return pred


class RecurrentCellCodeRetrievalModel(cell_code.CellCodeRetrievalModel):
  """Trainer-facing wrapper for the cell code retrieval model."""

  def build_flax_model(self) -> nn.Module:
    self.s2cell_ids = self.dataset_meta_data.get(
        's2cell_ids', self.config.s2cell_ids
    )
    self.s2cell_centers = jnp.asarray(
        self.dataset_meta_data['s2cell_centers_ecef'], dtype=self.dtype
    )
    self.cell_tree = self.dataset_meta_data['cell_tree']
    self.s2cell_ids_tree = self.dataset_meta_data['s2cell_ids_tree']
    return RecurrentCellCodeRetrieval(
        cell_tree=[jnp.asarray(x) for x in self.cell_tree],
        num_cells=len(self.s2cell_ids),
        config=self.config,
        dtype=self.dtype,
    )

  def init_state(
      self, model_params: types.ModelParams, model_state: types.ModelState
  ) -> tuple[types.ModelParams, types.ModelState]:
    model_params, model_state = super().init_state(model_params, model_state)
    return model_params, model_state

  def loss_metrics_function(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      model_params: jnp.ndarray | None = None,
      model_state: Any | None = None,
      learning_rates: types.LearningRates | None = None,
  ) -> types.LossMetricsTuple:
    losses, metrics = pred['losses'], pred['metrics']
    losses['total'] = types.compose_loss(losses, self.config.compose_loss)
    metrics = {
        **metrics,
        **self.distance_metrics(pred['best_cell_idx'], batch['cell_idx']),
    }
    return losses, metrics
