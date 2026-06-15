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

"""Code for Fusion of modalities."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt
import ml_collections
import numpy as np

from unigeoclip.scenic.models import vit


class FusionBlock(nn.Module):
  """Fuse modalities with self attention with registers."""

  config: ml_collections.ConfigDict
  dtype: jnp.dtype

  @nn.compact
  def __call__(
      self,
      features: jt.Float[jt.Array, '... M D'],
      *,
      train: bool = False,
  ) -> jt.Float[jt.Array, '... D']:

    if self.config.modality_encoding:
      mod_enc = self.param(
          'modality_encoding',
          nn.initializers.normal(stddev=1 / np.sqrt(features.shape[-1])),
          (1, features.shape[1], features.shape[-1]),
          self.dtype,
      )
      features = features + mod_enc

    batch_size, *_, c = features.shape

    if self.config.n_registers > 0:
      registers = self.param(
          'registers',
          nn.initializers.normal(stddev=1 / np.sqrt(c)),
          (1, self.config.n_registers, c),
          self.dtype,
      )
      batch_registers = jnp.repeat(registers, batch_size, axis=0)
      x = jnp.concatenate([batch_registers, features], axis=1)
    else:
      x = features

    for lyr in range(self.config.num_layers):
      x = vit.Encoder1DBlock(
          mlp_dim=self.config.mlp_dim,
          num_heads=self.config.num_heads,
          dropout_rate=self.config.dropout_rate,
          use_rope=False,
          norm_layer=vit._NORM_LAYER[self.config.norm_layer],
          n_registers=self.config.n_registers,
          attention_dropout_rate=self.config.attention_dropout_rate,
          stochastic_depth=(lyr / max(self.config.num_layers - 1, 1))
          * self.config.stochastic_depth,
          use_bias=self.config.use_bias,
          approximate_gelu=self.config.approximate_gelu,
          name=f'fusionblock_{lyr}',
          dtype=jax.dtypes.canonicalize_dtype(self.dtype),
      )(
          x,
          deterministic=not train,
          height=0,
          width=0,
      )
    if self.config.n_registers > 0:
      return x[:, 0, :]
    return x.max(axis=1)
