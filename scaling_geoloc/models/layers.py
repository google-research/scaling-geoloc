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

"""Frequently used layers."""

import functools
from etils.array_types import FloatArray
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections


class MLP(nn.Module):
  """A simple MLP."""

  config: ml_collections.ConfigDict
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      input_feats: FloatArray['... N'],
      train: bool = False,
  ) -> FloatArray['... D']:
    dense_layer = functools.partial(
        nn.Dense,
        kernel_init=jax.nn.initializers.glorot_uniform(),
        param_dtype=self.dtype,
    )
    activation = getattr(nn, self.config.activation)
    x = input_feats
    for i, d in enumerate(self.config.layers):
      if i > 0 or self.config.apply_input_activation:
        x = activation(x)
      x = dense_layer(d)(x)
    return x
