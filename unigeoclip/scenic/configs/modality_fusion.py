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

"""Config file for a modality fusion module."""

import ml_collections

_SIZES = {'Ti': 192, 'S': 384, 'B': 768, 'L': 1024, 'So400m': 1152, 'H': 1280}
_NUM_HEADS = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16, 'So400m': 16, 'H': 16}


def get_config(
    model_size: str = 'B',
) -> ml_collections.ConfigDict:
  """Returns the config of the modality fusion module."""
  config = ml_collections.ConfigDict()

  config.mlp_dim = _SIZES[model_size]
  config.num_heads = _NUM_HEADS[model_size]

  config.num_layers = 6
  config.dropout_rate = 0.1
  config.attention_dropout_rate = 0.1
  config.stochastic_depth = 0.1
  config.use_bias = True
  config.approximate_gelu = True
  config.modality_encoding = False

  config.norm_layer = 'layernorm'
  config.n_registers = 4
  config.name = 'modality_fusion'

  return config
