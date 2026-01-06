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

"""GeoCLIP config."""

from typing import Any

from ml_collections import config_dict

from scaling_geoloc.scenic.configs import sv_aerial


def get_config(
    location: str = 'bedenl',
    vit_pretrained: str = 'ibot_b16',
    cell_code_dim: int = 2048 + 128,
    add_evals: bool = True,
    link_cell_code_dim: bool = True,
) -> config_dict.ConfigDict:
  """Returns the config."""
  config = sv_aerial.get_config(
      location=location,
      vit_pretrained=vit_pretrained,
      cell_code_dim=cell_code_dim,
      add_evals=add_evals,
      link_cell_code_dim=link_cell_code_dim,
  )
  config.model_name = 'geoclip'
  config.model.compose_loss = 'sv_loc_nll'
  config.model.vit.head = 'geoclip'
  config.model.l2_normalize = False
  config.model.temperature = 0.07
  config.model.learned_temperature = True

  config.model.hidden_dim = 1024
  config.model.num_hidden_layers = 3
  config.model.sigmas = (2**4, 2**8, 2**12)
  config.model.learned_bias = True

  config.test_evals = config_dict.ConfigDict()
  config.eval_interval = 5_000
  return config


def get_hyper(h: Any) -> Any:
  return h.product([
      h.sweep('config.lr.base_learning_rate', [3e-4, 1e-2, 3e-3]),
  ])
