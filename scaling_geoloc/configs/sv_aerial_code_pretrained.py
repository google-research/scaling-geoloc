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

"""SV-aerial alignment config with mining."""

from typing import Any

import ml_collections

from scaling_geoloc.scenic.configs import sv_aerial_code


def get_config(
    location: str = 'bedenl',
    vit_pretrained: str = 'ibot_b16',
    cell_code_dim: int = 2048 + 128,
    add_evals: bool = True,
) -> ml_collections.ConfigDict:
  """Returns the config."""
  config = sv_aerial_code.get_config(
      location=location,
      vit_pretrained=vit_pretrained,
      cell_code_dim=cell_code_dim,
      add_evals=add_evals,
  )
  config.model_name = 'sv_aerial_code'
  config.lr.base_learning_rate = 1e-4
  config.num_training_steps = 100_000
  config.max_grad_norm = 10.0
  config.data.preprocess.s2cell_level = 15
  config.data.grain_num_workers = 16

  config.pretrained_xm = (164014236, 1)
  config.pretrained_load_meta_data = True
  config.pretrained_load_model_config = True

  # Mining configs.
  config.data.make_s2cell_data_source = False
  config.mining_interval = 100_000
  config.skip_first_mining_steps = 0
  config.log_train_viz_interval = 5_000
  config.skip_first_eval = True
  config.mining = ml_collections.ConfigDict()
  config.mining.batch_size = 64
  config.mining.min_distance_meters = 10_000
  config.lr.factors = 'constant * cosine_decay'
  config.overwrite_lr = {}
  config.data.s2cell_index_file_name = '{split}_grouped_s2cells.csv*'
  return config


def get_hyper(h: Any) -> Any:
  return h.product([
      h.sweep('config.data.make_s2cell_data_source', [True]),
      h.sweep('config.skip_first_mining_steps', [0]),
      h.sweep('config.mining.batch_size', [16, 64]),
  ])
