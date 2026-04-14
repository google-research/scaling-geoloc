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

"""Image retrieval config."""

from typing import Any
from ml_collections import config_dict
from scaling_geoloc.scenic.configs import defaults


def get_config(
    vit_name: str = 'dinov2_salad_b14', aggregator_name: str = 'salad'
) -> config_dict.ConfigDict:
  """Return the config."""
  config = defaults.base().unlock()
  config.model_name = 'image_retrieval'
  config.model = defaults.retrieval_pretrained(
      vit_name, aggregator_name
  ).unlock()

  config.data = defaults.query_map_scene().unlock()

  bs = 128
  config.batch_size = bs * config.get_ref('device_count')
  config.batch_size_eval = bs * config.get_ref('device_count')
  config.batch_size_viz = 2 * config.get_ref('device_count')
  config.lr = dict(
      learning_rate_schedule='compound',
      factors='constant * linear_warmup * rsqrt_decay * linear_cooldown',
      base_learning_rate=1e-3,
      warmup_steps=5_000,
      cooldown_steps=5_000,
      timescale=100_000,
      total_steps=1_000_000,
  )
  config.num_training_epochs = 100
  config.checkpoint_interval = 1_000
  config.log_summary_interval = 200
  config.log_eval_interval = 20_000
  config.checkpoint = True
  config.skip_first_eval = True
  config.skip_eval = False
  config.skip_train_viz = False
  config.log_train_viz_interval = 4000
  config.num_eval_visualizations = 1
  config.num_eval_steps = 16

  config.optimizer.optax_name = 'scale_by_adam'
  config.optimizer.optax_configs.b1 = 0.9
  config.optimizer.optax_configs.b2 = 0.95
  config.optimizer.weight_decay = 0.00005
  return config


def get_hyper(h: Any) -> Any:
  return h.product([])
