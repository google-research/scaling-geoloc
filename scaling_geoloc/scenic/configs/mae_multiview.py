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

"""Multiview MAE config."""

from ml_collections import config_dict

from scaling_geoloc.scenic.configs import defaults


def get_config() -> config_dict.ConfigDict:
  """Return the config."""
  config = defaults.base().unlock()
  config.model_name = 'mae_multiview'
  config.model = defaults.mae_fomo().unlock()
  config.model.align = 'image'
  config.model.stop_gradient_map = True
  config.model.subtract_center_from_map = True
  config.model.patch_temperature = 0.04
  config.model.encoding_dim = config_dict.placeholder(int)
  config.model.compose_loss = 'reconstruction + 1.0 * cross_entropy'
  config.model.lock()

  config.data = defaults.query_map_scene()
  config.data.grain_num_workers = 32
  config.data.preprocess.num_lidar_points = 0
  config.data.preprocess.sample_num_views = {'query': 1, 'map': 1}
  config.data.preprocess.add_grid_3d = True
  config.data.preprocess.voxelize_lidar_projections = True
  config.data.preprocess.num_dominant_voxels_per_patch = 4
  config.checkpoint = True
  config.batch_size = 32
  config.lr = dict(
      learning_rate_schedule='compound',
      factors='constant * linear_warmup * rsqrt_decay * linear_cooldown',
      base_learning_rate=1e-3,
      warmup_steps=10_000,
      cooldown_steps=10_000,
      timescale=10_000,
      total_steps=1_000_000,
  )
  config.num_training_epochs = 40
  config.checkpoint_interval = 2_000
  config.log_summary_interval = 100
  config.log_eval_interval = 1_000
  config.num_eval_steps = 100
  config.num_eval_visualizations = 8

  config.optimizer.optax_name = 'scale_by_adam'
  config.optimizer.optax_configs.b1 = 0.9
  config.optimizer.optax_configs.b2 = 0.95
  config.optimizer.weight_decay = 0.00005
  return config
