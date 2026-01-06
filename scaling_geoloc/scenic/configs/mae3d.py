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

"""3D MAE config."""

from ml_collections import config_dict

from scaling_geoloc.scenic.configs import defaults


def get_config() -> config_dict.ConfigDict:
  """Return the config."""
  config = defaults.base().unlock()
  config.model_name = 'mae3d'
  config.model = config_dict.ConfigDict()
  config.model.encoder = defaults.vit(
      'S', 16, output_dim=0, pooling='none'
  ).unlock()
  config.model.decoder = dict(
      mlp_dim=config.model.encoder.mlp_dim,
      num_layers=8,
      num_heads=config.model.encoder.num_heads,
      dropout_rate=config.model.encoder.dropout_rate,
      attention_dropout_rate=config.model.encoder.attention_dropout_rate,
      stochastic_droplayer_rate=config.model.encoder.stochastic_depth,
      dtype='float32',
  )
  config.model.masking_rate = 0.75

  config.data = defaults.query_map_scene().unlock()
  config.data.dirname = 'side_1q_2m_easy_stripped_depth'
  config.data.grain_num_workers = 24
  config.data.preprocess.num_lidar_points = 0
  config.data.preprocess.sample_num_views = {'query': 1, 'map': 2}
  config.data.preprocess.add_grid_3d = False
  config.data.preprocess.voxelize_lidar_projections = False
  config.data.preprocess.num_dominant_voxels_per_patch = 0
  config.data.preprocess.filter_keys = dict(keep=None)
  config.checkpoint = True
  config.skip_eval = True

  config.batch_size = 32 * config.get_ref('device_count')
  config.lr = dict(
      learning_rate_schedule='compound',
      factors='constant * linear_warmup * rsqrt_decay * linear_cooldown',
      base_learning_rate=1e-3,
      warmup_steps=10_000,
      cooldown_steps=10_000,
      timescale=10_000,
      total_steps=1_000_000,
  )
  config.num_training_epochs = 100
  config.checkpoint_interval = 5_000
  config.log_summary_interval = 100
  config.log_eval_interval = 10_000
  config.num_eval_steps = 4
  config.num_eval_visualizations = 4

  config.optimizer.optax_name = 'scale_by_adam'
  config.optimizer.optax_configs.b1 = 0.9
  config.optimizer.optax_configs.b2 = 0.95
  config.optimizer.weight_decay = 0.00005
  return config
