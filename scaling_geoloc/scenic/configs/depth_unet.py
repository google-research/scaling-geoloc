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

"""DepthUNet config."""

from ml_collections import config_dict

from scaling_geoloc.scenic.configs import defaults


def get_config() -> config_dict.ConfigDict:
  """Return the config."""
  config = defaults.base()
  config.model_name = 'depth_unet'
  config.model = defaults.depth_unet()
  config.data = defaults.query_map_scene()
  config.data.grain_num_workers = 32
  config.data.preprocess.num_lidar_points = 0
  config.data.preprocess.sample_num_views = {'query': 1, 'map': 1}
  config.checkpoint = True
  config.batch_size = 10
  config.lr.base_learning_rate = 1e-3
  config.num_training_epochs = 10
  config.checkpoint_interval = 5_000
  config.log_summary_interval = 500
  config.log_eval_interval = 2_500
  config.num_eval_steps = 1_000
  config.num_eval_visualizations = 8
  return config
