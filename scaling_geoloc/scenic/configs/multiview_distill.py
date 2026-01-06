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

"""Configuration file for iBOT pretraining."""

import ml_collections
from scaling_geoloc.scenic.configs import defaults
from scaling_geoloc.scenic.configs import dino_base


def get_config() -> ml_collections.ConfigDict:
  """Multiview distillation object."""
  config = dino_base.get_config().unlock()
  config.model_name = 'multiview_distill'
  config.num_eval_steps = 4
  config.batch_size_eval = config.get_ref('device_count')
  config.batch_size = 16 * config.get_ref('device_count')
  config.model.mask_ratio = -1
  config.model.vit = defaults.pretrained_backbone('dinov1_s8', 0)
  config.model.vit.dropout_rate = 0.0
  config.model.vit.attention_dropout_rate = 0.0
  config.model.vit.stochastic_depth = 0.0
  config.model.vit.freeze_backbone = False
  config.model.vit.pooling = 'none'
  config.model.align = 'gaussians'
  config.model.align_to = 'map+query'
  config.model.compose_loss = 'mse_l2'
  config.data.dirname = 'side_1q_2m_easy_stripped_depth'
  config.data.locations.train = ['from_snap_train']
  config.data.locations.valid = ['from_snap_val']
  config.data.preprocess.sample_num_views = {'query': 1, 'map': 2}
  config.data.preprocess.map_to_map = False
  config.model.clip_self_weight = 0.05
  config.data.preprocess.add_grid_3d = True
  config.data.preprocess.filter_keys = dict(keep=None)
  config.data.preprocess.voxelize_lidar_projections = False
  config.data.preprocess.filter_pixels_with_semantics = True
  config.data.preprocess.num_dominant_voxels_per_patch = 0
  config.data.preprocess.dino_crop = None
  config.rate_fns.params_momentum.base_learning_rate = 1.0  # fix teacher
  config.rate_fns.params_momentum.alpha = (
      1.0 / config.rate_fns.params_momentum.get_ref('base_learning_rate')
  )
  config.skip_eval = False
  return config
