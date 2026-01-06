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

"""Configuration file for iBOT pretraining on SV."""

import ml_collections
from scaling_geoloc.scenic.configs import defaults
from scaling_geoloc.scenic.configs import dino_base


def get_config() -> ml_collections.ConfigDict:
  config = dino_base.get_config().unlock()
  config.batch_size = 32 * config.get_ref('device_count')
  config.batch_size_eval = 2 * config.get_ref('device_count')
  config.model.mask_ratio = 0.3
  config.model.mask_prob = 0.5
  config.model.vit = defaults.vit('S', 16, 8192, 'none').unlock()
  config.model.vit.imagenet_normalization = True
  config.model.compose_loss = '0.5 * cls_all + patch_mask'
  return config.lock()
