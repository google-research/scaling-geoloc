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

"""Configuration file for iBOT training on SV scenes."""

from typing import Any
import ml_collections
from scaling_geoloc.scenic.configs import defaults
from scaling_geoloc.scenic.configs import dino_base


def get_config() -> ml_collections.ConfigDict:
  """Return base scenebot config."""
  config = dino_base.get_config().unlock()
  config.model_name = 'scenebot'
  config.batch_size_eval = config.get_ref('device_count')
  config.batch_size = 32 * config.get_ref('device_count')
  config.model.mask_ratio = 0.0
  config.model.vit = defaults.vit('S', 16, 8192, 'none')
  config.model.do_augment = True
  config.model.compose_loss = 'cls_global'
  config.data.locations.train = ['from_snap_train']
  config.data.locations.valid = ['from_snap_val']
  config.data.dirname = 'side_1q_2m_easy_stripped_depth'
  config.model.align_to = 'map+query'
  config.model.clip_self_weight = 0.05
  config.data.preprocess.sample_num_views = {'query': 1, 'map': 1}
  config.data.preprocess.filter_pixels_with_semantics = True
  config.data.preprocess.num_dominant_voxels_per_patch = 0
  config.data.preprocess.dino_crop = None
  config.skip_train_viz = False
  return config.lock()


def get_hyper(h: Any) -> Any:
  return h.product([
      h.sweep(
          'config.model.compose_loss',
          [
              'cls_global',
              'cls_global + student_patch_cross_mse_l2',
              'cls_global + self_similarity_nll + student_patch_cross_mse_l2',
              # 'cls_global + st_similarity_nll + patch_cross',
          ],
      ),
  ])
