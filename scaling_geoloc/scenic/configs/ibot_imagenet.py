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
  """Configuration to train iBOT on imagenet."""
  config = dino_base.get_config(1281167).unlock()
  config.batch_size_eval = 32 * config.get_ref('device_count')
  config.batch_size = 32 * config.get_ref('device_count')
  config.model.vit = defaults.vit('B', 8, 0, 'none').unlock()
  config.model.num_local_views = 8
  config.model.mask_ratio = 0.25
  config.model.mask_prob = 0.5
  config.model.vit.imagenet_normalization = True
  config.model.compose_loss = (
      'cls_all' if config.model.num_local_views > 0 else 'cls_global'
  ) + ' * 0.5 + patch_mask * 0.5'
  config.model.eval_crops = True
  config.model.stack_crops = True
  config.model.do_augment = False
  config.model.all_to_all = False
  config.shard_train_batch = False
  config.filter_train_batch = False
  config.skip_first_eval = True
  config.skip_train_viz = False
  config.data = defaults.dino_imagenet(
      config.model.num_local_views, not config.model.vit.imagenet_normalization
  )

  config.run_test_evals_in_process = True
  config.test_evals = ml_collections.ConfigDict()
  config.test_evals.knn_imagenet = defaults.knn_imagenet(
      config.get_ref('batch_size_eval'),
      not config.model.vit.imagenet_normalization,
  )
  config.test_evals.knn_imagenet.method = 'encode_multiview'
  config.test_evals.knn_imagenet.eval_interval = 25_000

  # For large scale training, we need to set the following.
  # config.lr.warmup_steps = 5_000
  # config.lr.base_learning_rate = 0.002
  return config.lock()
