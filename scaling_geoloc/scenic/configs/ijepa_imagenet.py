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

"""Configuration file for JEPA pretraining on ImageNet."""

import ml_collections
from scaling_geoloc.scenic.configs import defaults
from scaling_geoloc.scenic.configs import ijepa


def get_config() -> ml_collections.ConfigDict:
  """Configuration to train JEPA on imagenet."""
  config = ijepa.get_config(3 * 1281167).unlock()
  config.batch_size_eval = 4 * config.get_ref('device_count')
  config.batch_size = 32 * config.get_ref('device_count')
  config.model.num_local_views = 0
  config.model.vit.imagenet_normalization = True
  config.model.compose_loss = 'l1'
  config.model.do_augment = False
  config.shard_train_batch = True
  config.filter_train_batch = True
  config.skip_eval = True
  config.skip_first_eval = True
  config.skip_train_viz = True
  config.data.locations.train = ['imagenet2012']
  config.data.locations.valid = ['imagenet2012']
  augment_fn = (
      'inception_crop(224, 40, data_key="image_pixels")'
      '|flip_lr(data_key="image_pixels")'
  )
  config.data.preprocess.dino_crop = dict(
      augment=augment_fn,
      photo_augment=False,
  )
  config.data.preprocess.sample_num_views = {'query': 1}
  config.data.preprocess.jepa_mask = dict(
      patch_size=config.model.vit.patches.size[0],
  )

  config.run_test_evals_in_process = True
  config.test_evals = ml_collections.ConfigDict()
  config.test_evals.knn_imagenet = defaults.knn_imagenet(
      config.get_ref('batch_size_eval'),
      not config.model.vit.imagenet_normalization,
  )
  config.test_evals.knn_imagenet.method = 'encode_multiview'
  config.test_evals.knn_imagenet.eval_interval = 10_000

  # For large scale training, we need to set the following.
  return config.lock()
