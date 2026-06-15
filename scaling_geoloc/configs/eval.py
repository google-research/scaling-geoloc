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

"""Eval config for a pretrained GeoLocalizer model.

Usage:
    python -m scaling_geoloc.eval \\
        --config=scaling_geoloc.configs.eval \\
        --config.experiment=/path/to/checkpoint \\
        --config.test_evals.retrieval_cross_view.data.location=europe
"""

import ml_collections
from scaling_geoloc.configs import defaults


LOCATION = ml_collections.config_dict.placeholder(str)
BATCH_SIZE_PER_DEVICE = 128


def get_config() -> ml_collections.ConfigDict:
  config = ml_collections.ConfigDict()
  config.rng_seed = 42
  config.device_count = 1  # Automatically set in eval.py.

  # Checkpoint path. The eval script loads model config, weights, and
  # dataset_meta_data from here via trainer.reload_experiment.
  config.experiment = ml_collections.config_dict.placeholder(str)

  def _retrieval(task):
    cfg = defaults.retrieval_eval(LOCATION, BATCH_SIZE_PER_DEVICE).unlock()
    cfg.batch_size_per_device = BATCH_SIZE_PER_DEVICE
    cfg.batch_size = cfg.get_ref('batch_size_per_device') * config.get_ref('device_count')
    cfg.batch_size_eval = cfg.get_ref('batch_size')
    cfg.task = task
    cfg.db_split = 'train'
    return cfg

  config.test_evals = ml_collections.ConfigDict({
      'retrieval_cross_view': _retrieval('cross_view'),
      'retrieval_hybrid': _retrieval('hybrid'),
  })

  config.overwrite_existing_evals = False

  return config
