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

"""Train config for GeoLocalizerModel.

Usage:
    python -m scaling_geoloc.train \\
        --config=scaling_geoloc.configs.train \\
        --workdir=/path/to/workdir \\
        --config.data.location=europe
"""

import ml_collections
from scaling_geoloc.configs import defaults


LOCATION = ml_collections.config_dict.placeholder(str)
BATCH_SIZE_PER_DEVICE = 64
NUM_STEPS = 200_000
CELL_CODE_DIM = 2048 + 128


def get_config() -> ml_collections.ConfigDict:
  config = defaults.base().unlock()

  config.model_name = 'hybrid_geolocalization'
  config.rng_seed = 0

  # ---- data ----------------------------------------------------------------
  config.data = defaults.data(LOCATION).unlock()
  config.data.preprocess.size = (224, 224)
  config.data.preprocess.method = 'crop'

  # ---- training schedule ---------------------------------------------------
  config.batch_size_per_device = BATCH_SIZE_PER_DEVICE
  config.batch_size = config.get_ref('batch_size_per_device') * config.get_ref('device_count')
  config.num_training_steps = NUM_STEPS
  config.num_eval_steps = 50
  config.checkpoint_interval = 10_000
  config.log_eval_interval = 20_000
  config.log_summary_interval = 100

  config.lr.base_learning_rate = 3e-4
  config.lr.factors = 'constant * linear_warmup * cosine_decay'
  config.lr.warmup_steps = 5_000
  config.lr.start_decay_step = 0
  config.lr.steps_per_cycle = NUM_STEPS
  config.lr.alpha = 1e-6 / 3e-4  # min lr as fraction of base

  config.optimizer.weight_decay = 0.0
  config.max_grad_norm = 10.0

  # ---- model ---------------------------------------------------------------
  config.model = defaults.geo_localizer().unlock()

  config.model.vit = defaults.pretrained_backbone(
      'dinov2_b14',
      head_output_dim=CELL_CODE_DIM,
      bottleneck_dim=None,
  ).unlock()
  config.model.vit.train_last_layers = -1  # train all layers

  # Aerial backbone — patch on top of vit; only fields that differ are needed.
  config.model.vit_aerial = ml_collections.ConfigDict()

  config.model.prototype_dim = CELL_CODE_DIM
  config.model.temperature = 0.01
  config.model.l2_normalize = True
  config.model.store_half_precision = True

  config.model.do_augment = True
  config.model.same_cell_threshold = 200.0

  config.model.multi_similarity_alpha = 2.0
  config.model.multi_similarity_beta = 100.0
  config.model.multi_similarity_base = 0.2
  config.model.multi_similarity_margin = 0.0
  config.model.multi_similarity_factors = 'all'

  # ---- in-training evals ---------------------------------------------------
  config.test_evals = ml_collections.ConfigDict({
      'retrieval_classification': _retrieval('classification'),
  })

  return config


def _retrieval(task: str) -> ml_collections.ConfigDict:
  cfg = defaults.retrieval_eval(LOCATION, BATCH_SIZE_PER_DEVICE).unlock()
  cfg.task = task
  cfg.db_split = 'train'
  return cfg
