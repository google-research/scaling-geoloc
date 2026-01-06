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

"""DINO config."""

from ml_collections import config_dict

from scaling_geoloc.scenic.configs import defaults

_DATASET_SIZE = 1281167


def get_config(dataset_size: int = _DATASET_SIZE) -> config_dict.ConfigDict:
  """Return the dino config."""
  config = defaults.base().unlock()
  config.model_name = 'dino'
  config.model = config_dict.ConfigDict()
  config.model.vit = defaults.vit('S', 16, 65536, 'tok')
  config.model.vit.imagenet_normalization = False
  config.model.vit.dropout_rate = 0.0
  config.model.vit.attention_dropout_rate = 0.0
  config.model.compose_loss = 'cls_global'
  config.model.eval_crops = True
  config.model.stack_crops = True
  config.model.num_local_views = 0
  config.data = defaults.dino_imagenet(config.model.num_local_views)
  config.checkpoint = True
  config.batch_size = 512
  config.batch_size_eval = 512

  config.shard_train_batch = False
  config.filter_train_batch = False
  config.skip_first_eval = True

  config.num_training_epochs = 100

  steps_per_epoch = dataset_size // config.get_ref('batch_size')
  total_steps = config.num_training_epochs * steps_per_epoch

  config.checkpoint_interval = 5_000
  config.log_summary_interval = 100
  config.log_eval_interval = 5_000
  config.num_eval_steps = 1
  config.num_eval_visualizations = 1
  config.init_train = False

  config.lr = config_dict.ConfigDict()
  config.lr.learning_rate_schedule = 'compound'
  config.lr.factors = 'constant * linear_warmup * cosine_decay'
  config.lr.base_learning_rate = min(0.001 * config.batch_size / 512, 0.004)
  config.lr.warmup_steps = steps_per_epoch * 10
  config.lr.steps_per_cycle = total_steps
  config.lr.total_steps = total_steps
  config.lr.alpha = 0.000001 / config.lr.base_learning_rate

  config.max_grad_norm = None

  config.optimizer.optax_name = 'adamw_old'
  config.weight_decay = config_dict.ConfigDict()
  config.weight_decay.factors = 'constant*cosine_decay'
  config.weight_decay.base_learning_rate = 0.04
  config.weight_decay.steps_per_cycle = total_steps
  config.weight_decay.alpha = 10
  config.weight_decay.reg_wn_scale = True

  config.rate_fns = dict(
      teacher_temperature=dict(
          factors='constant',
          base_learning_rate=0.04,
          steps_per_cycle=total_steps,
      ),
      student_temperature=dict(
          factors='constant',
          base_learning_rate=0.1,
      ),
      params_momentum=dict(
          factors='constant * cosine_decay',
          base_learning_rate=0.996,
          alpha=1.0 / 0.996,
          steps_per_cycle=total_steps,
      ),
      center_momentum=dict(
          factors='constant',
          base_learning_rate=0.9,
          steps_per_cycle=total_steps,
      ),
      prototype_grad_scale=dict(
          factors='piecewise_constant',
          decay_events=[-1, steps_per_epoch.get()],
          decay_factors=[0.0, 1.0],
      ),
  )

  config.run_test_evals_in_process = True
  config.test_evals = config_dict.ConfigDict()
  config.test_evals.knn_imagenet = defaults.knn_imagenet(config.batch_size_eval)
  config.test_evals.knn_imagenet.method = 'encode_multiview'
  config.test_evals.knn_imagenet.eval_interval = 50_000
  return config.lock()
