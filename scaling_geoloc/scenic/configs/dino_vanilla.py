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

"""Default config for DINO ViT vanilla training on ImageNet2012."""

# pylint: disable=line-too-long
import ml_collections
from scaling_geoloc.scenic.configs import defaults


# Our dataset is roughly twice the size: 'num_train_examples': 2621527
_IMAGENET_TRAIN_SIZE = 1281167
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]
VARIANT = 'S/16'


# pylint: enable=line-too-long


def get_config(runlocal: bool = False) -> ml_collections.ConfigDict:
  """Returns the ViT experiment configuration for ImageNet."""

  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'gsv-dino_vit'
  # Dataset.
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.num_local_views = 8
  config.data = defaults.dino_imagenet(config.dataset_configs.num_local_views)

  config.overfit = False

  # Model.
  version, patch = VARIANT.split('/')
  config.model = ml_collections.ConfigDict()
  config.model_name = 'dino_baseline'
  config.model = defaults.vit(
      version, int(patch), output_dim=65536, pooling='tok'
  ).unlock()
  config.model.imagenet_normalization = True
  config.model_dtype_str = 'float32'
  config.model.temperature = 0.1
  config.model.center_momentum = 0.9
  config.model.pos_interpolation_method = 'bicubic'

  # Training.
  config.trainer_name = 'dino_trainer'
  config.max_grad_norm = 3
  config.num_training_epochs = 100
  config.batch_size = 512
  config.batch_size_eval = None
  steps_per_epoch = _IMAGENET_TRAIN_SIZE // config.batch_size + 1
  config.rng_seed = 42
  config.freeze_prototypes_steps = steps_per_epoch
  total_steps = config.num_training_epochs * steps_per_epoch

  # Learning rate.
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = steps_per_epoch * 10
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = 0.001
  config.lr_configs.alpha = 0.000001 / config.lr_configs.base_learning_rate

  # Weight decay.
  config.weight_decay = ml_collections.ConfigDict()
  config.weight_decay.factors = 'constant*cosine_decay'
  config.weight_decay.steps_per_cycle = total_steps
  config.weight_decay.base_learning_rate = 0.04
  config.weight_decay.alpha = 10

  # Teacher temperature.
  config.teacher_temperature = ml_collections.ConfigDict()
  config.teacher_temperature.factors = 'constant'
  config.teacher_temperature.steps_per_cycle = total_steps
  config.teacher_temperature.base_learning_rate = 0.04

  # Momentum rate scheduler.
  config.momentum_rate = ml_collections.ConfigDict()
  config.momentum_rate.factors = 'constant*cosine_decay'
  config.momentum_rate.steps_per_cycle = total_steps
  config.momentum_rate.base_learning_rate = 0.996
  config.momentum_rate.alpha = 1.0 / 0.996

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 5000
  config.log_summary_steps = 256

  config.knn = defaults.knn_imagenet(512)
  config.knn.eval_interval = 10_000
  config.knn.method = ''

  config.m = None  # Placeholder for randaug strength.
  config.l = None  # Placeholder for randaug layers.

  if runlocal:
    # Current implementation fails with UPTC.
    config.count_flops = False

  return config
