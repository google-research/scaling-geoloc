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

from typing import Any

from ml_collections import config_dict

from scaling_geoloc.scenic.configs import defaults

_DATASET_SIZE = 1281167  # ImageNet1K


def get_config(dataset_size: int = _DATASET_SIZE) -> config_dict.ConfigDict:
  """Return the dino config."""
  config = defaults.base().unlock()
  config.model_name = 'dino3d'
  config.model = config_dict.ConfigDict()
  config.model.vit = defaults.pretrained_backbone('dinov1_s16', 0)
  config.model.vit.pooling = 'none'
  config.model.vit.dropout_rate = 0.0
  config.model.vit.attention_dropout_rate = 0.0
  config.model.vit.stochastic_depth = 0.0
  config.model.vit.imagenet_normalization = True
  config.model.compose_loss = 'pairwise_matching'
  vits = defaults.vit('S', 16, 0, 'tok')
  config.model.decode = False
  config.model.decoder = dict(
      mlp_dim=vits.mlp_dim,
      num_layers=2,
      num_heads=vits.num_heads,
      dropout_rate=vits.dropout_rate,
      attention_dropout_rate=vits.attention_dropout_rate,
      stochastic_droplayer_rate=config.model.vit.stochastic_depth,
      dtype='float32',
  )
  config.model.mask_ratio = 0.0
  config.model.do_augment = False
  config.model.num_negative_samples = 100

  config.data = defaults.query_map_scene().unlock()
  config.data.dirname = 'side_1q_2m_easy_stripped_depth'
  config.data.grain_num_workers = 32
  config.data.valid_num_workers = 0
  config.data.preprocess.num_lidar_points = 0
  config.data.preprocess.sample_num_views = {'query': 1, 'map': 1}
  config.data.preprocess.add_grid_3d = False
  config.data.preprocess.voxelize_lidar_projections = False
  config.data.preprocess.filter_pixels_with_semantics = True
  config.data.preprocess.grid_3d = dict(
      extent_meters=(30, 30, 10),
      cell_size=0.25,
      z_offset=3.0,
      center_around_reference=False,
  )
  config.data.preprocess.num_dominant_voxels_per_patch = 0
  config.data.preprocess.patch_size = config.model.vit.patches.size[0]
  config.data.preprocess.filter_keys = dict(keep=None)
  config.checkpoint = True
  config.batch_size = 32 * config.get_ref('device_count')
  config.batch_size_eval = 8 * config.get_ref('device_count')
  num_training_epochs = 100
  steps_per_epoch = dataset_size // config.get_ref('batch_size')
  total_steps = num_training_epochs * steps_per_epoch
  config.num_training_steps = total_steps
  config.checkpoint_interval = 5_000
  config.log_summary_interval = 100
  config.log_eval_interval = 5_000
  config.num_eval_steps = 1
  config.num_eval_visualizations = 1

  config.lr = config_dict.ConfigDict()
  config.lr.learning_rate_schedule = 'compound'
  config.lr.factors = 'constant * linear_warmup * cosine_decay'
  config.lr.base_learning_rate = min(
      0.001 * config.get_ref('batch_size').to_float() / 512, 0.001
  )
  config.lr.warmup_steps = steps_per_epoch * 1
  config.lr.steps_per_cycle = total_steps
  config.lr.total_steps = total_steps
  config.lr.alpha = 0.000001 / config.lr.get('base_learning_rate')

  config.max_grad_norm = 3.0

  config.optimizer.optax_name = 'adamw_old'
  config.weight_decay = config_dict.ConfigDict()
  config.weight_decay.factors = 'constant*cosine_decay'
  config.weight_decay.base_learning_rate = 0.04
  config.weight_decay.steps_per_cycle = total_steps
  config.weight_decay.alpha = 10
  config.weight_decay.reg_wn_scale = True
  config.skip_first_eval = True
  config.skip_eval = True
  config.skip_train_viz = False

  momentum = 1.0
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
          base_learning_rate=momentum,
          alpha=1.0 / momentum,
          steps_per_cycle=total_steps,
      ),
      center_momentum=dict(
          factors='constant',
          base_learning_rate=1.0,
          steps_per_cycle=total_steps,
      ),
      prototype_grad_scale=dict(
          factors='piecewise_constant',
          decay_events=[-1, 2500],
          decay_factors=[0.0, 1.0],
      ),
  )

  config.run_test_evals_in_process = True
  config.test_evals = config_dict.ConfigDict()
  config.test_evals.knn_imagenet = defaults.knn_imagenet(
      config.get_ref('batch_size_eval'),
      False,
  )
  config.test_evals.knn_imagenet.method = 'encode_multiview'
  config.test_evals.knn_imagenet.eval_interval = 25_000
  config.test_evals.correspondences = defaults.correspondences(
      2 * config.get_ref('device_count'),
  )
  config.test_evals.correspondences.method = 'encode_multiview'
  config.test_evals.correspondences.eval_interval = 5_000
  config.test_evals.class_cooccurrence = defaults.class_cooccurrence(
      2 * config.get_ref('device_count'),
  )
  config.test_evals.class_cooccurrence.method = 'encode_multiview'
  config.test_evals.class_cooccurrence.eval_interval = 5_000
  return config.lock()


def get_hyper(hyper: Any) -> Any:
  """Grid search specification."""
  sweep_lr = hyper.sweep('config.lr.base_learning_rate', [0.001])
  sweep_decode = hyper.sweep('config.model.decode', [False, True])
  return hyper.product([sweep_lr, sweep_decode])
