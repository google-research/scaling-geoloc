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

_DATASET_SIZE = 1281167  # ImageNet1K


def get_config(dataset_size: int = _DATASET_SIZE) -> config_dict.ConfigDict:
  """Return the dino config."""
  config = defaults.base().unlock()
  config.model_name = 'scene_net'
  config.model = config_dict.ConfigDict()
  config.model.vit = defaults.pretrained_backbone('dinov1_s16', 0)
  # config.model.vit.pretrained_weights = None
  config.model.vit.imagenet_normalization = True
  config.model.compose_loss = 'log_depth_l1 + rendered_feature_similarity_l2'
  config.model.decoder = dict(
      mlp_dim=config.model.vit.mlp_dim,
      num_layers=4,
      num_heads=config.model.vit.num_heads,
      dropout_rate=config.model.vit.dropout_rate,
      attention_dropout_rate=config.model.vit.attention_dropout_rate,
      stochastic_droplayer_rate=config.model.vit.stochastic_depth,
      dtype='float32',
  )

  config.model.use_depth_prior = False
  config.model.depth_range = (
      (0.9, 1.1, 5) if config.model.use_depth_prior else (0.5, 1.8, 100)
  )
  config.model.max_num_voxels = 50000
  config.model.grid_extent = (100, 100, 40)
  config.model.render_stride = 4
  config.model.pad_grid = True
  config.model.predict_opacities = True
  config.model.num_depth_bins = 32  # log scale
  config.model.depth_min_max = (1.0, 80.0)

  config.model.mask_ratio = 0.0

  config.data = defaults.query_map_scene().unlock()
  config.data.data_dir = (
      'datasets/'
  )
  config.data.dirname = '4pano_1qv_depthonly'
  config.data.locations.train = ['from_snap_val']
  config.data.grain_num_workers = 32
  config.data.valid_num_workers = 0
  config.data.preprocess.num_lidar_points = 0
  config.data.preprocess.sample_num_views = {}
  config.data.preprocess.add_grid_3d = True
  config.data.preprocess.voxelize_lidar_projections = False
  config.data.preprocess.filter_pixels_with_semantics = True
  config.data.preprocess.grid_3d = dict(
      extent_meters=(30, 30, 20),
      cell_size=0.25,
      z_offset=4.0,
      center_around_reference=False,
  )
  config.data.preprocess.num_dominant_voxels_per_patch = 0
  config.data.preprocess.patch_size = config.model.vit.patches.size[0]
  config.data.preprocess.filter_keys = dict(keep=None)
  config.checkpoint = True
  config.batch_size = 64 * config.get_ref('device_count')
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
  config.init_train = False
  config.lr = config_dict.ConfigDict()
  config.lr.learning_rate_schedule = 'compound'
  config.lr.factors = 'constant * linear_warmup * cosine_decay'
  config.lr.base_learning_rate = min(
      0.001 * config.get_ref('batch_size').to_float() / 512, 0.005
  )
  config.lr.warmup_steps = steps_per_epoch * 10
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

  config.rate_fns = dict(
      params_momentum=dict(
          factors='constant * cosine_decay',
          base_learning_rate=0.996,
          alpha=1.0 / 0.996,
          steps_per_cycle=total_steps,
      ),
  )

  config.run_test_evals_in_process = True
  config.test_evals = config_dict.ConfigDict()
  config.test_evals.knn_imagenet = defaults.knn_imagenet(
      config.get_ref('batch_size_eval'),
      False,
  )
  config.test_evals.knn_imagenet.method = 'encode_multiview'
  config.test_evals.knn_imagenet.eval_interval = 50_000
  return config.lock()
