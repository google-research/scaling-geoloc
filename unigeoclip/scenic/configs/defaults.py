# Copyright 2026 Google LLC.
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

"""Default configs for modules and datasets."""

from typing import Final
import ml_collections
from ml_collections import config_dict

DATA_DIR = 'datasets/'

DATA_SPLITS_CITIES = dict(
    train=[
        'from_snap_train',
    ],
    valid=[
        'from_snap_val',
    ],
)

MEAN_RGB: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
STDDEV_RGB: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)


def base() -> ml_collections.ConfigDict:
  return ml_collections.ConfigDict(
      dict(
          model_name=config_dict.placeholder(str),
          device_count=1,  # Automatically set in train.py.
          model=config_dict.placeholder(ml_collections.ConfigDict),
          data=config_dict.placeholder(ml_collections.ConfigDict),
          # training
          batch_size=1,
          batch_size_eval=None,  # Use batch_size instead.
          rng_seed=0,
          scale_batch_size_by_device_count=True,
          shuffle_seed=1234567,
          data_dtype_str='float32',
          dtype_str='float32',
          lr=dict(
              learning_rate_schedule='compound',
              factors='constant',
              base_learning_rate=config_dict.placeholder(float),
              start_decay_step=0,
              steps_per_cycle=config_dict.placeholder(int),
          ),
          optimizer=dict(
              optax_name='scale_by_adam',
              optax_configs=dict(b1=0.9, b2=0.95),
              weight_decay=0.0,
          ),
          rate_fns={},  # extra learning rate fns, format: {name: config}
          max_grad_norm=config_dict.placeholder(float),
          num_training_steps=config_dict.placeholder(int),
          num_eval_steps=config_dict.placeholder(int),
          num_training_epochs=config_dict.placeholder(int),
          checkpoint=True,
          restore=True,  # whether to restore from checkpoint OR init_from
          init_from=dict(checkpoint_path=None, xm=None),
          checkpoint_interval=config_dict.placeholder(int),
          # logging
          log_eval_interval=5000,
          xprof_num_steps_after_validation=10,
          write_summary=True,
          skip_first_eval=False,
          log_summary_interval=config_dict.placeholder(int),
          debug_train=False,
          debug_eval=False,
          num_eval_visualizations=0,
          stack_visualizations_vertically=False,
          overfit=False,
          avoid_weight_decay_on_frozen_params=False,
      )
  ).lock()


def base_retrieval(total_steps: int = 100_000) -> config_dict.ConfigDict:
  """Base config for retrieval models."""
  config = base().unlock()

  # Data configs.
  config.data = query_map_scene().unlock()
  config.data.grain_num_threads = 16
  config.data.grain_num_workers_valid = 4
  config.data.preprocess.sample_num_views = {'query': 1}
  config.data.preprocess.filter_pixels_with_semantics = False
  config.data.preprocess.num_lidar_points = 0
  config.data.preprocess.add_grid_3d = False
  config.data.preprocess.voxelize_lidar_projections = False
  config.data.preprocess.filter_keys = dict(keep=None, exclude=('overlap',))
  config.data.preprocess.build_s2cell_index = True
  config.data.preprocess.build_s2cell_tree = False
  config.data.preprocess.min_s2cell_count = 1
  config.data.preprocess.center_crop = None

  config.data.make_s2cell_data_source = False
  config.data.deterministic = False
  config.data.grain_num_workers = 16
  config.data.s2cell_index_file_name = '{split}_grouped_s2cells16.csv*'

  # Logging configs.
  config.num_training_epochs = None
  config.checkpoint_interval = 5_000
  config.log_summary_interval = 200
  config.log_eval_interval = 10_000
  config.checkpoint = True
  config.skip_first_eval = True
  config.skip_eval = False
  config.skip_train_viz = False
  config.log_train_viz_interval = 5_000
  config.num_eval_visualizations = 1
  config.num_eval_steps = 16

  # Optimizer configs.
  config.optimizer.optax_name = 'scale_by_adam'
  config.optimizer.optax_configs.b1 = 0.9
  config.optimizer.optax_configs.b2 = 0.95
  config.optimizer.weight_decay = 0.0

  config.num_training_steps = total_steps
  config.pretrained_xm = None

  config.lr = ml_collections.ConfigDict()
  config.lr.learning_rate_schedule = 'compound'
  config.lr.factors = 'constant * linear_warmup * cosine_decay'
  config.lr.base_learning_rate = 3e-4
  config.lr.warmup_steps = 5_000
  config.lr.total_steps = config.get_ref('num_training_steps')
  config.lr.steps_per_cycle = config.get_ref('num_training_steps')
  config.lr.alpha = 1.0e-6 / config.lr.get('base_learning_rate')
  config.pretrained_load_meta_data = True
  config.pretrained_load_model_config = False
  config.pretrained_filter = config_dict.ConfigDict()
  config.pretrained_filter.ckpt_prefix_path = []
  config.pretrained_filter.model_prefix_path = []
  config.pretrained_filter.name_mapping = {}
  config.pretrained_filter.skip_regex = None
  config.batch_size_per_device = 1  # Set in get_config.
  config.batch_size = config.get_ref('batch_size_per_device') * config.get_ref(
      'device_count'
  )
  config.batch_size_eval = config.get_ref(
      'batch_size_per_device'
  ) * config.get_ref('device_count')
  config.batch_size_viz = 2 * config.get_ref('device_count')
  return config.lock()


def base_contrastive(total_steps: int = 100_000) -> config_dict.ConfigDict:
  """Base config for retrieval models."""
  config = base().unlock()
  # Data configs.
  config.data = query_map_scene().unlock()
  config.data.grain_num_threads = 32
  config.data.grain_num_workers_valid = 4
  config.data.grain_prefetch_buffer_size = 128
  config.data.preprocess.sample_num_views = {'query': 1}
  config.data.preprocess.filter_pixels_with_semantics = False
  config.data.preprocess.num_lidar_points = 0
  config.data.preprocess.add_grid_3d = False
  config.data.preprocess.voxelize_lidar_projections = False
  config.data.preprocess.filter_keys = dict(keep=None, exclude=('overlap',))
  config.data.preprocess.build_s2cell_index = True
  config.data.preprocess.build_s2cell_tree = False
  config.data.preprocess.min_s2cell_count = 1
  config.data.preprocess.center_crop = None

  config.data.make_s2cell_data_source = False
  config.data.deterministic = False
  config.data.grain_num_workers = 64
  config.data.s2cell_index_file_name = '{split}_grouped_s2cells16.csv*'

  # Logging configs.
  config.num_training_epochs = None
  config.checkpoint_interval = 5_000
  config.log_summary_interval = 200
  config.log_eval_interval = 10_000
  config.checkpoint = True
  config.skip_first_eval = True
  config.skip_eval = False
  config.skip_train_viz = False
  config.log_train_viz_interval = 5_000
  config.num_eval_visualizations = 1
  config.num_eval_steps = 16

  # Optimizer configs.
  config.optimizer.optax_name = 'scale_by_adam'
  config.optimizer.optax_configs.b1 = 0.9
  config.optimizer.optax_configs.b2 = 0.95
  config.optimizer.weight_decay = 0.0

  config.num_training_steps = total_steps
  config.pretrained_xm = None

  config.lr = ml_collections.ConfigDict()
  config.lr.learning_rate_schedule = 'compound'
  config.lr.factors = 'constant * linear_warmup * cosine_decay'
  config.lr.base_learning_rate = 3e-4
  config.lr.warmup_steps = 5_000
  config.lr.total_steps = config.get_ref('num_training_steps')
  config.lr.steps_per_cycle = config.get_ref('num_training_steps')
  config.lr.alpha = 1.0e-6 / config.lr.get('base_learning_rate')
  config.batch_size_per_device = 1  # Set in get_config.
  config.batch_size = config.get_ref('batch_size_per_device') * config.get_ref(
      'device_count'
  )
  config.batch_size_eval = config.get_ref(
      'batch_size_per_device'
  ) * config.get_ref('device_count')
  config.batch_size_viz = 2 * config.get_ref('device_count')
  return config.lock()


def query_map_scene() -> config_dict.ConfigDict:
  return config_dict.ConfigDict(
      dict(
          name='query_map',
          version=config_dict.placeholder(str),
          data_dir=DATA_DIR,
          dirname='side_lr_2m_1q_3sec_07_095',
          locations=dict(
              train=DATA_SPLITS_CITIES['train'],
              valid=DATA_SPLITS_CITIES['valid'],
          ),
          train_split='train',
          valid_split='train',
          shuffle_train=True,
          shuffle_valid=True,
          grain_num_workers=8,
          grain_num_workers_valid=4,
          grain_num_threads=16,
          grain_num_threads_valid=8,
          grain_prefetch_buffer_size=1,  # no prefetching
          preprocess=dict(
              sample_num_views={'query': 2, 'map': 4},
              num_lidar_points=100_000,
              filter_pixels_with_semantics=True,
              ignore_classes=['unknown'],
              mask_classes=['self', 'sky', 'lidar_artifact'],
              enforce_2d3d_semantic_consistency=True,
              add_grid_3d=True,
              grid_3d=dict(
                  extent_meters=(50, 50, 20),
                  cell_size=1.0,
                  z_offset=3.0,
                  center_around_reference=False,
              ),
              voxelize_lidar_projections=False,
              num_dominant_voxels_per_patch=0,
              patch_size=16,
          ),
      )
  ).lock()


def vit(
    version: str,
    patch: int,
    output_dim: int | None,
    pooling: str = 'tok',
    bottleneck_dim: int | None = 256,
    img_size: int = 224,
) -> ml_collections.ConfigDict:
  """Default config for ViT / B16."""
  config = ml_collections.ConfigDict()
  config.hidden_size = {
      'Ti': 192,
      'S': 384,
      'B': 768,
      'L': 1024,
      'So400m': 1152,
      'H': 1280,
  }[version]
  config.patches = ml_collections.ConfigDict()
  config.patches.size = [int(patch), int(patch)]
  config.posembs = (img_size // patch, img_size // patch)
  config.num_heads = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16, 'So400m': 16, 'H': 16}[
      version
  ]
  config.mlp_dim = {
      'Ti': 768,
      'S': 1536,
      'B': 3072,
      'L': 4096,
      'So400m': 4304,
      'H': 5120,
  }[version]
  config.num_layers = {
      'Ti': 12,
      'S': 12,
      'B': 12,
      'L': 24,
      'So400m': 27,
      'H': 32,
  }[version]

  config.head_hidden_dim = 2048
  config.head_output_dim = output_dim
  if bottleneck_dim is None:
    config.head_bottleneck_dim = output_dim
  else:
    config.head_bottleneck_dim = bottleneck_dim
  config.normalize_last_layer = True
  config.use_bn = False
  config.add_class_token = True
  config.n_registers = 4
  config.pooling = pooling
  config.pos_interpolation_method = 'bicubic'
  config.dropout_rate = 0.0
  config.attention_dropout_rate = 0.0
  config.stochastic_depth = 0.1
  config.use_bias = True
  config.pretrained_weights = None
  config.freeze_backbone = False
  config.imagenet_normalization = True
  config.train_last_layers = -1  # Train all layers.
  config.version = version
  config.positional_embedding = 'rope'
  config.norm_layer = 'layernorm'
  config.head_config = ml_collections.ConfigDict()
  config.head_config.pooling = 'concat'
  config.head_config.l2_normalize = False
  config.head_config.num_layers = 1
  config.head_config.dropout_rate = 0.0
  config.head_config.attention_dropout_rate = 0.0
  config.name = 'vit'
  return config.lock()


def siglip(
    version: str,
    patch: int = 16,
) -> ml_collections.ConfigDict:
  """Default config for ViT / B16."""
  config = ml_collections.ConfigDict()
  config.variant = f'{version}/{patch}'
  config.pool_type = 'map'
  config.head_zeroinit = False
  config.patches = ml_collections.ConfigDict()
  config.patches.size = [int(patch), int(patch)]

  config.name = 'siglip'
  return config.lock()


def eval_base(batch_size: int) -> ml_collections.ConfigDict:
  """Config for eval base."""
  config = ml_collections.ConfigDict()
  config.batch_size = batch_size
  config.batch_size_eval = batch_size
  config.eval_interval = 10_000
  config.log_progress_interval = 20
  return config


def retrieval(batch_size: int) -> ml_collections.ConfigDict:
  """Config for retrieval."""
  config = eval_base(batch_size)
  config.eval_name = 'geolocation_retrieval'
  config.method = config_dict.placeholder(str)
  config.num_samples = None
  config.num_plots = 10
  config.batch_size_retrieval = 128
  config.valid_distance_threshold = 100.0  # meters
  config.topk_recall_target = None

  config.data = query_map_scene().unlock()
  config.data.dirname = 'retrieval_l18'
  config.data.locations.train = ['v3_na_san_francisco']
  config.data.locations.valid = ['v3_na_san_francisco']
  config.data.shuffle_valid = True
  config.data.shuffle_train = False
  config.data.train_split = 'valid'
  config.data.valid_split = 'test'
  config.data.grain_num_workers_valid = 0
  config.data.grain_num_workers = 0
  config.data.preprocess.sample_num_views = {'query': 1}
  config.data.preprocess.num_lidar_points = 0
  config.data.preprocess.filter_pixels_with_semantics = False
  config.data.preprocess.mask_classes = []
  config.data.preprocess.enforce_2d3d_semantic_consistency = False
  config.data.preprocess.add_grid_3d = False
  config.data.preprocess.voxelize_lidar_projections = False
  config.data.preprocess.filter_keys = dict(keep=None, exclude=('overlap',))
  config.data.build_s2cell_index = True
  config.data.preprocess.center_crop = None
  config.data.preprocess.s2cell_level = 16
  config.data.preprocess.min_s2cell_count = 1
  config.data.preprocess.index_file_name = 'train_grouped_s2cells.csv*'
  config.data.make_s2cell_data_source = False
  config.data.preprocess.build_s2cell_index = True
  config.data.s2cell_index_file_name = '{split}_grouped_s2cells.csv*'
  return config


def downstream_eval(batch_size: int) -> ml_collections.ConfigDict:
  """Config for downstream tasks."""
  config = eval_base(batch_size)
  config.eval_name = 'classif_eval'
  config.method = config_dict.placeholder(str)

  config.data = ml_collections.ConfigDict()
  config.data.data_dir = (
      'datasets'
  )
  config.data.shuffle_valid = True
  config.data.shuffle_train = False
  config.data.train_split = 'train'
  config.data.valid_split = 'test'
  config.data.grain_prefetch_buffer_size = 1
  config.data.grain_num_workers = 8
  config.data.grain_num_workers_valid = 4
  config.data.grain_num_threads = 16
  config.data.grain_num_threads_valid = 8
  config.data.preprocess = ml_collections.ConfigDict()
  return config


def cross_view_retrieval(
    batch_size: int,
    num_workers: int,
    query: tuple[str, str, list[str]],  # pylint: disable=unused-argument
    database: tuple[str, str, list[str]],
    join_factor: float = -1.0,  # Do not join databases.
    join_normalize: bool = False,
) -> config_dict.ConfigDict:
  """Returns a config for cross-view retrieval."""
  config = retrieval(batch_size)
  config.data.grain_num_workers_valid = num_workers
  config.data.grain_num_workers = num_workers
  config.method = 'encode_multiview'
  config.method_db = 'encode_aerial'
  query_dirname, query_split, query_locations = query
  database_dirname, database_split, database_locations = database
  config.data.dirname = query_dirname
  config.data.locations.train = config.data.locations.valid = query_locations
  config.data.train_split = config.data.valid_split = query_split
  config.db_data = config_dict.ConfigDict()
  config.db_data.dirname = database_dirname
  config.db_data.locations = config_dict.ConfigDict()
  config.db_data.locations.train = config.db_data.locations.valid = (
      database_locations
  )
  config.db_data.train_split = config.db_data.valid_split = database_split
  config.plot = True
  config.db_data.preprocess = config_dict.ConfigDict()
  config.db_data.preprocess.from_aerial = True

  if join_factor > 0.0:
    config.join_databases = True
    config.join_factor = join_factor
    config.db_from_model = 'build_s2cell_database'
    config.join_normalize = join_normalize
  return config
