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

"""Default configs for modules and datasets."""

from typing import Final
import ml_collections
from ml_collections import config_dict

DATA_DIR = 'datasets/'

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
          restore=True,  # whether to restore from checkpoint
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


def base_retrieval(location: str, total_steps: int = 100_000) -> config_dict.ConfigDict:
  """Base config for retrieval models."""
  config = base().unlock()

  # Data configs.
  config.data = data(location).unlock()
  config.data.grain_num_threads = 16
  config.data.grain_num_threads_valid = 8
  config.data.grain_num_workers = 16
  config.data.grain_num_workers_valid = 4
  config.data.grain_prefetch_buffer_size = 32
  config.data.deterministic = False

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
  config.pretrained_path = None

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


def data(location: str) -> config_dict.ConfigDict:
  return config_dict.ConfigDict(
      dict(
          version=config_dict.placeholder(str),
          data_dir=DATA_DIR,
          location=location,
          index_subdir='index_level17',
          train_split='train',
          valid_split='valid',
          shuffle_train=True,
          shuffle_valid=True,
          grain_num_workers=8,
          grain_num_workers_valid=4,
          grain_num_threads=16,
          grain_num_threads_valid=8,
          grain_prefetch_buffer_size=1,  # no prefetching
          filter_invalid_cells=True,
          preprocess=dict(),
      )
  ).lock()


def vit_backbone(
    version: str,
    patch: int,
) -> ml_collections.ConfigDict:
  """Default config for ViT backbone."""
  config = ml_collections.ConfigDict()
  config.hidden_size = {'Ti': 192, 'S': 384, 'B': 768, 'L': 1024, 'H': 1280}[
      version
  ]
  config.patches = ml_collections.ConfigDict()
  config.patches.size = [int(patch), int(patch)]
  config.posembs = (224 // patch, 224 // patch)
  config.num_heads = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16, 'H': 16}[version]
  config.mlp_dim = {'Ti': 768, 'S': 1536, 'B': 3072, 'L': 4096, 'H': 5120}[
      version
  ]
  config.num_layers = {'Ti': 12, 'S': 12, 'B': 12, 'L': 24, 'H': 32}[version]
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
  return config


def encoder_head(
    output_dim: int | None,
    pooling: str = 'tok',
    bottleneck_dim: int | None = 256,
) -> ml_collections.ConfigDict:
  """Default config for an encoder head."""
  config = ml_collections.ConfigDict()
  config.head_output_dim = output_dim
  config.pooling = pooling
  config.head_config = ml_collections.ConfigDict()
  config.head_config.head_hidden_dim = 2048
  if bottleneck_dim is None:
    config.head_config.head_bottleneck_dim = output_dim
  else:
    config.head_config.head_bottleneck_dim = bottleneck_dim
  config.head_config.normalize_last_layer = True
  config.head_config.use_bn = False
  config.head_config.approximate_gelu = True
  config.head_config.pooling = 'concat'
  config.head_config.l2_normalize = False
  config.head_config.num_layers = 1
  config.head_config.dropout_rate = 0.0
  config.head_config.attention_dropout_rate = 0.0
  return config


def vit(
    version: str,
    patch: int,
    output_dim: int | None,
    pooling: str = 'tok',
    bottleneck_dim: int | None = 256,
) -> ml_collections.ConfigDict:
  """Default config for ViT / B16."""
  config = vit_backbone(version, patch).unlock()
  config.update(encoder_head(output_dim, pooling, bottleneck_dim))
  return config.lock()


def pretrained_backbone(
    name: str,
    head_output_dim: int | None = 8192,
    train_last_layers: int = -1,
    **kwargs,
) -> ml_collections.ConfigDict:
  """Creates a config to load a ViT backbone."""

  # Try to load a convnet first.
  version, patch = {
      'ibot_b16': ('B', 16),
      'ibot_l16': ('L', 16),
      'dinov1_b16': ('B', 16),
      'dinov1_b8': ('B', 8),
      'dinov1_s16': ('S', 16),
      'dinov1_s8': ('S', 8),
      'dinov2_s14': ('S', 14),
      'dinov2_b14': ('B', 14),
      'dinov2_l14': ('L', 14),
      'dinov3_s16': ('S', 16),
      'dinov3_b16': ('B', 16),
      'dinov3_l16': ('L', 16),
      'dinov3_satellite_l16': ('L', 16),
      'dinov2_salad_b14': ('B', 14),
  }[name]
  config = vit(
      version,
      patch,
      head_output_dim,
      pooling='tok',
      **kwargs,
  ).unlock()
  config.pretrained_weights = {
      'ibot_b16': 'ibot_b16',
      'ibot_l16': 'ibot_l16',
      'dinov1_b16': 'dino_official_b16',
      'dinov1_b8': 'dino_official_b8',
      'dinov1_s16': 'dino_official_s16',
      'dinov1_s8': 'dino_official_s8',
      'dinov2_s14': 'dinov2_vits14',
      'dinov2_b14': 'dinov2_vitb14',
      'dinov2_l14': 'dinov2_vitl14',
      'dinov3_s16': 'dinov3_vits16',
      'dinov3_b16': 'dinov3_vitb16',
      'dinov3_l16': 'dinov3_vitl16',
      # The satellite pretrained weights are available only for the L16 version.
      'dinov3_satellite_l16': 'dinov3_satellite_vitl16',
      'dinov2_salad_b14': 'dinov2_salad_b14',
  }[name]
  if 'satellite' in name:
    config.normalization = 'sat493m'
  else:
    config.normalization = 'imagenet'
  config.add_class_token = True
  config.train_last_layers = train_last_layers
  if 'dinov3' in name:
    config.num_storage_tokens = 4
    config.rope_config = ml_collections.ConfigDict(
        dict(
            use_rope=True,
            base=100.0,
            min_period=None,
            max_period=None,
            normalize_coords='max',
            shift_coords=0.1,
            jitter_coords=1.1,
            rescale_coords=2.0,  # as in the original Dinov3 paper
        )
    )
  return config.lock()


def retrieval_pretrained(
    backbone: str,
    aggregator: str,
) -> ml_collections.ConfigDict:
  """Creates a config to load a pretrained retrieval model."""
  config = ml_collections.ConfigDict()
  config.vit = pretrained_backbone(backbone, None)
  aggregator_configs = {
      'salad': dict(
          hidden_dim=512,
          token_dim=256,
          cluster_dim=128,
          num_clusters=64,
          pretrained=True,
      )
  }
  config.aggregator = ml_collections.ConfigDict(aggregator_configs[aggregator])
  config.aggregator_name = aggregator
  return config.lock()


def eval_base(batch_size: int) -> ml_collections.ConfigDict:
  """Config for eval base."""
  config = ml_collections.ConfigDict()
  config.batch_size = batch_size
  config.batch_size_eval = batch_size
  config.eval_interval = 10_000
  config.log_progress_interval = 20
  return config


def retrieval(location: str, batch_size: int) -> ml_collections.ConfigDict:
  """Config for retrieval."""
  config = eval_base(batch_size)
  config.eval_name = 'retrieval'
  config.method = config_dict.placeholder(str)
  config.num_samples = None
  config.num_plots = 10
  config.batch_size_retrieval = 64
  config.valid_distance_threshold = 100.0  # meters
  config.topk_recall_target = None

  config.data = data(location).unlock()
  config.data.shuffle_valid = True
  config.data.shuffle_train = False
  config.data.train_split = 'valid'
  config.data.valid_split = 'test'
  config.data.grain_num_workers_valid = 0
  config.data.grain_num_workers = 0
  return config


def cross_view_retrieval(
    batch_size: int,
    num_workers: int,
    query: tuple[str, str],  # pylint: disable=unused-argument
    database: tuple[str, str],
    join_factor: float = -1.0,  # Do not join databases.
    join_normalize: bool = False,
) -> config_dict.ConfigDict:
  """Returns a config for cross-view retrieval."""
  query_location, query_split = query
  database_location, database_split = database
  config = retrieval(query_location, batch_size)
  config.data.grain_num_workers_valid = num_workers
  config.data.grain_num_workers = num_workers
  config.method = 'encode_query'
  config.method_db = 'encode_aerial'
  config.data.train_split = config.data.valid_split = query_split
  config.db_data = config_dict.ConfigDict()
  config.db_data.location = database_location
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


def vpr_retrieval(
    batch_size: int,
    num_workers: int,
    query: tuple[str, str, list[str]],  # pylint: disable=unused-argument
    database: tuple[str, str, list[str]],
    join_factor: float = -1.0,  # Do not join databases.
) -> config_dict.ConfigDict:
  """Returns a config for IBR retrieval."""
  query_location, query_split = query
  database_location, database_split = database
  config = retrieval(query_location, batch_size)
  config.data.grain_num_workers_valid = num_workers
  config.data.grain_num_workers = num_workers
  config.method = 'encode_query'
  config.data.train_split = config.data.valid_split = query_split
  config.db_data = config_dict.ConfigDict()
  config.db_data.location = database_location
  config.db_data.train_split = config.db_data.valid_split = database_split
  config.plot = True

  if join_factor > 0.0:
    config.join_databases = True
    config.join_factor = join_factor
    config.db_from_model = 'build_s2cell_database'
  return config


def cell_code_retrieval(
    batch_size: int,
    num_workers: int,
    location: str,
    split: str,
) -> config_dict.ConfigDict:
  """Returns a config for cell code retrieval."""
  config = retrieval(location, batch_size)
  config.data.grain_num_workers_valid = num_workers
  config.data.grain_num_workers = num_workers
  config.method = 'encode_query'
  config.db_from_model = 'build_s2cell_database'
  config.data.train_split = config.data.valid_split = split
  return config


def geo_localizer() -> ml_collections.ConfigDict:
  """Default model config for GeoLocalizer / GeoLocalizerModel."""
  config = ml_collections.ConfigDict()
  config.temperature = 0.1
  config.prototype_dim = None  # None → use backbone output_dim
  config.crop = 'none'
  config.crop_size = (224, 224)
  config.shard_prototypes = False
  config.store_half_precision = False
  config.l2_normalize = False
  config.l2_normalize_per_head = False
  config.do_augment = False
  config.do_padding_augment = False
  config.do_stretching_augment = False
  config.sow_attention = False
  config.detach_query_prototypes = False
  config.detach_aerial_prototypes = False
  config.masked_expsum = None
  config.masked_expsum_ring = False
  config.filter_by_distance = False
  config.same_cell_threshold = 200.0
  # multi-similarity loss hyperparameters
  config.multi_similarity_alpha = 2.0
  config.multi_similarity_beta = 100.0
  config.multi_similarity_base = 0.2
  config.multi_similarity_margin = 0.0
  config.multi_similarity_factors = 'all'
  return config


def get_config() -> ml_collections.ConfigDict:
  """Dummy get_config for tests."""
  return base()
