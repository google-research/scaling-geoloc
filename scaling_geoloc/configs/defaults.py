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
from scaling_geoloc.scenic.configs import types as config_types

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
  config.data.grain_num_threads_valid = 8
  config.data.grain_num_workers = 16
  config.data.grain_num_workers_valid = 4
  config.data.grain_prefetch_buffer_size = 32
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


def dino_imagenet(
    num_local_views: int = 8, normalize: bool = True
) -> ml_collections.ConfigDict:
  """Config for DINO crops on ImageNet."""
  config = ml_collections.ConfigDict()
  config.dataset_name = 'dino_dataset'
  config.dataset = 'imagenet2012'
  config.train_split = 'train'
  config.prefetch_to_device = 2
  config.shuffle_buffer_size = 250_000
  config.num_local_views = num_local_views
  config.pp_train = (
      'decode'
      '|copy("image", "global_view0")'
      '|inception_crop(224, 40, data_key="global_view0")'
      '|flip_lr(data_key="global_view0")'
      '|value_range(0, 1, data_key="global_view0")'
      '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="global_view0")'
      '|random_grayscale(0.2, data_key="global_view0")'
      '|random_blur(1.0, data_key="global_view0")'
      + (
          f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="global_view0")'
          if normalize
          else ''
      )
      + '|copy("image", "global_view1")'
      '|inception_crop(224, 40, data_key="global_view1")'
      '|flip_lr(data_key="global_view1")'
      '|value_range(0, 1, data_key="global_view1")'
      '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="global_view1")'
      '|random_grayscale(0.2, data_key="global_view1")'
      '|random_blur(0.1, data_key="global_view1")'
      '|random_solarize(0.2, data_key="global_view1")'
      + (
          f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="global_view1")'
          if normalize
          else ''
      )
      + ''.join(
          [f'|copy("image", "local_view{i}")' for i in range(num_local_views)]
      )
      + ''.join([
          f'|inception_crop(96, 5, 40, data_key="local_view{i}")'
          for i in range(num_local_views)
      ])
      + ''.join([
          f'|flip_lr(data_key="local_view{i}")' for i in range(num_local_views)
      ])
      + ''.join([
          f'|value_range(0, 1, data_key="local_view{i}")'
          for i in range(num_local_views)
      ])
      + ''.join([
          '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1,'
          f' data_key="local_view{i}")'
          for i in range(num_local_views)
      ])
      + ''.join([
          f'|random_grayscale(0.2, data_key="local_view{i}")'
          for i in range(num_local_views)
      ])
      + ''.join([
          f'|random_blur(0.5, data_key="local_view{i}")'
          for i in range(num_local_views)
      ])
      + ''.join([
          f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="local_view{i}")'
          for i in range(num_local_views * normalize)
      ])
      + '|keep("global_view0", "global_view1"'
      + ''.join([f', "local_view{i}"' for i in range(num_local_views)])
      + ')'
  )
  return config


def depth_loss() -> config_dict.ConfigDict:
  return config_dict.ConfigDict(
      dict(
          fn='depth_mae',
          confidence_weight=1.0,
          confidence_type='semantic',
      )
  ).lock()


def depth_unet() -> ml_collections.ConfigDict:
  return ml_collections.ConfigDict(
      dict(
          image_encoder={
              'padding': 'SAME',
              'norm': 'GROUPNORM',
              'output_norm': 'NONE',
              'block_size': (64, 128, 256, 512),
          },
          loss=depth_loss(),
      )
  ).lock()


def depth_dpt() -> ml_collections.ConfigDict:
  return ml_collections.ConfigDict(
      dict(
          image_encoder={
              'mlp_dim': 1024,
              'num_layers': 12,
              'patch_size': 16,
              'hidden_size': 768,
              'num_heads': 16,
              'dropout_rate': 0.0,
              'attention_dropout_rate': 0.0,
              'return_layers': (1, 3, 8, 11),  # The layer indices to return
              'padding': 'SAME',
          },
          decoder={
              'block_size': (96, 192, 384, 768),  # Intermediate dims in upscale
              'block_scale': (3, 2, 1, 0),  # Upscale factors 2**f
              'feature_dim': 256,  # Decoder dims
              'num_convs': 2,  # Number of convolutions in decoder blocks
              'padding': 'SAME',
          },
          loss=depth_loss(),
      )
  ).lock()


def mae_fomo() -> ml_collections.ConfigDict:
  """Config for MAE from fomo."""
  config = ml_collections.ConfigDict()

  config.masking_rate = 0.75
  config.hidden_size = 768

  config.encoder = config_dict.ConfigDict()
  config.encoder.mlp_dim = 3072
  config.encoder.num_layers = 12
  config.encoder.num_heads = 12

  config.decoder = config_dict.ConfigDict()
  config.decoder.mlp_dim = 3072
  config.decoder.num_layers = 12
  config.decoder.num_heads = 12

  config.patch_height = 16
  config.patch_width = 16

  config.dropout_rate = 0.1
  config.attention_dropout_rate = 0.1
  config.stochastic_droplayer_rate = 0
  config.include_unmasked_tokens_in_loss = False
  config.use_passthrough_input_for_unmasked_tokens = True
  return config.lock()


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


def convnet_pretrained(
    name: str,
    head_output_dim: int | None = 8192,
    train_last_layers: int = -1,
    **kwargs,
) -> ml_collections.ConfigDict | None:
  """Creates a config to load a ViT backbone."""
  version = {
      'convnext_b': 'B',
      'convnext_l': 'L',
  }

  version = version.get(name, None)
  if version is None:
    return None

  config = encoder_head(
      head_output_dim,
      pooling='gap',
      **kwargs,
  ).unlock()
  config.version = version
  config.pretrained_weights = {
      'convnext_b': 'convnext_base_in22k',
      'convnext_l': 'convnext_large_in22k',
  }[name]

  config.backbone_type = config_types.BackboneTypes.CONVNET
  config.imagenet_normalization = True
  config.train_last_layers = train_last_layers
  return config.lock()


def pretrained_backbone(
    name: str,
    head_output_dim: int | None = 8192,
    train_last_layers: int = -1,
    **kwargs,
) -> ml_collections.ConfigDict:
  """Creates a config to load a ViT backbone."""

  # Try to load a convnet first.
  config = convnet_pretrained(
      name, head_output_dim, train_last_layers, **kwargs
  )
  if config is not None:
    return config

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
      'beit_clip_b16': ('B', 16),
      'dinov2_salad_b14': ('B', 14),
      'siglipv2_b16': ('B', 16),
      'siglipv2_l16': ('L', 16),
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
      'beit_clip_b16': 'beitv2_vqkd_encoder_base_decoder_1x768x12_clip',
      'dinov2_salad_b14': 'dinov2_salad_b14',
      'siglipv2_b16': 'SigLIP B/16 224',
      'siglipv2_l16': 'SigLIP L/16 256',
  }[name]

  config.use_big_vision_backbone = 'siglip' in name
  if config.use_big_vision_backbone:
    config.normalization = 'none'
  elif 'satellite' in name:
    config.normalization = 'sat493m'
  else:
    config.normalization = 'imagenet'
  config.add_class_token = True
  config.train_last_layers = train_last_layers
  config.backbone_type = config_types.BackboneTypes.VIT
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


def knn_imagenet(
    batch_size: int,
    normalize: bool = True,
) -> ml_collections.ConfigDict:
  """Returns a standard knn eval configuration."""
  config = eval_base(batch_size)
  config.eval_interval = 10_000
  config.ks = [10, 20, 50]
  config.temperature = 0.07
  config.dataset_name = 'knn_imagenet_dataset'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.prefetch_to_device = 2
  config.dataset_configs.training = False
  config.dataset_configs.shuffle_buffer_size = 250_000
  config.dataset_configs.pp_train = (
      'decode|resize_small(256)|central_crop(224)|value_range(0, 1)'
      + (f'|standardize({MEAN_RGB}, {STDDEV_RGB})' if normalize else '')
      + '|keep("image", "label")'
  )
  config.dataset_configs.pp_eval = (
      'decode|resize_small(256)|central_crop(224)|value_range(0, 1)'
      + (f'|standardize({MEAN_RGB}, {STDDEV_RGB})' if normalize else '')
      + '|keep("image", "label")'
  )
  config.method = config_dict.placeholder(str)
  return config


def retrieval(batch_size: int) -> ml_collections.ConfigDict:
  """Config for retrieval."""
  config = eval_base(batch_size)
  config.eval_name = 'retrieval'
  config.method = config_dict.placeholder(str)
  config.num_samples = None
  config.num_plots = 10
  config.batch_size_retrieval = 64
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
  config.data.build_s2cell_index = False
  config.data.preprocess.center_crop = None
  return config


def correspondences(batch_size: int) -> ml_collections.ConfigDict:
  """Config for retrieval."""
  config = eval_base(batch_size)
  config.eval_name = 'correspondences'
  config.method = config_dict.placeholder(str)
  config.num_samples = 1_024
  config.num_plots = 8
  config.data = query_map_scene().unlock()
  config.data.dirname = 'side_1q_2m_easy_stripped_depth'
  config.data.locations.train = ['from_snap_val']
  config.data.shuffle_valid = True
  config.data.grain_num_workers_valid = 0
  config.data.grain_num_workers = 0
  config.data.preprocess.sample_num_views = {'query': 1, 'map': 1}
  config.data.preprocess.num_lidar_points = 0
  config.data.preprocess.filter_pixels_with_semantics = True
  config.data.preprocess.mask_classes = []
  config.data.preprocess.enforce_2d3d_semantic_consistency = True
  config.data.preprocess.add_grid_3d = False
  config.data.preprocess.voxelize_lidar_projections = False
  config.data.preprocess.filter_keys = dict(keep=None)
  config.resize_factor = 1.0
  config.matching = dict(threshold=0.0, mutual_check=False)
  return config


def class_cooccurrence(batch_size: int) -> ml_collections.ConfigDict:
  """Config for class-cooccurrence."""
  config = eval_base(batch_size)
  config.eval_name = 'class_cooccurrence'
  config.method = config_dict.placeholder(str)
  config.num_samples = 1_000
  config.num_plots = 1
  config.data = query_map_scene().unlock()
  config.data.dirname = 'side_1q_2m_easy_stripped_depth'
  config.data.locations.train = ['from_snap_val']
  config.data.shuffle_valid = True
  config.data.grain_num_workers_valid = 0
  config.data.grain_num_workers = 0
  config.data.preprocess.sample_num_views = {'query': 1, 'map': 1}
  config.data.preprocess.num_lidar_points = 0
  config.data.preprocess.filter_pixels_with_semantics = True
  config.data.preprocess.ignore_classes = ['unknown']
  config.data.preprocess.mask_classes = ['self', 'lidar_artifact']
  config.data.preprocess.enforce_2d3d_semantic_consistency = True
  config.data.preprocess.add_grid_3d = False
  config.data.preprocess.voxelize_lidar_projections = False
  config.data.preprocess.filter_keys = dict(keep=None)
  config.resize_factor = 1.0
  config.n_feat_sample = 200
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


def vpr_retrieval(
    batch_size: int,
    num_workers: int,
    query: tuple[str, str, list[str]],  # pylint: disable=unused-argument
    database: tuple[str, str, list[str]],
    join_factor: float = -1.0,  # Do not join databases.
) -> config_dict.ConfigDict:
  """Returns a config for IBR retrieval."""
  config = retrieval(batch_size)
  config.data.grain_num_workers_valid = num_workers
  config.data.grain_num_workers = num_workers
  config.method = 'encode_multiview'
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

  if join_factor > 0.0:
    config.join_databases = True
    config.join_factor = join_factor
    config.db_from_model = 'build_s2cell_database'
  return config


def cell_code_retrieval(
    batch_size: int,
    num_workers: int,
    dirname: str,
    split: str,
    locations: list[str],
) -> config_dict.ConfigDict:
  """Returns a config for cell code retrieval."""
  config = retrieval(batch_size)
  config.data.grain_num_workers_valid = num_workers
  config.data.grain_num_workers = num_workers
  config.method = 'encode_multiview'
  config.db_from_model = 'build_s2cell_database'
  config.data.dirname = dirname
  config.data.locations.train = config.data.locations.valid = locations
  config.data.train_split = config.data.valid_split = split
  return config


def get_config() -> ml_collections.ConfigDict:
  """Dummy get_config for tests."""
  return base()
