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

"""Cell-code retrieval config."""

from typing import Any

import ml_collections
from ml_collections import config_dict

from scaling_geoloc.scenic.configs import defaults


def apply_locations(
    location: str,
    config: config_dict.ConfigDict,
    add_evals: bool = True,
):
  """Applies locations to config."""
  config.test_evals = ml_collections.ConfigDict()
  config.run_test_evals_in_process = add_evals

  if location == 'western_continental_europe':
    config.data.dirname = 'retrieval_l14_split'
    config.data.locations.train = ['western_continental_europe_1sample_re']
    config.data.locations.valid = ['western_continental_europe_1sample_re']
    config.data.valid_split = 'test'
    config.data.preprocess.s2cell_level = 14
    config.data.grain_num_workers = 64
    config.data.preprocess.index_file_name = 'train_grouped_s2cells.csv*'
  elif location == 'western_continental_europe80m':
    config.data.dirname = 'retrieval_l14_split'
    config.data.locations.train = ['western_continental_europe80m']
    config.data.locations.valid = ['western_continental_europe80m']
    config.data.valid_split = 'test'
    config.data.preprocess.s2cell_level = 14
    config.data.grain_num_workers = 64
    config.data.preprocess.index_file_name = 'train_grouped_s2cells.csv*'
  elif location == 'bedenl_metros':
    config.data.dirname = 'retrieval_l14_split'
    config.data.locations.train = ['bedenl', 'bedenl_metros']
    config.data.locations.valid = ['bedenl', 'bedenl_metros']
    config.data.valid_split = 'test'
    config.data.preprocess.s2cell_level = 14
    config.data.grain_num_workers = 64
    config.data.preprocess.index_file_name = 'train_grouped_s2cells.csv*'

    if add_evals:
      config.test_evals = ml_collections.ConfigDict()
      config.test_evals.bedenl = defaults.cross_view_retrieval(
          8 * config.get_ref('device_count'),
          8,
          ('retrieval_l14_split', 'test', ['bedenl']),
          ('retrieval_l14_split', 'test_aerial', ['bedenl/aerial']),
      )
      config.test_evals.bedenl.num_samples = 100_000
      config.test_evals.bedenl.eval_interval = 20_000
  elif location.startswith('bedenl_highres'):
    match location:
      case 'bedenl_highres_448_old':
        config.data.dirname = 'retrieval_l14_split_448'
        loc = 'bedenl'
      case 'bedenl_highres_448_new':
        config.data.dirname = 'retrieval_l14_split_448_new'
        loc = 'bedenl80m'
      case 'bedenl_highres_720':
        config.data.dirname = 'retrieval_l14_split_720'
        loc = 'bedenl80m'
      case _:
        raise ValueError(f'Unknown bedenl_highres location: {location}')

    config.data.locations.train = [loc]
    config.data.locations.valid = [loc]

    if add_evals:
      config.test_evals = ml_collections.ConfigDict()
      config.test_evals.bedenl = defaults.cross_view_retrieval(
          8 * config.get_ref('device_count'),
          8,
          (config.data.dirname, 'test', [loc]),
          (
              config.data.dirname,
              'test_aerial',
              [loc + '/aerial'],
          ),
      )
      config.test_evals.bedenl.num_samples = 100_000
      config.test_evals.bedenl.eval_interval = 20_000
    config.data.valid_split = 'test'
    config.data.preprocess.s2cell_level = 14
    config.data.grain_num_workers = 64
    config.data.preprocess.index_file_name = 'train_grouped_s2cells.csv*'
  elif location.startswith('bedenl'):
    config.data.dirname = 'retrieval_l14_split'
    config.data.locations.train = [location]
    config.data.locations.valid = [location]
    config.data.valid_split = 'test'
    config.data.preprocess.s2cell_level = 14
    config.data.grain_num_workers = 64
    config.data.preprocess.index_file_name = 'train_grouped_s2cells.csv*'

    if add_evals:
      config.test_evals = ml_collections.ConfigDict()
      config.test_evals.bedenl = defaults.cross_view_retrieval(
          8 * config.get_ref('device_count'),
          8,
          ('retrieval_l14_split', 'test', [location]),
          ('retrieval_l14_split', 'test_aerial', [f'{location}/aerial']),
      )
      config.test_evals.bedenl.num_samples = 100_000
      config.test_evals.bedenl.eval_interval = 20_000
  else:
    raise ValueError(f'Unknown location: {location}')


def get_config(
    location: str = 'bedenl80m',
    vit_pretrained: str = 'ibot_b16',
    cell_code_dim: int = 2048 + 128,
    add_evals: bool = True,
    link_cell_code_dim: bool = True,
    finetune: bool = False,
) -> config_dict.ConfigDict:
  """Return the config."""
  config = defaults.base_retrieval().unlock()
  apply_locations(location, config, add_evals)
  config.data.make_s2cell_data_source = False
  config.data.preprocess.build_s2cell_index = True

  config.model_name = 'sv_aerial'
  config.model = config_dict.ConfigDict()
  config.model.cell_code_dim = cell_code_dim
  config.model.vit = defaults.pretrained_backbone(
      vit_pretrained,
      head_output_dim=config.model.get_ref('cell_code_dim')
      if link_cell_code_dim
      else config.model.cell_code_dim,
      bottleneck_dim=None,
  ).unlock()
  config.model.vit_aerial = ml_collections.ConfigDict()  # patches model.vit
  if vit_pretrained == 'dinov3_l16':
    config.model.vit_aerial = defaults.pretrained_backbone(
        'dinov3_satellite_l16',
        head_output_dim=config.model.get_ref('cell_code_dim')
        if link_cell_code_dim
        else config.model.cell_code_dim,
        bottleneck_dim=None,
    ).unlock()
    config.model.vit_aerial.head = 'salad'
    config.model.vit_aerial.freeze_backbone = False
    config.model.vit_aerial.train_last_layers = -1  # Train all layers.
    config.model.vit_aerial.head_config.num_dustbins = 1
    config.model.vit_aerial.head_config.l2_normalize = False
  config.model.vit.head = 'salad'
  config.model.vit.freeze_backbone = False
  config.model.vit.train_last_layers = -1  # Train all layers.
  config.model.vit.head_config.num_dustbins = 1
  config.model.vit.head_config.l2_normalize = False
  config.model.embed_aerial_coords = False
  config.model.center_aerial_logits = False
  config.model.center_aerial_logits_momentum = 0.99
  config.model.encode_intrinsics = []  # list of EnumIntrinsics.

  # Variable model configs.
  config.model.compose_loss = 'nll + aerial_sv_nll'
  config.model.l2_normalize = True
  config.model.temperature = 0.01
  config.model.learned_temperature = False
  config.model.aerial_pooling = 'mean'
  config.model.do_augment = True
  config.model.do_padding_augment = False
  config.model.do_stretching_augment = False
  config.model.memorize_steps = 0
  config.model.multi_similarity_base = 0.0
  config.model.crop = 'none'
  config.model.haversine_temperature = 100
  config.model.same_cell_threshold = 200.0
  config.model.label_smoothing = 0.0
  config.model.decouple_softmax = False

  # Ordered sequence of (height, width) tuples or None (no resizing).
  # Example: [None, (224, 224), (256, 256)]
  config.model.resize_batch_sequence = []
  batch_size_map = {'S': 128, 'B': 64, 'L': 32}
  if location == 'bedenl_highres_720':
    batch_size_map = {'S': 64, 'B': 32, 'L': 16}
  config.batch_size_per_device = batch_size_map[config.model.vit.version]
  config.lr.base_learning_rate = 3e-4
  config.pretrained_xm = None
  if finetune:
    config.pretrained_xm = (164085658, 1)  # Momentum 0.99 / df8x8.
    config.pretrained_load_meta_data = False  # No need to load meta data.
    # Do not load the cell codes and the head.
    config.pretrained_filter.skip_regex = '.*(cell_codes|output_projection).*'
    config.model.vit.train_last_layers = -1
    config.num_training_steps = 50_000
    config.lr.warmup_steps = 1_000  # Shorter warmup.
    config.avoid_weight_decay_on_frozen_params = False
  return config


def get_hyper(h: Any) -> Any:
  # Config for reproducing fervers et al.
  return h.product([
      h.zipit([
          h.sweep('config.model.vit.head', ['fervers']),
          h.sweep('config.model.compose_loss', ['nll + aerial_sv_nll']),
          h.sweep('config.model.label_smoothing', [0.1]),
      ]),
      h.sweep('config.model.temperature', [1.0 / 36]),
      h.sweep('config.num_training_steps', [100_000]),
      h.sweep('config.model.learned_temperature', [False]),
      h.sweep('config.model.decouple_softmax', [True]),
  ])
