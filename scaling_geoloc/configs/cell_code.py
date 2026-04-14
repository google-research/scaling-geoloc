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

from ml_collections import config_dict

from scaling_geoloc.scenic.configs import defaults


def apply_locations(
    setting: str, config: config_dict.ConfigDict, add_evals: bool = True
) -> None:
  """Applies the locations to the config."""
  config.data.valid_split = 'valid'
  if setting.startswith('bedenl'):
    config.data.dirname = 'retrieval_l14_split'
    train_locations = [setting]
    valid_locations = [setting]
    config.data.valid_split = 'test'
    config.data.preprocess.s2cell_level = 15
    config.data.preprocess.index_file_name = 'train_grouped_s2cells.csv*'

    if add_evals:
      config.test_evals = config_dict.ConfigDict()
      config.test_evals.bedenl = defaults.cell_code_retrieval(
          8 * config.get_ref('device_count'),
          8,
          'retrieval_l14_split',
          'test',
          ['bedenl'],
      )
      config.test_evals.bedenl.num_samples = 100_000
      config.test_evals.bedenl.eval_interval = 20_000
      config.test_evals.bedenl.valid_distance_threshold = 500  # meters
  elif setting == 'western_continental_europe':
    config.data.dirname = 'retrieval_l14_split'
    train_locations = ['western_continental_europe_1sample_re']
    valid_locations = ['western_continental_europe_1sample_re']
    config.data.valid_split = 'test'
    config.data.preprocess.s2cell_level = 14
    config.data.grain_num_workers = 64
    config.data.preprocess.index_file_name = 'train_grouped_s2cells.csv*'
    config.test_evals = config_dict.ConfigDict()
    config.test_evals.wce_cc = defaults.cell_code_retrieval(
        8 * config.get_ref('device_count'),
        8,
        'retrieval_l14_split',
        'test',
        ['western_continental_europe_1sample_re'],
    )
    config.test_evals.wce_cc.num_samples = 100_000
    config.test_evals.wce_cc.eval_interval = 20_000
    config.test_evals.wce_cc.valid_distance_threshold = 200  # meters
  else:
    raise ValueError(f'Unknown setting: {setting}')
  config.data.locations.train = train_locations
  config.data.locations.valid = valid_locations


def get_config(
    location: str = 'western_continental_europe',
    vit_pretrained: str = 'ibot_b16',
    cell_code_dim: int = 2048 + 128,
    add_evals: bool = True,  # pylint: disable=unused-argument
    finetune: bool = True,
) -> config_dict.ConfigDict:
  """Return the config."""
  config = defaults.base_retrieval(200_000).unlock()
  apply_locations(location, config, add_evals=add_evals)
  config.data.grain_num_workers = 32
  config.data.grain_num_workers_valid = 0
  config.data.preprocess.build_s2cell_tree = True
  config.data.preprocess.cell_tree_max_height = 4
  config.data.preprocess.cell_tree_step = 1
  config.data.cell_tree_num_neighbors = 0

  config.model_name = 'cell_code'
  config.model = config_dict.ConfigDict()
  config.model.cell_code_dim = cell_code_dim
  config.model.vit = defaults.pretrained_backbone(
      vit_pretrained,
      head_output_dim=config.model.get_ref('cell_code_dim'),
      bottleneck_dim=None,
  ).unlock()

  # Variable model configs.
  config.model.compose_loss = 'hierarchical_nll_weighted'
  config.model.vit.head = 'salad'
  config.model.l2_normalize = True
  config.model.temperature = 0.01
  config.model.learned_temperature = True
  config.model.shard_codes = True
  config.model.do_augment = True
  config.model.crop = 'none'
  config.model.cell_codes_momentum = 0.0
  config.model.ema_parameterfree = False
  config.model.l2_normalize_per_head = False
  config.model.num_heads_and_offset = (1, 0)
  config.model.haversine_temperature = 500
  config.model.haversine_temperatures = (10_000, 1_000, 100)
  config.model.similarity_bias = 0.2
  config.model.learned_similarity_bias = False

  config.batch_size_per_device = {'S': 128, 'B': 64, 'L': 48}[
      config.model.vit.version
  ]
  config.lr.base_learning_rate = 3e-4
  config.overwrite_lr = {'cell_codes': 1e-2}
  config.num_training_steps = 100_000
  config.checkpoint_interval = 20_000
  config.skip_first_eval = True
  config.run_test_evals_in_process = True

  if finetune:
    assert location == 'western_continental_europe'
    config.pretrained_xm = (158790813, 1)
    config.overwrite_lr = {'output_projection|cell_codes': 1e-2}
    config.pretrained_load_meta_data = False
    config.pretrained_filter.skip_regex = '.*(cell_codes|output_projection).*'
    config.model.vit.train_last_layers = -1
    config.num_training_steps = 100_000
    config.lr.warmup_steps = 1_000  # Shorter warmup.
    config.avoid_weight_decay_on_frozen_params = False
  return config


def get_hyper(h: Any) -> Any:
  return h.product([
      h.sweep(
          'config.model.compose_loss',
          [
              'multi_similarity',
              'haversine_nll',
              'hierarchical_nll',
          ],
      ),
      h.sweep('config.data.preprocess.s2cell_level', [15]),
  ])
