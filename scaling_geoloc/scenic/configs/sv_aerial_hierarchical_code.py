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

"""SV-Aerial hierarchical code model config."""

from typing import Any
import ml_collections
from scaling_geoloc.scenic.configs import sv_aerial_code


def get_config(
    location: str = 'bedenl',
    vit_pretrained: str = 'ibot_b16',
    cell_code_dim: int = 4096 + 128,
    add_evals: bool = True,
    height: int = 2,
    finetune: bool = True,
) -> ml_collections.ConfigDict:
  """Returns the config."""
  config = sv_aerial_code.get_config(
      location=location,
      vit_pretrained=vit_pretrained,
      cell_code_dim=cell_code_dim,
      add_evals=add_evals,
  )
  config.model_name = 'sv_aerial_hierarchical_code'
  if cell_code_dim == 8192 + 256 and height == 4:
    config.model.cell_code_dims = [4096, 2048, 1024 + 512 + 256, 512]
  elif cell_code_dim == 8192 + 256 and height == 3:
    config.model.cell_code_dims = [4096 + 2048 + 512, 1024 + 256, 512]

  elif cell_code_dim == 8192 + 256 and height == 2:
    config.model.cell_code_dims = [4096 + 2048 + 512, 1024 + 256 + 512]
  elif cell_code_dim == 2048 + 128 and height == 4:
    config.model.cell_code_dims = [1024, 512, 256 + 128, 256]
  elif cell_code_dim == 2048 + 128 and height == 3:
    config.model.cell_code_dims = [1024 + 256, 512 + 128, 256]
  elif cell_code_dim == 2048 + 128 and height == 3:
    config.model.cell_code_dims = [512, 512 + 128, 1024]
  elif cell_code_dim == 2048 + 128 and height == 2:
    config.model.cell_code_dims = [1024 + 128, 1024]
  elif cell_code_dim == 4096 + 128 and height == 2:
    config.model.cell_code_dims = [1024 + 2048, 1024 + 128]
  else:
    raise ValueError(
        f'Unsupported cell_code_dim ({cell_code_dim}) and height ({height})'
    )
  config.model.shard_codes = False  # Currently not supported.
  config.model.shard_last_n_layers = 1  # Only 1 supported.
  config.model.cascade = False

  if height != len(config.model.cell_code_dims):
    raise ValueError(
        f'height ({height}) != len(cell_code_dims)'
        f' ({len(config.model.cell_code_dims)})'
    )

  config.data.preprocess.build_s2cell_tree = True
  config.data.preprocess.cell_tree_max_height = height
  config.data.preprocess.cell_tree_step = 4
  config.data.preprocess.cell_tree_num_neighbors = 0
  config.data.preprocess.interpolate_cell_tree = False
  config.data.grain_num_workers = 32
  config.skip_first_eval = True

  if finetune:
    if 'benelux' in location:
      config.pretrained_xm = (156169309, 1)
    elif 'bedenl' in location:
      config.pretrained_xm = (159886564, 1)
    else:
      config.pretrained_xm = (158790813, 1)
    config.pretrained_load_meta_data = False  # No need to load meta data.
    # Do not load the cell codes and the head.
    config.pretrained_filter.skip_regex = '.*(cell_codes|output_projection).*'
    config.model.vit.train_last_layers = -1
    config.num_training_steps = 50_000
    config.lr.warmup_steps = 1_000  # Shorter warmup.
    config.avoid_weight_decay_on_frozen_params = False
  config.test_evals = {}
  return config


def get_hyper(h: Any) -> Any:
  return h.product([
      h.sweep('config.model.vit.train_last_layers', [-1]),
      h.sweep('config.data.preprocess.s2cell_level', [15]),
      h.sweep('config.model.vit.head', ['salad_nol2']),
      h.sweep('config.model.cascade', [True]),
      h.sweep('config.model.shard_last_n_layers', [1]),
  ])
