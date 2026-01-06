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

"""Hierarchical cell code retrieval config."""

from ml_collections import config_dict
from scaling_geoloc.scenic.configs import cell_code


def get_config(
    location: str = 'western_continental_europe',
    mode: str = 'exponential',
    cell_code_dim: int = 8192 + 256,
) -> config_dict.ConfigDict:
  """Return the config."""
  config = cell_code.get_config(location).unlock()
  config.model_name = 'hierarchical_cell_code'
  config.model.use_patch_probs = False  # Not supported.
  config.model.compose_loss = 'cell_nll_tree'
  config.model.temperature = 0.01
  h = 4
  config.model.vit.head_output_dim = cell_code_dim
  config.model.cell_code_dim = cell_code_dim
  ccd_ref = config.model.cell_code_dim
  if mode == 'exponential':
    config.model.cell_code_dims = [ccd_ref * 2 ** (h - i - 1) for i in range(h)]
  elif mode == 'linear':
    config.model.cell_code_dims = [ccd_ref * (h - i) for i in range(h)]
  elif mode == 'equal':
    config.model.cell_code_dims = [ccd_ref for _ in range(h)]
  else:
    raise ValueError(f'Unknown mode: {mode}')

  if cell_code_dim == 8192 + 256 and h == 4:
    config.model.cell_code_dims = [4096, 2048, 1024 + 512 + 512, 256]
  elif cell_code_dim == 2048 + 128 and h == 4:
    config.model.cell_code_dims = [1024, 512, 256 + 128, 256]
  else:
    raise ValueError(
        f'Unsupported cell_code_dim ({cell_code_dim}) and height ({h})'
    )
  config.model.global_contrastive = True
  config.model.shared_projections = True
  config.model.shard_codes = False  # Currently not supported.
  config.model.cumprod = False
  config.model.learned_temperature = False
  config.data.preprocess.cell_tree_max_height = h
  config.data.preprocess.cell_tree_step = 2
  config.data.preprocess.cell_tree_num_neighbors = (
      1 - config.model.get_ref('global_contrastive')
  ) * 5
  config.data.preprocess.interpolate_cell_tree = False
  config.data.grain_num_workers = 32
  return config


def get_hyper(h):
  return h.product([
      h.sweep(
          'config.model.compose_loss',
          ['haversine_nll_mask_weighted'],
      ),
      h.sweep('config.batch_size_per_device', [96]),
      h.sweep('config.data.preprocess.interpolate_cell_tree', [False]),
  ])
