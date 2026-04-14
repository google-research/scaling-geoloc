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

"""Config for code-based implicit model for Street View and Aerial data."""

from typing import Any

import ml_collections

from scaling_geoloc.scenic.configs import sv_aerial


def get_config(
    location: str = 'bedenl',
    vit_pretrained: str = 'ibot_b16',
    output_dim: int = 2048 + 128,
    cell_code_dim: int = 512,
    add_evals: bool = True,
    finetune: bool = True,
) -> ml_collections.ConfigDict:
  """Return the config."""
  config = sv_aerial.get_config(
      location=location,
      vit_pretrained=vit_pretrained,
      cell_code_dim=output_dim,
      add_evals=add_evals,
      link_cell_code_dim=False,
  )
  config.model_name = 'sv_aerial_code_implicit'
  config.model.implicit_cell_code_dim = cell_code_dim
  config.model.shard_codes = True
  config.model.implicit_cell_codes_dropout = 0.0
  config.model.compose_loss = 'nll + aerial_sv_nll'

  if finetune:
    if 'benelux' in location:
      config.pretrained_xm = (156169309, 1)  # Momentum 0.99 / df8x8.
    else:
      config.pretrained_xm = (158790813, 1)  # Momentum 0.99 / df8x8.
    config.pretrained_load_meta_data = False  # No need to load meta data.
    # Do not load the cell codes and the head.
    config.pretrained_filter.skip_regex = '.*(cell_codes|output_projection).*'
    config.model.vit.train_last_layers = -1
    config.num_training_steps = 50_000
    config.lr.warmup_steps = 1_000  # Shorter warmup.
    config.avoid_weight_decay_on_frozen_params = False
  return config


def get_hyper(h: Any) -> Any:
  return h.product([
      h.zipit([
          h.sweep('config.model.cell_code_dim', [0, 512]),
          h.sweep('config.model.implicit_cell_codes_dropout', [0.0, 0.0]),
      ]),
      h.sweep('config.model.center_aerial_logits_momentum', [0.9]),
      h.sweep('config.model.center_aerial_logits', [True]),
  ])
