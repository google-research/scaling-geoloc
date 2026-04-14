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

"""SV-aerial alignment finetune config."""

from typing import Any

import ml_collections

from scaling_geoloc.scenic.configs import sv_aerial_code


def get_config(
    location: str = 'bedenl80m',
    vit_pretrained: str = 'ibot_b16',
    cell_code_dim: int = 2048 + 128,
    add_evals: bool = True,
    link_cell_code_dim: bool = False,
) -> ml_collections.ConfigDict:
  """Returns the config."""
  config = sv_aerial_code.get_config(
      location=location,
      vit_pretrained=vit_pretrained,
      cell_code_dim=cell_code_dim,
      add_evals=add_evals,
      link_cell_code_dim=link_cell_code_dim,
  )

  # We freeze the backbone by default (Just train heads and cell codes).
  config.model.vit.train_last_layers = -1
  config.model.vit.head_output_dim = cell_code_dim

  # The current best model.
  if 'benelux' in location:
    config.pretrained_xm = (156169309, 1)
  elif 'bedenl' in location:
    config.pretrained_xm = (159886564, 1)
    # config.pretrained_xm = (163723443, 1)
    # config.pretrained_xm = (159949745, 1)
  else:
    config.pretrained_xm = (158790813, 1)  # WCE SV-A / InfoNCE model.
  config.pretrained_load_meta_data = False  # No need to load meta data.
  # Do not load the cell codes and the head.
  config.pretrained_filter.skip_regex = '.*(cell_codes|output_projection).*'
  config.lr.base_learning_rate = 3e-4
  config.overwrite_lr = {'output_projection|cell_codes': 1e-2}
  config.num_training_steps = 100_000
  config.lr.warmup_steps = 1_000  # Shorter warmup.
  config.avoid_weight_decay_on_frozen_params = False
  config.model.checkpoint_interval = 20_000
  return config


def get_hyper(h: Any) -> Any:
  return h.product([
      h.sweep('config.model.vit.head', ['salad_nol2']),
      h.sweep('config.data.preprocess.s2cell_level', [15]),
      h.sweep('config.model_name', ['sv_aerial_code_largescale']),
      h.sweep('config.model.store_half_precision', [True]),
      h.sweep('config.model.sv_multi_similarity_proxy_positives', ['frustum']),
      h.sweep(
          'config.model.aerial_multi_similarity_proxy_positives', ['adjacent']
      ),
      h.sweep(
          'config.model.multi_similarity_factors',
          ['all'],
      ),
      h.sweep('config.model.masked_expsum', [False]),
      h.sweep('config.model.topk', [128]),
      h.sweep('config.model.multi_similarity_mean_pos_sim', [True]),
      h.sweep('config.model.detach_aerial_cell_codes', [True]),
  ])
