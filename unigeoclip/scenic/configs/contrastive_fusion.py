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

"""Config file for contrastive fusion model."""

from typing import Any, Mapping, Sequence

import ml_collections

from unigeoclip.scenic.configs import contrastive_merge
from unigeoclip.scenic.configs import modality_fusion


def get_config(
    data_repo: str = 'area_summaries_sharded',
    location: str = 'area_summaries',
    *,
    model_names: Mapping[str, str] | None = None,
    modalities: Sequence[str] | None = None,
    model_size: str = 'B',
    loss_name: str = 'InfoNCE',
    num_workers: int = 32,
    add_evals: bool = True,
    two_stage_training: bool = True,
) -> ml_collections.ConfigDict:
  """Returns the config."""
  config = contrastive_merge.get_config(
      data_repo=data_repo,
      location=location,
      model_names=model_names,
      modalities=modalities,
      model_size=model_size,
      loss_name=loss_name,
      num_workers=num_workers,
      add_evals=add_evals,
  )
  config.batch_size_per_device = 256
  config.model_name = 'contrastive_fusion'
  config.model.mask_negatives_in_same_s2cell = True
  config.model.two_stage_training = two_stage_training
  config.prob = 0.7
  config.data.grain_num_workers = 16
  config.model.modality_fusion_block = modality_fusion.get_config()

  config.test_evals.areas_dataset_geoloc.eval_name = (
      'geolocation_retrieval_fusion'
  )
  config.test_evals.ood_geoloc.eval_name = (
      'geolocation_retrieval_fusion'
  )

  config.data.preprocess.tokenize_area_summary = True

  if two_stage_training:
    config.pretrained_xm = (209065570, 1)
    config.pretrained_load_meta_data = True
    config.num_training_steps = 100_000
    config.lr.warmup_steps = 1_000  # Shorter warmup.
    config.avoid_weight_decay_on_frozen_params = True
    config.lr.base_learning_rate = 1e-3

  return config


def get_hyper(h: Any) -> Any:
  # Run against our own main baselines.
  return h.product([])
