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

"""Config file for geoclip model."""

import importlib
from typing import Any, Final, Mapping, Sequence

import ml_collections

from unigeoclip.configs import defaults
from unigeoclip.configs import helper_evals as evals

_PATH: Final[str] = (
    'unigeoclip.configs'
)
_MODELS: Mapping[str, str] = {
    'sv': 'siglip_sv',
    'latlng': 'rff_geoclip'
}
_MODALITIES: Sequence[str] = ['sv', 'latlng']
_BATCH_SIZES: Mapping[str, int] = {'S': 512, 'B': 256, 'L': 128, 'So400m': 128}


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
) -> ml_collections.ConfigDict:
  """Returns the config."""
  config = defaults.base_contrastive().unlock()

  if modalities is None:
    modalities = _MODALITIES

  config.model_name = 'contrastive_merge'

  # Data config
  config.data.dirname = data_repo
  config.data.locations.train = [location]
  config.data.locations.valid = [location]
  config.data.valid_split = 'valid'
  config.data.preprocess.s2cell_level = 16
  config.data.preprocess.index_file_name = 'train_grouped_s2cells.csv*'
  config.data.grain_num_workers = 32
  config.data.grain_num_workers_valid = 4
  config.data.make_s2cell_data_source = False
  config.data.preprocess.build_s2cell_index = True

  # Models configs
  config.model = ml_collections.ConfigDict()
  config.model.modalities = modalities

  if model_names is None:
    model_names = _MODELS

  for modality in modalities:
    if model_names[modality] is not None:
      config.model[f'{modality}_encoder'] = importlib.import_module(
          f'{_PATH}.{model_names[modality]}'
      ).get_config(
          model_size=model_size,
      )

  config.model.exclude_pairs = []
  config.model.crop = 'none'
  config.model.temperature = 0.01
  config.model.learned_temperature = False
  config.model.l2_normalize = True
  config.model.l2_normalize_per_head = False
  config.model.compose_loss = loss_name
  config.model.latlng_proj = False
  config.model.text_proj = False
  config.model.do_augment = True
  config.model.normalize_output_embeddings = True
  config.model.all2all_proj = False
  config.model.latlng_noise_meters = 50.0

  # Training config
  config.batch_size_per_device = _BATCH_SIZES[model_size]
  config.lr.base_learning_rate = 2e-4
  config.num_training_steps = 200_000
  config.lr.warmup_steps = 4_000  # Shorter warmup.
  config.log_eval_interval = 5_000
  config.checkpoint_interval = 5_000

  if add_evals:
    config.run_test_evals_in_process = True
    config.test_evals = ml_collections.ConfigDict()
    config.test_evals.areas_dataset_geoloc = evals.cross_view_retrieval(
        128 * config.get_ref('device_count'),
        num_workers,
        ('area_summaries_sharded', 'test', ['area_summaries']),
        ('area_summaries_sharded', 'test_aerial', ['area_summaries']),
    )
    config.test_evals.areas_dataset_geoloc.retrieval_batch_size = 512
    config.test_evals.areas_dataset_geoloc.cell_codes = [16]
    config.test_evals.areas_dataset_geoloc.cache_results = False
    config.test_evals.areas_dataset_geoloc.restrict_to_s2cells = 'data'
    config.test_evals.areas_dataset_geoloc.eval_interval = (
        config.num_training_steps
    )

    config.test_evals.ood_geoloc = evals.cross_view_retrieval(
        32 * config.get_ref('device_count'),
        num_workers,
        ('amsterdam_10m_aerial_aug_sv_bk', 'test', ['amsterdam']),
        ('amsterdam_10m_aerial_aug_sv_bk', 'test_aerial', ['amsterdam']),
    )
    config.test_evals.ood_geoloc.retrieval_batch_size = 512
    config.test_evals.ood_geoloc.cell_codes = []
    config.test_evals.ood_geoloc.cache_results = False
    config.test_evals.ood_geoloc.restrict_to_s2cells = 'data'
    config.test_evals.ood_geoloc.eval_interval = config.num_training_steps

    config.test_evals.pv4ger = evals.classif_eval(
        16 * config.get_ref('device_count'),
        'valid',
        num_workers,
        'pv4ger_classification',
        num_classes=2,
    )
    config.test_evals.pv4ger.eval_interval = config.num_training_steps

    config.test_evals.chesapeake = evals.semseg_eval(
        2 * config.get_ref('device_count'),
        'test',
        num_workers,
        'm-chesapeake',
        num_classes=7,
    )
    config.test_evals.chesapeake.eval_interval = config.num_training_steps

  return config


def get_hyper(h: Any) -> Any:
  # Run against our own main baselines.
  return h.product([])
