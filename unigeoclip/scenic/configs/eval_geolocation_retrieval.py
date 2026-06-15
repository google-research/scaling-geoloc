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

"""Config for evaluating IBR on SV-Aerial datasets."""

from ml_collections import config_dict

from unigeoclip.scenic.configs import helper_evals as helper


def get_config(num_workers: int = 32) -> config_dict.ConfigDict:
  """Return the config."""
  config = config_dict.ConfigDict()
  config.overwrite_existing_evals = False
  config.rng_seed = 42
  config.data_dtype_str = 'float32'
  config.dtype_str = 'float32'
  config.device_count = config_dict.placeholder(
      int
  )  # Initialized by eval script.
  config.experiments = {'final_merge': (209065570, 1)}

  config.test_evals = config_dict.ConfigDict()
  config.test_evals.ood_geoloc = helper.cross_view_retrieval(
      32 * config.get_ref('device_count'),
      num_workers,
      ('amsterdam_10m_aerial_aug_sv_bk', 'test', ['amsterdam']),
      ('amsterdam_10m_aerial_aug_sv_bk', 'test_aerial', ['amsterdam']),
  )
  config.test_evals.ood_geoloc.retrieval_batch_size = 512
  config.test_evals.ood_geoloc.cell_codes = []
  config.test_evals.ood_geoloc.cache_results = False
  config.test_evals.ood_geoloc.restrict_to_s2cells = 'data'

  config.test_evals.areas_dataset_geoloc = helper.cross_view_retrieval(
      128 * config.get_ref('device_count'),
      num_workers,
      ('area_summaries_sharded', 'test', ['area_summaries']),
      ('area_summaries_sharded', 'test_aerial', ['area_summaries']),
  )
  config.test_evals.areas_dataset_geoloc.retrieval_batch_size = 512
  config.test_evals.areas_dataset_geoloc.cell_codes = [16]
  config.test_evals.areas_dataset_geoloc.cache_results = False
  config.test_evals.areas_dataset_geoloc.restrict_to_s2cells = 'data'

  config.test_evals.pv4ger = helper.classif_eval(
      16 * config.get_ref('device_count'),
      'test',
      num_workers,
      'pv4ger_classification',
      num_classes=2,
  )

  config.test_evals.chesapeake = helper.semseg_eval(
      2 * config.get_ref('device_count'),
      'test',
      num_workers,
      'm-chesapeake',
      num_classes=7,
  )

  config.test_evals.mdas_dsm = helper.semseg_eval(
      16 * config.get_ref('device_count'),
      'test',
      num_workers,
      'mdas',
      num_classes=13,
  )
  config.test_evals.mdas_dsm.eval_name = 'dsm_eval'
  return config
