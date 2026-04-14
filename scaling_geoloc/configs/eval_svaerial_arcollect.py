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

"""Config for evaluating IBR on SV-Aerial datasets."""

from ml_collections import config_dict

from scaling_geoloc.scenic.configs import defaults


def get_config(num_workers: int = 64) -> config_dict.ConfigDict:
  """Return the config."""
  config = config_dict.ConfigDict()
  config.overwrite_existing_evals = False
  config.rng_seed = 42
  config.data_dtype_str = 'float32'
  config.dtype_str = 'float32'
  config.device_count = config_dict.placeholder(
      int
  )  # Initialized by eval script.
  config.experiments = {
      # 'Fervers/Exact': (164085658, 2),
      # 'SV-A/SALAD/Smoothing': (163700491, 1),
      # 'SV-A/Metros/Finetuned': (165133763, 1),
      # 'SV-A/80m/Finetuned': (165252883, 1),
      # 'Old-SV-A-CC': (159949745, 1),  # Very old.
      # 'SV-A-CC/L15/Frustum': (162073073, 1),  # Weak ARCollect.
      # 'SV-A-CC/Full': (164887458, 1),
      # 'SV-A/SV-CC': (164887458, 2),
      # 'SV-CC/A-CC': (164887458, 3),
      # 'SV-A/A-CC': (164887458, 4),
      # 'SV-A-CC/DetachCC-A': (165136660, 1),
      # 'SV-A-CC/L15/DetachCC-A/Fast': (165406847, 1),
      # 'SV-CC/Metros/Finetuned': (165133059, 1),
      # 'SV-A-CC/L15/Metros/Fast': (164180964, 1),
      # 'SV-CC/L15': (163723443, 1),
      'dinobase': (183109369, 1),
      'dinolarge': (183111181, 1),
  }

  config.test_evals = config_dict.ConfigDict()

  config.test_evals.arcollect = defaults.cross_view_retrieval(
      32 * config.get_ref('device_count'),
      64,  # num_workers,
      (
          '../../ttl=360d/arcollect',
          'test',
          ['bedenl_l16_25_to_100_examples_per_cell-250424/'],
      ),
      ('retrieval_l14_split', 'test_aerial', ['bedenl_tinyoffset/aerial']),
  )
  config.test_evals.arcollect.cache_results = False
  config.test_evals.arcollect.restrict_to_s2cells = (
      'retrieval_l14_split/bedenl/*_grouped_s2cells.csv*'
  )

  config.test_evals.arcollect.data.preprocess.from_arcollect = True
  config.test_evals.arcollect.db_data.preprocess.from_arcollect = False
  # config.test_evals.arcollect.data.preprocess.center_crop = (360, 360)
  config.test_evals.arcollect.data.preprocess.resize_pad = (224, 224)
  # config.test_evals.arcollect.data.preprocess.stretch_size = (224, 224)
  config.test_evals.arcollect.db_data.preprocess.center_crop = None
  config.test_evals.arcollect.db_data.preprocess.resize_pad = None
  config.test_evals.arcollect.db_data.preprocess.stretch_size = None
  config.test_evals.arcollect.db_data.grain_num_workers_valid = 64
  config.test_evals.arcollect.db_data.grain_num_workers = 64

  config.test_evals.arcollect_cc = defaults.cell_code_retrieval(
      16 * config.get_ref('device_count'),
      num_workers,
      '../../ttl=360d/arcollect',
      'test',
      ['bedenl_l16_25_to_100_examples_per_cell-250424/'],
  )
  config.test_evals.arcollect_cc.cache_results = False
  config.test_evals.arcollect_cc.data.preprocess.from_arcollect = True
  # config.test_evals.arcollect_cc.data.preprocess.center_crop = (360, 360)
  config.test_evals.arcollect_cc.data.preprocess.resize_pad = (224, 224)
  # config.test_evals.arcollect_cc.data.preprocess.stretch_size = (224, 224)
  config.test_evals.arcollect_cc.valid_distance_threshold = 500

  return config
