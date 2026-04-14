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
      'dinobasepadding': (186677052, 1),
      'dinolargepadding': (186868823, 1),
      # 'dinolargestretching': (186882226, 1),
      # 'convnextbase': (185780555, 1),
      # 'convnextlarge': (185795768, 1),
      # 'dinobasestretching': (186677138, 1),
  }

  config.test_evals = config_dict.ConfigDict()
  config.test_evals.osv5m = defaults.cross_view_retrieval(
      32 * config.get_ref('device_count'),
      64,  # num_workers,
      (
          '../../ttl=360d/osv-5m',
          'train',  # train split contains more images
          ['bedenlpadding-250904/'],  # or bedenlstretching-250905/
      ),
      ('retrieval_l14_split', 'test_aerial', ['bedenl_tinyoffset/aerial']),
  )
  config.test_evals.osv5m.cache_results = False
  config.test_evals.osv5m.restrict_to_s2cells = (
      'retrieval_l14_split/bedenl/*_grouped_s2cells.csv*'
  )

  config.test_evals.osv5m.data.preprocess.from_osv5m = True
  config.test_evals.osv5m.db_data.preprocess.from_osv5m = False

  # stretch_size or resize_pad are equal transformations for osv5m evaluation
  # since the images have been already preprocessed (stretched or padded) to be
  # square during the data generation. Prefer using stretch_size since it
  # is faster than resize_pad.
  config.test_evals.osv5m.data.preprocess.stretch_size = (224, 224)
  config.test_evals.osv5m.db_data.preprocess.center_crop = None
  config.test_evals.osv5m.db_data.preprocess.resize_pad = None
  config.test_evals.osv5m.db_data.preprocess.stretch_size = None
  config.test_evals.osv5m.db_data.grain_num_workers_valid = 64
  config.test_evals.osv5m.db_data.grain_num_workers = 64

  config.test_evals.osv5m_cc = defaults.cell_code_retrieval(
      16 * config.get_ref('device_count'),
      num_workers,
      '../../ttl=360d/osv-5m',
      'train',
      ['bedenlpadding-250904/'],  # or bedenlstretching-250905/
  )
  config.test_evals.osv5m_cc.cache_results = False
  config.test_evals.osv5m_cc.data.preprocess.from_osv5m = True
  config.test_evals.osv5m_cc.data.preprocess.stretch_size = (224, 224)
  config.test_evals.osv5m_cc.valid_distance_threshold = 500

  return config
