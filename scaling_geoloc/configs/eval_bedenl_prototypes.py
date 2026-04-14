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

"""Cell code evaluation config on BEDENL (Trekker + normal)."""

from ml_collections import config_dict

from scaling_geoloc.scenic.configs import defaults


def get_config(num_workers: int = 64) -> config_dict.ConfigDict:
  """Return the config."""
  config = config_dict.ConfigDict()
  config.overwrite_existing_evals = False
  config.rng_seed = 42
  config.data_dtype_str = 'float32'
  config.dtype_str = 'float32'
  config.device_count = 1  # Initialized by training script.
  config.experiments = {
      'Ours (best, old)': (162073073, 1),
      'Ours (best, detach)': (165252758, 2),
      'Ours (prototypes-only)': (163723443, 1),
      'Ours (best, old, bedenl+metros)': (164180964, 1),
      'Ours (fixed80m, detach)': (165406847, 1),
  }
  config.test_evals = config_dict.ConfigDict()

  config.test_evals.bedenl_cc = defaults.cell_code_retrieval(
      64 * config.get_ref('device_count'),
      num_workers,
      'retrieval_l14_split',
      'test',
      ['bedenl'],
  )
  config.test_evals.bedenl_cc.cache_results = False
  # Set this large enough, i.e. larger than the tested cell extent.
  config.test_evals.bedenl_cc.valid_distance_threshold = 500

  config.test_evals.bedenl_trekker_cc = defaults.cell_code_retrieval(
      64 * config.get_ref('device_count'),
      num_workers,
      'retrieval_l14_split_trekker',
      'train',
      ['europe'],
  )
  config.test_evals.bedenl_trekker_cc.valid_distance_threshold = 500

  return config
