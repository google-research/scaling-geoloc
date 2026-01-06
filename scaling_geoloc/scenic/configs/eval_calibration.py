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
      # 'SV-A-CC/L15/Frustum': (162073073, 1),
      'SV-A-CC/L15/DetachCC-A/Fast': (165252758, 2),
      # 'SV-A-CC/L15/80m/DetachCC-A/Fast': (165406847, 1),
  }

  config.test_evals = config_dict.ConfigDict()

  config.test_evals.bedenl_large_accsc1 = defaults.cross_view_retrieval(
      32 * config.get_ref('device_count'),
      num_workers,
      ('retrieval_l14_split', 'test', ['bedenl']),
      ('retrieval_l14_split', 'test_aerial', ['bedenl_tinyoffset/aerial']),
      join_factor=1.0,
  )
  config.test_evals.bedenl_large_accsc1.restrict_to_s2cells = 'data'

  config.test_evals.bedenl_large_accsc125 = defaults.cross_view_retrieval(
      32 * config.get_ref('device_count'),
      num_workers,
      ('retrieval_l14_split', 'test', ['bedenl']),
      ('retrieval_l14_split', 'test_aerial', ['bedenl_tinyoffset/aerial']),
      join_factor=1.25,
  )
  config.test_evals.bedenl_large_accsc125.restrict_to_s2cells = 'data'

  config.test_evals.bedenl_large_acc = defaults.cross_view_retrieval(
      32 * config.get_ref('device_count'),
      num_workers,
      ('retrieval_l14_split', 'test', ['bedenl']),
      ('retrieval_l14_split', 'test_aerial', ['bedenl_tinyoffset/aerial']),
      join_factor=1.5,
  )
  config.test_evals.bedenl_large_acc.restrict_to_s2cells = 'data'

  config.test_evals.bedenl_large_accsc175 = defaults.cross_view_retrieval(
      32 * config.get_ref('device_count'),
      num_workers,
      ('retrieval_l14_split', 'test', ['bedenl']),
      ('retrieval_l14_split', 'test_aerial', ['bedenl_tinyoffset/aerial']),
      join_factor=1.75,
  )
  config.test_evals.bedenl_large_accsc175.restrict_to_s2cells = 'data'

  config.test_evals.bedenl_large_accsc2 = defaults.cross_view_retrieval(
      32 * config.get_ref('device_count'),
      num_workers,
      ('retrieval_l14_split', 'test', ['bedenl']),
      ('retrieval_l14_split', 'test_aerial', ['bedenl_tinyoffset/aerial']),
      join_factor=2.0,
  )
  config.test_evals.bedenl_large_accsc2.restrict_to_s2cells = 'data'

  return config
