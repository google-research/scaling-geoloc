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


def cross_view_retrieval(
    batch_size: int,
    num_workers: int,
    query: tuple[str, str, list[str]],  # pylint: disable=unused-argument
    database: tuple[str, str, list[str]],
    join_factor: float = -1.0,  # Do not join databases.
    join_normalize: bool = False,
) -> config_dict.ConfigDict:
  """Returns a config for cross-view retrieval."""
  config = defaults.retrieval(batch_size)
  config.data.grain_num_workers_valid = num_workers
  config.data.grain_num_workers = num_workers
  config.method = 'encode_multiview'
  config.method_db = 'encode_aerial'
  query_dirname, query_split, query_locations = query
  database_dirname, database_split, database_locations = database
  config.data.dirname = query_dirname
  config.data.locations.train = config.data.locations.valid = query_locations
  config.data.train_split = config.data.valid_split = query_split
  config.db_data = config_dict.ConfigDict()
  config.db_data.dirname = database_dirname
  config.db_data.locations = config_dict.ConfigDict()
  config.db_data.locations.train = config.db_data.locations.valid = (
      database_locations
  )
  config.db_data.train_split = config.db_data.valid_split = database_split
  config.plot = True
  config.db_data.preprocess = config_dict.ConfigDict()
  config.db_data.preprocess.from_aerial = True

  if join_factor > 0.0:
    config.join_databases = True
    config.join_factor = join_factor
    config.db_from_model = 'build_s2cell_database'
    config.join_normalize = join_normalize
  return config


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
  config.experiments = {
      # 'SV-A/MS': (159886564, 1),  # best baseline
      # 'SV-A/SALAD/Smoothing': (163700491, 1),
      # 'SV-A/Fervers/SingleLOD': (163583833, 1),
      # 'SV-A-CC/L15/Frustum': (162073073, 1),
      # 'SV-A/Fervers/MultiLOD': (164049528, 1),
      'SV-A/Fervers/MultiLOD/80m': (165477802, 1),
      'SV-A/Fervers/SingleLOD/80m': (165651835, 1),
      # 'SV-A/SALAD/Mining': (163893886, 1),
      # 'SV-A/MS/EMA99': (164014236, 1),
      # 'SV-A/NCE/EMA99': (164014236, 2),
      # 'Fervers/Exact': (164085658, 2),
  }

  config.test_evals = config_dict.ConfigDict()
  config.test_evals.bedenl_multi = cross_view_retrieval(
      32 * config.get_ref('device_count'),
      num_workers,
      ('retrieval_l14_split', 'test', ['bedenl']),
      ('retrieval_l14_split', 'test_aerial', ['bedenl_multil16_80m/aerial']),
  )
  config.test_evals.bedenl_multi.cache_results = False
  config.test_evals.bedenl_multi.restrict_to_s2cells = 'data'

  config.test_evals.bedenl_large = cross_view_retrieval(
      32 * config.get_ref('device_count'),
      num_workers,
      ('retrieval_l14_split', 'test', ['bedenl']),
      ('retrieval_l14_split', 'test_aerial', ['bedenl_tinyoffset/aerial']),
  )
  config.test_evals.bedenl_large.cache_results = False
  config.test_evals.bedenl_large.restrict_to_s2cells = 'data'

  config.test_evals.bedenl_multi_old = cross_view_retrieval(
      32 * config.get_ref('device_count'),
      num_workers,
      ('retrieval_l14_split', 'test', ['bedenl']),
      ('retrieval_l14_split', 'test_aerial', ['bedenl_multi/aerial']),
  )
  config.test_evals.bedenl_multi_old.cache_results = False
  config.test_evals.bedenl_multi_old.restrict_to_s2cells = 'data'

  return config
