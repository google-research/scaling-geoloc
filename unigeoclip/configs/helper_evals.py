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

"""Helper functions for eval configs."""

from ml_collections import config_dict

from unigeoclip.configs import defaults


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


def classif_eval(
    batch_size: int,
    split_test: str,
    num_workers: int,
    name: str,
    num_classes: int,
) -> config_dict.ConfigDict:
  """Returns a config for classification eval."""
  config = defaults.downstream_eval(batch_size)
  config.data.valid_split = split_test
  config.data.grain_num_workers_valid = num_workers
  config.data.grain_num_workers = num_workers
  config.data.dataset_name = name
  config.data.data_dir = config.data.data_dir + '/' + name
  config.num_classes = num_classes
  config.batch_sizes_eval = [512, 128, 32, 8, 2048]
  config.learning_rates = [2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5]
  return config


def semseg_eval(
    batch_size: int,
    split_test: str,
    num_workers: int,
    name: str,
    num_classes: int,
) -> config_dict.ConfigDict:
  """Returns a config for segmentation eval."""
  config = defaults.downstream_eval(batch_size)
  config.eval_name = 'semseg_eval'
  config.data.valid_split = split_test
  config.data.grain_num_workers_valid = num_workers
  config.data.grain_num_workers = num_workers
  config.data.dataset_name = name
  config.data.data_dir = config.data.data_dir + '/' + name
  config.num_classes = num_classes
  config.batch_sizes_eval = [512, 256, 128, 64]
  config.learning_rates = [2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5]
  return config
