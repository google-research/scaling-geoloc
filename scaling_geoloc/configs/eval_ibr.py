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

"""IBR evaluation config."""

from ml_collections import config_dict
from scaling_geoloc.scenic.configs import defaults
from scaling_geoloc.scenic.configs import image_retrieval_pretrained


def get_xm_config(
    xm_id: int,
    work_unit_id: int,
    base_config: config_dict.ConfigDict,
) -> config_dict.ConfigDict:
  """Returns an XM config."""
  base_config.init_from = config_dict.ConfigDict()
  base_config.init_from.xm = (xm_id, work_unit_id)
  return base_config


BASEPATH = 'scaling_geoloc.scenic.configs.{}'


def get_config() -> config_dict.ConfigDict:
  """Return the config."""
  config = config_dict.ConfigDict()
  config.overwrite_existing_evals = False
  config.rng_seed = 42
  config.data_dtype_str = 'float32'
  config.dtype_str = 'float32'
  config.device_count = 1  # Initialized by training script.
  config.experiments = {
      'Ours (best, old)': (162073073, 1),  # Best BEDENL model.
      'dinov2_salad_b14': image_retrieval_pretrained.get_config(
          'dinov2_salad_b14', 'salad'
      ),
      'Ours (prototypes-only)': (163723443, 1),  # Best SV-CC model, BEDENL.
      'Ours (aerial-only, retrained)': (
          164014236,
          1,
      ),  # Best SV-A model BEDENL.
      'Ours (best, detached)': (165252758, 2),  # New detached (clean).
  }

  num_workers = 16
  config.test_evals = config_dict.ConfigDict()

  config.test_evals.bedenl = defaults.vpr_retrieval(
      32 * config.get_ref('device_count'),
      num_workers,
      ('retrieval_l14_split', 'test', ['bedenl']),
      ('retrieval_l14_split', 'train', ['bedenl']),
  )

  config.test_evals.uk_ireland = defaults.vpr_retrieval(
      8 * config.get_ref('device_count'),
      num_workers,
      ('retrieval_l14_split_re', 'test', ['united_kingdom_ireland']),
      ('retrieval_l14_split_re', 'train', ['united_kingdom_ireland']),
  )
  config.test_evals.uk_ireland.cache_results = False

  config.test_evals.portugal = defaults.vpr_retrieval(
      8 * config.get_ref('device_count'),
      num_workers,
      ('retrieval_l14_split', 'test', ['portugal']),
      ('retrieval_l14_split', 'train', ['portugal']),
  )
  config.test_evals.portugal.cache_results = False

  config.test_evals.trekker_ibr = defaults.vpr_retrieval(
      16 * config.get_ref('device_count'),
      num_workers,
      ('retrieval_l14_split_trekker', 'train', ['europe']),
      ('retrieval_l14_split', 'train', ['bedenl']),
  )
  config.test_evals.trekker_ibr.db_data.grain_num_workers_valid = 64
  config.test_evals.trekker_ibr.db_data.grain_num_workers = 64
  return config
