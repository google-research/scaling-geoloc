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


def get_config(
    num_workers: int = 64, kappa: float = 1.5
) -> config_dict.ConfigDict:
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
      'InfoNCE-Baseline': (158790813, 1),
      'Ours (full, L14, very old)': (162076391, 2),
      'Ours (best, L15, paper)': (163765400, 1),
      'Ours (prototypes-only)': (164550496, 1),
      'Ours (aerial-only)': (164550496, 2),
  }

  config.test_evals = config_dict.ConfigDict()

  config.test_evals.wce_aerial = defaults.cross_view_retrieval(
      64 * config.get_ref('device_count'),
      num_workers,
      ('retrieval_l14_split_re', 'test', ['western_continental_europe']),
      (
          'retrieval_l14_split_re',
          'test_aerial',
          ['western_continental_europe/aerial'],
      ),
  )
  config.test_evals.wce_aerial.restrict_to_s2cells = 'data'

  config.test_evals.wce_aerial = defaults.cross_view_retrieval(
      64 * config.get_ref('device_count'),
      num_workers,
      ('retrieval_l14_split_re', 'test', ['western_continental_europe']),
      (
          'retrieval_l14_split_re',
          'test_aerial',
          ['western_continental_europe/aerial'],
      ),
      join_factor=kappa,
  )
  config.test_evals.wce_aerial.restrict_to_s2cells = 'data'

  config.test_evals.wce_cc = defaults.cell_code_retrieval(
      8 * config.get_ref('device_count'),
      num_workers,
      'retrieval_l14_split',
      'test',
      ['western_continental_europe_1sample_re'],
  )
  config.test_evals.wce_cc.valid_distance_threshold = 500
  config.test_evals.wce_cc.restrict_to_s2cells = 'data'

  config.test_evals.uk_ireland = defaults.cross_view_retrieval(
      32 * config.get_ref('device_count'),
      num_workers,
      ('retrieval_l14_split_re', 'test', ['united_kingdom_ireland']),
      (
          'retrieval_l14_split_re',
          'test_aerial',
          ['united_kingdom_ireland/aerial'],
      ),
  )

  return config
