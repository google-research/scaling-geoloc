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

"""Config for evaluating SV-Aerial on BEDENL."""

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
      'Ours (aerial-only, old)': (159886564, 1),  # best baseline
      'Ours (aerial-only, retrained)': (164014236, 1),
      'Ours (aerial-only, fixed80m, finetuned)': (165252883, 1),
      'Ours (aerial-only, fixed80m, scratch)': (165616045, 1),
      # 'Fervers (baseline)': (164085658, 2),  # Might not work anymore -> ask!
      'Fervers (fixed80m, finetuned)': (165189312, 1),
      'Fervers (fixed80m, scratch)': (165658427, 1),
      'Ours (full, fixed80m, finetuned)': (165252883, 2),
      'Ours (full, fixed80m, detach, scratch)': (165406847, 1),
      'Ours (full, detach)': (165252758, 2),
      'Ours (full, best, old)': (162073073, 1),
  }

  config.test_evals = config_dict.ConfigDict()

  # Main eval.
  config.test_evals.bedenl_aerial = defaults.cross_view_retrieval(
      64 * config.get_ref('device_count'),
      num_workers,
      ('retrieval_l14_split', 'test', ['bedenl']),
      ('retrieval_l14_split', 'test_aerial', ['bedenl_tinyoffset/aerial']),
  )
  # Restrict to train set.
  config.test_evals.bedenl_aerial.restrict_to_s2cells = (
      'retrieval_l14_split/bedenl/*_grouped_s2cells.csv*'
  )

  # Supplementary, gaps.
  config.test_evals.bedenl_aerial_extended = defaults.cross_view_retrieval(
      64 * config.get_ref('device_count'),
      num_workers,
      ('retrieval_l14_split_re', 'test', ['bedenl']),
      ('retrieval_l14_split_re', 'test_aerial', ['bedenl/aerial']),
  )

  # Trekker.
  config.test_evals.bedenl_trekker_aerial = defaults.cross_view_retrieval(
      64 * config.get_ref('device_count'),
      num_workers,
      ('retrieval_l14_split_trekker', 'train', ['europe']),
      ('retrieval_l14_split', 'test_aerial', ['bedenl_tinyoffset/aerial']),
  )
  # Restrict to train set.
  config.test_evals.bedenl_trekker_aerial.restrict_to_s2cells = (
      'retrieval_l14_split/bedenl/*_grouped_s2cells.csv*'
  )
  return config
