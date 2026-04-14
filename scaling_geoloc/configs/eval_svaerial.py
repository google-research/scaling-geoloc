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
      # 'SV-A': (154996612, 1),
      # 'SV-CC': (154996612, 2),
      # 'SV-A-CC': (154996612, 3),
      # 'SV-A-CC-MultiSim-from-SV-A': (155403090, 2),
      # 'SV-A-CC-Mine64': (155342231, 2),
      # 'SV-A-CC-MultiSim': (156078401, 1),
      # 'SV-A-CC-MultiSim-Mask': (156078401, 2),
      # 'SV-A-CC-detach-A': (156169309, 1),
      # 'SV-A-CC-Momentum099': (156169686, 1),
      # 'SV-A-CC-MultiSim-Mask-Frustum': (156817614, 1),
      # 'SV-A-CC-MultiSim-Mask-Frustum-NewHead': (156890620, 1),
      # 'SV-A-Tuned': (154716680, 2),
      # 'WesternContinentalEurope-V0': (158790813, 1),
      # 'SV-A-CC/L15': (159949745, 1),
      'SV-A/MS': (159886564, 1),  # best baseline
      # 'SV-CC/L15': (163723443, 1),
      # 'SV-A-CC/L14': (159886564, 2),
      # 'SV-A-CC/L15': (160940008, 1),
      # 'SV-A-CC/L12': (161504116, 3),
      # 'SV-A-CC/L15/Clean': (161583943, 1),
      # 'SV-CC/L15/Clean': (161583943, 2),
      # 'SV-A-CC/L15/Frustum': (162073073, 1),
      # 'SV-A-CC/L15': (163364528, 1),
      # 'SV-A-CC/L16': (163542164, 1),
      # 'SV-A/SALAD/Smoothing': (163700491, 1),
      # 'SV-A/Fervers/SingleLOD': (163583833, 1),
      # 'SV-A/Fervers/MultiLOD': (164049528, 1),
      # 'SV-A/SALAD/SALAD/Mining': (163893886, 1),
      # 'SV-A-CC/L15/FrustumMargin': (162166865, 1),
      # 'SV-A-CC/L14/WCE': (162076391, 2),  # WCE
      # 'SV-A-CC/L15/LS': (163198204, 1),
      # 'WCE-L15-Frustum': (163765400, 1),
      # 'SV-A/MS/EMA99': (164014236, 1),
      # 'SV-A/NCE/EMA99': (164014236, 2),
      # 'Fervers/Exact': (164085658, 2),
      # 'SV-A-CC/L15/Metros/Fast': (164180964, 1),
      # 'SV-A-CC/L15/Metros': (164097305, 1),
      # 'Area/15/NN/NN': (162471320, 1),
      # 'Area/15/F/I': (162471320, 2),
      # 'Area/15/F/NN': (162471320, 3),
      # 'Area/15/NN/I': (162471320, 4),
      # 'Area/15/I/I': (162471320, 5),
      # 'SV-A/SV-CC': (163469190, 2),
      # 'SV-CC/A-CC': (163469190, 3),
      # 'SV-A-CC/Frustum/Scratch': (163767433, 2),
      # 'SV-A-CC/L16/Frustum/1K': (164441212, 1),
      # 'SV-A-CC/L15/Frustum/1K': (164441212, 2),
      # 'SV-A-CC/L14/Frustum/4K': (164441212, 3),
      # 'SV-A-CC/L13/Frustum/8K': (164327755, 2),
      # 'SV-A-CC/Hierarchical/4K': (164522340, 1),
      # 'Fervers/80m/Finetuned': (165189312, 1),
      # 'SV-A/80m/Finetuned': (165252883, 1),
      # 'SV-A-CC/L15/80m/Finetuned': (165252883, 2),
      # 'SV-A-CC/Full': (164887458, 1),
      # 'SV-A/SV-CC': (164887458, 2),
      # 'SV-CC/A-CC': (164887458, 3),
      # 'SV-A/A-CC': (164887458, 4),
      # 'SV-A-CC/DetachCC-A': (165136660, 1),
      'SV-A-CC/L15/DetachCC-A/Fast': (165252758, 2),
      # 'SV-CC/Metros/Finetuned': (165133059, 1),
      # 'SV-A/Metros/Finetuned': (165133763, 1),
      # 'SV-A-CC/L15/80m/DetachCC-A/Fast': (165406847, 1),
      # 'SV-A-CC/L15/Metros/Finetuned': (164180964, 1),
      # 'SV-A/Fervers/SingleLOD/80m': (165658427, 1),
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
  config.test_evals.bedenl_cc.valid_distance_threshold = 500

  config.test_evals.bedenl_large = defaults.cross_view_retrieval(
      64 * config.get_ref('device_count'),
      num_workers,
      ('retrieval_l14_split', 'test', ['bedenl']),
      ('retrieval_l14_split', 'test_aerial', ['bedenl_tinyoffset/aerial']),
  )
  config.test_evals.bedenl_large.cache_results = False
  config.test_evals.bedenl_large.restrict_to_s2cells = 'data'

  # config.test_evals.portugal_aerial =defaults.cross_view_retrieval(
  #     64 * config.get_ref('device_count'),
  #     num_workers,
  #     ('retrieval_l14_split', 'test', ['portugal']),
  #     ('retrieval_l14_split', 'test_aerial', ['portugal_aerial/aerial']),
  # )
  # config.test_evals.portugal_aerial.cache_results = False
  # config.test_evals.portugal_aerial.num_samples = 200_000

  # config.test_evals.bedenl_large_acc =defaults.cross_view_retrieval(
  #     32 * config.get_ref('device_count'),
  #     num_workers,
  #     ('retrieval_l14_split', 'test', ['bedenl']),
  #     ('retrieval_l14_split', 'test_aerial', ['bedenl_tinyoffset/aerial']),
  #     join_factor=1.5,
  # )
  # config.test_evals.bedenl_large_acc.cache_results = False
  # config.test_evals.bedenl_large_acc.restrict_to_s2cells = 'data'

  config.test_evals.bedenl_large_new = defaults.cross_view_retrieval(
      64 * config.get_ref('device_count'),
      num_workers,
      ('retrieval_l14_split_re', 'test', ['bedenl']),
      ('retrieval_l14_split_re', 'test_aerial', ['bedenl/aerial']),
  )
  config.test_evals.bedenl_large_new.cache_results = False

  # config.test_evals.bedenl_large_new_acc =defaults.cross_view_retrieval(
  #     32 * config.get_ref('device_count'),
  #     num_workers,
  #     ('retrieval_l14_split_re', 'test', ['bedenl']),
  #     ('retrieval_l14_split_re', 'test_aerial', ['bedenl/aerial']),
  #     join_factor=1.5,
  # )
  # config.test_evals.bedenl_large_new_acc.cache_results = False

  # config.test_evals.bedenl_multi =defaults.cross_view_retrieval(
  #     64 * config.get_ref('device_count'),
  #     num_workers,
  #     ('retrieval_l14_split', 'test', ['bedenl']),
  #     ('retrieval_l14_split', 'test_aerial', ['bedenl_multi/aerial']),
  # )
  # config.test_evals.bedenl_multi.cache_results = False
  # # config.test_evals.bedenl_large.valid_distance_threshold = 500
  # config.test_evals.bedenl_multi.restrict_to_s2cells = 'data'

  # config.test_evals.uk_ireland =defaults.cross_view_retrieval(
  #     8 * config.get_ref('device_count'),
  #     num_workers,
  #     ('retrieval_l14_split_re', 'test', ['united_kingdom_ireland']),
  #     (
  #         'retrieval_l14_split_re',
  #         'test_aerial',
  #         ['united_kingdom_ireland/aerial'],
  #     ),
  # )
  # config.test_evals.uk_ireland.cache_results = False

  # config.test_evals.bedenl_trekker =defaults.cross_view_retrieval(
  #     64 * config.get_ref('device_count'),
  #     num_workers,
  #     ('retrieval_l14_split_trekker', 'train', ['europe']),
  #     ('retrieval_l14_split', 'test_aerial', ['bedenl_tinyoffset/aerial']),
  # )

  # config.test_evals.bedenl_trekker_cc = defaults.cell_code_retrieval(
  #     64 * config.get_ref('device_count'),
  #     num_workers,
  #     'retrieval_l14_split_trekker',
  #     'train',
  #     ['europe'],
  # )
  # config.test_evals.bedenl_trekker_cc.valid_distance_threshold = 500

  # config.test_evals.bedenl_metros_acc =defaults.cross_view_retrieval(
  #     64 * config.get_ref('device_count'),
  #     num_workers,
  #     ('retrieval_l14_split_trekker', 'train', ['europe']),
  #     ('retrieval_l14_split', 'test_aerial', ['bedenl_tinyoffset/aerial']),
  #     join_factor=1.5,
  # )
  # config.test_evals.bedenl_metros_acc.num_samples = 100_000

  # config.test_evals.bedenl_metros_cc = defaults.cell_code_retrieval(
  #     64 * config.get_ref('device_count'),
  #     num_workers,
  #     'retrieval_l14_split',
  #     'test',
  #     ['bedenl_metros'],
  # )
  # config.test_evals.bedenl_metros_cc.num_samples = 100_000
  # config.test_evals.bedenl_metros_cc.valid_distance_threshold = 500  # meters

  return config
