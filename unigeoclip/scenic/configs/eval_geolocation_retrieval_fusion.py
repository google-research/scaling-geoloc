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

"""Config for evaluating IBR on SV-Aerial datasets."""

from ml_collections import config_dict

from unigeoclip.scenic.configs import helper_evals as helper


def get_config(num_workers: int = 32) -> config_dict.ConfigDict:
  """Return the config."""
  config = config_dict.ConfigDict()
  config.overwrite_existing_evals = True
  config.rng_seed = 42
  config.data_dtype_str = 'float32'
  config.dtype_str = 'float32'
  config.device_count = config_dict.placeholder(
      int
  )  # Initialized by eval script.
  config.experiments = {
      # 'fusion_notext': (199674201, 1),
      # 'text_frozen': (200235511, 1),
      # 'ema_notext': (199659952, 1),
      # 'fusion_ml_lineartext': (201745780, 1),
      # 'fusion_ema': (201753164, 1),
      # '2 stages': (202698260, 1),
      # '2 stages harder': (202897885, 1),
      # 'loss10': (202460668, 4),
      # 'loss10_1e-4': (202460668, 5)
      # 'fusion_out-mid': (203227516, 1),
      # 'fusion_EMA_linearLE_out-mid-nonl_vf': (203415174, 1),
      # 'fusion_linearLE_out-mid-nonl_noisell_vf': (203454676, 1)
      # 'fusion_text_ft': (204045650, 1)
      # '2 stage othermask': (204101643, 1),
      # '2 stage 1 block': (204093658, 1),
      # '2 stage 3 blocks': (204093658, 2),
      # '2 stage 6 blocks': (204093658, 3),
      # '2 stage 12 blocks': (204093658, 4),
      # '2 stage intraloss': (204091376, 1),
      # '2 stage posemb': (204267290, 1),
      # '2 stage otherll': (204278188, 1),
      # '2 stage throughfusion': (204279017, 1),
      # '2 stage closer': (204285788, 1),
      # '2 stage noproj': (204297143, 1),
      # '2 stage all pairs': (204325603, 1),
      # 'test througfusion': (204572773, 1),
      # 'test': (204574395, 1),
      # 'test no proj': (204584319, 1),
      # 'test no proj througfusion': (204585884, 1),
      # 'test no proj througfusion wl': (204604493, 1),
      # 'zfull no proj througfusion': (204588166, 1),
      # 'MultipleNCE': (204724421, 1),
      # 'MultipleNCE2': (204727055, 1),
      # 'MultipleNCE no wl': (204741638, 1),
      # 'MultipleNCE on backbones': (204741638, 1)
      # 'MultipleNCE afix no fusion': (204787269, 1),
      # 'MultipleNCE fix': (204835393, 1),
      # 'MultipleNCE fix no text': (204836174, 1)
      # 'mNCE': (204956202, 1),
      # 'mNCE text ft': (204957077, 1),
      # 'mNCE nolinear2': (204962384, 1),
      # 'mNCE nolinear no text': (204973059, 1),
      # 'mNCE nolinear So400': (205016242, 1)
      # 'mNCE2_nolll_textrecons': (205501641, 1),
      # 'mNCE2_nolll_norecons_oll': (205517613, 1),
      # 'mNCE2_nolll_fix': (205554125, 1),
      # 'mNCE2_nolll_fix_MAE': (205554125, 1),
      # 'mNCE2_nolll_fix_MAE_oll_noproj': (205559389, 1),
      # 'mNCE2_nolll_fix_MAE_textdirect': (205671230, 1),
      # 'mNCE_nolll_fix_sr': (207343058, 1),
      # 'mNCE_nolll_fix_sr_maskcell': (207460281, 1),
      # 'mNCE_nolll_fix_sr_maskcell_ft': (207951669, 1),
      # 'mNCE1': (208182600, 1),
      # 'fullsent_ft': (208777030, 1),
      # 'fullsent': (208783585, 1),
      # 'masclose': (209228896, 1),
      # 'masclose_1000': (209234275, 1),
      # 'masclose_1000_ft': (209252532, 1),
      # '2_stage_fusion_mod_encoding': (209813050, 1),
      # '2_stage_fusion': (209813050, 2),
      '2_stage_fusion_textrecons': (209818207, 1),
      '2_stage_fusion_focus': (209942713, 1),
  }

  config.test_evals = config_dict.ConfigDict()
  config.test_evals.ood_geoloc = helper.cross_view_retrieval(
      32 * config.get_ref('device_count'),
      num_workers,
      ('amsterdam_10m_aerial_aug_sv_bk', 'test', ['amsterdam']),
      ('amsterdam_10m_aerial_aug_sv_bk', 'test_aerial', ['amsterdam']),
  )
  config.test_evals.ood_geoloc.eval_name = 'geolocation_retrieval_fusion'
  config.test_evals.ood_geoloc.retrieval_batch_size = 512
  config.test_evals.ood_geoloc.cell_codes = []
  config.test_evals.ood_geoloc.cache_results = False
  config.test_evals.ood_geoloc.restrict_to_s2cells = 'data'

  config.test_evals.areas_dataset_geoloc = helper.cross_view_retrieval(
      128 * config.get_ref('device_count'),
      num_workers,
      ('area_summaries_sharded', 'test', ['area_summaries']),
      ('area_summaries_sharded', 'test_aerial', ['area_summaries']),
  )
  config.test_evals.areas_dataset_geoloc.eval_name = (
      'geolocation_retrieval_fusion'
  )
  config.test_evals.areas_dataset_geoloc.retrieval_batch_size = 512
  config.test_evals.areas_dataset_geoloc.cell_codes = [16]
  config.test_evals.areas_dataset_geoloc.cache_results = False
  config.test_evals.areas_dataset_geoloc.restrict_to_s2cells = 'data'

  config.test_evals.pv4ger = helper.classif_eval(
      16 * config.get_ref('device_count'),
      'test',
      num_workers,
      'pv4ger_classification',
      num_classes=2,
  )
  config.test_evals.pv4ger = helper.classif_eval(
      16 * config.get_ref('device_count'),
      'test',
      num_workers,
      'pv4ger_classification',
      num_classes=2,
  )

  config.test_evals.chesapeake = helper.semseg_eval(
      2 * config.get_ref('device_count'),
      'test',
      num_workers,
      'm-chesapeake',
      num_classes=7,
  )
  config.test_evals.chesapeake = helper.semseg_eval(
      2 * config.get_ref('device_count'),
      'test',
      num_workers,
      'm-chesapeake',
      num_classes=7,
  )

  config.test_evals.mdas_dsm = helper.semseg_eval(
      16 * config.get_ref('device_count'),
      'test',
      num_workers,
      'mdas',
      num_classes=13,
  )
  config.test_evals.mdas_dsm.eval_name = 'dsm_eval'
  config.test_evals.mdas_dsm = helper.semseg_eval(
      16 * config.get_ref('device_count'),
      'test',
      num_workers,
      'mdas',
      num_classes=13,
  )
  config.test_evals.mdas_dsm.eval_name = 'dsm_eval'
  return config
