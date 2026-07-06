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

"""Downstream eval config: pv4ger, chesapeake, MDAS and PDFM.

Evaluates one trained checkpoint (``config.experiment``) on every task whose
encoder method its model implements (classification, aerial/DSM segmentation and
location regression). The checkpoint's model must implement the encoder interface
(see ``unigeoclip/scenic/models/contrastive_merge.py``); keep only the ``test_evals``
your checkpoint supports (see the support matrix in the README):

    python -m unigeoclip.eval \
        --config=unigeoclip/scenic/configs/eval_downstream.py \
        --workdir=/tmp/eval \
        --config.experiment=/path/to/checkpoint

Generate the fake datasets first with
``python -m unigeoclip.data.generate_fake_downstream_data``, or point
``--config.test_evals.<task>.data.data_dir`` at the real datasets.
"""

import ml_collections
from ml_collections import config_dict

_DATA_DIR = 'datasets'


def _data(name: str, valid_split: str) -> ml_collections.ConfigDict:
  cfg = ml_collections.ConfigDict()
  cfg.dataset_name = name
  cfg.data_dir = f'{_DATA_DIR}/{name}'
  cfg.train_split = 'train'
  cfg.valid_split = valid_split
  cfg.shuffle_train = False
  cfg.shuffle_valid = False
  cfg.grain_num_workers = 4
  cfg.grain_num_workers_valid = 4
  cfg.grain_num_threads = 8
  cfg.grain_num_threads_valid = 8
  cfg.grain_prefetch_buffer_size = 1
  cfg.preprocess = ml_collections.ConfigDict()
  return cfg


def _classif(name: str, valid_split: str, num_classes: int) -> ml_collections.ConfigDict:
  cfg = ml_collections.ConfigDict()
  cfg.eval_name = 'classif_eval'
  cfg.batch_size = 16
  cfg.batch_size_eval = 16
  cfg.eval_interval = 1_000_000
  cfg.log_progress_interval = 20
  cfg.num_classes = num_classes
  cfg.batch_sizes_eval = [512, 128, 32]
  cfg.learning_rates = [2e-3, 1e-3, 5e-4, 1e-4]
  cfg.n_epochs = 500
  cfg.k_values = (1, 5, 10)
  cfg.data = _data(name, valid_split)
  return cfg


def _semseg(name: str, valid_split: str, num_classes: int) -> ml_collections.ConfigDict:
  cfg = ml_collections.ConfigDict()
  cfg.eval_name = 'semseg_eval'
  cfg.batch_size = 8
  cfg.batch_size_eval = 8
  cfg.eval_interval = 1_000_000
  cfg.log_progress_interval = 20
  cfg.num_classes = num_classes
  cfg.batch_sizes_eval = [256, 64]
  cfg.learning_rates = [2e-3, 1e-3, 5e-4]
  cfg.n_epochs = 50
  cfg.data = _data(name, valid_split)
  return cfg


def _dsm(name: str, valid_split: str, num_classes: int) -> ml_collections.ConfigDict:
  cfg = ml_collections.ConfigDict()
  cfg.eval_name = 'dsm_eval'
  cfg.batch_size = 16
  cfg.batch_size_eval = 16
  cfg.eval_interval = 1_000_000
  cfg.log_progress_interval = 20
  cfg.num_classes = num_classes
  cfg.batch_sizes_eval = [256, 64]
  cfg.learning_rates = [2e-3, 1e-3, 5e-4]
  cfg.n_epochs = 200
  cfg.data = _data(name, valid_split)
  return cfg


def _pdfm() -> ml_collections.ConfigDict:
  cfg = ml_collections.ConfigDict()
  cfg.eval_name = 'pdfm_eval'
  cfg.batch_size = 256
  cfg.batch_size_eval = 256
  cfg.eval_interval = 1_000_000
  cfg.data = ml_collections.ConfigDict()
  cfg.data.data_dir = f'{_DATA_DIR}/pdfm'  # contains conus27.csv / pdfm_benchmark.csv
  return cfg


def get_config() -> ml_collections.ConfigDict:
  """Returns the downstream eval config."""
  cfg = ml_collections.ConfigDict()
  cfg.experiment = config_dict.placeholder(str)  # checkpoint workdir
  cfg.overwrite_existing_evals = False
  cfg.dtype_str = 'float32'

  cfg.test_evals = ml_collections.ConfigDict()
  cfg.test_evals.pv4ger = _classif('pv4ger_classification', 'valid', num_classes=2)
  cfg.test_evals.chesapeake = _semseg('m-chesapeake', 'test', num_classes=7)
  # MDAS DSM segmentation needs a model with a `dsm` modality (contrastive_merge).
  cfg.test_evals.mdas = _dsm('mdas', 'valid', num_classes=13)
  # PDFM needs a model with a `latlng` modality (e.g. contrastive_merge).
  cfg.test_evals.pdfm = _pdfm()
  return cfg
