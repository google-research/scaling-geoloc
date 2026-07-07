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

"""Geolocation-retrieval eval config (reproducible fake-data path).

Runs cross-modal retrieval (street-view <-> aerial <-> location) and Haversine
geolocation metrics for one trained checkpoint (``config.experiment``) over the
procedural ``MultiModalDummyDataSource``, so it runs on CPU without any files:

    python -m unigeoclip.eval \
        --config=unigeoclip/configs/eval_geolocation_retrieval.py \
        --workdir=/tmp/eval \
        --config.experiment=/path/to/checkpoint

``cell_codes`` are grid-cell sizes in degrees: for each, the evaluation
locations are binned onto a regular lat/lng grid and retrieval is scored against
the distinct cell centers (coarser cells -> a smaller, coarser gallery). Leave
it empty to run only the full-set retrieval metrics. To evaluate on real data,
write shards with ``unigeoclip.data.generate_fake_multimodal_data`` (or any
dataset with the same 5-modality spec) and set ``--config.test_evals.geoloc.
data.data_dir`` / ``.data.location``.
"""

import ml_collections
from ml_collections import config_dict


def _data() -> ml_collections.ConfigDict:
  cfg = ml_collections.ConfigDict()
  # ``location`` unset -> the procedural MultiModalDummyDataSource (CPU, no
  # files). Set it (plus ``data_dir``) to read on-disk 5-modality shards.
  cfg.location = config_dict.placeholder(str)
  cfg.data_dir = 'datasets'
  cfg.num_train_examples = 64
  cfg.num_valid_examples = 64
  cfg.num_views = 1
  cfg.sv_image_size = 224
  cfg.aerial_image_size = 256
  cfg.dsm_image_size = 256
  cfg.text_dim = 768
  cfg.shuffle_train = False
  cfg.shuffle_valid = False
  cfg.grain_num_workers = 0
  cfg.grain_num_workers_valid = 0
  cfg.grain_num_threads = 1
  cfg.grain_prefetch_buffer_size = 1
  return cfg


def _geoloc() -> ml_collections.ConfigDict:
  cfg = ml_collections.ConfigDict()
  cfg.eval_name = 'geolocation_retrieval'
  cfg.multimodal_dummy = True  # build the dataset from the dummy source
  cfg.batch_size_eval = 16
  cfg.log_progress_interval = 20
  cfg.cell_codes = [10.0]  # grid-cell sizes in degrees ([] -> full-set only)
  cfg.data = _data()
  return cfg


def get_config() -> ml_collections.ConfigDict:
  """Returns the geolocation-retrieval eval config."""
  cfg = ml_collections.ConfigDict()
  cfg.experiment = config_dict.placeholder(str)  # checkpoint workdir
  cfg.overwrite_existing_evals = False
  cfg.dtype_str = 'float32'
  cfg.test_evals = ml_collections.ConfigDict()
  cfg.test_evals.geoloc = _geoloc()
  return cfg
