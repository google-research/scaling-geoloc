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

"""Shared fixtures for training and inference tests."""

import jax
import ml_collections
import numpy as np
import pytest

from scaling_geoloc.configs import defaults
from scaling_geoloc.data.dummy_data_source import DummyDataSource
from scaling_geoloc.models.hybrid_geolocalization import GeoLocalizerModel

# Keep everything tiny so tests run fast on CPU.
NUM_CELLS = 32
BATCH_SIZE = 2
IMAGE_SIZE = 32  # 2×2 patches with patch_size=16


def _make_model_config() -> ml_collections.ConfigDict:
    cfg = ml_collections.ConfigDict()
    # Smallest ViT variant; no pretrained weights so init is random.
    cfg.vit = defaults.vit('Ti', 16, output_dim=64, pooling='tok', bottleneck_dim=None)
    # Aerial backbone is vit + this patch (empty = same as query backbone).
    cfg.vit_aerial = ml_collections.ConfigDict()
    # Simple loss that is always computed (no multi-similarity path).
    cfg.compose_loss = 'query_sim_l2 + aerial_sim_l2'
    return cfg


def _make_train_config(model_cfg: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
    cfg = ml_collections.ConfigDict()
    cfg.device_count = jax.device_count()
    cfg.batch_size = BATCH_SIZE
    cfg.batch_size_eval = BATCH_SIZE
    cfg.model = model_cfg
    cfg.model_name = 'hybrid_geolocalization'
    cfg.lr = ml_collections.ConfigDict({
        'learning_rate_schedule': 'compound',
        'factors': 'constant',
        'base_learning_rate': 1e-3,
        'start_decay_step': 0,
        'steps_per_cycle': 100,
    })
    cfg.optimizer = ml_collections.ConfigDict({
        'optax_name': 'scale_by_adam',
        'optax_configs': ml_collections.ConfigDict({'b1': 0.9, 'b2': 0.95}),
        'weight_decay': 0.0,
    })
    cfg.rate_fns = {}
    cfg.max_grad_norm = None
    cfg.num_training_steps = 10
    cfg.debug_train = False
    cfg.debug_eval = False
    cfg.avoid_weight_decay_on_frozen_params = False
    cfg.overfit = False
    return cfg


def _make_batch(batch_size: int = BATCH_SIZE) -> dict:
    source = DummyDataSource(
        num_examples=batch_size * 4,
        num_cells=NUM_CELLS,
        sv_image_size=IMAGE_SIZE,
        aerial_image_size=IMAGE_SIZE,
    )
    examples = [source[i] for i in range(batch_size)]
    batch = jax.tree.map(lambda *xs: np.stack(xs), *examples)
    # Normalize uint8 → float32 [0, 1] (normally done by NormalizePixels transform).
    batch['query']['image'] = batch['query']['image'].astype(np.float32) / 255.0
    batch['aerial']['image'] = batch['aerial']['image'].astype(np.float32) / 255.0
    return batch


@pytest.fixture(scope='session', autouse=True)
def disable_jit():
    with jax.disable_jit():
        yield


@pytest.fixture(scope='session')
def cell_centers_ll() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform([-90, -180], [90, 180], (NUM_CELLS, 2)).astype(np.float32)


@pytest.fixture(scope='session')
def model_config() -> ml_collections.ConfigDict:
    return _make_model_config()


@pytest.fixture(scope='session')
def train_config(model_config) -> ml_collections.ConfigDict:
    return _make_train_config(model_config)


@pytest.fixture(scope='session')
def dataset_meta_data(cell_centers_ll) -> dict:
    batch = _make_batch()
    return {
        'cell_centers_ll': cell_centers_ll,
        'num_train_examples': 16,
        'num_valid_examples': 4,
        'dummy_batch_fn': lambda: batch,
    }


@pytest.fixture(scope='session')
def geo_model(model_config, dataset_meta_data) -> GeoLocalizerModel:
    return GeoLocalizerModel(model_config, dataset_meta_data)


@pytest.fixture(scope='class')
def dummy_batch() -> dict:
    return _make_batch()
