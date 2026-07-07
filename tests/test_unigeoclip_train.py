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

"""Train-step smoke test for the UniGeoCLIP multimodal contrastive model.

Runs on CPU with two virtual devices (set via XLA_FLAGS in pytest.ini) over the
procedural multimodal fake data, exercising the sv / aerial / latlng modalities
and a single ``Experiment.train_step`` end to end.
"""

import jax
import ml_collections
import numpy as np
import pytest
from scenic.dataset_lib import dataset_utils

from unigeoclip.data.multimodal_dummy_source import MultiModalDummyDataSource
from unigeoclip.models.contrastive_merge import ContrastiveModel
from unigeoclip import trainer

# Tiny so the test runs fast on CPU.
EMB_DIM = 32
IMAGE_SIZE = 32  # 2x2 patches with patch_size=16
BATCH_SIZE = 2
TEXT_DIM = 16
MODALITIES = ['sv', 'aerial', 'latlng']


def _vit_encoder(img_size: int, imagenet_norm: bool) -> ml_collections.ConfigDict:
  cfg = ml_collections.ConfigDict()
  cfg.name = 'vit'
  cfg.hidden_size = 48
  cfg.patches = ml_collections.ConfigDict()
  cfg.patches.size = [16, 16]
  cfg.posembs = (img_size // 16, img_size // 16)
  cfg.num_heads = 3
  cfg.mlp_dim = 96
  cfg.num_layers = 2
  cfg.head = 'linear'
  cfg.head_hidden_dim = 64
  cfg.head_bottleneck_dim = 32
  cfg.head_output_dim = EMB_DIM
  cfg.normalize_last_layer = True
  cfg.use_bn = False
  cfg.add_class_token = True
  cfg.n_registers = 2
  cfg.pooling = 'tok'
  cfg.pos_interpolation_method = 'bicubic'
  cfg.dropout_rate = 0.0
  cfg.attention_dropout_rate = 0.0
  cfg.stochastic_depth = 0.0
  cfg.use_bias = True
  cfg.pretrained_weights = None
  cfg.freeze_backbone = False
  cfg.imagenet_normalization = imagenet_norm
  cfg.train_last_layers = -1
  cfg.positional_embedding = 'rope'
  cfg.norm_layer = 'layernorm'
  cfg.head_config = ml_collections.ConfigDict()
  return cfg


def _rff_encoder() -> ml_collections.ConfigDict:
  cfg = ml_collections.ConfigDict()
  cfg.model_name = 'rff2'
  cfg.cell_code_dim = EMB_DIM
  cfg.output_dim = EMB_DIM
  cfg.hidden_dim = EMB_DIM
  cfg.sigmas = (2**0, 2**4, 2**8)
  cfg.num_hidden_layers = 2
  cfg.num_heads = 2
  cfg.n_registers = 2
  cfg.norm_layer = 'layernorm'
  return cfg


def _model_config() -> ml_collections.ConfigDict:
  cfg = ml_collections.ConfigDict()
  cfg.modalities = MODALITIES
  cfg.sv_encoder = _vit_encoder(IMAGE_SIZE, imagenet_norm=True)
  cfg.aerial_encoder = _vit_encoder(IMAGE_SIZE, imagenet_norm=False)
  cfg.latlng_encoder = _rff_encoder()
  cfg.temperature = 0.01
  cfg.learned_temperature = False
  cfg.l2_normalize = False
  cfg.l2_normalize_per_head = False
  cfg.normalize_output_embeddings = True
  cfg.compose_loss = 'InfoNCE'
  cfg.exclude_pairs = []
  cfg.latlng_proj = False
  cfg.text_proj = False
  cfg.all2all_proj = False
  cfg.do_augment = False
  cfg.crop = 'none'
  cfg.crop_size = (IMAGE_SIZE, IMAGE_SIZE)
  cfg.latlng_noise_meters = 0.0
  return cfg


def _train_config(model_cfg: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
  cfg = ml_collections.ConfigDict()
  cfg.device_count = jax.device_count()
  cfg.batch_size = BATCH_SIZE
  cfg.batch_size_eval = BATCH_SIZE
  cfg.model = model_cfg
  cfg.model_name = 'contrastive_merge'
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
  source = MultiModalDummyDataSource(
      num_examples=batch_size * 4,
      sv_image_size=IMAGE_SIZE,
      aerial_image_size=IMAGE_SIZE,
      dsm_image_size=IMAGE_SIZE,
      text_dim=TEXT_DIM,
  )
  examples = [source[i] for i in range(batch_size)]
  return jax.tree.map(lambda *xs: np.stack(xs), *examples)


@pytest.fixture(scope='module')
def model_config() -> ml_collections.ConfigDict:
  return _model_config()


@pytest.fixture(scope='module')
def dummy_batch() -> dict:
  return _make_batch()


@pytest.fixture(scope='module')
def model(model_config) -> ContrastiveModel:
  return ContrastiveModel(model_config, {})


@pytest.fixture(scope='module')
def experiment(model, dummy_batch, model_config):
  rng = jax.random.PRNGKey(0)
  return trainer.Experiment.init(
      rng=rng,
      config=_train_config(model_config),
      model=model,
      dummy_input=dummy_batch,
  )


def test_init_succeeds(experiment):
  assert experiment is not None
  assert experiment.train_state is not None
  assert len(jax.tree.leaves(experiment.train_state.params)) > 0


def test_train_step_returns_total_loss(experiment, dummy_batch):
  batch = dataset_utils.shard(dummy_batch)
  metrics, _ = experiment.train_step(batch, shard=False)
  assert 'loss/total' in metrics


def test_step_counter_increments(experiment, dummy_batch):
  before = int(jax.device_get(experiment.train_state.global_step[0]))
  batch = dataset_utils.shard(dummy_batch)
  experiment.train_step(batch, shard=False)
  after = int(jax.device_get(experiment.train_state.global_step[0]))
  assert after == before + 1


def test_generate_and_load_five_modalities(tmp_path):
  """The on-disk generator emits + round-trips all five modalities."""
  from unigeoclip.data.generate_fake_multimodal_data import (  # noqa: E501
      write_multimodal_dataset,
  )
  from unigeoclip import datasets

  write_multimodal_dataset(
      str(tmp_path / 'mm'),
      num_train=8,
      num_valid=4,
      sv_image_size=IMAGE_SIZE,
      aerial_image_size=IMAGE_SIZE,
      dsm_image_size=IMAGE_SIZE,
      text_dim=TEXT_DIM,
  )

  cfg = ml_collections.ConfigDict()
  cfg.data_dir = str(tmp_path)
  cfg.location = 'mm'
  cfg.train_split = 'train'
  cfg.valid_split = 'valid'
  cfg.shuffle_train = False
  cfg.shuffle_valid = False
  cfg.grain_num_workers = 0
  cfg.grain_num_workers_valid = 0
  cfg.grain_num_threads = 1
  cfg.grain_num_threads_valid = 1
  cfg.grain_prefetch_buffer_size = 1

  ds = datasets.build_dataset(cfg, batch_size=4, batch_size_eval=4, seed=0)
  batch = next(ds.train_iter())
  assert 'image_pixels' in batch['query']['views']         # sv
  assert {'rgb', 'center_ll'} <= set(batch['aerial'])       # aerial
  assert {'dsm', 'center_ll'} <= set(batch['dsm'])          # dsm
  assert 'latlng' in batch                                  # latlng
  assert 'siglip_embeddings' in batch                       # text
