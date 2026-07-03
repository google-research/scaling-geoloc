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

"""End-to-end downstream eval smoke test on tiny fake data.

Runs the pv4ger classification (k-NN + linear probe) and chesapeake
segmentation (DPT) evals against the reference ViT encoder, exercising the
documented encoder interface (``get_aerial_embeddings`` /
``get_dense_aerial_embeddings``). Kept tiny: a 4-layer encoder, 1-epoch probes.
"""

import numpy as np
import jax
import ml_collections
import pytest

from unigeoclip.data import generate_fake_downstream_data as gen
from unigeoclip.scenic.evals.classif_eval import ClassificationEvaluation
from unigeoclip.scenic.evals.dsm_eval import DsmSegmentationEvaluation
from unigeoclip.scenic.evals.pdfm_eval import PDFMEvaluation
from unigeoclip.scenic.evals.semseg_eval import SegmentationEvaluation
from unigeoclip.scenic.configs import defaults
from unigeoclip.scenic.models.contrastive_merge import ContrastiveModel
from unigeoclip.scenic import trainer
from tests import test_unigeoclip_train as train_test


def _data_config(data_dir: str, name: str, valid_split: str) -> ml_collections.ConfigDict:
  cfg = ml_collections.ConfigDict()
  cfg.dataset_name = name
  cfg.data_dir = f'{data_dir}/{name}'
  cfg.train_split = 'train'
  cfg.valid_split = valid_split
  cfg.shuffle_train = False
  cfg.shuffle_valid = False
  cfg.grain_num_workers = 0
  cfg.grain_num_workers_valid = 0
  cfg.grain_num_threads = 1
  cfg.grain_num_threads_valid = 1
  cfg.grain_prefetch_buffer_size = 1
  cfg.preprocess = ml_collections.ConfigDict()
  return cfg


def _eval_config(data_cfg, num_classes: int) -> ml_collections.ConfigDict:
  cfg = ml_collections.ConfigDict()
  cfg.batch_size = 2
  cfg.batch_size_eval = 2
  cfg.log_progress_interval = 100
  cfg.num_classes = num_classes
  cfg.batch_sizes_eval = [2]
  cfg.learning_rates = [1e-3]
  # DPT segmentation uses a 1-epoch warmup, so n_epochs must be > 1 for a
  # positive cosine-decay horizon.
  cfg.n_epochs = 2
  cfg.k_values = (1,)
  cfg.data = data_cfg
  return cfg


@pytest.fixture(scope='module')
def fake_datasets(tmp_path_factory):
  out = str(tmp_path_factory.mktemp('downstream'))
  rng = np.random.default_rng(0)
  gen._write_pv4ger(out, image_size=64, rng=rng)
  gen._write_chesapeake(out, size=32, num_classes=7, rng=rng)
  return out


def _contrastive_state(model_cfg):
  """Initialises a ContrastiveModel for the eval harness (no training needed)."""
  model = ContrastiveModel(model_cfg, {})
  exp = trainer.Experiment.init(
      rng=jax.random.PRNGKey(0),
      config=train_test._train_config(model_cfg),
      model=model,
      dummy_input=train_test._make_batch(),
  )
  return model, exp.train_state


@pytest.fixture(scope='module')
def vit_aerial_state():
  """contrastive_merge with the in-repo ViT aerial tower (pooled embeddings)."""
  return _contrastive_state(train_test._model_config())


@pytest.fixture(scope='module')
def siglip_aerial_state():
  """contrastive_merge with a SigLIP2 aerial tower for the chesapeake DPT eval.

  Chesapeake's DPT head needs multi-depth dense features, which the SigLIP
  backbone exposes via ``encode_dense(get_depths=...)`` (the in-repo ViT's dense
  path is single-depth). ``load_pretrained=False`` random-initialises the tower
  so the test needs no SigLIP2 checkpoint on disk.
  """
  model_cfg = train_test._model_config()
  model_cfg.modalities = ['sv', 'aerial']
  siglip_cfg = defaults.siglip(version='Ti', patch=16).unlock()
  siglip_cfg.model_name = 'SigLIP2 Ti/16 256'
  siglip_cfg.load_pretrained = False
  model_cfg.aerial_encoder = siglip_cfg
  return _contrastive_state(model_cfg)


def test_pv4ger_classification(fake_datasets, vit_aerial_state):
  model, state = vit_aerial_state
  cfg = _eval_config(
      _data_config(fake_datasets, 'pv4ger_classification', 'valid'),
      num_classes=2,
  )
  evaluator = ClassificationEvaluation(model, cfg)
  _, metrics, _ = evaluator.evaluate(state)
  assert 'linear_probing' in metrics
  assert 'knn_accuracy@1' in metrics


def test_chesapeake_segmentation(fake_datasets, siglip_aerial_state):
  model, state = siglip_aerial_state
  cfg = _eval_config(
      _data_config(fake_datasets, 'm-chesapeake', 'test'), num_classes=7
  )
  evaluator = SegmentationEvaluation(model, cfg)
  _, metrics, _ = evaluator.evaluate(state)
  assert 'DTP' in metrics


def test_mdas_dsm_segmentation(tmp_path):
  """MDAS: DSM dense embeddings -> linear land-use segmentation probe."""
  out = str(tmp_path)
  gen._write_mdas(out, dsm_size=512, num_classes=13, rng=np.random.default_rng(0))

  # dsm_eval needs an encoder with get_dense_dsm_embeddings -> contrastive_merge
  # with the dsm modality (dsm tower at 256px, patch 16 -> a 16x16 token grid
  # that the eval upsamples to the 32x32 label crop).
  model_cfg = train_test._model_config()
  model_cfg.modalities = ['sv', 'aerial', 'latlng', 'dsm']
  model_cfg.dsm_encoder = train_test._vit_encoder(256, imagenet_norm=False)
  model = ContrastiveModel(model_cfg, {})
  exp = trainer.Experiment.init(
      rng=jax.random.PRNGKey(0),
      config=train_test._train_config(model_cfg),
      model=model,
      dummy_input=train_test._make_batch(),
  )

  cfg = _eval_config(_data_config(out, 'mdas', 'valid'), num_classes=13)
  _, metrics, _ = DsmSegmentationEvaluation(model, cfg).evaluate(exp.train_state)
  assert 'mIoU' in metrics


def test_pdfm_regression(tmp_path):
  """PDFM: location-embedding -> socioeconomic-target ridge regression."""
  out = str(tmp_path)
  gen._write_pdfm(out, num_regions=80, num_targets=3, rng=np.random.default_rng(0))

  # PDFM needs an encoder with get_location_embeddings -> use contrastive_merge
  # (its tiny config includes the latlng modality).
  model_cfg = train_test._model_config()
  model = ContrastiveModel(model_cfg, {})
  exp = trainer.Experiment.init(
      rng=jax.random.PRNGKey(0),
      config=train_test._train_config(model_cfg),
      model=model,
      dummy_input=train_test._make_batch(),
  )

  cfg = ml_collections.ConfigDict()
  cfg.batch_size_eval = 16
  cfg.data = ml_collections.ConfigDict()
  cfg.data.data_dir = f'{out}/pdfm'
  _, metrics, _ = PDFMEvaluation(model, cfg).evaluate(exp.train_state)
  assert 'r2_mean' in metrics
