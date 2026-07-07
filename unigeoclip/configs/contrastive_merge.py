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

"""Training config for the UniGeoCLIP multimodal contrastive model.

Public release: backbones default to the in-repo ViT (no ``big_vision``), the
text modality consumes precomputed SigLIP embeddings through a linear
projection, and the data is the procedural ``MultiModalDummyDataSource`` so the
whole pipeline runs on CPU. Override any field from the command line, e.g.::

    python -m unigeoclip.train \
        --config=unigeoclip/configs/contrastive_merge.py \
        --workdir=/tmp/wd \
        --config.num_training_steps=1000 \
        --config.batch_size_per_device=64

Pass ``--config=...:B`` (or ``S``/``L``) to select a larger ViT preset.
"""

from collections.abc import Sequence

import ml_collections
from ml_collections import config_dict

from unigeoclip.configs import defaults

_HIDDEN = {'tiny': 64, 'Ti': 192, 'S': 384, 'B': 768, 'L': 1024}
_LAYERS = {'tiny': 2, 'Ti': 12, 'S': 12, 'B': 12, 'L': 24}
_HEADS = {'tiny': 2, 'Ti': 3, 'S': 6, 'B': 12, 'L': 16}
_MODALITIES = ('sv', 'aerial', 'latlng', 'dsm', 'text')


def _vit_encoder(
    preset: str, img_size: int, *, imagenet_norm: bool
) -> ml_collections.ConfigDict:
  """A ViT image encoder config (sv / aerial / dsm)."""
  cfg = ml_collections.ConfigDict()
  cfg.name = 'vit'
  cfg.hidden_size = _HIDDEN[preset]
  cfg.patches = ml_collections.ConfigDict()
  cfg.patches.size = [16, 16]
  cfg.posembs = (img_size // 16, img_size // 16)
  cfg.num_heads = _HEADS[preset]
  cfg.mlp_dim = 4 * _HIDDEN[preset]
  cfg.num_layers = _LAYERS[preset]
  cfg.head = 'linear'
  cfg.head_hidden_dim = 2048
  cfg.head_bottleneck_dim = 256
  cfg.head_output_dim = _HIDDEN[preset]
  cfg.normalize_last_layer = True
  cfg.use_bn = False
  cfg.add_class_token = True
  cfg.n_registers = 4
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


def _rff_encoder(preset: str) -> ml_collections.ConfigDict:
  """A random-Fourier-feature transformer location encoder (latlng)."""
  cfg = ml_collections.ConfigDict()
  cfg.model_name = 'rff2'
  cfg.cell_code_dim = _HIDDEN[preset]
  cfg.output_dim = _HIDDEN[preset]
  cfg.hidden_dim = _HIDDEN[preset]
  cfg.sigmas = (2**0, 2**4, 2**8, 2**12)
  cfg.num_hidden_layers = max(2, _LAYERS[preset] // 2)
  cfg.num_heads = _HEADS[preset]
  cfg.n_registers = 8
  cfg.norm_layer = 'layernorm'
  return cfg


def model_config(
    preset: str, modalities: Sequence[str]
) -> ml_collections.ConfigDict:
  """Assembles the contrastive model config for the given modalities."""
  cfg = ml_collections.ConfigDict()
  cfg.modalities = list(modalities)
  if 'sv' in modalities:
    cfg.sv_encoder = _vit_encoder(preset, 224, imagenet_norm=True)
  if 'aerial' in modalities:
    cfg.aerial_encoder = _vit_encoder(preset, 256, imagenet_norm=False)
  if 'dsm' in modalities:
    cfg.dsm_encoder = _vit_encoder(preset, 256, imagenet_norm=False)
  if 'latlng' in modalities:
    cfg.latlng_encoder = _rff_encoder(preset)
  if 'text' in modalities:
    # ``None`` => consume precomputed SigLIP text embeddings via a linear
    # projection. Set a ``siglip_text`` encoder config to fine-tune a text tower
    # (requires the optional ``big_vision`` package).
    cfg.text_encoder = config_dict.placeholder(ml_collections.ConfigDict)
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
  cfg.do_augment = True
  cfg.crop = 'none'
  cfg.crop_size = (224, 224)
  cfg.latlng_noise_meters = 50.0
  return cfg


def data_config(preset: str) -> ml_collections.ConfigDict:
  """Multimodal data config.

  Procedural by default; set ``data_dir`` + ``location`` to read the 5-modality
  ArrayRecord shards written by ``unigeoclip.data.generate_fake_multimodal_data``.
  """
  img = 64 if preset == 'tiny' else 224
  cfg = ml_collections.ConfigDict()
  cfg.dataset_name = config_dict.placeholder(str)  # None => multimodal foundation
  # File-backed source (optional): <data_dir>/<location>/{train,valid}-*.arrayrecord
  cfg.data_dir = 'datasets'
  cfg.location = config_dict.placeholder(str)  # None => procedural dummy source
  cfg.train_split = 'train'
  cfg.valid_split = 'valid'
  cfg.num_train_examples = 1024
  cfg.num_valid_examples = 256
  cfg.num_views = 1
  cfg.sv_image_size = img
  cfg.aerial_image_size = img
  cfg.dsm_image_size = img
  cfg.text_dim = 768
  cfg.shuffle_train = True
  cfg.shuffle_valid = True
  cfg.grain_num_workers = 0
  cfg.grain_num_workers_valid = 0
  cfg.grain_num_threads = 1
  cfg.grain_num_threads_valid = 1
  cfg.grain_prefetch_buffer_size = 1
  return cfg


def get_config(preset: str = 'tiny') -> ml_collections.ConfigDict:
  """Returns the training config. ``preset`` in {tiny, Ti, S, B, L}."""
  cfg = defaults.base().unlock()
  cfg.model_name = 'contrastive_merge'
  cfg.model = model_config(preset, _MODALITIES)
  cfg.data = data_config(preset)

  # Schedule / optimizer.
  cfg.lr = ml_collections.ConfigDict({
      'learning_rate_schedule': 'compound',
      'factors': 'constant * linear_warmup * cosine_decay',
      'base_learning_rate': 2e-4,
      'warmup_steps': 1_000,
      'steps_per_cycle': 200_000,
      'total_steps': 200_000,
  })
  cfg.optimizer = ml_collections.ConfigDict({
      'optax_name': 'scale_by_adam',
      'optax_configs': ml_collections.ConfigDict({'b1': 0.9, 'b2': 0.95}),
      'weight_decay': 0.0,
  })
  cfg.max_grad_norm = 1.0
  cfg.num_training_steps = 200_000

  # Batch size: total = per-device * device_count (set in train.py).
  cfg.batch_size_per_device = 8 if preset == 'tiny' else 64
  cfg.batch_size = cfg.get_ref('batch_size_per_device') * cfg.get_ref(
      'device_count'
  )
  cfg.batch_size_eval = cfg.get_ref('batch_size_per_device') * cfg.get_ref(
      'device_count'
  )

  # Logging / checkpointing (small + cheap by default for the fake-data smoke).
  cfg.log_summary_interval = 50
  cfg.log_eval_interval = 5_000
  cfg.checkpoint_interval = 5_000
  cfg.skip_first_eval = True
  cfg.skip_eval = True
  cfg.skip_train_viz = True
  cfg.xprof = False
  cfg.checkpoint = True
  return cfg.lock()
