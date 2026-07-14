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

"""Procedural multimodal fake data for UniGeoCLIP contrastive training.

Each example bundles the five UniGeoCLIP modalities co-located at one point on
Earth: street-view (``sv``), aerial RGB, a DSM height tile, the lat/lng itself,
and a precomputed SigLIP *text* embedding. The nested layout matches exactly
what ``contrastive_merge.flatten_batch`` / ``process_batch`` consume after Grain
batches a list of these dicts (leading batch dim ``B`` added by Grain):

    {
      'latlng':  float32 [2],                          # shared ground-truth lat/lng
      'query':  {'image': float32 [Hs, Ws, 3], 'latlng': float32 [2]},  # -> sv
      'aerial': {'image': float32 [Ha, Wa, 3], 'latlng': float32 [2]},
      'dsm':    {'image': float32 [Hd, Wd],    'latlng': float32 [2]},
      'text':   {'siglip_embeddings': float32 [Dt]},   # -> text (no text backbone)
    }

Image pixels are float32 in ``[0, 1]`` (the normalization a real loader would
apply), so no extra transform is required to feed the ViT backbones. Example
``i`` is fully determined by seed ``i`` so the source supports Grain's
random-access interface.
"""

import dataclasses

import grain.python as grain
import numpy as np


@dataclasses.dataclass
class MultiModalDummyDataSource(grain.RandomAccessDataSource):
  """Random-access multimodal dummy source for contrastive training/tests.

  Attributes:
    num_examples: Number of examples the source exposes.
    sv_image_size: Street-view image height/width.
    aerial_image_size: Aerial tile height/width.
    dsm_image_size: DSM tile height/width.
    text_dim: Dimensionality of the precomputed SigLIP text embedding.
  """

  num_examples: int = 64
  sv_image_size: int = 32
  aerial_image_size: int = 32
  dsm_image_size: int = 32
  text_dim: int = 64

  def __len__(self) -> int:
    return self.num_examples

  def __getitem__(self, index: int) -> dict:
    rng = np.random.default_rng(seed=index)
    # One ground-truth location shared by every co-located modality.
    latlng = rng.uniform([-90.0, -180.0], [90.0, 180.0]).astype(np.float32)

    sv = rng.random(
        (self.sv_image_size, self.sv_image_size, 3),
        dtype=np.float32,
    )
    aerial = rng.random(
        (self.aerial_image_size, self.aerial_image_size, 3), dtype=np.float32
    )
    dsm = rng.standard_normal(
        (self.dsm_image_size, self.dsm_image_size)
    ).astype(np.float32)
    siglip_embeddings = rng.standard_normal(self.text_dim).astype(np.float32)

    return {
        'latlng': latlng,
        'query': {'image': sv, 'latlng': latlng.copy()},
        'aerial': {'image': aerial, 'latlng': latlng.copy()},
        'dsm': {'image': dsm, 'latlng': latlng.copy()},
        'text': {'siglip_embeddings': siglip_embeddings},
    }
