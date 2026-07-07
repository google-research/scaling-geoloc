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

"""Config file for a ViT backbone for aerial modlaity."""

import ml_collections

from unigeoclip.configs import defaults

_SIZES = {'Ti': 192, 'S': 384, 'B': 768, 'L': 1024, 'H': 1280}


def get_config(
    model_size: str = 'B',
) -> ml_collections.ConfigDict:
  """Returns the config of the SV ViT."""
  config = defaults.vit(
      version=model_size,
      patch=16,
      output_dim=_SIZES[model_size],
      pooling='tok',
      img_size=256,
  ).unlock()

  config.head = 'linear'
  config.imagenet_normalization = False

  return config
