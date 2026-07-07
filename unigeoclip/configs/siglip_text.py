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

"""Config file for a pretrained SigLIP2 backbone for text modality."""

import ml_collections

_SIZES = {'Ti': 192, 'S': 384, 'B': 768, 'L': 1024, 'So400m': 1152, 'H': 1280}


def get_config(
    model_size: str = 'B',
) -> ml_collections.ConfigDict:
  """Returns the config of the siglip backbone."""
  config = ml_collections.ConfigDict()
  config.variant = model_size
  config.pool_type = 'last'
  config.vocab_size = 256_000
  config.output_dim = _SIZES[model_size]
  config.seqlen = 64
  if model_size == 'B':
    config.model_name = 'SigLIP2 B/16 224'
  elif model_size == 'So400m':
    config.model_name = 'SigLIP2 So400m/16 256'
  else:
    raise ValueError(f'Unsupported model size: {model_size}')
  config.name = 'siglip_text'

  return config
