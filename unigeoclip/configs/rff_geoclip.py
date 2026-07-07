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

"""Config file for a RFF Encoder on latlng modality."""

import ml_collections


_SIZES = {'Ti': 192, 'S': 384, 'B': 768, 'L': 1024, 'So400m': 1152, 'H': 1280}


def get_config(
    model_size: str = 'B',
) -> ml_collections.ConfigDict:
  """Returns the config of the RFF Encoder."""
  size = _SIZES[model_size]

  config = ml_collections.ConfigDict()
  config.cell_code_dim = size
  # Sigma controls the spread of sampled frequencies. However, due to
  # coordinate transformations, it is difficult to directly map this parameter
  # to a specific spatial resolution.
  config.sigmas = (2**-4, 2**0, 2**4, 2**8, 2**12)
  config.hidden_dim = size
  config.output_dim = size

  config.model_name = 'rff_geoclip'
  return config
