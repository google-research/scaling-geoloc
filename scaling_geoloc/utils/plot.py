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

"""Plotting utilities with matplotlib dependencies."""


from etils.array_types import BoolArray
from etils.array_types import FloatArray
import jax.numpy as jnp
import matplotlib.pyplot as plt


def apply_colormap(
    data: FloatArray['...'],
    mask: BoolArray['...'] | None = None,
    cmap: str = 'turbo_r',
    vmin: float | None = None,
    vmax: float | None = None,
) -> FloatArray['... 3']:
  """Applies a colormap to each element in an array."""
  clipped_data = jnp.clip(data, vmin, vmax)
  if vmin is None:
    vmin = (
        jnp.min(clipped_data) if mask is None else jnp.min(clipped_data[mask])
    )
  if vmax is None:
    vmax = (
        jnp.max(clipped_data) if mask is None else jnp.max(clipped_data[mask])
    )
  normalized_data = (clipped_data - vmin) / max(vmax - vmin, 1.0e-10)

  colors = plt.get_cmap(cmap)(normalized_data)[..., :3].astype(data.dtype)

  if mask is not None:
    colors = colors * mask[..., None].astype(colors.dtype)
  return colors
