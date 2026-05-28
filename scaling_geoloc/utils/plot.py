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

"""Plotting and image visualisation utilities."""

from collections.abc import Sequence

from etils.array_types import BoolArray
from etils.array_types import FloatArray
from etils.array_types import IntArray
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition

from scaling_geoloc.utils.misc import grid_indices, l2_normalize


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


def alpha_blend(
    data_i: FloatArray['...'],
    data_j: FloatArray['...'],
    alpha: FloatArray['...'] | float,
) -> FloatArray['...']:
  """Alpha blends two data sources."""
  return data_i * (1.0 - alpha) + data_j * alpha


def side_by_side(
    images: Sequence[FloatArray['... H W 3']],
) -> FloatArray['... H W 3']:
  """Concatenates images side by side."""
  return jnp.concatenate(images, axis=-3)


def features_to_rgb(
    features: FloatArray['... C'],
    normalize: bool = True,
    mask: BoolArray['...'] | None = None,
    offset: int = 0,
    renormalize: bool = True,
) -> FloatArray['... 3']:
  """Project a c-dimensional feature map to RGB colors using PCA."""
  pca = decomposition.PCA(n_components=offset + 3)
  features_flat = features.reshape([-1, features.shape[-1]])
  features_fit = features[mask] if mask is not None else features_flat
  if normalize:
    features_fit = l2_normalize(features_fit)
  features_pca = (
      pca.fit(features_fit)
      .transform(features_flat)
      .reshape(features.shape[:-1] + (pca.n_components,))
  )[..., -3:]
  if renormalize:
    features_pca = l2_normalize(features_pca)
    features_pca = (features_pca + 1) / 2
  else:
    features_pca = (features_pca - features_pca.min()) / (
        np.ptp(features_pca) + np.finfo(np.float32).eps
    )
  return features_pca


def draw_image_border(
    image: FloatArray['... H W C'],
    border_color: FloatArray['... 3'],
    border_width: int = 1,
) -> FloatArray['... H W C']:
  """Draws a border around an image."""
  h, w = image.shape[-3:-1]
  coords = grid_indices(h, w, image.shape[:-3])
  border_mask = (
      (coords < border_width).any(-1)
      | (coords[..., 0] >= w - border_width)
      | (coords[..., 1] >= h - border_width)
  )[..., None]
  return jnp.where(border_mask, border_color, image)


def label_image_border(
    image: FloatArray['... H W C'],
    labels: IntArray['...'],
    colors_per_label: FloatArray['NL 3'],
    border_width: int = 1,
) -> FloatArray['H W C']:
  """Labels an image with a colored border per label."""
  for label, label_color in enumerate(colors_per_label):
    image = jnp.where(
        labels[..., None, None, None] == label,
        draw_image_border(image, label_color, border_width=border_width),
        image,
    )
  return image
