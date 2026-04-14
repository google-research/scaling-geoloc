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

"""2D visualizations in image space."""

from collections.abc import Callable, Sequence
from typing import Any

from etils.array_types import Array
from etils.array_types import BoolArray
import matplotlib.pyplot as plt
import numpy as np
from plotly import subplots as plotly_subplots
import plotly.graph_objects as go


def plot_images(
    imgs: Sequence[np.ndarray],
    titles: Sequence[str] | None = None,
    cmaps: str = 'gray',
    dpi: int = 100,
    pad: float = 0.5,
    adaptive: bool = True,
    fig_edge_size: float = 4,
    **kwargs,
) -> plt.Figure:
  """Plot a list of images."""
  n = len(imgs)
  if not isinstance(cmaps, (list, tuple)):
    cmaps = [cmaps] * n

  if adaptive:
    ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
  else:
    ratios = [4 / 3] * n
  figsize = [sum(ratios) * fig_edge_size, fig_edge_size]
  fig, ax = plt.subplots(
      1, n, figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': ratios}
  )
  if n == 1:
    ax = [ax]
  for i in range(n):
    ax[i].imshow(imgs[i], cmap=cmaps[i], **kwargs)
    ax[i].get_yaxis().set_ticks([])
    ax[i].get_xaxis().set_ticks([])
    ax[i].set_axis_off()
    for spine in ax[i].spines.values():  # remove frame
      spine.set_visible(False)
    if titles:
      ax[i].set_title(titles[i])
  fig.tight_layout(pad=pad)
  return fig


def plot_heatmaps(
    heatmaps: Sequence[np.ndarray],
    vmin: float = 0.0,
    vmax: float | None = None,
    cmap: str = 'Spectral',
    alpha: np.ndarray | float | Sequence[np.ndarray] | Sequence[float] = 0.5,
    axes: Sequence[plt.Axes] | None = None,
) -> Sequence[plt.Artist]:
  """Plot image-like heatmaps."""
  if axes is None:
    axes = plt.gcf().axes
  artists = []
  for i, ax in enumerate(axes):
    alpha_ = alpha if isinstance(alpha, float) else alpha[i]
    art = ax.imshow(
        heatmaps[i].squeeze(-1),
        alpha=((heatmaps[i] > vmin).astype(np.float32) * alpha_).squeeze(-1),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    artists.append(art)
  return artists


class Plotter2D:
  """Helper class to plot 2D visualizations on top of images."""

  def __init__(
      self,
      images: Sequence[Array['H W 3|1']],
      height: int = 300,
      width: int | None = None,
      show_axes: bool = False,
      showlegend: bool = True,
  ):
    self.num_images = len(images) + 1
    self.images = images
    ratios = [i.shape[1] / i.shape[0] for i in images]  # W / H
    ratios = ratios + [ratios[-1]]
    self.fig = plotly_subplots.make_subplots(
        rows=1,
        cols=self.num_images,
        row_heights=[height],
        column_widths=[r * height for r in ratios],
        horizontal_spacing=0.00,
    )
    self._num_groups = 0  # For auto grouping
    self._group_to_color = {}

    for i, image in enumerate(images):
      image = image * 255 if image.dtype == 'float32' else image
      self.fig.add_trace(
          go.Image(z=image, hoverinfo='none'),
          row=1,
          col=i + 1,
      )
      self.fig.update_xaxes(
          visible=show_axes,
          row=1,
          col=i + 1,
          range=[0, image.shape[1] - 1],
          autorange=False,
      )
      self.fig.update_yaxes(
          visible=show_axes,
          row=1,
          col=i + 1,
          range=[image.shape[0] - 1, 0],
          autorange=False,
      )

    self.fig.update_layout(
        height=height,
        width=width or sum(ratios) * height,
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        legend=dict(
            orientation='v',
            yanchor='middle',
            y=0.5,
            xanchor='right',
            x=0.99,
            itemsizing='constant',
        ),
        showlegend=showlegend,
        hoverlabel_namelength=-1,
        overwrite=True,
    )

  def add_heatmap(
      self,
      idx: int,
      labels: Array['H W'] | Array['H W 1'],
      mask: BoolArray['H W'] | BoolArray['H W'] | None = None,
      colorscale: str = 'Turbo',
      showscale: bool = False,
      label_convert_fn: Callable[[Any], str] = str,
      opacity: float = 0.5,
      showlegend: bool = True,
      name: str | None = None,
      legendgroup: str | None = None,
      **kwargs,
  ):
    """Add labels to an image."""
    if labels.ndim == 3:
      labels = labels.squeeze(-1)
    if label_convert_fn:
      text_labels = np.vectorize(label_convert_fn)(labels)
    else:
      text_labels = None
    if mask is not None:
      if mask.ndim == 3:
        mask = mask.squeeze(-1)
      labels = np.array(labels.astype(np.float32))
      labels[~mask] = np.nan
    self.fig.add_trace(
        go.Heatmap(
            z=labels,
            colorscale=colorscale,
            showscale=showscale,
            legendgroup=legendgroup if legendgroup is not None else name,
            opacity=opacity,
            name=name,
            hoverongaps=False,
            showlegend=showlegend,
            customdata=np.dstack([text_labels]) if label_convert_fn else None,
            hovertemplate='Label: %{z} (%{customdata[0]})'
            if label_convert_fn
            else None,
            **kwargs,
        ),
        row=1,
        col=idx + 1,
    )

  def add_heatmaps(
      self,
      labels: Sequence[Array['H W'] | Array['H W 1']],
      mask: Sequence[BoolArray['H W'] | BoolArray['H W']] | None = None,
      colorscale: str = 'Turbo',
      label_convert_fn: Callable[[Any], str] | None = None,
      opacity: float = 0.5,
      name: str | None = None,
      legendgroup: str | None = None,
      visible: str | bool = 'legendonly',
      **kwargs,
  ):
    """Add labels to each image."""
    for i, labels_i in enumerate(labels):
      self.add_heatmap(
          i,
          labels_i,
          mask[i] if mask is not None else None,
          colorscale=colorscale,
          label_convert_fn=label_convert_fn,
          opacity=opacity,
          name=name,
          showlegend=not i,
          legendgroup=legendgroup,
          visible=visible,
          **kwargs,
      )

  def add_gridlines(
      self,
      ps: int,
      visible: str | bool = 'legendonly',
      color: str = 'grey',
      name: str = 'gridlines',
      **kwargs,
  ):
    """Add gridlines to each image (ps pixels apart)."""
    for idx, image in enumerate(self.images):
      h, w = image.shape[:2]
      coords = []
      for i in range(1, h // ps + 1):
        coords.extend([[0, i * ps], [w, i * ps], [np.nan, np.nan]])
      for i in range(1, w // ps + 1):
        coords.extend([[i * ps, 0], [i * ps, h], [np.nan, np.nan]])
      coords = np.array(coords)
      self.fig.add_trace(
          go.Scatter(
              x=coords[..., 0],
              y=coords[..., 1],
              mode='lines',
              line=dict(color=color, width=1),
              name=name,
              legendgroup=name,
              zorder=1000,
              visible=visible,
              showlegend=not idx,
              **kwargs,
          ),
          row=1,
          col=idx + 1,
      )
