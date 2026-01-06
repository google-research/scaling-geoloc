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

"""Geo plotting."""

from collections.abc import Sequence
from typing import Any

from etils.array_types import FloatArray
from etils.array_types import StrArray
import numpy as np
import plotly.graph_objects as go

from scaling_geoloc.viz import viz3d
from google3.util.geometry.python import s2


def _rgb_to_hex(rgb_n: FloatArray['3']) -> str:
  r, g, b = (rgb_n.clip(min=0, max=0.999) * 255).astype(int).tolist()
  return '#%02x%02x%02x' % (r, g, b)


def rgb_to_hex(color: FloatArray['... 3']) -> StrArray['...']:
  return np.apply_along_axis(
      _rgb_to_hex,
      -1,
      np.array(color),
  )


class GeoPlotter:
  """Plot latlong data."""

  def __init__(
      self,
      zoom: int = 12,
      mapbox_style: str = 'carto-positron',
      colorway: Sequence[str] = viz3d._DEFAULT_COLORWAY,
      **kwargs,
  ):
    self.fig = go.Figure()
    self.fig.update_layout(
        mapbox_style=mapbox_style,
        autosize=True,
        mapbox_zoom=zoom,
        margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
        showlegend=True,
        colorway=colorway,
        legend=dict(itemsizing='constant', x=0.9, y=0.9),
        template=dict(
            layout=dict(
                colorway=colorway,
            )
        ),
        overwrite=True,
        **kwargs,
    )
    self._num_groups = 0  # for auto grouping
    self._group_to_color = {}

  def next_color(self, group: Any | None = None) -> str:
    """Returns the next color or the color for a group."""
    if group in self._group_to_color:
      return self._group_to_color[group]
    colors = self.fig.layout['template']['layout']['colorway']
    color = colors[self._num_groups % len(colors)]
    self._num_groups += 1
    if group is not None:
      self._group_to_color[group] = color
    return color

  def plot(
      self,
      latlngs: FloatArray['N 2'],
      color: str | np.ndarray | None = None,
      text: str | list[str] | None = None,
      name: str | None = None,
      legendgroup: str | None = None,
      size: int = 5,
      mode: str = 'markers',
      max_num_points: int | None = None,
      **kwargs,
  ) -> None:
    """Plot point data."""
    latlngs = np.asarray(latlngs)

    if (
        isinstance(color, np.ndarray)
        and np.ndim(color) == 2
        and color.shape[-1] == 3
    ):
      color = rgb_to_hex(color)

    elif color is None and isinstance(text, (np.ndarray, list)):
      color = np.array([self.next_color(label) for label in text])

    elif color is None and name is not None:
      color = self.next_color(name)

    if max_num_points is not None and len(latlngs) > max_num_points:
      selected = np.random.choice(
          latlngs.shape[0], size=max_num_points, replace=False
      )
      latlngs = latlngs[selected]
      if isinstance(color, np.ndarray):
        color = color[selected]
      if isinstance(text, (np.ndarray, list)):
        text = np.asarray(text)[selected]

    marker = {
        'size': size,
        'color': color,
    }

    if 'marker' in kwargs:
      marker = {**marker, **kwargs.pop('marker')}

    self.fig.add_trace(
        go.Scattermapbox(
            lat=latlngs[..., 0],
            lon=latlngs[..., 1],
            mode=mode,
            text=text,
            marker=marker,
            name=name,
            legendgroup=legendgroup if legendgroup is not None else name,
            **kwargs,
        )
    )
    center = latlngs.reshape(-1, 2).mean(0)
    self.fig.update_layout(
        mapbox_center=dict(zip(('lat', 'lon'), center)),
    )

  def plot_polygons(
      self,
      polygons: Sequence[FloatArray['N 2']],
      name: str,
      labels: Sequence[str] | None = None,
      close: bool = False,
      mode: str = 'lines',
      **kwargs,
  ) -> None:
    """Convenience method to plot multiple elements."""
    center = np.concatenate(polygons, axis=0).mean(0)

    if kwargs.get('color') is None:
      kwargs['color'] = self.next_color()
    if close:
      polygons = [np.concatenate([p, p[0:1, :]], axis=0) for p in polygons]
    # Split polygons
    polygons = [
        np.concatenate([p, [[np.nan, np.nan]]], axis=0) for p in polygons
    ]

    if labels is not None:
      if len(labels) != len(polygons):
        raise ValueError('Labels must have the same length as polygons.')
      labels = np.concatenate(
          [[labels[i] for _ in p] for i, p in enumerate(polygons)]
      )

    flattened_vertices = np.concatenate(polygons, axis=0)
    self.plot(
        flattened_vertices,
        mode=mode,
        legendgroup=name,
        legendgrouptitle_text=name,
        showlegend=True,
        name=name,
        hovertext=labels,
        **kwargs,
    )
    self.fig.update_layout(
        mapbox_center=dict(zip(('lat', 'lon'), center)),
    )

  def plot_s2cells(
      self,
      s2cell_tokens: Sequence[str],
      color: np.ndarray | str | None = 'black',
      mode: str = 'lines',
      plot_polygons: bool = True,
      name: str = 's2cells',
      marker_size: int = 8,
      **kwargs,
  ) -> None:
    """Convenience method to plot S2 cells."""

    all_s2cell_corners = []
    all_s2cell_centers = []
    for token in s2cell_tokens:
      s2cell = s2.S2Cell(s2.S2CellId.from_token(token.encode('utf-8')))
      s2cell_center = s2.S2LatLng.from_point(s2cell.center())
      all_s2cell_centers.append(
          [s2cell_center.lat.degrees, s2cell_center.lng.degrees]
      )
      corners = []
      for i in range(4):
        corner_latlng = s2.S2LatLng.from_point(s2cell.vertex(i))
        corners.append([corner_latlng.lat.degrees, corner_latlng.lng.degrees])
      all_s2cell_corners.append(np.array(corners))

    if color is None and 'text' not in kwargs:
      color = self.next_color()
    if plot_polygons:
      self.plot_polygons(
          all_s2cell_corners,
          labels=s2cell_tokens,
          name=name,
          mode=mode,
          color=color,
          close=True,
          **kwargs,
      )
    self.plot(
        np.array(all_s2cell_centers),
        hovertext=kwargs.get('labels', s2cell_tokens),
        color=color,
        size=marker_size,
        name=name,
        showlegend=True,
        visible='legendonly' if plot_polygons else True,
        opacity=1.0,
        **kwargs,
    )
