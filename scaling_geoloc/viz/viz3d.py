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

"""3D plotting utils."""

from collections.abc import Callable, Sequence
from typing import Any, Final

from etils.array_types import Array
from etils.array_types import FloatArray
import jax
import numpy as np
import plotly
import plotly.graph_objects as go

from scaling_geoloc.viz import utils

_DEFAULT_COLORWAY: Final[Sequence[str]] = sum(
    [
        getattr(plotly.colors.qualitative, name)
        for name in ['Plotly', 'Light24', 'Dark24']
    ],
    [],
)


class Plotter3D:
  """3D plotting tool for cameras and point clouds.

  Attributes:
    fig: The plotly figure. Can be customized directly.
    scene_t_w: The SE(3) transformation from an arbitrary world coordinate
      system to the scene coordinate system for plotting. All points and cameras
      are assumed to be in world coordinates and will betransformed to scene
      coordinates for plotting.
  """

  def __init__(
      self,
      height: int | None = 600,
      width: int | None = None,
      show_axes: bool = False,
      scene_t_w: FloatArray['4 4'] = np.eye(4),
      colorway: Sequence[str] = _DEFAULT_COLORWAY,
  ):
    """Initializes the 3D plotter.

    Args:
      height: The height of the figure in px (auto if None).
      width: The width of the figure in px (auto if None).
      show_axes: Whether to show the xyz axes.
      scene_t_w: The SE(3) transformation from an arbitrary world coordinate
        system to the scene coordinate system for plotting. Defaults to
        identity.
      colorway: The colorway to use for plotting.
    """
    self.fig = go.Figure()
    axes = dict(
        visible=show_axes,
    )
    self.fig.update_layout(
        height=height,
        width=width,
        scene_camera=dict(
            eye=dict(x=0.0, y=-0.1, z=-2),
            up=dict(x=0, y=-1.0, z=0),
            projection=dict(type='orthographic'),
        ),
        scene=dict(
            xaxis=axes,
            yaxis=axes,
            zaxis=axes,
            aspectmode='data',
            dragmode='orbit',
        ),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        legend=dict(
            orientation='h',
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.1,
            itemsizing='constant',
        ),
        hoverlabel_namelength=-1,
        colorway=colorway,
        template=dict(
            layout=dict(
                colorway=colorway,
            )
        ),
        overwrite=True,
    )
    self.scene_t_w = scene_t_w
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

  def add_camera(
      self,
      w_t_cam: FloatArray['4 4'],
      intrinsics: FloatArray['3 3'],
      color: str | None = None,
      name: str | None = None,
      legendgroup: str | None = None,
      size: float = 1.0,
      line_width: float = 5,
      text: str | None = None,
      showlegend: bool = False,
      **kwargs,
  ):
    """Plot a camera frustum from pose and intrinsic matrix."""
    scene_t_cam = self.scene_t_w @ w_t_cam
    scene_r_cam, scene_p_cam = scene_t_cam[:3, :3], scene_t_cam[:3, 3]
    width, height = intrinsics[0, 2] * 2, intrinsics[1, 2] * 2
    corners = np.array([
        [0, 0, 1],
        [width, 0, 1],
        [width, height, 1],
        [0, height, 1],
        [0, 0, 1],
    ])
    if size is not None:
      image_extent = max(size * width / 1024.0, size * height / 1024.0)
      world_extent = (
          max(width, height) / (intrinsics[0, 0] + intrinsics[1, 1]) / 0.5
      )
      scale = 0.5 * image_extent / world_extent
    else:
      scale = 1.0
    corners = corners @ np.linalg.inv(intrinsics).T
    corners = (corners / 2 * scale) @ scene_r_cam.T + scene_p_cam
    legendgroup = legendgroup if legendgroup is not None else name

    # Pyramid faces defined by vertices[i], vertices[j] and vertices[k].
    i = [0, 0, 0, 0]
    j = [1, 2, 3, 4]
    k = [2, 3, 4, 1]

    triangles = np.vstack((i, j, k)).T
    vertices = np.concatenate(([scene_p_cam], corners))
    tri_points = np.array([vertices[i] for i in triangles.reshape(-1)])
    x, y, z = tri_points.T

    color = color if color is not None else self.next_color()
    pyramid = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='lines',
        name=name,
        legendgroup=legendgroup if legendgroup is not None else name,
        line=dict(color=color, width=line_width),
        showlegend=showlegend,
        hovertemplate=text.replace('\n', '<br>') if text else None,
        **kwargs,
    )
    self.fig.add_trace(pyramid)

  def add_cameras(
      self,
      w_t_cams: FloatArray['N 4 4'],
      intrinsics: FloatArray['N 3 3'],
      **kwargs,
  ):
    """Plot a set of cameras."""
    for i in range(w_t_cams.shape[0]):
      self.add_camera(
          w_t_cams[i],
          intrinsics[i],
          **kwargs,
      )

  def add_points(
      self,
      w_points: FloatArray['N 3'],
      color: str | np.ndarray | None = None,
      name: str | None = None,
      legendgroup: str | None = None,
      max_num_samples: int | None = 10_000,
      colorscale: str | None = 'Turbo',
      cmin: float | None = None,
      cmax: float | None = None,
      showscale: bool = False,
      size: int = 1,
      hoverinfo: str = 'skip',
      **kwargs,
  ):
    """Plot a camera frustum from pose and intrinsic matrix."""
    if max_num_samples and max_num_samples < w_points.shape[0]:
      rng = np.random.default_rng(0)
      idx = rng.choice(w_points.shape[0], size=max_num_samples, replace=False)
      w_points = w_points[idx]
      if color is not None or not isinstance(color, str):
        color = color[idx]

    scene_points = w_points @ self.scene_t_w[:3, :3].T + self.scene_t_w[:3, -1]
    self.fig.add_traces(
        go.Scatter3d(
            x=scene_points[:, 0],
            y=scene_points[:, 1],
            z=scene_points[:, 2],
            mode='markers',
            marker=dict(
                size=size,
                color=color if color is not None else self.next_color(),
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                opacity=1,
                showscale=showscale,
                line=dict(width=0),
            ),
            name=name,
            legendgroup=legendgroup if legendgroup is not None else name,
            hoverinfo=hoverinfo,
            **kwargs,
        )
    )

  def add_labelled_points(
      self,
      w_points: FloatArray['N 3'],
      labels: Array['N'],
      label_convert_fn: Callable[[Any], str] = str,
      legendgroup: str | None = None,
      max_num_samples: int = 10_000,
      **kwargs,
  ):
    """Plot a set of labeled points."""
    if 'color' in kwargs:
      raise ValueError('Color is autodetermined by label and cannot be set.')
    unique_labels = sorted(np.unique(labels))
    if max_num_samples and max_num_samples < w_points.shape[0]:
      rng = np.random.default_rng(0)
      idx = rng.choice(w_points.shape[0], size=max_num_samples, replace=False)
      w_points = w_points[idx]
      labels = labels[idx]

    for label in unique_labels:
      label_name = label_convert_fn(label)
      self.add_points(
          w_points[labels == label],
          name=label_name,
          legendgroup=legendgroup if legendgroup is not None else label_name,
          max_num_samples=None,
          showlegend=True,
          color=self.next_color(label_name),
          **kwargs,
      )

  def add_voxel_grid(
      self,
      grid_xyz: FloatArray['N 3'],
      w_t_grid: FloatArray['4 4'],
      cell_size: float,
      color: str | np.ndarray | None = 'black',
      contour_visible: str | bool = True,
      add_mesh: bool = False,
      mesh_color: Array['N 3'] | str | None = None,
      mesh_visible: str | bool = 'legendonly',
      name: str | None = None,
      legendgroup: str | None = None,
      transform: bool = True,
      **kwargs,
  ):
    """Add sparse voxel grid."""
    grid_vertices, indices = utils.triangulate_cube_faces(grid_xyz, cell_size)

    if transform:
      scene_t_grid = self.scene_t_w @ w_t_grid
    else:
      scene_t_grid = w_t_grid
    scene_vertices = (
        grid_vertices @ scene_t_grid[:3, :3].T + scene_t_grid[:3, -1]
    )
    name = name + '_grid' if name is not None else 'grid'

    triangles = np.vstack(indices).T
    tri_points = scene_vertices[triangles]

    c_name = name + '_contour' if name else 'contour'
    c_legendgroup = legendgroup if legendgroup is not None else c_name
    contour_color = color or self.next_color()
    xe = []
    ye = []
    ze = []
    for tri in tri_points:
      xe.extend([tri[k % 3][0] for k in range(4)] + [None])
      ye.extend([tri[k % 3][1] for k in range(4)] + [None])
      ze.extend([tri[k % 3][2] for k in range(4)] + [None])

    self.fig.add_traces(
        go.Scatter3d(
            x=xe,
            y=ye,
            z=ze,
            mode='lines',
            name=c_name,
            legendgroup=c_legendgroup,
            visible=contour_visible,
            line=dict(color=contour_color, width=1),
            showlegend=True,
            **kwargs,
        )
    )

    face_color = None
    if add_mesh:
      if isinstance(mesh_color, str) and mesh_color == 'same':
        mesh_color = contour_color
      elif mesh_color is None:
        mesh_color = self.next_color()
      elif not isinstance(mesh_color, str):
        face_color = jax.lax.collapse(
            np.repeat(
                mesh_color[:, None],
                axis=1,
                repeats=12,  # 6 * 2 triangles per cube
            ),
            0,
            2,
        )
        mesh_color = None
      self.fig.add_traces(
          go.Mesh3d(
              x=scene_vertices[:, 0],
              y=scene_vertices[:, 1],
              z=scene_vertices[:, 2],
              i=indices[0],
              j=indices[1],
              k=indices[2],
              flatshading=True,
              name=name,
              color=mesh_color,
              facecolor=face_color,
              showlegend=True,
              visible=mesh_visible,
              legendgroup=legendgroup if legendgroup is not None else name,
              **kwargs,
          )
      )

      self.fig.data[-1].update(
          lighting=dict(
              ambient=0.5, diffuse=1, fresnel=4, specular=0.5, roughness=0.5
          )
      )
