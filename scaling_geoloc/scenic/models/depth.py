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

"""Base class for monocular depth models."""

from typing import Mapping, Optional

from etils.array_types import FloatArray
import flax.linen as nn
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.models import base
from scaling_geoloc.utils import misc
from scaling_geoloc.utils import plot


class DepthModel(base.BaseModel):
  """Base class for monocular depth models. Adds losses, metrics and plots."""

  @classmethod
  def num_output_channels(cls, config: ml_collections.ConfigDict) -> int:
    """Returns the number of output channels."""
    geometry_channels = 3 if config.loss.fn == 'scene_coordinate_mse' else 1
    return geometry_channels + (config.loss.confidence_type is not None)

  @classmethod
  def logits_to_predictions(
      cls, config: ml_collections.ConfigDict, logits: FloatArray['B H W C']
  ) -> types.Predictions:
    """Converts the logits to predictions."""
    if config.loss.fn == 'scene_coordinate_mse':
      pred = {'depth': logits[..., 2:3], 'scene_coordinates': logits[..., :3]}
    else:
      pred = {'depth': logits[..., :1]}
    if config.loss.confidence_type is not None:
      confidence_logits = logits[..., -1:]
      if config.loss.confidence_type == 'semantic':
        confidence = nn.sigmoid(confidence_logits)
      elif config.loss.confidence_type == 'exp':
        confidence = 1 + jnp.exp(confidence_logits)
      else:
        raise ValueError(config.loss.confidence_type)

      pred['depth_confidence_logits'] = confidence_logits
      pred['depth_confidence'] = confidence
    return pred

  def loss_metrics_function(
      self,
      pred: types.Predictions,
      data: types.Batch,
      model_params: Optional[jnp.ndarray] = None,
  ) -> types.LossMetricsTuple:

    depth = pred['depth']
    eps = 1e-2

    norm_axes = (-4, -3, -2, -1)
    depth_gt = misc.concat_all('lidar_depth', data, axis=-4)
    depth_gt_mask = misc.concat_all('lidar_mask', data, axis=-4)
    depth_gt_mask = depth_gt_mask & (depth_gt > 0)

    delta_depth = jnp.abs((depth_gt - depth))
    delta_log_depth = jnp.abs(
        jnp.log(depth.clip(min=eps)) - jnp.log(depth_gt.clip(min=eps))
    )

    max_ratio = jnp.maximum(
        depth / depth_gt.clip(min=eps), depth_gt / depth.clip(min=eps)
    )

    pixel_metrics = {
        'depth_mae': delta_depth,
        'depth_mse': delta_depth**2,
        'log_depth_mae': delta_log_depth,
        'ratio_125': max_ratio < 1.25,
        'rel_depth_error': delta_depth / depth_gt.clip(min=eps),
    }

    gt_depth_coverage = jnp.mean(depth_gt_mask, axis=norm_axes)

    metrics = {
        'gt_depth_coverage': gt_depth_coverage,
        **{
            metric_name: misc.masked_mean(metric, depth_gt_mask, axis=norm_axes)
            for metric_name, metric in pixel_metrics.items()
        },
    }

    losses = {}
    if self.config.loss.confidence_type is None:
      loss_scale_fn = lambda x: x
    elif self.config.loss.confidence_type == 'exp':
      confid = 1.0 + jnp.exp(pred['depth_confidence_logits'])
      alpha = self.config.loss.confidence_weight
      loss_scale_fn = lambda x: x * confid - alpha * jnp.log(confid)
    elif self.config.loss.confidence_type == 'semantic':
      loss_scale_fn = lambda x: x
    else:
      raise ValueError(self.config.loss.confidence_type)

    if self.config.loss.fn in pixel_metrics:
      losses['total'] = misc.masked_mean(
          loss_scale_fn(pixel_metrics[self.config.loss.fn]),
          depth_gt_mask,
          axis=norm_axes,
      )
    elif self.config.loss.fn == 'silog':
      # https://papers.nips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf
      variance_focus = 0.5
      log_depth_mse = misc.masked_mean(
          delta_log_depth**2,
          depth_gt_mask,
          axis=norm_axes,
      )
      losses['total'] = (
          log_depth_mse - variance_focus * metrics['log_depth_mae'] ** 2
      )
    elif self.config.loss.fn == 'scene_coordinate_mse':
      scene_coordinates_gt = misc.concat_all('lidar_points', data, axis=-4)
      delta_scene_coordinates = jnp.sum(
          (scene_coordinates_gt - pred['scene_coordinates']) ** 2,
          axis=-1,
          keepdims=True,
      )
      metrics['scene_coordinates_mse'] = misc.masked_mean(
          delta_scene_coordinates,
          depth_gt_mask,
          axis=norm_axes,
      )
      losses['total'] = misc.masked_mean(
          loss_scale_fn(delta_scene_coordinates),
          depth_gt_mask,
          axis=norm_axes,
      )
    else:
      raise ValueError(f'Unknown loss function: {self.config.loss.fn}')

    pixel_mask = misc.concat_all('pixel_mask', data, axis=-4)

    if self.config.loss.confidence_type == 'semantic':
      losses['confidence_nll'] = misc.masked_mean(
          optax.sigmoid_binary_cross_entropy(
              pred['depth_confidence_logits'],
              pixel_mask > 0,
          ),
          mask=pixel_mask > -1,
          axis=norm_axes,
      )
      losses['total'] += (
          self.config.loss.confidence_weight * losses['confidence_nll']
      )
    return losses, metrics

  def visualize(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      vmax: float | None = 30.0,  # Only used when log_depth is False
      stack_plot_axis: int | None = -3,
      stack_views_horizontally: bool = False,
      max_num_plots: int = 8,
      sample_seed: int | None = None,
      log_depth: bool = True,
      stack_batch_axis: int | None = -2,
  ) -> Mapping[str, FloatArray['B H W 3']]:
    image_pixels = misc.concat_all('image_pixels', batch, axis=-4)
    depth = pred['depth']
    depth_gt = misc.concat_all('lidar_depth', batch, axis=-4)
    depth_gt_mask = misc.concat_all('lidar_mask', batch, axis=-4)
    depth_gt_mask = depth_gt_mask & (depth_gt > 0)

    depth_image = plot.apply_colormap(
        (depth if not log_depth else jnp.log10(depth))[..., 0],
        vmin=0.0,
        vmax=vmax if not log_depth else 2.0,
    )
    alpha = pred.get('depth_confidence', 1.0)
    if self.config.loss.confidence_type == 'exp':
      alpha = 1.0  # Do not apply alpha blending for exp confidences
    depth_overlay = misc.alpha_blend(image_pixels, depth_image, alpha * 0.5)

    depth_error = jnp.abs(depth - depth_gt)

    depth_error_image = plot.apply_colormap(
        jnp.log10(depth_error)[..., 0],
        mask=depth_gt_mask[..., 0],
        cmap='turbo',
        vmin=-2,
        vmax=1,
    )

    depth_error_overlay = misc.alpha_blend(
        misc.to_grayscale(image_pixels) * 0.4,
        depth_error_image,
        depth_gt_mask,
    )

    relative_depth_error = depth_error / depth_gt.clip(min=1e-2)
    relative_depth_error_image = plot.apply_colormap(
        relative_depth_error[..., 0],
        mask=depth_gt_mask[..., 0],
        cmap='turbo',
        vmin=0,
        vmax=1,
    )

    relative_depth_error_overlay = misc.alpha_blend(
        misc.to_grayscale(image_pixels) * 0.4,
        relative_depth_error_image,
        depth_gt_mask,
    )

    if stack_plot_axis:
      visualizations = {
          'depth': jnp.concatenate(
              [
                  depth_overlay,
                  depth_error_overlay,
                  relative_depth_error_overlay,
              ],
              axis=stack_plot_axis,
          )
      }
    else:
      visualizations = {
          'depth_overlay': depth_overlay,
          'relative_depth_error_overlay': relative_depth_error_overlay,
          'depth_error_overlay': depth_error_overlay,
      }

    if self.config.loss.confidence_type == 'exp':
      visualizations['confidence'] = plot.apply_colormap(
          jnp.log(pred['depth_confidence'][..., 0]),
          cmap='turbo',
      )

    stack_axis = -2 if stack_views_horizontally else -4
    visualizations = misc.map_dict(
        lambda k, v: jnp.concatenate(v.swapaxes(0, 1), axis=stack_axis),
        visualizations,
    )

    # Sample views to not overload the visualizations.
    visualizations = misc.map_dict(
        lambda k, v: misc.sample(
            v,
            max_num_plots,
            axis=0,
            rng=np.random.default_rng(sample_seed) if sample_seed else None,
        ),
        visualizations,
    )

    if stack_batch_axis:
      visualizations = misc.map_dict(
          lambda k, v: jnp.concatenate(v, axis=stack_batch_axis)[None],
          visualizations,
      )

    return visualizations
