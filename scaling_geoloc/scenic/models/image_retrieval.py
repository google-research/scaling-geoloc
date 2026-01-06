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

"""Base class for image retrieval models."""

from typing import Mapping, Optional

from absl import logging
from etils.array_types import FloatArray
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from scaling_geoloc.scenic import load_params
from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.models import base
from scaling_geoloc.scenic.models import heads
from scaling_geoloc.scenic.models import vit


class ImageRetrievalModel(base.BaseModel):
  """Base class for image retrieval models."""

  required_rngs: tuple[str, ...] = ('dropout',)

  def build_flax_model(self) -> nn.Module:
    return vit.ImageRetrievalViT(
        config=self.config.get('model', self.config),
        dtype=self.dtype,
    )

  def loss_metrics_function(
      self,
      pred: types.Predictions,
      data: types.Batch,
      model_params: Optional[jnp.ndarray] = None,
  ) -> types.LossMetricsTuple:

    losses = {}
    metrics = {}
    return losses, metrics

  def visualize(
      self,
      pred: types.Predictions,
      batch: types.Batch,
  ) -> Mapping[str, FloatArray['B H W 3']]:
    visualizations = {}
    return visualizations

  def init_state(
      self, model_params: types.ModelParams, model_state: types.ModelState
  ) -> tuple[types.ModelParams, types.ModelState]:
    pretrained_weights = self.config.vit.get('pretrained_weights')
    pretrained_checkpoint = self.config.vit.get('pretrained_checkpoint')
    if pretrained_weights is not None or pretrained_checkpoint is not None:
      if pretrained_weights:
        logging.info(
            'Loading pretrained teacher weights: %s', pretrained_weights
        )
      else:
        logging.info(
            'Loading pretrained teacher checkpoint: %s', pretrained_checkpoint
        )

      pretrained_params = flax.core.unfreeze(
          load_params.load_params(
              pretrained_weights,
              pretrained_checkpoint,
              model_params['backbone'],
          )
      )

      def check_shape_equal(x: FloatArray, y: FloatArray) -> bool:
        is_shape_equal = x.shape == y.shape
        if not is_shape_equal:
          raise ValueError(f'Shapes are not identical: {x.shape} != {y.shape}')
        return is_shape_equal

      jax.tree.map(
          check_shape_equal,
          model_params['backbone'],
          pretrained_params,
      )
      model_params['backbone'] = pretrained_params

      aggregator_params = heads.SALAD.load_pretrained_params()
      jax.tree.map(
          check_shape_equal,
          model_params['aggregator'],
          aggregator_params,
      )
      model_params['aggregator'] = aggregator_params
    return model_params, model_state
