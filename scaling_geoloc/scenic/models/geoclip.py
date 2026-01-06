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

"""Reinterpretation of the GeoClip model (WARNING: not fully tested!)."""

from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.models import location_encoder
from scaling_geoloc.scenic.models import sv_aerial
from scaling_geoloc.utils import misc


class GeoCLIP(sv_aerial.SVAerialRetrieval):
  """GeoClip model."""

  @nn.compact
  def __call__(
      self,
      batch: types.Batch,
      *,
      train: bool,
      debug: bool = False,
  ) -> types.Predictions:
    pred = super().__call__(batch, train=train, debug=debug)
    batch = sv_aerial.flatten_batch(batch)
    location_features = location_encoder.LocationEncoder(
        output_dim=self.config.cell_code_dim,
        sigmas=self.config.get('sigmas', (2**4, 2**8, 2**12)),
        hidden_dim=self.config.hidden_dim,
        num_hidden_layers=self.config.num_hidden_layers,
    )(batch['latlng'])
    if self.config.get('l2_normalize', False):
      location_features = misc.l2_normalize(location_features)
    pred['location_features'] = location_features

    pred['sv_loc_probs'] = self.stochastic_probs(
        location_features,
        batch['latlng'],
        pred['logits'],
        batch['latlng'],
        batch,
        target_name='sv_loc',
    )
    pred['aerial_loc_probs'] = self.stochastic_probs(
        location_features,
        batch['latlng'],
        pred['aerial_logits_p'],
        batch['latlng'],
        batch,
        target_name='aerial_loc',
    )
    return pred


class GeoCLIPModel(sv_aerial.SVAerialRetrievalModel):
  """GeoClip model."""

  def build_flax_model(self) -> nn.Module:
    self.s2cell_centers_ll = jnp.asarray(
        self.dataset_meta_data['s2cell_centers_ll'], dtype=self.dtype
    )
    return GeoCLIP(
        self.config,
        self.dtype,
        cell_centers=self.s2cell_centers_ll,
    )

  def _loss_metrics_function(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      model_params: jnp.ndarray | None = None,
      model_state: Any | None = None,
      learning_rates: types.LearningRates | None = None,
  ) -> types.LossMetricsTuple:
    losses, metrics = super()._loss_metrics_function(
        pred,
        batch,
    )

    svloc_losses, svloc_metrics = self.stochastic_loss_metrics_function(
        pred['sv_loc_probs'], batch, prefix='sv_loc_'
    )
    aerial_loc_losses, aerial_loc_metrics = (
        self.stochastic_loss_metrics_function(
            pred['aerial_loc_probs'], batch, prefix='aerial_loc_'
        )
    )
    losses.update(svloc_losses)
    metrics.update(svloc_metrics)
    losses.update(aerial_loc_losses)
    metrics.update(aerial_loc_metrics)

    return losses, metrics
