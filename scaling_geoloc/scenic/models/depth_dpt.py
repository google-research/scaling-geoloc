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

"""Predict per-point occupancy from an image-based scene representation."""

import functools

import flax.linen as nn
import jax.numpy as jnp
import ml_collections

from scaling_geoloc.scenic import types
import scaling_geoloc.scenic.configs.defaults as default_configs
from scaling_geoloc.scenic.models import dpt
from scaling_geoloc.scenic.models.depth import DepthModel  # pylint: disable=g-importing-member
from scaling_geoloc.utils import misc
from google3.pyglib import classproperties


class DepthDPT(nn.Module):
  """Predict monocular depth from an image."""

  config: ml_collections.ConfigDict
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    norm = functools.partial(nn.GroupNorm, epsilon=1e-6)
    self.image_encoder = dpt.ViTEncoder(
        mlp_dim=self.config.image_encoder.mlp_dim,
        num_layers=self.config.image_encoder.num_layers,
        num_heads=self.config.image_encoder.num_heads,
        patch_size=self.config.image_encoder.patch_size,
        hidden_size=self.config.image_encoder.hidden_size,
        dropout_rate=self.config.image_encoder.dropout_rate,
        attention_dropout_rate=self.config.image_encoder.attention_dropout_rate,
        return_layers=self.config.image_encoder.return_layers,
        padding=self.config.image_encoder.padding,
        dtype=self.dtype,
    )

    self.decoder = dpt.DPTHead(
        num_classes=DepthModel.num_output_channels(self.config),
        block_size=self.config.decoder.block_size,
        block_scale=self.config.decoder.block_scale,
        feature_dim=self.config.decoder.feature_dim,
        num_convs=self.config.decoder.num_convs,
        padding=self.config.decoder.padding,
        norm=norm,
    )

  def __call__(
      self, data: types.Batch, train: bool = False, debug: bool = False
  ) -> types.Predictions:
    # get all image pixels from the query and map views
    image_pixels = misc.concat_all('image_pixels', data, axis=-4)
    h, w = image_pixels.shape[-3:-1]

    image_pixels_flat = jnp.reshape(
        image_pixels, (-1, *image_pixels.shape[-3:])
    )  # [BN, H, W, 3]
    _, layers = self.image_encoder(image_pixels_flat, train=train)

    logits_flat = self.decoder(layers)
    logits = jnp.reshape(
        logits_flat,
        image_pixels.shape[:-3] + logits_flat.shape[-3:],
    )[..., :h, :w, :]

    return DepthModel.logits_to_predictions(self.config, logits)

  @classproperties.classproperty
  @classmethod
  def default_config(cls) -> ml_collections.ConfigDict:
    return default_configs.depth_unet()


class DepthDPTModel(DepthModel):
  """Trainer-facing wrapper for the DepthDPT model."""

  def build_flax_model(self) -> nn.Module:
    return DepthDPT(self.config, self.dtype)

  @classmethod
  def default_flax_model_config(cls) -> ml_collections.ConfigDict:
    return DepthDPT.default_config  # pytype: disable=bad-return-type
