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

"""Predict per-point depth with a UNet."""

import flax.linen as nn
import jax.numpy as jnp
import ml_collections

from scaling_geoloc.scenic import types
import scaling_geoloc.scenic.configs.defaults as default_configs
from scaling_geoloc.scenic.models import unet
from scaling_geoloc.scenic.models.depth import DepthModel  # pylint: disable=g-importing-member
from scaling_geoloc.utils import misc
from google3.pyglib import classproperties


class DepthUNet(nn.Module):
  """Predict monocular depth from an image."""

  config: ml_collections.ConfigDict
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.image_encoder = unet.UNet(
        num_classes=DepthModel.num_output_channels(self.config),
        padding=self.config.image_encoder.padding,
        norm=self.config.image_encoder.norm,
        output_norm=self.config.image_encoder.output_norm,
        block_size=self.config.image_encoder.block_size,
    )

  def __call__(
      self, data: types.Batch, train: bool = False, debug: bool = False
  ) -> types.Predictions:

    # get all image pixels from the query and map views
    image_pixels = misc.concat_all('image_pixels', data, axis=-4)

    image_pixels_flat = jnp.reshape(
        image_pixels, (-1, *image_pixels.shape[-3:])
    )  # [BN, H, W, 3]
    logits_flat = self.image_encoder(image_pixels_flat, train=train)
    logits = jnp.reshape(
        logits_flat,
        image_pixels.shape[:-3] + logits_flat.shape[-3:],
    )

    return DepthModel.logits_to_predictions(self.config, logits)

  @classproperties.classproperty
  @classmethod
  def default_config(cls) -> ml_collections.ConfigDict:
    return default_configs.depth_unet()


class DepthUNetModel(DepthModel):
  """Trainer-facing wrapper for the DepthUNet model."""

  def build_flax_model(self) -> nn.Module:
    return DepthUNet(self.config, self.dtype)

  @classmethod
  def default_flax_model_config(cls) -> ml_collections.ConfigDict:
    return DepthUNet.default_config  # pytype: disable=bad-return-type
