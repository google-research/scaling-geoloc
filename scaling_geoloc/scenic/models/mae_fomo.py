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

"""MAE model from fomo."""

from collections.abc import Mapping

from etils.array_types import FloatArray
import flax.linen as nn
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils

from google3.location.visualmapping.experimental.fomo.scenic import trainer as fomo_trainer
from google3.location.visualmapping.experimental.fomo.scenic.models import mae_model
from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.models import base
from scaling_geoloc.utils import misc


class MaskedAutoEncoder(nn.Module):
  """Masked Auto Encoder model."""

  config: ml_collections.ConfigDict
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self, batch: types.Batch, *, train: bool, debug: bool = False
  ) -> types.Predictions:
    image_pixels = misc.concat_all('image_pixels', batch, axis=-4) * 2.0 - 1.0

    image_pixels_flat = jnp.reshape(
        image_pixels, (-1, *image_pixels.shape[-3:])
    )

    image_pixels_flat = misc.crop_to_multiple(
        image_pixels_flat, self.config.patch_height
    )
    h, w, c = image_pixels_flat.shape[-3:]
    n_tokens = (h // self.config.patch_height) * (w // self.config.patch_width)
    n_masked = np.floor(n_tokens * self.config.masking_rate).astype('int32')

    mae = mae_model.MaskedAutoEncoder(
        input_kernel=None,  # No downsampling
        encoder_config=self.config.encoder,
        decoder_config=self.config.decoder,
        patches_hw=(self.config.patch_height, self.config.patch_width),
        hidden_size=self.config.hidden_size,
        dropout_rate=self.config.dropout_rate,
        attention_dropout_rate=self.config.attention_dropout_rate,
        stochastic_droplayer_rate=self.config.stochastic_droplayer_rate,
        n_tokens=n_tokens + 1,
        n_masked=n_masked,
        num_image_channels=c,
        dtype=self.dtype,
    )

    x_hat, x_rep, mask = mae.encode_masked(image_pixels_flat, train=train)

    return {
        'x_hat': x_hat,
        'x_rep': x_rep,
        'token_masks': mask,
        'image_pixels': image_pixels_flat,
    }


class MaskedAutoEncoderModel(base.BaseModel):
  """Masked Auto Encoder model."""

  required_rngs: tuple[str, ...] = ('masking_rng', 'dropout')

  def build_flax_model(self) -> nn.Module:
    return MaskedAutoEncoder(
        config=self.config,
        dtype=self.dtype,
    )

  def loss_metrics_function(
      self,
      pred: types.Predictions,
      data: types.Batch,
      model_params: jnp.ndarray | None = None,
  ) -> types.LossMetricsTuple:
    x, weights = fomo_trainer._patch_inputs(  # pylint: disable=protected-access
        pred['image_pixels'],
        pred['token_masks'],
        patch_height=self.config.patch_height,
        patch_width=self.config.patch_width,
        include_unmasked_tokens_in_loss=self.config.include_unmasked_tokens_in_loss,
    )
    loss = fomo_trainer.regression_loss_with_token_masking(
        predictions=pred['x_hat'],
        targets=x,
        weights=weights,
    )

    losses = {}
    losses['total'] = loss
    return losses, {}

  def visualize(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      max_num_plots: int = 4,
      **kwargs,
  ) -> Mapping[str, FloatArray['B H W 3']]:
    visualizations = {}
    visualizations['fomo'] = fomo_trainer._compose_images(  # pylint: disable=protected-access
        dataset_utils.shard(pred['image_pixels']),
        dataset_utils.shard(pred['x_hat']),
        config=ml_collections.ConfigDict(
            dict(
                use_passthrough_input_for_unmasked_tokens=self.config.use_passthrough_input_for_unmasked_tokens,
                include_unmasked_tokens_in_loss=self.config.include_unmasked_tokens_in_loss,
                model=dict(
                    patch_height=self.config.patch_height,
                    patch_width=self.config.patch_width,
                ),
                max_images_summary=max_num_plots,
            ),
        ),
        masked_token_indices=dataset_utils.shard(pred['token_masks']),
    )

    return visualizations
