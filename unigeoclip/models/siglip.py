# Copyright 2026 Google LLC.
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

"""SigLIP2 model for image embeddings."""

from typing import Any, Sequence

from big_vision.models import vit
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt
import ml_collections


class PreTrainedSigLip2Model(nn.Module):
  """SigLIP2 model."""

  variant: str = "B"
  pool_type: str = "map"
  head_zeroinit: bool = False
  model_name: str = "SigLIP2 B/16 224"
  norm_input: bool = False

  def setup(self):
    self.imagenet_mean = jnp.array([0.485, 0.456, 0.406])
    self.imagenet_std = jnp.array([0.229, 0.224, 0.225])
    self.model = vit.Model(
        variant=self.variant,
        pool_type=self.pool_type,
        head_zeroinit=self.head_zeroinit,
    )

  def __call__(
      self, image: jt.Float[jt.Array, "... H W 3"], *, train: bool = False
  ) -> jt.Float[jt.Array, "... D"]:
    """Computes the image embeddings for the given image.

    Args:
      image: Batch of images to compute embeddings for.
      train: Whether to use training mode.

    Returns:
      The image embeddings.
    """
    if self.norm_input:
      image = (image - self.imagenet_mean) / self.imagenet_std
    else:
      image = (image - 0.5) * 2.0
    return self.model(image, train=train)

  @nn.compact
  def encode_multiview(
      self,
      x: jt.Float[jt.Array, "... H W 3"],
      *,
      train: bool = False,
  ) -> jt.Float[jt.Array, "... D"]:
    return self(jax.lax.collapse(x, 0, -3), train=train)[0]

  @nn.compact
  def encode_dense(
      self,
      x: jt.Float[jt.Array, "... H W 3"],
      get_depths: Sequence[int] = (),
  ) -> jt.Float[jt.Array, "B N D"]:
    x = jax.lax.collapse(x, 0, -3)
    if self.norm_input:
      x = (x - self.imagenet_mean) / self.imagenet_std
    else:
      x = (x - 0.5) * 2.0
    # Get all intermediate outputs from the encoder.
    intermediate_outputs = self.model(x)[1]
    stack = [intermediate_outputs["encoded"]]
    for depth in get_depths:
      stack.append(intermediate_outputs["encoder"][f"block{depth:02d}"]["+mlp"])
    return jnp.stack(stack, axis=1)

  @classmethod
  def load_pretrained_params(cls, model_str: str) -> Any:
    """Loads pre-trained parameters."""
    # Load the parameters from the checkpoint.
    loaded_params = vit.load(None, model_str, {})
    final_params = {"model": loaded_params}
    return final_params


def from_config(config: ml_collections.ConfigDict) -> "PreTrainedSigLip2Model":
  return PreTrainedSigLip2Model(
      variant=config.variant,
      pool_type=config.pool_type,
      head_zeroinit=config.head_zeroinit,
      model_name=config.model_name,
  )
