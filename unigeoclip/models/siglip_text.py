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

from typing import Any

from big_vision.models.proj.image_text import text_transformer
from big_vision.models.proj.image_text import two_towers
import flax.linen as nn
import jaxtyping as jt
import ml_collections


class PreTrainedSigLip2TextModel(nn.Module):
  """SigLIP2 model."""

  variant: str = "B"
  output_dim: int = 768
  pool_type: str = "last"
  vocab_size: int = 256_000
  model_name: str = "SigLIP2 B/16 224"

  def setup(self):
    self.model = text_transformer.Model(
        self.output_dim,
        variant=self.variant,
        pool_type=self.pool_type,
        vocab_size=self.vocab_size,
    )

  def __call__(
      self, texts: jt.Float[jt.Array, "... D"], *, train: bool = False
  ) -> jt.Float[jt.Array, "... D"]:
    """Computes the text embeddings for the given texts.

    Args:
      texts: Batch of text to compute embeddings for.
      train: Whether to use training mode.

    Returns:
      The texts embeddings.
    """
    ztxt, _ = self.model(texts, train=train)
    return ztxt

  @classmethod
  def load_pretrained_params(cls, init_params: Any) -> Any:
    """Loads pre-trained parameters."""
    # Load the parameters from the checkpoint.
    init_files = two_towers.VANITY_NAMES.get(cls.model_name, cls.model_name)
    init_files = {k: f"{init_files}:{k}" for k in ("img", "txt", "t")}
    txt_init = init_files.pop("txt", None)
    loaded_params = text_transformer.load(
        init_params["text_backbone"]["model"],
        txt_init,
        {},
    )
    final_params = {"model": loaded_params}
    return final_params


def from_config(
    config: ml_collections.ConfigDict,
) -> "PreTrainedSigLip2TextModel":
  return PreTrainedSigLip2TextModel(
      variant=config.variant,
      pool_type=config.pool_type,
      vocab_size=config.vocab_size,
      output_dim=config.output_dim,
      model_name=config.model_name,
  )
