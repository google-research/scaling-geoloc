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

"""TFDS feature spec for the multimodal (5-modality) training data.

Mirrors exactly the nested example produced by ``MultiModalDummyDataSource`` so
the generated ArrayRecord shards round-trip through
``tfds.builder_from_directory(...).as_data_source(split)`` back into the same
structure the contrastive model consumes. It follows the ``scaling_geoloc``
per-modality ``{image, latlng}`` convention, extended with ``dsm`` and ``text``:
``query/{image,latlng}`` -> sv, ``aerial/{image,latlng}``, ``dsm/{image,latlng}``,
``text/siglip_embeddings`` -> text, plus the shared top-level ``latlng``.
"""

import tensorflow as tf
import tensorflow_datasets as tfds


def multimodal_feature_spec(
    sv_image_size: int = 224,
    aerial_image_size: int = 256,
    dsm_image_size: int = 256,
    text_dim: int = 768,
) -> dict:
  """Returns the nested TFDS feature spec for one multimodal example."""
  tensor = tfds.features.Tensor

  def latlng() -> tfds.features.Tensor:
    return tensor(shape=(2,), dtype=tf.float32)

  return {
      'latlng': latlng(),
      'query': {
          'image': tensor(
              shape=(sv_image_size, sv_image_size, 3), dtype=tf.float32
          ),
          'latlng': latlng(),
      },
      'aerial': {
          'image': tensor(
              shape=(aerial_image_size, aerial_image_size, 3), dtype=tf.float32
          ),
          'latlng': latlng(),
      },
      'dsm': {
          'image': tensor(
              shape=(dsm_image_size, dsm_image_size), dtype=tf.float32
          ),
          'latlng': latlng(),
      },
      'text': {
          'siglip_embeddings': tensor(shape=(text_dim,), dtype=tf.float32),
      },
  }
