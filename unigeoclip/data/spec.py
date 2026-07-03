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
structure the contrastive model consumes (``query/views/image_pixels`` -> sv,
``aerial/{rgb,center_ll}``, ``dsm/{dsm,center_ll}``, ``latlng``,
``siglip_embeddings`` -> text).
"""

import tensorflow as tf
import tensorflow_datasets as tfds


def multimodal_feature_spec(
    num_views: int = 1,
    sv_image_size: int = 224,
    aerial_image_size: int = 256,
    dsm_image_size: int = 256,
    text_dim: int = 768,
) -> dict:
  """Returns the nested TFDS feature spec for one multimodal example."""
  tensor = tfds.features.Tensor
  return {
      'latlng': tensor(shape=(2,), dtype=tf.float32),
      'query': {
          'views': {
              'image_pixels': tensor(
                  shape=(num_views, sv_image_size, sv_image_size, 3),
                  dtype=tf.float32,
              ),
          },
      },
      'aerial': {
          'rgb': tensor(
              shape=(aerial_image_size, aerial_image_size, 3), dtype=tf.float32
          ),
          'center_ll': tensor(shape=(2,), dtype=tf.float32),
      },
      'dsm': {
          'dsm': tensor(
              shape=(dsm_image_size, dsm_image_size), dtype=tf.float32
          ),
          'center_ll': tensor(shape=(2,), dtype=tf.float32),
      },
      'siglip_embeddings': tensor(shape=(text_dim,), dtype=tf.float32),
  }
