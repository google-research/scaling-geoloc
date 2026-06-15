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

"""TFDS-style data specification for the hybrid geo-localization dataset.

Each dataset example pairs one query (Street View) image with one aerial tile
covering the same geographic location.  Only raw image pixels and GPS
coordinates are stored on disk; cell indices and interpolation weights are
added during preprocessing.

On-disk example
---------------
Each stored example has the following structure (using TFDS feature notation)::

    {
      'query': QuerySpec,
      'aerial': AerialSpec,
    }

See :class:`QuerySpec` and :class:`AerialSpec` for the per-modality fields.
"""

import dataclasses

import tensorflow as tf
import tensorflow_datasets as tfds


@dataclasses.dataclass
class QuerySpec:
  """Feature specification for a single query (Street View) image.

  Attributes:
    image_size: Height and width of the stored image in pixels.
    encoding_format: JPEG or PNG encoding used in the TFRecord, or None for raw.
  """

  image_size: int = 224
  encoding_format: str | None = 'jpeg'

  def spec(self) -> dict:
    """Returns the TFDS feature spec for one query example."""
    return {
        'image': tfds.features.Image(
            shape=(self.image_size, self.image_size, 3),
            dtype=tf.uint8,
            encoding_format=self.encoding_format,
        ),
        # GPS position of the camera (WGS-84 degrees).
        'latlng': tfds.features.Tensor(shape=(2,), dtype=tf.float32),
    }


@dataclasses.dataclass
class AerialSpec:
  """Feature specification for one aerial tile.

  Attributes:
    image_size: Height and width of the stored tile in pixels.
    encoding_format: JPEG or PNG encoding, or None for raw.
  """

  image_size: int = 256
  encoding_format: str | None = 'jpeg'

  def spec(self) -> dict:
    """Returns the TFDS feature spec for one aerial tile example."""
    return {
        'image': tfds.features.Image(
            shape=(self.image_size, self.image_size, 3),
            dtype=tf.uint8,
            encoding_format=self.encoding_format,
        ),
        # GPS coordinates of the tile centre (WGS-84 degrees).
        'latlng': tfds.features.Tensor(shape=(2,), dtype=tf.float32),
    }


@dataclasses.dataclass
class ExampleSpec:
  """Feature specification for one paired query + aerial training example.

  Attributes:
    query: Configuration for the query image.
    aerial: Configuration for the aerial tile.
  """

  query: QuerySpec = dataclasses.field(default_factory=QuerySpec)
  aerial: AerialSpec = dataclasses.field(default_factory=AerialSpec)

  def spec(self) -> dict:
    """Returns the full TFDS feature spec for one training example."""
    return {
        'query': self.query.spec(),
        'aerial': self.aerial.spec(),
    }


# Number of adjacent cells used for bilinear interpolation in cell space.
NUM_ADJACENT_CELLS = 4


@dataclasses.dataclass
class ModalityIndexSpec:
  """Feature spec for per-modality cell assignments in index_level<N>/<modality>/.

  Each shard stores a top-level ``cell_idx`` plus interpolation neighbours
  nested under the modality key (``'query'`` or ``'aerial'``).  When merged
  via ``JoinedDataSource(..., key=None)`` the ``cell_idx`` surfaces at the
  top level and the neighbours deep-merge into the existing modality sub-dict.

  On-disk structure (e.g. modality='aerial')::

      {
        'cell_idx': int32 [],
        'aerial': {
          'adjacent_cell_idx': int32  [NUM_ADJACENT_CELLS],
          'adjacent_weights':  float32 [NUM_ADJACENT_CELLS],
        }
      }

  Attributes:
    modality: Either ``'query'`` or ``'aerial'``.
  """

  modality: str  # 'query' | 'aerial'

  def spec(self) -> dict:
    return {
        'cell_idx': tfds.features.Tensor(shape=(), dtype=tf.int32),
        self.modality: {
            'adjacent_cell_idx': tfds.features.Tensor(
                shape=(NUM_ADJACENT_CELLS,), dtype=tf.int32
            ),
            'adjacent_weights': tfds.features.Tensor(
                shape=(NUM_ADJACENT_CELLS,), dtype=tf.float32
            ),
        },
    }


@dataclasses.dataclass
class TrainExampleSpec:
  """Feature spec for a full paired training example (images + cell index).

  Combines :class:`ExampleSpec` (images, latlng) with precomputed cell
  assignments.  On-disk structure::

      {
        'cell_idx': int32 [],
        'query': {
          'image':              uint8   [H, W, 3],
          'latlng':             float32 [2],
          'adjacent_cell_idx':  int32   [NUM_ADJACENT_CELLS],
          'adjacent_weights':   float32 [NUM_ADJACENT_CELLS],
        },
        'aerial': {
          'image':              uint8   [H, W, 3],
          'latlng':             float32 [2],
          'adjacent_cell_idx':  int32   [NUM_ADJACENT_CELLS],
          'adjacent_weights':   float32 [NUM_ADJACENT_CELLS],
        },
      }

  Attributes:
    query: Query image spec.
    aerial: Aerial image spec.
  """

  query: QuerySpec = dataclasses.field(default_factory=QuerySpec)
  aerial: AerialSpec = dataclasses.field(default_factory=AerialSpec)

  def spec(self) -> dict:
    adjacent = {
        'adjacent_cell_idx': tfds.features.Tensor(
            shape=(NUM_ADJACENT_CELLS,), dtype=tf.int32
        ),
        'adjacent_weights': tfds.features.Tensor(
            shape=(NUM_ADJACENT_CELLS,), dtype=tf.float32
        ),
    }
    return {
        'cell_idx': tfds.features.Tensor(shape=(), dtype=tf.int32),
        'query': {**self.query.spec(), **adjacent},
        'aerial': {**self.aerial.spec(), **adjacent},
    }
