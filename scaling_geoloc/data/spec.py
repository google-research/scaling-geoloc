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
class TrainExampleSpec:
  """Feature spec for the precomputed cell index paired with an ExampleSpec.

  This spec is stored in a separate index dataset (e.g. ``index_level17/``)
  alongside the image shards.  It records, for each example:

  * A single shared ``cell_idx`` (the S2 cell containing both the query and
    aerial location — they share a cell since they are co-located).
  * Per-modality ``adjacent_cell_idx`` and optionally ``adjacent_weights`` for
    bilinear interpolation of the cell-code embedding.

  On-disk structure::

      {
        'cell_idx':  int32  []        # index into the cell grid
        'query': {
          'adjacent_cell_idx':  int32   [NUM_ADJACENT_CELLS]
          'adjacent_weights':   float32 [NUM_ADJACENT_CELLS]  # optional
        }
        'aerial': {
          'adjacent_cell_idx':  int32   [NUM_ADJACENT_CELLS]
          'adjacent_weights':   float32 [NUM_ADJACENT_CELLS]  # optional
        }
      }

  Attributes:
    store_adjacent_weights: Whether to store bilinear interpolation weights.
      When False, only ``adjacent_cell_idx`` is stored; weights are computed
      on-the-fly during training or omitted entirely.
  """

  store_adjacent_weights: bool = True

  def _modality_spec(self) -> dict:
    spec = {
        'adjacent_cell_idx': tfds.features.Tensor(
            shape=(NUM_ADJACENT_CELLS,), dtype=tf.int32
        ),
    }
    if self.store_adjacent_weights:
      spec['adjacent_weights'] = tfds.features.Tensor(
          shape=(NUM_ADJACENT_CELLS,), dtype=tf.float32
      )
    return spec

  def spec(self) -> dict:
    """Returns the TFDS feature spec for one index example."""
    return {
        'cell_idx': tfds.features.Tensor(shape=(), dtype=tf.int32),
        'query': self._modality_spec(),
        'aerial': self._modality_spec(),
    }
