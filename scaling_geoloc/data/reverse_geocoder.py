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

"""Reverse geocoder utils."""

import os
from typing import Final, Mapping, Sequence

from etils.array_types import BoolArray
from etils.array_types import FloatArray
from etils.array_types import IntArray
from etils.array_types import StrArray
import numpy as np

from google3.location.country.reverse.python import s2range_reverse_geocoder
from google3.pyglib import gfile
from google3.sstable.python import sstable
from google3.util.geometry.python import s2

S2RANGE_DATA_PATH: Final[str] = (
    '/google_src/head/depot/google3/googledata/location/country/reverse/'
)

AVAILABLE_DATA_STREAMS: Mapping[str, str] = {
    'metro': 'metros-level13.s2range',
    'metro_area': 'metro-areas-level13.s2range',
    'country': 'allcountries-level12.s2range',
    'county': 'counties-level14.s2range',
    'province': 'provinces-level14.s2range',
    'language': 'bestlanguages-level12.s2range',
}

INVALID_LABEL: Final[str] = ''
INVALID_LABEL_ID: Final[int] = -1


class ReverseGeocoder:
  """Reverse geocoder for one datastream."""

  def __init__(self, entity: str):
    self.data_stream = os.path.join(
        S2RANGE_DATA_PATH, AVAILABLE_DATA_STREAMS[entity]
    )
    self._geocoder = s2range_reverse_geocoder.S2RangeReverseGeocoder()
    self._geocoder.LoadDataFile(self.data_stream)

    # We initialize this lazily to avoid loading the sstable if it is not used.
    self._label_to_id = {}
    self._id_to_label = np.array([])

  @classmethod
  def list_all_files(cls) -> list[str]:
    """Returns a list of available files."""
    paths = gfile.Glob(os.path.join(S2RANGE_DATA_PATH, '*.s2range'))
    return [os.path.basename(path) for path in paths]

  def get_label_for_latlng(self, lat: float, lng: float) -> str:
    """Returns the labels for the given latitude and longitude."""
    result = self._geocoder.LookupLatLng(lat, lng)
    return result

  def get_label_for_latlngs(self, latlng: FloatArray['N 2']) -> StrArray['N']:
    """Returns the labels for the given latitude and longitude."""
    labels = np.apply_along_axis(
        lambda x: np.array(self.get_label_for_latlng(*x)).astype(np.object_),
        -1,
        latlng,
    )
    return labels

  def get_label_for_s2cell(self, s2cell_id: s2.S2CellId) -> str:
    """Returns the labels for the given s2cell."""
    s2_latlng = s2.S2LatLng.from_point(s2cell_id.to_point())
    return self.get_label_for_latlng(
        s2_latlng.lat.degrees, s2_latlng.lng.degrees
    )

  def get_label_for_s2cell_token(self, s2cell_token: bytes) -> str:
    """Returns the labels for the given s2cell token."""
    return self.get_label_for_s2cell(s2.S2CellId.from_token(s2cell_token))

  def get_label_for_s2cell_tokens(
      self, s2cell_tokens: StrArray['N']
  ) -> StrArray['N']:
    """Returns the labels for the given s2cells."""
    labels = np.apply_along_axis(
        self.get_label_for_s2cell_token, -1, s2cell_tokens
    )
    return labels

  def check_valid(self, labels: str | StrArray['N 2']) -> bool | BoolArray['N']:
    """Checks whether the labels are valid."""
    return labels != INVALID_LABEL

  def _load_id_map(self):
    """Loads the id map."""
    if self._label_to_id:
      return
    all_labels: list[bytes] = sorted(
        np.unique(list(sstable.SSTable(self.data_stream).values())).tolist()
    )
    self._id_to_label = [label.decode('utf-8') for label in all_labels]
    self._label_to_id = {
        label: idx for idx, label in enumerate(self._id_to_label)
    }
    self._label_to_id[INVALID_LABEL] = INVALID_LABEL_ID
    self._id_to_label = np.array(self._id_to_label + [INVALID_LABEL])

  def label_to_id(self, label: str | StrArray['N']) -> int | IntArray['N']:
    """Returns the label to index mapping."""
    self._load_id_map()
    if isinstance(label, str):
      return self._label_to_id[label]
    else:
      return np.vectorize(lambda x: self._label_to_id[x])(label).astype(int)

  def id_to_label(self, label_id: int | IntArray['N']) -> str | StrArray['N']:
    """Returns the label to index mapping."""
    self._load_id_map()
    if isinstance(label_id, int):
      return self._id_to_label[label_id].item()
    else:
      return self._id_to_label[label_id]


class CombinedReverseGeocoder:
  """Reverse geocoder for multiple datastreams."""

  def __init__(self, entities: Sequence[str] | None = None):
    self._entities = entities or list(AVAILABLE_DATA_STREAMS.keys())
    self._geocoders = {
        entity: ReverseGeocoder(entity) for entity in self._entities
    }

  def get_label_for_latlng(
      self, lat: float, lng: float, entity: str
  ) -> StrArray['N']:
    """Returns the labels for the given latitude and longitude."""
    return self._geocoders[entity].get_label_for_latlng(lat, lng)

  def get_labels_for_latlng(
      self, lat: float, lng: float, entities: Sequence[str] | None = None
  ) -> dict[str, str]:
    """Returns the labels for the given latitude and longitude."""
    if entities is None:
      entities = self._geocoders.keys()
    return {
        entity: self._geocoders[entity].get_label_for_latlng(lat, lng)
        for entity in entities
    }

  def get_label_for_latlngs(
      self, latlng: FloatArray['N 2'], entity: str
  ) -> StrArray['N']:
    """Returns the labels for the given latitude and longitude."""
    return self._geocoders[entity].get_label_for_latlngs(latlng)

  def get_labels_for_latlngs(
      self, latlng: FloatArray['N 2'], entities: Sequence[str] | None = None
  ) -> dict[str, StrArray['N']]:
    """Returns the labels for the given latitude and longitude."""
    if entities is None:
      entities = self._geocoders.keys()
    return {
        entity: self._geocoders[entity].get_label_for_latlngs(latlng)
        for entity in entities
    }

  def check_valid(self, labels: str | StrArray['N']) -> bool | BoolArray['N']:
    """Returns the labels for the given latitude and longitude."""
    return labels != INVALID_LABEL

  def label_to_id(self, labels: str | StrArray['N'], entity: str) -> int:
    """Returns the label to index mapping."""
    return self._geocoders[entity].label_to_id(labels)

  def labels_to_ids(
      self, label_dict: dict[str, StrArray['N'] | str]
  ) -> dict[str, IntArray['N'] | int]:
    """Returns the label to index mapping."""
    return {
        entity: self._geocoders[entity].label_to_id(labels)
        for entity, labels in label_dict.items()
    }

  def ids_to_labels(
      self, ids_dict: dict[str, IntArray['N'] | int]
  ) -> dict[str, StrArray['N'] | str]:
    """Returns the label to index mapping."""
    return {
        entity: self._geocoders[entity].id_to_label(ids_)
        for entity, ids_ in ids_dict.items()
    }
