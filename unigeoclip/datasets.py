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

"""Dataset for training foundation models."""

from collections.abc import Mapping
import copy
import functools
import json
import os
from typing import Any

import grain.python as grain
import h5py
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import rasterio
import tensorflow_datasets as tfds

from scaling_geoloc.data import datasets
from scaling_geoloc.utils import distributed
from scaling_geoloc import types
from unigeoclip.data.multimodal_dummy_source import MultiModalDummyDataSource


class Pv4GerDataSource(grain.RandomAccessDataSource):
  """Data source for the PV4Ger dataset."""

  def __init__(self, data_path: str, split: str):
    """Initializes the Pv4GerDataSource.

    Args:
        data_path: The path to the directory containing the .h5 files and
          labels.json.
        split: The split of the dataset to load.
    """
    self.data_path = data_path
    self._metadata = []

    partition_path = os.path.join(self.data_path, 'default_partition.json')
    with open(partition_path, 'r') as file:
      partition = json.load(file)[split]

    labels_path = os.path.join(self.data_path, 'label_map.json')

    with open(labels_path, 'r') as file:
      self.labels_map = json.load(file)

    # Create a flat list of metadata for every single item in the dataset.
    for label in list(self.labels_map.keys()):
      for filename in self.labels_map[label]:
        if filename not in partition:
          continue
        file_path = os.path.join(self.data_path, filename) + '.hdf5'
        self._metadata.append({'file_path': file_path, 'label': int(label)})

  def __len__(self) -> int:
    """Returns the total number of data points across all files."""
    return len(self._metadata)

  def __getitem__(self, index: int) -> Mapping[str, Any]:
    """Loads a single data point.

    This method is called for each index in the dataset. It opens the
    corresponding H5 file, reads the data at the specific inner index,
    and returns it along with its label.

    Args:
        index: The global index of the data point to retrieve.

    Returns:
        A dictionary containing the data and its label.
    """

    item_meta = self._metadata[index]
    file_path = item_meta['file_path']

    with open(file_path, 'rb') as file:
      with h5py.File(file, 'r') as hf:
        band_names = ['Red', 'Green', 'Blue']
        bands = [jnp.array(hf[band_name][:]) for band_name in band_names]

        # Concatenate the bands along a new axis to create an image
        image = jnp.stack(bands, axis=-1)
        image = image.astype(jnp.float32) / 255.0

    return {'input': image, 'label': np.array(item_meta['label'])}


class ChesapeakeDataSource(grain.RandomAccessDataSource):
  """Data source for the Chesapeake dataset."""

  def __init__(self, data_path: str, split: str):
    """Initializes the ChesapeakeDataSource.

    Args:
        data_path: The path to the directory containing the .h5 files and
          labels.json.
        split: The split of the dataset to load.
    """
    self.data_path = data_path
    self._metadata = []

    partition_path = os.path.join(self.data_path, 'default_partition.json')
    with open(partition_path, 'r') as file:
      partition = json.load(file)[split]

    for filename in partition:
      file_path = os.path.join(self.data_path, filename) + '.hdf5'
      self._metadata.append({'file_path': file_path})

  def __len__(self) -> int:
    """Returns the total number of data points across all files."""
    return len(self._metadata)

  def __getitem__(self, index: int) -> Mapping[str, Any]:
    """Loads a single data point.

    This method is called for each index in the dataset. It opens the
    corresponding H5 file, reads the data at the specific inner index,
    and returns it along with its label.

    Args:
        index: The global index of the data point to retrieve.

    Returns:
        A dictionary containing the data and its label.
    """

    item_meta = self._metadata[index]
    file_path = item_meta['file_path']

    with open(file_path, 'rb') as file:
      with h5py.File(file, 'r') as hf:
        red_band = jnp.array(hf['Red'][:])
        blue_band = jnp.array(hf['Blue'][:])
        green_band = jnp.array(hf['Green'][:])

        # Concatenate the bands along a new axis to create an image
        image = jnp.stack([red_band, green_band, blue_band], axis=-1)
        label = jnp.array(hf['label'][:])

    return {'input': image, 'label': label}


class MDASDataSource(grain.RandomAccessDataSource):
  """Data source for the MDAS dataset."""

  def __init__(self, data_path: str, split: str):
    """Initializes the MDASDataSource.

    Args:
        data_path: The path to the directory containing the .h5 files and
          labels.json.
        split: The split of the dataset to load.
    """
    partition = {
        'train': 'sub_area_1',
        'valid': 'sub_area_2',
        'test': 'sub_area_3',
    }

    self.data_path = os.path.join(data_path, partition[split])
    self._metadata = []

    name = partition[split][:-2] + partition[split][-1]
    label_path = os.path.join(self.data_path, f'osm_landuse_{name}.tif')
    with open(label_path, 'rb') as file:
      with rasterio.open(file) as src:
        labels_map = src.read(1)

    labels_map = np.array(
        jax.image.resize(
            labels_map,
            (1250, 1500),
            jax.image.ResizeMethod.NEAREST,
        )
    )
    labels_map[labels_map < 0] = 0
    # Shift the labels by 7200 because MDAS labels start at 7200.
    labels_map[labels_map > 0] = labels_map[labels_map > 0] - 7200
    # Discard labels that are not in all label maps.
    indices_to_zero = [10, 13, 14, 15, 19]
    for index in indices_to_zero:
      labels_map[labels_map == index] = 0
    # Shift the labels to fill the gaps in the labels.
    labels_map[labels_map == 17] = 10
    labels_map[labels_map == 18] = 13
    # Setting the no label class to -1 to discard it and shifting the labels.
    labels_map -= 1
    self.labels_map = labels_map

    file_path = os.path.join(self.data_path, f'3K_DSM_{name}.tif')
    with open(file_path, 'rb') as file:
      with rasterio.open(file) as src:
        self.dsm_map = src.read(1)

    for i in range(self.dsm_map.shape[0] // 256):
      for j in range(self.dsm_map.shape[1] // 256):
        self._metadata.append((i, j))

  def __len__(self) -> int:
    """Returns the total number of data points across all files."""
    return len(self._metadata)

  def __getitem__(self, index: int) -> Mapping[str, Any]:
    """Loads a single data point.

    This method is called for each index in the dataset. It opens the
    corresponding crops of the DSM and labels and returns them.

    Args:
        index: The global index of the data point to retrieve.

    Returns:
        A dictionary containing the data and its label.
    """

    i, j = self._metadata[index]
    size_dsm = 256
    size_label = 32
    image = self.dsm_map[
        i * size_dsm : i * size_dsm + size_dsm,
        j * size_dsm : j * size_dsm + size_dsm,
    ]
    label = self.labels_map[
        i * size_label : i * size_label + size_label,
        j * size_label : j * size_label + size_label,
    ]

    return {'input': np.expand_dims(image, axis=0), 'label': label}


def create_downstream_dataset(
    dataset_config: ml_collections.ConfigDict,
    dataset_name: str,
    batch_size: int,
    batch_size_eval: int,
    seed: int,
    transformations: grain.Transformations = (),
    meta_data: types.DataDict | None = None,
) -> types.Dataset:
  """Create a dataset for training."""

  process_batch_size = batch_size
  process_batch_size_eval = batch_size_eval

  train_transformations = copy.deepcopy(transformations)
  valid_transformations = copy.deepcopy(transformations)

  if dataset_name == 'pv4ger_classification':
    train_data_source = Pv4GerDataSource(
        dataset_config.data_dir, dataset_config.train_split
    )
    valid_data_source = Pv4GerDataSource(
        dataset_config.data_dir, dataset_config.valid_split
    )
  elif dataset_name == 'm-chesapeake':
    train_data_source = ChesapeakeDataSource(
        dataset_config.data_dir, dataset_config.train_split
    )
    valid_data_source = ChesapeakeDataSource(
        dataset_config.data_dir, dataset_config.valid_split
    )
  elif dataset_name == 'mdas':
    train_data_source = MDASDataSource(
        dataset_config.data_dir, dataset_config.train_split
    )
    valid_data_source = MDASDataSource(
        dataset_config.data_dir, dataset_config.valid_split
    )
  else:
    raise ValueError(f'Unsupported dataset name: {dataset_name}')

  dummy_batch_fn = functools.partial(
      datasets.get_dummy_batch,
      train_data_source,
      train_transformations,
      batch_size,
      seed=999,
  )

  meta_data = {
      'num_train_examples': len(train_data_source),
      'num_valid_examples': len(valid_data_source),
      'dummy_batch_fn': dummy_batch_fn,
      **(meta_data if meta_data is not None else {}),
  }

  train_loader = datasets.dataloader_provider_fn(
      source=train_data_source,
      num_epochs=1,
      shuffle=dataset_config.shuffle_train,
      seed=seed,
      drop_remainder=False,
      transformations=train_transformations,
      batch_size=process_batch_size,
      worker_count=dataset_config.grain_num_workers,
      read_options=grain.ReadOptions(
          prefetch_buffer_size=dataset_config.grain_prefetch_buffer_size,
          num_threads=dataset_config.grain_num_threads,
      ),
      mining_indices=meta_data.get(
          f'{dataset_config.train_split}_mining_indices', None
      ),
  )

  valid_loader = datasets.dataloader_provider_fn(
      source=valid_data_source,
      num_epochs=1,
      seed=seed,
      drop_remainder=False,
      shuffle=dataset_config.shuffle_valid,
      transformations=valid_transformations,
      batch_size=process_batch_size_eval,
      worker_count=dataset_config.grain_num_workers_valid,
      read_options=grain.ReadOptions(
          prefetch_buffer_size=dataset_config.grain_prefetch_buffer_size,
          num_threads=dataset_config.grain_num_threads_valid,
      ),
      mining_indices=meta_data.get(
          f'{dataset_config.valid_split}_mining_indices', None
      ),
  )

  return types.Dataset(
      train_iter=train_loader,
      valid_iter=valid_loader,
      meta_data=meta_data,
      train_ds=train_data_source,
      valid_ds=valid_data_source,
  )


def create_multimodal_dataset(
    dataset_config: ml_collections.ConfigDict,
    batch_size: int,
    batch_size_eval: int,
    seed: int,
    meta_data: types.DataDict | None = None,
) -> types.Dataset:
  """Create the multimodal contrastive (sv/aerial/dsm/text/latlng) dataset.

  Two backends:

  * ``location`` unset (default) -> the procedural ``MultiModalDummyDataSource``
    so training runs on CPU without any files.
  * ``location`` set -> read the 5-modality ArrayRecord shards written by
    ``unigeoclip.data.generate_fake_multimodal_data`` (or any real dataset with
    the same TFDS spec) from ``<data_dir>/<location>``.
  """
  process_batch_size = distributed.get_process_batch_size(batch_size)
  process_batch_size_eval = distributed.get_process_batch_size(batch_size_eval)

  location = dataset_config.get('location')
  if location:
    loc_dir = os.path.join(dataset_config.data_dir, location)
    builder = tfds.builder_from_directory(loc_dir)
    train_data_source = builder.as_data_source(
        dataset_config.get('train_split', 'train')
    )
    valid_data_source = builder.as_data_source(
        dataset_config.get('valid_split', 'valid')
    )
  else:
    source_kwargs = dict(
        num_views=dataset_config.get('num_views', 1),
        sv_image_size=dataset_config.get('sv_image_size', 224),
        aerial_image_size=dataset_config.get('aerial_image_size', 256),
        dsm_image_size=dataset_config.get('dsm_image_size', 256),
        text_dim=dataset_config.get('text_dim', 768),
    )
    train_data_source = MultiModalDummyDataSource(
        num_examples=dataset_config.get('num_train_examples', 1024),
        **source_kwargs,
    )
    valid_data_source = MultiModalDummyDataSource(
        num_examples=dataset_config.get('num_valid_examples', 256),
        **source_kwargs,
    )

  dummy_batch_fn = functools.partial(
      datasets.get_dummy_batch, train_data_source, (), batch_size, seed=999
  )

  meta_data = {
      'num_train_examples': len(train_data_source),
      'num_valid_examples': len(valid_data_source),
      'dummy_batch_fn': dummy_batch_fn,
      **(meta_data if meta_data is not None else {}),
  }

  def _loader(source, shuffle, bs, workers):
    return datasets.dataloader_provider_fn(
        source=source,
        shuffle=shuffle,
        seed=seed,
        shard_options=grain.ShardByJaxProcess(drop_remainder=True),
        drop_remainder=True,
        transformations=(),
        batch_size=bs,
        worker_count=workers,
        read_options=grain.ReadOptions(
            prefetch_buffer_size=dataset_config.grain_prefetch_buffer_size,
            num_threads=dataset_config.grain_num_threads,
        ),
    )

  return types.Dataset(
      train_iter=_loader(
          train_data_source,
          dataset_config.shuffle_train,
          process_batch_size,
          dataset_config.grain_num_workers,
      ),
      valid_iter=_loader(
          valid_data_source,
          dataset_config.shuffle_valid,
          process_batch_size_eval,
          dataset_config.grain_num_workers_valid,
      ),
      meta_data=meta_data,
      train_ds=train_data_source,
      valid_ds=valid_data_source,
  )


def build_dataset(
    dataset_config: ml_collections.ConfigDict,
    batch_size: int,
    batch_size_eval: int,
    seed: int,
    meta_data: types.DataDict | None = None,
) -> types.Dataset:
  """Create a dataset from config."""
  dataset_name = dataset_config.get('dataset_name')
  if dataset_name in ('pv4ger_classification', 'm-chesapeake', 'mdas'):
    return create_downstream_dataset(
        dataset_config, dataset_name, batch_size, batch_size_eval, seed
    )
  return create_multimodal_dataset(
      dataset_config, batch_size, batch_size_eval, seed, meta_data=meta_data
  )
