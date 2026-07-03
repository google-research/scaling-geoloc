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

"""Generate tiny fake pv4ger / chesapeake / MDAS datasets for the downstream evals.

The files match exactly the on-disk layout the data sources in
``unigeoclip/scenic/datasets.py`` expect, so the classification and segmentation
evals run on CPU without the real datasets:

  <out>/pv4ger_classification/
    default_partition.json   # {'train': [...], 'valid': [...]}
    label_map.json           # {'0': [...filenames...], '1': [...]}
    <name>.hdf5              # datasets Red/Green/Blue [H, W] uint8

  <out>/m-chesapeake/
    default_partition.json   # {'train': [...], 'test': [...]}
    <name>.hdf5              # Red/Green/Blue [H, W] uint8 + label [H, W] int

  <out>/mdas/sub_area_{1,2,3}/     # train / valid / test sub-areas
    3K_DSM_sub_area{1,2,3}.tif     # single-band float DSM (heights)
    osm_landuse_sub_area{1,2,3}.tif  # single-band OSM land-use codes

The chesapeake tile size is chosen so the DPT head's upsampled output
(16x the patch grid) matches the label resolution: with a patch size of 16, a
32x32 tile gives a 2x2 patch grid and a 32x32 DPT output == 32x32 label.

The MDAS land-use labels are written in the raw OSM encoding (offset 7200) that
``MDASDataSource`` decodes, so the fake data flows through the identical
transform as the real `MDAS <https://mediatum.ub.tum.de/1657312>`_ dataset.
"""

import json
import os

from absl import app
from absl import flags
import h5py
import numpy as np
import pandas as pd
import rasterio

FLAGS = flags.FLAGS

# Inverse of ``MDASDataSource``'s label remap: raw OSM codes that decode to the
# contiguous final classes 0..12 (raw 0 stays background -> ignored as -1). The
# two gaps (raw 7210 / 7213 are dropped by the source) are reached through the
# source's 17->10 and 18->13 remaps.
_MDAS_RAW_FOR_CLASS = (
    7201, 7202, 7203, 7204, 7205, 7206, 7207, 7208, 7209,  # -> classes 0..8
    7217,  # -> class 9
    7211, 7212,  # -> classes 10, 11
    7218,  # -> class 12
)

# NB: flags are defined under ``__main__`` (not at import) so importing the
# reusable writer functions from tests doesn't register module-level flags
# (which would collide with the other generator's identically-named flags).


def _write_pv4ger(out_dir: str, image_size: int, rng: np.random.Generator):
  """Binary solar-panel classification (2 classes)."""
  root = os.path.join(out_dir, 'pv4ger_classification')
  os.makedirs(root, exist_ok=True)

  splits = {'train': 8, 'valid': 4}
  partition, label_map = {}, {'0': [], '1': []}
  for split, n in splits.items():
    names = []
    for i in range(n):
      name = f'{split}_{i:04d}'
      label = int(i % 2)
      bands = rng.integers(0, 256, size=(3, image_size, image_size), dtype=np.uint8)
      with h5py.File(os.path.join(root, name + '.hdf5'), 'w') as hf:
        hf.create_dataset('Red', data=bands[0])
        hf.create_dataset('Green', data=bands[1])
        hf.create_dataset('Blue', data=bands[2])
      names.append(name)
      label_map[str(label)].append(name)
    partition[split] = names

  with open(os.path.join(root, 'default_partition.json'), 'w') as f:
    json.dump(partition, f, indent=2)
  with open(os.path.join(root, 'label_map.json'), 'w') as f:
    json.dump(label_map, f, indent=2)
  print(f'Wrote pv4ger fake data to {root}')


def _write_chesapeake(
    out_dir: str, size: int, num_classes: int, rng: np.random.Generator
):
  """Land-cover segmentation (num_classes classes)."""
  root = os.path.join(out_dir, 'm-chesapeake')
  os.makedirs(root, exist_ok=True)

  splits = {'train': 5, 'test': 3}
  partition = {}
  for split, n in splits.items():
    names = []
    for i in range(n):
      name = f'{split}_{i:04d}'
      bands = rng.integers(0, 256, size=(3, size, size), dtype=np.uint8)
      label = rng.integers(0, num_classes, size=(size, size), dtype=np.int32)
      with h5py.File(os.path.join(root, name + '.hdf5'), 'w') as hf:
        hf.create_dataset('Red', data=bands[0])
        hf.create_dataset('Green', data=bands[1])
        hf.create_dataset('Blue', data=bands[2])
        hf.create_dataset('label', data=label)
      names.append(name)
    partition[split] = names

  with open(os.path.join(root, 'default_partition.json'), 'w') as f:
    json.dump(partition, f, indent=2)
  print(f'Wrote chesapeake fake data to {root}')


def _write_mdas(
    out_dir: str, dsm_size: int, num_classes: int, rng: np.random.Generator
):
  """DSM land-use segmentation (single-band GeoTIFFs, raw OSM label encoding).

  ``MDASDataSource`` resizes each label map to a fixed 1250x1500 grid and tiles
  the DSM into 256x256 crops (each paired with a 32x32 label crop), so the DSM
  side length controls the number of tiles: ``(dsm_size // 256) ** 2`` per
  sub-area.
  """
  root = os.path.join(out_dir, 'mdas')
  raw_codes = np.array((0,) + _MDAS_RAW_FOR_CLASS[:num_classes], dtype=np.int32)
  # Label is NEAREST-resized to 1250x1500 by the source; a small raster suffices.
  label_h, label_w = 200, 240
  for area in (1, 2, 3):
    sub_dir = os.path.join(root, f'sub_area_{area}')
    os.makedirs(sub_dir, exist_ok=True)
    name = f'sub_area{area}'

    dsm = rng.standard_normal((dsm_size, dsm_size)).astype(np.float32)
    labels = rng.choice(raw_codes, size=(label_h, label_w)).astype(np.int32)
    _write_geotiff(os.path.join(sub_dir, f'3K_DSM_{name}.tif'), dsm)
    _write_geotiff(os.path.join(sub_dir, f'osm_landuse_{name}.tif'), labels)
  print(f'Wrote MDAS fake data to {root}')


def _write_geotiff(path: str, array: np.ndarray) -> None:
  """Writes a single-band GeoTIFF (no CRS/transform; the source only reads)."""
  with rasterio.open(
      path,
      'w',
      driver='GTiff',
      height=array.shape[0],
      width=array.shape[1],
      count=1,
      dtype=array.dtype,
      transform=rasterio.Affine.identity(),
  ) as dst:
    dst.write(array, 1)


def _write_pdfm(
    out_dir: str, num_regions: int, num_targets: int, rng: np.random.Generator
):
  """Region-level (latlng -> socioeconomic targets) regression CSV."""
  root = os.path.join(out_dir, 'pdfm')
  os.makedirs(root, exist_ok=True)
  data = {
      'place': [f'zcta_{i:05d}' for i in range(num_regions)],
      'lat': rng.uniform(-60.0, 70.0, num_regions).astype(np.float32),
      'lng': rng.uniform(-180.0, 180.0, num_regions).astype(np.float32),
  }
  for t in range(num_targets):
    col = rng.standard_normal(num_regions).astype(np.float32)
    # Sprinkle a few NaNs to exercise the per-region masking.
    col[rng.random(num_regions) < 0.05] = np.nan
    data[f'target_{t}'] = col
  pd.DataFrame(data).to_csv(os.path.join(root, 'pdfm_benchmark.csv'), index=False)
  print(f'Wrote PDFM fake data to {root}')


def main(argv) -> None:
  del argv
  rng = np.random.default_rng(FLAGS.seed)
  _write_pv4ger(FLAGS.output_dir, FLAGS.pv4ger_size, rng)
  _write_chesapeake(FLAGS.output_dir, FLAGS.chesapeake_size, 7, rng)
  _write_mdas(FLAGS.output_dir, FLAGS.mdas_dsm_size, num_classes=13, rng=rng)
  _write_pdfm(FLAGS.output_dir, num_regions=200, num_targets=4, rng=rng)


if __name__ == '__main__':
  flags.DEFINE_string('output_dir', 'datasets', 'Root output directory.')
  flags.DEFINE_integer('pv4ger_size', 64, 'pv4ger image side length.')
  flags.DEFINE_integer(
      'chesapeake_size', 32, 'chesapeake tile/label side length.'
  )
  flags.DEFINE_integer(
      'mdas_dsm_size', 768, 'MDAS DSM side length ((size // 256) ** 2 tiles).'
  )
  flags.DEFINE_integer('seed', 0, 'Random seed.')
  app.run(main)
