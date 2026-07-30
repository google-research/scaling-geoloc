# Scaling Image Geo-Localization to Continent Level

This repository contains the code for the following publications:
- [Scaling Image Geo-Localization to Continent Level](https://arxiv.org/pdf/2510.26795) by *Philipp Lindenberger, Paul-Edouard Sarlin, Jan Hosang, Matteo Balice, Marc Pollefeys, Simon Lynen, Eduard Trulls*, published at NeurIPS 2025.
- [UniGeoCLIP: Unified Geospatial Contrastive Learning](https://gastruc.github.io/unigeoclip) by *Guillaume Astruc, Eduard Trulls, Jan Hosang, Loic Landrieu, Paul-Edouard Sarlin*, published at the EarthVision workshop at CVPR 2026.

If you use any of this code, consider citing the relevant papers:
```bibtex
@inproceedings{lindenberger2026scaling,
  title     = {{Scaling Image Geo-Localization to Continent Level}},
  author    = {Lindenberger, Philipp and Sarlin, Paul-Edouard and Hosang, Jan
               and Pollefeys, Marc and Lynen, Simon and Trulls, Eduard},
  booktitle = {NeurIPS},
  year      = {2026}
}
@inproceedings{astruc2026unigeoclip,
  title     = {UniGeoCLIP: Unified Geospatial Contrastive Learning},
  author    = {Astruc, Guillaume and Trulls, Eduard and Hosang, Jan
               and Landrieu, Lo{\"i}c and Sarlin, Paul-Edouard},
  booktitle = {EarthVision Workshop, CVPR},
  year      = {2026}
}
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Data

Data is organized by location. The root directory is controlled by `config.data.data_dir` (default: `datasets/`) and the location by `config.data.location`.

### Directory layout

```
datasets/
  <location>/
    query/
      dataset_info.json
      train-*.arrayrecord       # ground-view images, training
      val-*.arrayrecord
      test-*.arrayrecord        # test queries
    aerial/
      dataset_info.json
      train-*.arrayrecord       # aerial tile images, training
      val-*.arrayrecord
      test-*.arrayrecord        # test database
    index_level15/              # precomputed S2 cell assignments at level 15
      query/
        dataset_info.json
        train-*.arrayrecord
        val-*.arrayrecord
        test-*.arrayrecord
      aerial/
        dataset_info.json
        train-*.arrayrecord
        val-*.arrayrecord
        test-*.arrayrecord
      metadata.npz              # cell_centers_latlng: float32 [N_cells, 2]
```

Images and cell index are stored separately so multiple `index_level<N>/` directories can coexist without duplicating image data. The active index is set via `config.data.index_subdir` (default: `index_level15`).

### Shard specs

**`query/` and `aerial/` shards** store images and positions:
```
{'image': uint8 [H, W, 3], 'latlng': float32 [2]}
```

**`index_level<N>/query/` and `index_level<N>/aerial/` shards** (`ModalityIndexSpec`) store per-image cell assignments. Each shard has a top-level `cell_idx` and interpolation neighbours nested under its modality key:
```
# index_level<N>/query/              # index_level<N>/aerial/
{                                    {
  'cell_idx': int32 [],                'cell_idx': int32 [],
  'query': {                           'aerial': {
    'adjacent_cell_idx': int32 [4],      'adjacent_cell_idx': int32 [4],
    'adjacent_weights':  float32 [4],    'adjacent_weights':  float32 [4],
  }                                    }
}                                    }
```

When four sources are joined for training (`JoinedDataSource` order: query images → aerial images → aerial index → query index), the query's `cell_idx` takes precedence and the neighbours deep-merge into their respective sub-dicts.

**`metadata.npz`** contains `cell_centers_latlng: float32 [N_cells, 2]` — lat/lng of each cell centre.

### Per-example fields after joining

`JoinedDataSource` merges all four sources into a single example per index:

| Field | Shape | Description |
|---|---|---|
| `cell_idx` | `int32` | S2 cell index (shared by query and aerial) |
| `query/image` | `uint8 [H, W, 3]` | Ground-view image |
| `query/latlng` | `float32 [2]` | Ground-truth lat/lng |
| `query/adjacent_cell_idx` | `int32 [4]` | Neighbouring cell indices for query |
| `query/adjacent_weights` | `float32 [4]` | Bilinear interpolation weights for query |
| `aerial/image` | `uint8 [H, W, 3]` | Aerial tile image |
| `aerial/latlng` | `float32 [2]` | Aerial tile lat/lng |
| `aerial/adjacent_cell_idx` | `int32 [4]` | Neighbouring cell indices for aerial |
| `aerial/adjacent_weights` | `float32 [4]` | Bilinear interpolation weights for aerial |

---

## Training

```bash
python -m scaling_geoloc.train \
    --config=scaling_geoloc.configs.train \
    --workdir=/path/to/workdir \
    --config.data.location=<location>
```

Key config overrides:

| Flag | Default | Description |
|---|---|---|
| `--config.data.location` | *(required)* | Location name matching a subdirectory in `data_dir` |
| `--config.data.data_dir` | `datasets/` | Root data directory |
| `--config.data.index_subdir` | `index_level15` | Which cell index to use |
| `--config.batch_size_per_device` | `64` | Per-device batch size (total = × device count) |
| `--config.lr.base_learning_rate` | `3e-4` | Peak learning rate |
| `--config.num_training_steps` | `200000` | Total training steps |
| `--config.model.vit.pretrained_weights` | `dinov2_vitb14` | Query backbone weights |

To fine-tune from a checkpoint:

```bash
python -m scaling_geoloc.train \
    --config=scaling_geoloc.configs.train \
    --workdir=/path/to/finetune_workdir \
    --config.data.location=<location> \
    --config.pretrained_path=/path/to/pretrained_workdir
```

---

## Evaluation

```bash
python -m scaling_geoloc.eval \
    --config=scaling_geoloc.configs.eval \
    --workdir=/path/to/eval_output \
    --config.experiment=/path/to/checkpoint \
    --config.test_evals.retrieval_cross_view.data.location=<location>
```

The eval script runs each eval in `config.test_evals` and writes results under `workdir/<eval_name>/`. Already-evaluated runs are skipped unless `--config.overwrite_existing_evals=True`.

### Eval tasks

| Task (`config.task`) | Query encoder | Database |
|---|---|---|
| `cross_view` | `encode_query` | Aerial images via `encode_aerial` |
| `classification` | `encode_query` | Learned cell prototypes + `metadata.npz` cell centres |
| `hybrid` | `encode_query` | Aerial images blended with cell prototypes via `cell_idx` |

> **Index level consistency for `hybrid`:**
> Each aerial embedding is blended with its cell prototype using `cell_idx`. The `index_subdir` used at eval must match the cell resolution the model was trained with, so that `cell_idx` values align with the prototype table. Many-to-one is fine — multiple aerial images can share the same `cell_idx`. Set `cell_idx = -1` on an aerial example to skip prototype blending for that entry.

### Reported metrics

Results are averaged over all queries:

- `recall@k@<threshold>` — fraction of queries whose top-k retrieval is within threshold (50m, 100m, 200m, 500m, 1km, 2km, 5km, 10km, 25km, 50km, 100km, 200km, 500km, 750km, 1000km)
- `distance@k` — mean haversine distance to the closest top-k result

---

## Configs

All configs live in `scaling_geoloc/configs/` and use [`ml_collections`](https://github.com/google/ml_collections). Any config field can be overridden from the command line with `--config.<field>=<value>`.

| File | Purpose |
|---|---|
| `configs/train.py` | Training config — model, data, schedule |
| `configs/eval.py` | Evaluation config — experiment path + eval tasks |
| `configs/defaults.py` | Shared factory functions for data, model, eval sub-configs |

### Defaults helpers

```python
from scaling_geoloc.configs import defaults

defaults.data(location)                  # data loading config
defaults.preprocess()                    # image preprocessing sub-config
defaults.geo_localizer()                 # model hyperparameters
defaults.pretrained_backbone(name, ...)  # ViT backbone with pretrained weights
defaults.retrieval_eval(location, bs)    # retrieval eval sub-config
```

---

## Code structure

```
scaling_geoloc/
  train.py               # Training entry point
  eval.py                # Evaluation entry point
  trainer.py             # Experiment class: init, train loop, checkpointing
  load.py                # Model and eval registry
  types.py               # Shared type aliases

  configs/
    train.py             # Training config
    eval.py              # Eval config
    defaults.py          # Shared config factories

  data/
    datasets.py          # build_dataset: Grain data pipeline
    joined_data_source.py  # Merges 4 TFDS sources into one
    transforms.py        # Grain transforms (normalize, resize, filter)
    samplers.py          # Custom Grain samplers
    spec.py              # Dataset field spec

  models/
    hybrid_geolocalization.py  # Main dual-encoder model (GeoLocalizerModel)
    vit.py               # ViT backbone with DINO weight loading
    heads.py             # Projection heads
    base.py              # BaseModel abstract class

  evals/
    retrieval.py         # RetrievalEvaluation: build_index, evaluate_retrieval
    base_eval.py         # BaseEval abstract class

  utils/
    geo.py               # Haversine distance, distance_metrics
    misc.py              # Multi-similarity loss, distributed top-k
    distributed.py       # Sharded similarity search
    augmentations.py     # JAX image augmentations
    grids.py             # S2 cell grid utilities
    io.py                # Checkpoint I/O helpers
```

---

## Tests

```bash
# Default test suite (sets 2 virtual devices via XLA_FLAGS):
python -m pytest

# All tests including train step:
XLA_FLAGS=--xla_force_host_platform_device_count=2 python -m pytest tests/
```

---

## UniGeoCLIP

[`unigeoclip/`](unigeoclip/README.md) contains the code for [UniGeoCLIP](https://gastruc.github.io/unigeoclip),
built on top of this library: a multimodal contrastive model
(street-view / aerial / DSM / text / location) and a downstream evaluation
harness so follow-up work can benchmark their own encoders on **pv4ger**,
**chesapeake** and **PDFM**. Like the rest of the repo, it runs end to end on
CPU against fake data. See [`unigeoclip/README.md`](unigeoclip/README.md).

---

*This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).*
