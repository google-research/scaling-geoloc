# UniGeoCLIP: Unified Geospatial Contrastive Learning

[![arXiv](https://img.shields.io/badge/arXiv-2604.11668-b31b1b.svg)](https://arxiv.org/abs/2604.11668)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://gastruc.github.io/unigeoclip)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](../LICENSE)

*EarthVision Workshop @ CVPR 2026* — Guillaume Astruc, Eduard Trulls, Jan Hosang,
Loïc Landrieu, Paul-Edouard Sarlin

Official code for **UniGeoCLIP**, built on the public `scaling_geoloc` library. It
provides a multimodal contrastive model that aligns five co-located modalities —
street-view (`sv`), aerial RGB, DSM height (`dsm`), text, and location (`latlng`)
— and a downstream evaluation harness.

The full train + eval pipeline runs on CPU against procedurally generated fake
data, so it is reproducible without any private data.

## Setup

Run all commands from the repository root (the parent of `unigeoclip/`).

```bash
pip install -r requirements.txt
```

`scenic` and [`big_vision`](https://github.com/google-research/big_vision) are
source-tree packages; if the git install fails, clone them onto `PYTHONPATH`.
Backbones default to the in-repo ViT — `big_vision` is only needed for the
SigLIP2 towers.

## Quickstart (fake data, CPU)

[`demo.ipynb`](demo.ipynb) walks through the whole flow. From the shell:

```bash
# fake downstream data -> datasets/{pv4ger_classification, m-chesapeake, mdas, pdfm}/
python -m unigeoclip.data.generate_fake_downstream_data --output_dir=datasets

# train (procedural fake data, writes a checkpoint to --workdir)
python -m unigeoclip.train \
    --config=unigeoclip/configs/contrastive_merge.py:tiny \
    --workdir=/tmp/uni_run --config.num_training_steps=20 \
    --config.batch_size_per_device=4 --config.checkpoint_interval=20

# evaluate that checkpoint
python -m unigeoclip.eval \
    --config=unigeoclip/configs/eval_downstream.py \
    --workdir=/tmp/uni_eval --config.experiment=/tmp/uni_run
```

The `tiny` (ViT) checkpoint covers pv4ger + MDAS + PDFM; drop `chesapeake` from
`eval_downstream.py`'s `get_config()`, as it needs a SigLIP aerial tower (below).

## Training

```bash
python -m unigeoclip.train \
    --config=unigeoclip/configs/contrastive_merge.py:<preset> \
    --workdir=/tmp/uni_run
```

`<preset>` ∈ {`tiny`, `Ti`, `S`, `B`, `L`}; any field is overridable (e.g.
`--config.num_training_steps`, `--config.model.compose_loss=multisim`).
Modalities come from the procedural `MultiModalDummyDataSource` by default; to
train from files, write on-disk shards with
`unigeoclip.data.generate_fake_multimodal_data` and set `--config.data.location`.
For real data, provide a `grain.RandomAccessDataSource` yielding the TFDS spec in
`data/spec.py` (`sv`, `aerial`, `dsm`, `latlng`, precomputed `text`).

## Evaluation

`python -m unigeoclip.eval` runs each task in `test_evals` against one
checkpoint, pulling features through the model's encoder interface:

| Eval (dataset) | Model method | Metric |
|---|---|---|
| `classif_eval` (pv4ger) | `get_aerial_embeddings` | linear probe, kNN@{1,5,10} |
| `semseg_eval` (chesapeake) | `get_dense_aerial_embeddings` | mIoU (DPT head) |
| `dsm_eval` (MDAS) | `get_dense_dsm_embeddings` | mIoU, acc |
| `pdfm_eval` (PDFM) | `get_location_embeddings` | mean test R² |

`contrastive_merge` runs all four with a SigLIP aerial tower
(`configs/siglip_aerial.py`); with the in-repo ViT it covers pv4ger + MDAS + PDFM
(chesapeake's DPT head needs the SigLIP tower's multi-depth dense features). Edit
`eval_downstream.py`'s `get_config()` to keep the tasks your checkpoint supports.
To evaluate your own encoder, implement these methods on a
`scaling_geoloc.models.base.BaseModel` and register it in `load.py`
(`models/contrastive_merge.py` is the reference).

### Geolocation retrieval

`configs/eval_geolocation_retrieval.py` scores cross-modal retrieval (street-view
↔ aerial ↔ location) and Haversine geolocation over the procedural multimodal
data, so it needs a `contrastive_merge` checkpoint with the `latlng` modality:

```bash
python -m unigeoclip.eval \
    --config=unigeoclip/configs/eval_geolocation_retrieval.py \
    --workdir=/tmp/eval_geoloc --config.experiment=/tmp/uni_run
```

`cell_codes` are grid-cell sizes in degrees: each bins the evaluation locations
onto a regular lat/lng grid and retrieves against the distinct cell centers (a
coarser grid → a smaller, coarser gallery). Set
`--config.test_evals.geoloc.data.location` to read real 5-modality shards instead
of the dummy source.

### Real datasets

Point each task's `data.data_dir` at data laid out like the fake generator emits:

- **pv4ger / chesapeake** — per-tile `.hdf5` (`Red`/`Green`/`Blue` [+ `label`]) +
  `default_partition.json` (+ `label_map.json` for pv4ger).
- **MDAS** — per-sub-area GeoTIFFs `mdas/sub_area_{1,2,3}/{3K_DSM,osm_landuse}_*.tif`
  from the open [MDAS](https://mediatum.ub.tum.de/1657312) dataset.
- **PDFM** — the public
  [`conus27.csv`](https://github.com/google-research/population-dynamics/blob/master/data/benchmarks/conus27.csv)
  dropped into `datasets/pdfm/`.

## Tests

```bash
python -m pytest tests/test_unigeoclip_train.py tests/test_unigeoclip_eval.py
```

Two CPU devices are set via `pytest.ini`; the chesapeake DPT probe takes ~2 min.

## Citation

```bibtex
@inproceedings{astruc2026unigeoclip,
  title     = {UniGeoCLIP: Unified Geospatial Contrastive Learning},
  author    = {Astruc, Guillaume and Trulls, Eduard and Hosang, Jan
               and Landrieu, Lo{\"i}c and Sarlin, Paul-Edouard},
  booktitle = {EarthVision Workshop, CVPR},
  year      = {2026}
}
```

## License

Released under the [Apache 2.0 License](../LICENSE).

## Acknowledgements

Builds on the `scaling_geoloc` library in this repo,
[scenic](https://github.com/google-research/scenic),
[big_vision](https://github.com/google-research/big_vision), and
[GeoCLIP](https://github.com/VicenteVivan/geo-clip).
