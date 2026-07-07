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

"""PDFM downstream eval: location-embedding -> socioeconomic-target regression.

Ported from the paper's `eval.ipynb`. For each region it extracts a location
embedding via the model's ``get_location_embeddings`` method, then for every
socioeconomic target it fits a ridge regression on the train split, sweeps the
regularization strength, selects the best on the validation split and reports
the corresponding test R^2. The reported metric is the mean test R^2 over all
targets (NaN targets are masked per-region).

The evaluated model must expose ``get_location_embeddings(latlng) -> [N, D]``
(e.g. the ``latlng`` modality of ``contrastive_merge``).
"""

from absl import logging
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.train_lib import train_utils
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from scaling_geoloc import types
from scaling_geoloc.evals import base_eval
from scaling_geoloc.models import base
from unigeoclip.data.pdfm_source import PDFMDataSource

# Ridge regularization strengths swept per target (as in eval.ipynb).
_ALPHAS = [10 ** (i - 1) for i in range(10)]


def _location_embeddings(
    train_state: train_utils.TrainState,
    model: base.BaseModel,
    latlngs: np.ndarray,
    batch_size: int,
) -> np.ndarray:
  """Extracts location embeddings for an array of latlngs (batched pmap)."""

  def _reps(train_state, latlngs):
    variables = {'params': train_state.params, **train_state.model_state}
    return model.flax_model.apply(
        variables, latlngs=latlngs,
        method=model.flax_model.get_location_embeddings,
    )

  pmapped = jax.pmap(_reps, axis_name='batch')
  num_devices = jax.local_device_count()
  out = []
  for i in range(0, latlngs.shape[0], batch_size):
    chunk = np.asarray(latlngs[i : i + batch_size], np.float32)
    n = chunk.shape[0]
    pad = (-n) % num_devices
    if pad:
      chunk = np.concatenate([chunk, np.zeros((pad, 2), np.float32)], axis=0)
    reps = dataset_utils.unshard(pmapped(train_state, dataset_utils.shard(chunk)))
    out.append(np.asarray(reps)[:n])
  return np.concatenate(out, axis=0)


class PDFMEvaluation(base_eval.BaseEval):
  """Ridge-regression probe of location embeddings against PDFM targets."""

  def __init__(self, model: base.BaseModel, config: ml_collections.ConfigDict):
    self.model = model
    self.config = config

  def evaluate(
      self,
      train_state: train_utils.TrainState,
      writer: metric_writers.MetricWriter | None = None,
      workdir: str | None = None,
  ) -> types.EvalResults:
    source = PDFMDataSource(self.config.data.data_dir)
    latlngs = source.latlngs
    targets = source.targets  # [N, T], may contain NaNs

    logging.info('Extracting PDFM location embeddings for %d regions.', len(latlngs))
    reps = _location_embeddings(
        train_state, self.model, latlngs, self.config.get('batch_size_eval', 256)
    )

    # Same splits as eval.ipynb: 20% test, then 12.5% of the rest as validation.
    idx = np.arange(reps.shape[0])
    train_val_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.125, random_state=42
    )

    per_target_r2 = []
    for t, name in enumerate(source.target_names):
      yt = targets[:, t]

      def _split(split_idx, yt=yt):
        m = split_idx[~np.isnan(yt[split_idx])]
        return m

      tr, va, te = _split(train_idx), _split(val_idx), _split(test_idx)
      if len(tr) == 0 or len(va) == 0 or len(te) == 0:
        continue

      r2_vals, r2_tests = [], []
      for alpha in _ALPHAS:
        reg = Ridge(alpha=alpha).fit(reps[tr], yt[tr])
        r2_vals.append(r2_score(yt[va], reg.predict(reps[va])))
        r2_tests.append(r2_score(yt[te], reg.predict(reps[te])))
      best = r2_tests[int(np.argmax(r2_vals))]
      per_target_r2.append(best)
      logging.info('PDFM target %s: test R^2 = %.4f', name, best)

    metrics = {'r2_mean': float(np.mean(per_target_r2)) if per_target_r2 else 0.0}
    return {}, metrics, {}
