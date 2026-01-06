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

"""Class-cooccurrence evaluation."""

import collections
from collections.abc import Callable
import functools
from typing import Any, TypeAlias
from absl import logging
from clu import metric_writers
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.train_lib import train_utils
from sklearn import metrics as skmetrics
from scaling_geoloc.scenic import datasets
from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.evals import base_eval
from scaling_geoloc.scenic.models import base
from scaling_geoloc.utils import misc

Batch: TypeAlias = Any
Predictions: TypeAlias = Any


def inference_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    method: str,
    model_rngs: tuple[str, ...] = (),
    debug: bool | None = False,
    capture_intermediates: bool = True,
) -> Predictions:
  """Encodes a set of images."""
  new_rng, _ = jax.random.split(train_state.rng)
  rng_dict = dict(zip(model_rngs, jax.random.split(new_rng, len(model_rngs))))
  for k in rng_dict:
    rng_dict[k] = train_utils.bind_rng_to_host_device(
        rng_dict[k], axis_name="batch", bind_to="device"
    )
  variables = {"params": train_state.params, **train_state.model_state}
  extra_mutable = ["intermediates"] if capture_intermediates else []
  (_, backbone_output), _ = flax_model.apply(
      variables,
      batch,
      train=False,
      method=method,
      mutable=["batch_stats"] + extra_mutable,
      debug=debug,
      rngs=rng_dict,
  )

  return backbone_output[..., 1:, :]


def prepare_inference_step(
    inference_step_fn: Callable[..., Any],
    train_state: train_utils.TrainState,
) -> Callable[[Batch], Predictions]:
  """Prepare simple inference step on a single image."""

  def _inference_step(batch: Batch) -> Predictions:
    batch = dataset_utils.shard(batch)
    pred = inference_step_fn(
        train_state,
        batch,
    )
    pred = dataset_utils.unshard(pred)
    return pred

  return _inference_step


class ClassCooccurrenceEvaluation(base_eval.BaseEval):
  """Class-cooccurrence evaluator."""

  def __init__(self, model: base.BaseModel, config: ml_collections.ConfigDict):
    self.config = config
    self.model = model
    transformations, metadata = datasets.transformations_from_config(
        config.data
    )
    self._dataset = datasets.create_dataset(
        config.data,
        batch_size=config.batch_size_eval,
        batch_size_eval=config.batch_size_eval,
        transformations=transformations,
        meta_data=metadata,
        seed=155,
    )

    self.eval_iter = self._dataset.valid_iter()
    self.initial_iter_state = (
        self.eval_iter.get_state()  # pytype: disable=attribute-error
    )

    self.num_steps = (
        config.get("num_samples", self._dataset.meta_data["num_valid_examples"])
        // config.batch_size_eval
    )

    self.num_classes = 46

    self.inference_step_pmapped = jax.pmap(
        functools.partial(
            inference_step,
            flax_model=model.flax_model,
            method=self.config.method,
            capture_intermediates=False,
            model_rngs=(),
        ),
        axis_name="batch",
        # We can donate the eval_batch's buffer.
        donate_argnums=(1,),
    )

  def evaluate(
      self,
      train_state: train_utils.TrainState,
      writer: metric_writers.MetricWriter | None = None,
      workdir: str | None = None,
  ) -> types.EvalResults:
    """Evaluates the model."""
    results = collections.defaultdict(list)
    summaries = collections.defaultdict(list)
    visualizations = collections.defaultdict(list)
    representation_fn = prepare_inference_step(
        self.inference_step_pmapped, train_state
    )

    # ensure that we always start from the beginning of the dataset
    logging.info("Resetting eval iterator.")
    self.eval_iter.set_state(
        self.initial_iter_state
    )  # pytype: disable=attribute-error

    logging.info("Running Class-Cooccurrence Evaluation...")
    rng = jax.random.PRNGKey(155)
    for i in range(self.num_steps):
      batch = next(self.eval_iter)
      if i % 20 == 0:
        logging.info("Extracting... [%d/%d]", i, self.num_steps)
      batch = misc.filter_batch_for_jit(batch)
      if self.config.get("resize_factor", 1.0) > 1.0:
        batch = misc.resize_image_like(
            batch,
            self.config.resize_factor,
            antialias=self.config.get("antialias", True),
        )
      batch = jax.tree.map(
          lambda x, y: jax.lax.collapse(jnp.concatenate([x, y], axis=1), 0, 2),
          batch["map"]["views"],
          batch["query"]["views"],
      )

      valid_points = batch["pixel_mask"].clip(min=0)
      ph, pw = self.model.config.vit.patches.size
      features = representation_fn(batch["image_pixels"])
      labels = misc.crop_to_multiple(batch["semantic_raster"], ph)
      h, w = labels.shape[-3:-1]
      nh, nw = h // ph, w // pw
      features = misc.sequence_to_image(features, nh, nw)
      features = jax.image.resize(
          features, labels.shape[:-1] + features.shape[-1:], "linear"
      )

      b = labels.shape[0]
      valid_points = valid_points.reshape(b, -1)
      labels = labels.reshape(b, -1)
      features = features.reshape(b, -1, features.shape[-1])
      rng, rng_perm = jax.random.split(rng)
      isample = jax.random.permutation(rng_perm, labels.shape[-1])  # h * w
      isample1 = isample[: self.config.n_feat_sample]
      isample2 = isample[
          self.config.n_feat_sample : 2 * self.config.n_feat_sample
      ]

      sample1 = jnp.take(features, isample1, axis=-2)
      sample2 = jnp.take(features, isample2, axis=-2)
      del features

      sample1 /= jnp.linalg.norm(sample1, axis=-1, keepdims=True)
      labels1 = jnp.take(labels, isample1, axis=-1)
      labels1 = jax.nn.one_hot(labels1, self.num_classes)
      valid_points1 = jnp.take(valid_points, isample1, axis=-1)[:, :, None]
      sample2 /= jnp.linalg.norm(sample2, axis=-1, keepdims=True)
      labels2 = jnp.take(labels, isample2, axis=-1)
      labels2 = jax.nn.one_hot(labels2, self.num_classes)
      valid_points2 = jnp.take(valid_points, isample2, axis=-1)[:, :, None]

      correlations = jnp.einsum("bnc,bmc->bnm", sample1, sample2).reshape(-1)
      gtcorrelations = jnp.einsum("bnc,bmc->bnm", labels1, labels2).reshape(-1)
      all_valid_points = jnp.einsum(
          "bnc,bmc->bnm", valid_points1, valid_points2
      ).reshape(-1)
      results["preds"].append(np.array(correlations))
      results["targets"].append(np.array(gtcorrelations))
      results["all_valid_points"].append(np.array(all_valid_points))
    preds = np.concatenate(results["preds"], axis=0)
    targets = np.concatenate(results["targets"], axis=0)
    if "all_valid_points" in results:
      all_valid_points = np.concatenate(results["all_valid_points"], axis=0)
      idx = all_valid_points == 1
      preds = preds[idx]
      targets = targets[idx]
    preds -= preds.min()
    preds /= preds.max()
    precisions, recalls, _ = skmetrics.precision_recall_curve(
        targets,
        preds,
    )
    f1_scores = 2 * precisions * recalls / (precisions + recalls)
    average_precision = skmetrics.average_precision_score(targets, preds)

    summaries = {
        k: np.concatenate(v, axis=0).mean().item() for k, v in summaries.items()
    }

    summaries["AP"] = average_precision.item()
    summaries["AUC-ROC"] = skmetrics.roc_auc_score(targets, preds).item()
    summaries["max-F1"] = f1_scores.max().item()

    summaries = {k: round(v * 100, 1) for k, v in summaries.items()}
    visualizations = {
        k: np.concatenate(v, axis=0) for k, v in visualizations.items()
    }

    return {}, summaries, visualizations
