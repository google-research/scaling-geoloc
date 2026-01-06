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

"""Correspondence evaluation."""

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
from scaling_geoloc.scenic import datasets
from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.evals import base_eval
from scaling_geoloc.scenic.models import base
from scaling_geoloc.utils import gaussians
from scaling_geoloc.utils import misc
from scaling_geoloc.utils import plot

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

  features = backbone_output[..., 1:, :]
  return features


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


class CorrespondenceEvaluation(base_eval.BaseEval):
  """Correspondence evaluator."""

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
        seed=config.get("seed", 155),
    )

    self.eval_iter = self._dataset.valid_iter()
    # pytype: disable=attribute-error
    self.initial_iter_state = self.eval_iter.get_state()
    # pytype: enable=attribute-error

    self.num_steps = (
        config.get("num_samples", self._dataset.meta_data["num_valid_examples"])
        // config.batch_size_eval
    )

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
    summaries = collections.defaultdict(list)
    visualizations = collections.defaultdict(list)
    representation_fn = prepare_inference_step(
        self.inference_step_pmapped, train_state
    )

    # ensure that we always start from the beginning of the dataset
    logging.info("Resetting eval iterator.")
    # pytype: disable=attribute-error
    self.eval_iter.set_state(self.initial_iter_state)
    # pytype: enable=attribute-error

    logging.info("Running Correspondence Evaluation...")
    for i in range(self.num_steps):
      batch = next(self.eval_iter)
      if i % 20 == 0:
        logging.info("Extracting... [%d/%d]", i, self.num_steps)
      batch = misc.filter_batch_for_jit(batch)
      if self.config.get("resize_factor", 1.0) != 1.0:
        batch = misc.resize_image_like(
            batch,
            self.config.resize_factor,
            antialias=self.config.get("antialias", True),
        )
      maps = jax.tree.map(lambda x: x[:, 0], batch["map"]["views"])
      query = jax.tree.map(lambda x: x[:, 0], batch["query"]["views"])
      features = representation_fn(
          jnp.stack([maps["image_pixels"], query["image_pixels"]], axis=1)
      )
      map_features = features[:, 0]
      query_features = features[:, 1]
      for scene in [maps, query]:
        scene["w_lidar_points"] = misc.lidar_depth_to_xyz(
            scene["lidar_depth"],
            types.create_jaxcam(
                scene["cam_t_w"],
                scene["camera"],
                use_inverted_distortion=True,
            ),
        )

      h, w = maps["image_pixels"].shape[-3:-1]
      ph, pw = self.model.config.vit.patches.size
      nh, nw = h // ph, w // pw
      # L2 normalize and reshape to image-like.
      map_features = misc.l2_normalize(
          misc.sequence_to_image(map_features, nh, nw)
      )
      query_features = misc.l2_normalize(
          misc.sequence_to_image(query_features, nh, nw)
      )
      map_gaussians = gaussians.gaussians_from_points(
          w_points=misc.patchify(maps["w_lidar_points"], ph, pw),
          weights=misc.patchify(maps["lidar_mask"], ph, pw),
      )
      query_gaussians = gaussians.gaussians_from_points(
          w_points=misc.patchify(query["w_lidar_points"], ph, pw),
          weights=misc.patchify(query["lidar_mask"], ph, pw),
      )

      weights = gaussians.batched_probabilities_between_gaussians(
          map_gaussians, query_gaussians
      )  # b x lm x lq

      pred_matches = misc.match_features(
          map_features,
          query_features,
          **self.config.matching.to_dict(),
      )
      gt_matches = misc.matches_from_similarity(
          weights, threshold=0.001, mutual_check=True
      )

      true_positive = (gt_matches["matches_i"] == pred_matches["matches_i"]) & (
          gt_matches["matches_i"] > -1
      )
      summaries["true_positive"].append(true_positive.sum((-1, -2)))
      recall = (
          misc.masked_mean(
              true_positive,
              gt_matches["matches_i"] > -1,
              axis=(-1, -2),
          )
          * 100.0
      )

      precision = (
          misc.masked_mean(
              true_positive,
              pred_matches["matches_i"] > -1,
              axis=(-1, -2),
          )
          * 100.0
      )
      summaries["recall"].append(recall)
      summaries["precision"].append(precision)
      summaries["num_matches"].append(
          (pred_matches["matches_i"] > -1).sum((-1, -2))
      )
      summaries["num_gt_matches"].append(
          (gt_matches["matches_i"] > -1).sum((-1, -2))
      )

      kp1 = pred_matches["keypoints_j"]  # Grid centers.
      m_kp1 = np.take_along_axis(kp1, pred_matches["matches_i"], axis=-2)
      m_kp1_gt = np.take_along_axis(kp1, gt_matches["matches_i"], axis=-2)

      valid = pred_matches["matches_i"] > -1
      valid_gt = gt_matches["matches_i"] > -1

      d_px = np.sqrt(((m_kp1 - m_kp1_gt) ** 2).sum(-1, keepdims=True))

      for th in [1, 1.5, 2]:
        summaries[f"recall@{th}"].append(
            misc.masked_mean(
                (d_px <= (th + 1.0e-5)) & valid, valid_gt, axis=(-1, -2)
            )
            * 100.0
        )
        summaries[f"precision@{th}"].append(
            misc.masked_mean(
                (d_px <= (th + 1.0e-5)) & valid_gt, valid, axis=(-1, -2)
            )
            * 100.0
        )

      if i * self.config.batch_size_eval < self.config.get("num_plots", 1):
        images = np.concatenate(
            [maps["image_pixels"], query["image_pixels"]], axis=-2
        )
        joined = jax.tree.map(
            lambda x, y: np.stack([x, y], axis=0), maps, query
        )
        visualizations["correspondences"].append(
            misc.batch_draw_matches(
                images,
                pred_matches["keypoints_i"] * ph,
                pred_matches["keypoints_j"] * pw,
                pred_matches["matches_i"],
                np.where(
                    pred_matches["matches_i"] == gt_matches["matches_i"],
                    1.0,
                    -1.0,
                )
                * (pred_matches["matches_i"] > -1),
                draw="points+lines",
                color_by_coords=False,
            )
        )

        visualizations["all_correspondences"].append(
            misc.batch_draw_matches(
                images,
                pred_matches["keypoints_i"] * ph,
                pred_matches["keypoints_j"] * pw,
                pred_matches["matches_i"],
                pred_matches["matches_i"] > -1,
                draw="points+lines",
                color_by_coords=True,
            )
        )

        visualizations["gt_correspondences"].append(
            misc.batch_draw_matches(
                images,
                pred_matches["keypoints_i"] * ph,
                pred_matches["keypoints_j"] * pw,
                gt_matches["matches_i"],
                gt_matches["matches_i"] > -1,
                draw="points+lines",
                color_by_coords=True,
            )
        )

        visualizations["gt_depth"].append(
            np.concatenate(
                misc.alpha_blend(
                    misc.to_grayscale(joined["image_pixels"]),
                    plot.apply_colormap(
                        joined["lidar_depth"][..., 0],
                        cmap="turbo_r",
                        vmin=0.0,
                        vmax=30.0,
                    ),
                    joined["lidar_mask"],
                ),
                axis=-2,
            )
        )

        features = np.concatenate([map_features, query_features], axis=-2)
        features_pca = misc.features_to_rgb(features, normalize=False)
        features_pca = jax.image.resize(
            features_pca,
            images.shape,
            "nearest",
            True,
        )

        visualizations["features_pca"].append(features_pca)

    summaries = {
        k: np.concatenate(v, axis=0).mean().item() for k, v in summaries.items()
    }

    visualizations = {
        k: np.concatenate(v, axis=0)[: self.config.get("num_plots", 1)]
        for k, v in visualizations.items()
    }

    return {}, summaries, visualizations
