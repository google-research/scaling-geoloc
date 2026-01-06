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

"""KNN evaluation on ImageNet."""

from collections.abc import Mapping
import functools

from clu import metric_writers
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scenic.projects.dino.datasets import knn_imagenet_dataset  # pylint: disable=unused-import
from scenic.projects.dino.evaluators import eval_knn
from scenic.train_lib import train_utils

from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.evals import base_eval
from scaling_geoloc.scenic.models import base


def get_representation(
    train_state: train_utils.TrainState,
    batch: Mapping[str, jnp.ndarray],
    *,
    flax_model: nn.Module,
    method: str,
    gather_to_host: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Feeds the inputs to the model and returns their representations.

  Args:
    train_state: Dict, the state of training including the current. global_step,
      model_state, rng, and optimizer. The buffer of this argument can be
      donated to the computation.
    batch: A single batch of data from the dataset.
    flax_model: A Flax model.
    method: The method to call on the model.
    gather_to_host: Whether to gather results from all devices to the host,
      rather than leaving them distributed.

  Returns:
    Representation learned by the model for the given inputs.
    If `gather_to_host` is True, these are collected from all hosts.
  """
  variables = {'params': train_state.params, **train_state.model_state}

  _, features = flax_model.apply(
      variables, batch['inputs'], train=False, method=method
  )
  representation = features[..., 0, :]
  if gather_to_host:
    representation = jax.lax.all_gather(representation, 'batch')
    batch = jax.lax.all_gather(batch, 'batch')
  return representation, batch['label'], batch['batch_mask']


class KNNImageNetEval(base_eval.BaseEval):
  """KNN evaluation on ImageNet1K."""

  def __init__(self, model: base.BaseModel, config: ml_collections.ConfigDict):
    self.model = model
    self.config = config

    self.representation_fn = functools.partial(
        get_representation,
        flax_model=model.flax_model,
        method=config.method,
    )

    self.knn_classifier = eval_knn.KnnEvaluator(self.representation_fn, config)

  def evaluate(
      self,
      train_state: train_utils.TrainState,
      writer: metric_writers.MetricWriter | None = None,
      workdir: str | None = None,
  ) -> types.EvalResults:
    summaries = self.knn_classifier.run(train_state)
    visualizations = {}
    return {}, summaries, visualizations
