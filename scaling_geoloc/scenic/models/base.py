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

"""A better base model."""

from typing import Any, Callable, Mapping, Optional

from etils.array_types import Array
import flax
from flax import jax_utils
import flax.linen as nn
import grain.python as grain
import jax
import jax.numpy as jnp
import ml_collections
from scenic.train_lib import train_utils

from scaling_geoloc.scenic import types


class BaseModel:
  """Defines commonalities between all models."""

  required_rngs: tuple[str, ...] = ()

  def __init__(
      self,
      config: ml_collections.ConfigDict,
      dataset_meta_data: dict[str, Any],
      dtype: jnp.dtype = jnp.float32,
  ):
    self.config = config
    self.dataset_meta_data = dataset_meta_data
    self.dtype = dtype
    self.flax_model = self.build_flax_model()

  def loss_metrics_function(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      model_params: Optional[jnp.ndarray] = None,
      model_state: Optional[Any] = None,
      learning_rates: Optional[Mapping[str, float]] = None,
  ) -> types.LossMetricsTuple:
    """Returns the loss and metric dictionaries."""
    raise NotImplementedError('Subclasses must implement metrics.')

  def eval_loss_metrics_function(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      model_params: Optional[jnp.ndarray] = None,
  ) -> types.LossMetricsTuple:
    """Returns the loss and metric dictionaries."""
    return self.loss_metrics_function(pred, batch, model_params=model_params)

  def scale_grad(
      self, grad: types.Grad, learning_rates: types.LearningRates  # pylint: disable=unused-argument
  ) -> types.Grad:
    return grad

  def init_state(
      self,
      model_params: types.ModelParams,
      model_state: types.ModelState,
  ) -> tuple[types.ModelParams, types.ModelState]:
    """Initializes the model state and params (No-op by default)."""
    return model_params, model_state

  def update_state(
      self,
      pred: types.Predictions,  # pylint: disable=unused-argument
      model_params: types.ModelParams,
      model_state: types.ModelState,
      learning_rates: types.LearningRates,  # pylint: disable=unused-argument
  ) -> tuple[types.ModelParams, types.ModelState]:
    """Updates the model state (No-op by default)."""
    return model_params, model_state

  def build_flax_model(self) -> nn.Module:
    raise NotImplementedError('Subclasses must implement build_flax_model().')

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    """Default config for the flax model that is built in `build_flax_model`.

    This function in particular serves the testing functions and supposed to
    provide config that are passed to the flax_model when it's built in
    `build_flax_model` function, e.g., `model_dtype_str`.
    """
    raise NotImplementedError(
        'Subclasses must implement default_flax_model_config().'
    )

  def visualize(
      self, pred: types.Predictions, batch: types.Batch
  ) -> Mapping[str, Array['H W 3'] | Array['B H W 3']]:
    """Visualize the predictions of the model."""
    raise NotImplementedError('Subclasses must implement visualize().')

  def mine_examples(
      self,
      train_state: train_utils.TrainState,
      data_source: grain.RandomAccessDataSource,
      **kwargs,
  ) -> types.DataDict:
    """Mine examples for mining."""
    raise NotImplementedError('Subclasses must implement mine_examples().')

  def replicate_state(
      self,
      train_state: train_utils.TrainState,
  ) -> train_utils.TrainState:
    """Replicate train state."""
    return jax_utils.replicate(train_state)

  def unreplicate_state(
      self,
      train_state: train_utils.TrainState,
  ) -> train_utils.TrainState:
    """Unreplicate train state."""
    return jax.tree.map(lambda x: jax.device_get(x[0]), train_state)

  def inference_fn(
      self,
      method: str,
  ) -> Callable[..., Any]:
    """Inference function."""

    def _inference_fn(
        train_state: train_utils.TrainState,  # replicated
        *args,
        **kwargs,
    ) -> types.Predictions:
      """Inference function."""
      new_rng, _ = jax.random.split(train_state.rng)
      rng_dict = dict(
          zip(
              self.required_rngs,
              jax.random.split(new_rng, len(self.required_rngs)),
          )
      )
      for k in rng_dict:
        rng_dict[k] = train_utils.bind_rng_to_host_device(
            rng_dict[k], axis_name='batch', bind_to='device'
        )
      return self.flax_model.apply(
          {'params': train_state.params, **train_state.model_state},
          *args,
          method=method,
          **kwargs,
      )

    return _inference_fn

  def sync_model_state_across_replicas(
      self,
      train_state: train_utils.TrainState,
  ) -> types.ModelState:
    """Sync the model_state across replicas.

    Forked from scenic.train_lib.train_utils.sync_model_state_across_replicas to
    sync the entire model state (not just batch_stats).

    Args:
      train_state: TrainState; Current state of training.

    Returns:
      Updated model_state that is synced across replicas.
    """

    updated = {
        k: train_utils.pmap_mean(v) for k, v in train_state.model_state.items()
    }
    return flax.core.copy(
        train_state.model_state,
        updated,
    )

  def aggregate_grads(self, grad: types.Grad) -> types.Grad:
    """Aggregate gradients."""
    return jax.lax.pmean(grad, axis_name='batch')
