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

"""Trainer for LaMa models."""

from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
import functools
import gc
import os
import time
from typing import Any

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
import flax
from flax import jax_utils
import flax.linen as nn
from flax.training import checkpoints
import grain.python as grain
import jax
from jax.example_libraries import optimizers as jax_optimizers
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from scenic.common_lib import debug_utils
from scenic.dataset_lib import dataset_utils
from scenic.google.xm import xm_utils
from scenic.model_lib.base_models import model_utils
from scenic.train_lib import lr_schedules
from scenic.train_lib import optax as scenic_optax
from scenic.train_lib import optimizers
from scenic.train_lib import pretrain_utils as scenic_pretrain_utils
from scenic.train_lib import train_utils

from scaling_geoloc.scenic import load
from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.evals import base_eval
from scaling_geoloc.scenic.models import base
from scaling_geoloc.utils import io as lama_io
from scaling_geoloc.utils import misc
from google3.pyglib import gfile

# Always use Orbax checkpointing.
flax.config.update('flax_use_orbax_checkpointing', True)


def initialize_model(
    *,
    model_def: nn.Module,
    dummy_input: types.Batch,
    rng: jnp.ndarray,
    model_rngs: tuple[str, ...] = (),
    freeze: bool = True,
    train: bool = True,
    method: Callable[..., Any] | str | None = None,
) -> tuple[types.PyTree, types.PyTree, int]:
  """Initializes parameters and model state.

  Differently from train_utils.initialize_model, expects a dummy batch as input
  instead of building one from (shape, dtype) specs. This allows us to use batch
  dictionaries with arbitrary objects, not only jax arrays.

  Args:
    model_def: Definition of a model.
    dummy_input: Batch dictionary.
    rng: Jax rng key.
    model_rngs: The rngs to use for the model.
    freeze: Whether to freeze the model state and params.
    train: Whether to initialize the model in train mode.
    method: The method to initialize. Defaults to `__call__`.

  Returns:
    A tuple [init_params, init_model_state, num_trainable_params].
  """

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @functools.partial(jax.jit, backend='cpu')
  def _initialize_model(rngs):
    """Initialization function to be jitted."""
    init_model_state, init_params = flax.core.pop(
        model_def.init(
            rngs,
            dummy_input,
            train=train,
            debug=False,
            method=method,
        ),
        'params',
    )
    init_model_state = {
        'model_stats': {},
        **init_model_state,
    }
    return init_params, init_model_state

  rngs = dict(
      zip(('params',) + model_rngs, jax.random.split(rng, 1 + len(model_rngs)))
  )
  init_params, init_model_state = _initialize_model(rngs)
  # Pop out params rng:
  rngs.pop('params')

  # Count number of trainable parameters:
  num_trainable_params = debug_utils.log_param_shapes(init_params)
  if freeze:
    init_params = flax.core.freeze(init_params)
    init_model_state = flax.core.freeze(init_model_state)
  return init_params, init_model_state, num_trainable_params


def reduce_metrics(
    metrics: types.MetricsDict, mask: jnp.ndarray | None = None
) -> types.AggregatedMetricsDict:
  """Aggregates the metrics across examples and accelerators."""
  aggregated = {}
  for k, v in metrics.items():
    mask = mask if mask is not None else jnp.ones_like(v)
    metric_mask = jnp.logical_and(mask.astype(bool), jnp.isfinite(v))
    num_examples = metric_mask.sum()
    metric_tuple = (jnp.sum(v, where=metric_mask), num_examples)
    aggregated[k] = model_utils.psum_metric_normalizer(metric_tuple)
  return aggregated  # pytype: disable=bad-return-type  # jax-ndarray


def train_step(
    train_state: train_utils.TrainState,
    batch: types.Batch,
    *,
    flax_model: nn.Module,
    loss_metrics_fn: types.LossMetricsFn,
    state_update_fn: types.UpdateFn,
    aggregate_grads_fn: types.AggregateGradsFn,
    scale_grad_fn: types.ScaleGradFn,
    lr_fns: Mapping[str, optax.Schedule],
    config: ml_collections.ConfigDict,
    model_rngs: tuple[str, ...] = (),
    debug: bool | None = False,
) -> tuple[train_utils.TrainState, types.AggregatedMetricsDict, dict[str, Any]]:
  """Runs a single step of training.

  Given the state of the training and a batch of data, computes
  the loss and updates the parameters of the model.

  Note that in this code, the buffers of the first (train_state) and second
  (batch) arguments are donated to the computation.

  Args:
    train_state: The state of training including the current global_step,
      model_state, rng, params, and optimizer. The buffer of this argument can
      be donated to the computation.
    batch: A single batch of data. The buffer of this argument can be donated to
      the computation.
    flax_model: A Flax model.
    loss_metrics_fn: A function that, given predictions, a batch, and parameters
      of the model, calculates the loss and metrics dictionaries.
    state_update_fn: A function that updates parameters and model states.
    aggregate_grads_fn: A function that aggregates the gradients over devices.
    scale_grad_fn: A function that scales the gradients.
    lr_fns: The learning rate fns used for logging.
    config: Configurations of the experiment.
    model_rngs: The rngs to use for the model.
    debug: Whether the debug mode is enabled during training. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Updated state of training and computed metrics and some training logs.
  """
  new_rng, _ = jax.random.split(train_state.rng)

  global_step = jnp.array([train_state.global_step])
  lr_params = {name: rate_fn(global_step) for name, rate_fn in lr_fns.items()}

  rng_dict = dict(zip(model_rngs, jax.random.split(new_rng, len(model_rngs))))
  for k in rng_dict:
    rng_dict[k] = train_utils.bind_rng_to_host_device(
        rng_dict[k], axis_name='batch', bind_to='device'
    )

  def training_loss_fn(params):
    variables = {'params': params, **train_state.model_state}
    pred, new_model_state = flax_model.apply(
        variables,
        batch,
        mutable=['batch_stats'],
        train=True,
        debug=debug,
        rngs=rng_dict,
    )
    new_model_state = {**train_state.model_state, **new_model_state}
    losses, metrics = loss_metrics_fn(
        pred,
        batch,
        params,
        new_model_state,
        lr_params | rng_dict,
    )
    loss = losses['total'].mean(where=batch.get('batch_mask'))
    return loss, (pred, new_model_state, losses, metrics)

  grad_fn = jax.grad(training_loss_fn, has_aux=True)
  grad, (pred, new_model_state, losses, metrics) = grad_fn(train_state.params)

  grad = aggregate_grads_fn(grad)

  if config.get('max_grad_norm') is not None:
    grad = jax_optimizers.clip_grads(grad, config.max_grad_norm)

  grad = scale_grad_fn(grad, lr_params)
  if train_state.tx is None:
    raise ValueError('train_state.tx cannot be None.')
  updates, new_opt_state = train_state.tx.update(
      grad, train_state.opt_state, train_state.params
  )

  if config.get('avoid_weight_decay_on_frozen_params', False):
    updates = jax.tree_util.tree_map(
        lambda u, g: (optax.global_norm(g) > 0.0) * u,
        updates,
        grad,
    )

  new_params = optax.apply_updates(train_state.params, updates)

  # Apply updates like EMA and moving averages.
  new_params, new_model_state = state_update_fn(
      pred,
      new_params,
      new_model_state,
      lr_params,
  )

  training_logs = {
      'l2/grads': optax.global_norm(grad),
      'l2/updates': optax.global_norm(jax.tree_util.tree_leaves(updates)),
  }

  # Log the update norms of student and teacher if available.
  for module_name in ['student', 'teacher']:
    if module_name in updates:
      training_logs[f'l2/updates/{module_name}'] = optax.global_norm(
          jax.tree_util.tree_leaves(updates[module_name])
      )

  if 'model_stats' in new_model_state:
    for k, v in new_model_state['model_stats'].items():
      if isinstance(v, jnp.ndarray) and v.dtype == jnp.float32:
        training_logs[f'model_stats/{k}_norm'] = jnp.linalg.norm(v)
        training_logs[f'model_stats/{k}_abs_mean'] = jnp.abs(v).mean()
        training_logs[f'model_stats/{k}_std'] = jnp.std(v)
        training_logs[f'model_stats/{k}_delta_norm'] = jnp.linalg.norm(
            v - train_state.model_state['model_stats'][k]
        )

  for name, lr_fn in lr_fns.items():
    training_logs[f'learning_rate/{name}'] = lr_fn(global_step)

  for k, v in losses.items():
    metrics[f'loss/{k}'] = v
  metrics = reduce_metrics(metrics, batch.get('batch_mask'))

  is_fin = [jax.lax.is_finite(g).all() for g in jax.tree_util.tree_leaves(grad)]
  is_fin = jnp.all(jnp.stack(is_fin))

  # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
  # params should be restored (= skip this step).
  new_opt_state = jax.tree_util.tree_map(
      functools.partial(jnp.where, is_fin),
      new_opt_state,
      train_state.opt_state,
  )
  new_params = jax.tree_util.tree_map(
      functools.partial(jnp.where, is_fin), new_params, train_state.params
  )
  training_logs['is_finite'] = is_fin

  # Plot l2 norm of params after we got rid of inf and NaN.
  training_logs['l2/params'] = optax.global_norm(
      jax.tree_util.tree_leaves(new_params)
  )

  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=train_state.global_step + 1,
      opt_state=new_opt_state,
      params=new_params,
      model_state=new_model_state,
      rng=new_rng,
  )

  return new_train_state, metrics, training_logs


def resize_batch(
    batch: types.Batch,
    step: int,
    resize_sequence: list[tuple[int, int] | None],
) -> types.Batch:
  """Resizes images in the batch based on a cycling `resize_sequence`.

  This is useful for training at different sizes. On each training step, a size
  is selected from `resize_sequence` (with `None` to represent no
  resizing) based on the current step number.

  Args:
    batch: The input batch of data.
    step: The current training step.
    resize_sequence: A list of `(height, width)` tuples or None, defining the
      resize cycle. If empty, no resizing is performed.

  Returns:
    The batch with potentially resized images.
  """
  if not resize_sequence:
    return batch

  cycle_index = step % len(resize_sequence)
  target_size = resize_sequence[cycle_index]

  if target_size is not None:
    new_height, new_width = target_size
    img = batch['query']['views']['image_pixels']
    target_shape = (
        *img.shape[:-3],
        new_height,
        new_width,
        img.shape[-1],
    )
    batch['query']['views']['image_pixels'] = jax.image.resize(
        img,
        target_shape,
        method='nearest',
    )
  return batch


def eval_step(
    train_state: train_utils.TrainState,
    batch: types.Batch,
    *,
    flax_model: nn.Module,
    loss_metrics_fn: types.LossMetricsFn,
    model_rngs: tuple[str, ...] = (),
    debug: bool | None = False,
    capture_intermediates: bool = True,
    gather_to_host: bool = False,
) -> tuple[types.Predictions, types.AggregatedMetricsDict]:
  """Runs a single step of evaluation.

  Args:
    train_state: TrainState, the state of training including the current
      global_step, model_state, rng, params and optimizer state. The buffer of
      this argument can be donated to the computation.
    batch: A single batch of data.
    flax_model: A Flax model.
    loss_metrics_fn: A function that, given predictions, a batch, and parameters
      of the model, calculates the loss and metrics dictionaries.
    model_rngs: The rngs to use for the model.
    debug: Whether the debug mode is enabled during evaluation. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.
    capture_intermediates: Whether intermediate values should be collected.
    gather_to_host: Whether predictions should be gathered to host device.

  Returns:
    Predictions and calculated metrics.
  """
  new_rng, _ = jax.random.split(train_state.rng)
  rng_dict = dict(zip(model_rngs, jax.random.split(new_rng, len(model_rngs))))
  for k in rng_dict:
    rng_dict[k] = train_utils.bind_rng_to_host_device(
        rng_dict[k], axis_name='batch', bind_to='device'
    )
  variables = {'params': train_state.params, **train_state.model_state}
  extra_mutable = ['intermediates'] if capture_intermediates else []
  pred, model_state = flax_model.apply(
      variables,
      batch,
      train=False,
      mutable=['batch_stats'] + extra_mutable,
      debug=debug,
      rngs=rng_dict,
  )
  if 'intermediates' in model_state:
    pred['intermediates'] = {
        k: v
        for k, v in model_state['intermediates'].items()
        if isinstance(v, dict)
    }
  if gather_to_host:
    pred = jax.lax.all_gather(pred, 'batch')
  losses, metrics = loss_metrics_fn(
      pred, batch, model_params=train_state.params
  )
  for k, v in losses.items():
    metrics[f'loss/{k}'] = v
  metrics = reduce_metrics(metrics, batch.get('batch_mask'))
  return pred, metrics


def optimizer_from_config(
    config: ml_collections.ConfigDict,
    params: Any,
) -> tuple[
    optax.GradientTransformation,
    Mapping[str, optax.ScalarOrSchedule],
    ml_collections.ConfigDict,
]:
  """Creates an optimizer from the config."""
  lr_schedule = config.get('schedule')
  if lr_schedule is None:
    ignore_lr_keys = ['all/.*']
    lr_configs = config.get('lr') or config.get('lr_configs')
    schedules = {}
    if config.get('overwrite_lr') is not None:
      for pattern, base_lr in config.overwrite_lr.items():
        overwrite_lr_configs = lr_configs.copy_and_resolve_references()
        overwrite_lr_configs.base_learning_rate = base_lr
        schedules[pattern] = ml_collections.ConfigDict(
            dict(
                re=rf'.*({pattern}).*',
                lr_configs=overwrite_lr_configs,
            )
        )
        ignore_lr_keys.append(pattern)
    schedules['all'] = ml_collections.ConfigDict(
        dict(
            re=rf'(?!.*({"|".join(ignore_lr_keys)}))(.*)', lr_configs=lr_configs
        )
    )
    lr_schedule = ml_collections.ConfigDict(schedules)
  if config.optimizer.optax_name == 'adamw_old':
    weight_decay_fn = lr_schedules.compound_lr_scheduler(config.weight_decay)

    # Old interface to optimizer
    if config.weight_decay.reg_wn_scale:
      mask_func = (
          lambda x, s: (
              x.ndim != 1
              and 'bias' not in s
              and 'scale' not in s
              and 'teacher' not in s
          )
          or 'weightnorm_scale' in s
      )
    else:
      mask_func = lambda x, s: True
    weight_decay_mask = optimizers.tree_map_with_names_values(mask_func, params)
    learning_rate_fn = lr_schedules.get_learning_rate_fn(lr_schedule.all)
    tx = optax.inject_hyperparams(optax.adamw)(
        learning_rate=learning_rate_fn,
        weight_decay=weight_decay_fn,
        mask=weight_decay_mask,
    )

    lr_fns = {
        'base_learning_rate': learning_rate_fn,
        'weight_decay': weight_decay_fn,
    }
  else:
    schedule_fns = scenic_optax.make_schedule(lr_schedule)
    lr_fns = {name: lr_fn for _, name, (lr_fn, _) in schedule_fns}
    tx, _ = scenic_optax.make(config.optimizer, schedule_fns, params)
  return tx, lr_fns, lr_schedule


def load_from_xm(
    xid: int,
    wid: int,
    config: ml_collections.ConfigDict | None = None,
) -> tuple[
    ml_collections.ConfigDict,
    base.BaseModel,
    train_utils.TrainState,
]:
  """Loads model, config and train state from XM."""
  reload_config, init_checkpoint_path = xm_utils.get_info_from_xmanager(
      xid, wid
  )

  if config is None:
    config = reload_config
  else:
    config.update(reload_config)

  dataset_meta_data = lama_io.npload(
      os.path.join(init_checkpoint_path, 'dataset_meta_data.npz'),
      allow_pickle=True,
  )

  model_cls = load.get_model(config.model_name)

  model = model_cls(
      config.model,
      dataset_meta_data,
      getattr(jnp, config.dtype_str),
  )

  restored_train_state = scenic_pretrain_utils.restore_pretrained_checkpoint(
      init_checkpoint_path, assert_exist=True
  )

  restored_train_state = restored_train_state.replace(metadata={})  # pytype: disable=attribute-error

  return config, model, restored_train_state


def init_from_pretrain_state(
    existing_train_state: train_utils.TrainState,  # unreplicated
    pretrain_state: train_utils.TrainState,  # unreplicated
    ckpt_prefix_path: list[str] | None = None,
    model_prefix_path: list[str] | None = None,
    name_mapping: Mapping[str, str] | None = None,
    skip_regex: str | None = None,
) -> train_utils.TrainState:
  """(Partially) initializes the train state."""
  new_train_state = scenic_pretrain_utils.init_from_pretrain_state(
      existing_train_state,
      pretrain_state,
      ckpt_prefix_path=ckpt_prefix_path,
      model_prefix_path=model_prefix_path,
      name_mapping=name_mapping,
      skip_regex=skip_regex,
  )
  # Warning: The model_state of the pretrain_state is properly transferred.
  # Warning: Especially custom model_stats are affected by this.
  new_train_state = new_train_state.replace(  # pytype: disable=attribute-error
      params=flax.core.unfreeze(new_train_state.params),
      model_state=flax.core.unfreeze(new_train_state.model_state),
  )
  del existing_train_state
  return new_train_state


class Experiment:
  """Experiment class for training and evaluation of models."""

  def __init__(
      self,
      rng: jnp.ndarray,
      config: ml_collections.ConfigDict,
      model: base.BaseModel,
      train_state: train_utils.TrainState,
      lr_fns: Mapping[str, optax.Schedule] | None = None,
      lr_schedule: ml_collections.ConfigDict | None = None,
      checkpoint_dir: str | None = None,
  ):
    self.config = config
    self.lead_host = jax.process_index() == 0
    self.host_cpu = jax.devices('cpu')[0]
    self.rng = rng
    self.model = model
    self.model_cls = model.__class__
    self.chrono = train_utils.Chrono()

    if checkpoint_dir is not None:
      train_state = self._maybe_init_from_checkpoint(
          checkpoint_dir, train_state
      )
    self.train_state = self.model.replicate_state(train_state)

    if lr_fns is None and lr_schedule is None:
      _, lr_fns, lr_schedule = optimizer_from_config(config, train_state.params)
    assert lr_fns is not None and lr_schedule is not None
    self.lr_fns = lr_fns
    self.lr_schedule = lr_schedule

    if self.config.get('run_test_evals_in_process'):
      self.test_evals = self.create_test_evals(
          self.config.get('test_evals', {})
      )
    else:
      self.test_evals = {}
    self.recompile()

  @classmethod
  def init(
      cls,
      rng: jnp.ndarray,
      config: ml_collections.ConfigDict,
      model: base.BaseModel,
      dummy_input: types.Batch,
      init_train: bool = True,
      pretrained_train_state: train_utils.TrainState | None = None,
      checkpoint_dir: str | None = None,
  ):
    """Initializes the experiment with a dummy input."""
    # Initialize model.
    rng, init_rng = jax.random.split(rng)
    params, model_state, num_trainable_params = initialize_model(
        model_def=model.flax_model,
        dummy_input=dummy_input,
        rng=init_rng,
        model_rngs=model.required_rngs,
        freeze=False,  # Allow models to update the param dict manually
        train=init_train,
    )

    # Let the model overwrite the initial params and model state
    params, model_state = model.init_state(params, model_state)

    logging.info('Number of trainable parameters: %d', num_trainable_params)

    tx, lr_fns, lr_schedule = optimizer_from_config(config, params)

    # We jit this, such that the arrays that are created on the same device as
    # the input is, in this case the CPU. Else they'd be on device[0].
    opt_state = jax.jit(tx.init, backend='cpu')(params)

    _, train_rng = jax.random.split(rng, 2)

    train_state = train_utils.TrainState(
        global_step=0,
        opt_state=opt_state,
        tx=tx,
        params=params,
        model_state=model_state,
        rng=train_rng,
        metadata={},
    )

    if pretrained_train_state is not None:
      train_state = init_from_pretrain_state(
          train_state,
          pretrained_train_state,
          **config.get('pretrained_filter', {}).to_dict(),
      )
    del pretrained_train_state
    del params  # Do not keep a copy of the initial params.

    return Experiment(
        rng=rng,
        config=config,
        model=model,
        train_state=train_state,
        lr_fns=lr_fns,
        lr_schedule=lr_schedule,
        checkpoint_dir=checkpoint_dir,
    )

  def recompile(
      self, model: ml_collections.ConfigDict | base.BaseModel | None = None
  ):
    """Recompile eval and train steps."""
    if model is None:
      model = self.model
    elif isinstance(model, (ml_collections.ConfigDict, dict)):
      # Update the model config.
      self.config.model.update(model)
      self.model = model = self.model_cls(
          self.config.model, self.model.dataset_meta_data
      )
    elif isinstance(model, self.model_cls):
      self.model = model
    else:
      raise ValueError(f'Unsupported model type: {type(model)}')

    # Empty init rate fns
    extra_lr_fns = self.create_extra_lr_fns(self.lr_schedule.get('total_steps'))

    lr_fns = {**self.lr_fns, **extra_lr_fns}

    self.train_step_pmapped = jax.pmap(
        functools.partial(
            train_step,
            flax_model=model.flax_model,
            loss_metrics_fn=model.loss_metrics_function,
            state_update_fn=model.update_state,
            aggregate_grads_fn=model.aggregate_grads,
            scale_grad_fn=model.scale_grad,
            lr_fns=lr_fns,
            config=self.config,
            debug=self.config.debug_train,
            model_rngs=model.required_rngs,
        ),
        axis_name='batch',
        # We can donate both buffers of train_state and train_batch.
        donate_argnums=(0, 1),
    )
    self.eval_step_pmapped = jax.pmap(
        functools.partial(
            eval_step,
            flax_model=model.flax_model,
            loss_metrics_fn=model.eval_loss_metrics_function,
            debug=self.config.debug_eval,
            capture_intermediates=True,
            model_rngs=model.required_rngs,
        ),
        axis_name='batch',
        # We can donate the eval_batch's buffer.
        donate_argnums=(1,),
    )

  def create_test_evals(
      self, eval_configs: Mapping[str, ml_collections.ConfigDict]
  ) -> Mapping[str, base_eval.BaseEval]:
    """Creates test evals."""
    test_evals = {}
    for eval_name, test_eval_config in eval_configs.items():
      if self.lead_host:
        logging.info(
            'Adding test eval: %s every %d steps.',
            eval_name,
            test_eval_config.eval_interval,
        )
      test_evals[eval_name] = load.get_eval(test_eval_config.eval_name)(
          self.model,
          test_eval_config,
      )
    return test_evals

  def create_extra_lr_fns(
      self, total_steps: int
  ) -> Mapping[str, optax.Schedule]:
    """Creates extra LR functions from the config."""
    extra_lr_fns = {}
    # For backwards compatibility.
    extra_lr_fn_dict = self.config.get('extra_lr_fns') or self.config.get(
        'rate_fns'
    )
    if extra_lr_fn_dict is None:
      return {}
    for param_name, rate_config in extra_lr_fn_dict.items():
      rate_config = rate_config.copy_and_resolve_references()
      rate_config.unlock()
      rate_config.steps_per_cycle = rate_config.get(
          'steps_per_cycle',
          total_steps,
      )
      extra_lr_fns[param_name] = lr_schedules.compound_lr_scheduler(rate_config)
    return extra_lr_fns

  def _maybe_init_from_checkpoint(
      self,
      workdir: str,
      train_state: train_utils.TrainState,  # unreplicated
  ) -> train_utils.TrainState:  # unreplicated
    """Initializes the train state from a checkpoint if it exists."""
    train_state = train_state.replace(  # pytype: disable=attribute-error
        metadata={'chrono': self.chrono.save()},
    )  # pytype: enable=attribute-error

    if self.config.checkpoint:
      train_state, _ = train_utils.restore_checkpoint(workdir, train_state)

    self.chrono.load(train_state.metadata['chrono'])
    train_state = train_state.replace(metadata={})
    return train_state

  def maybe_init_from_checkpoint(self, workdir: str):
    """Initializes the train state from a checkpoint if it exists."""
    train_state = self.model.unreplicate_state(self.train_state)
    del self.train_state
    jax.clear_caches()
    train_state = self._maybe_init_from_checkpoint(workdir, train_state)
    # Replicate the optimizer, state, and rng.
    self.train_state = self.model.replicate_state(train_state)
    del train_state

  def init_from_pretrain_state(
      self,
      pretrain_state: train_utils.TrainState,
      ckpt_prefix_path: list[str] | None = None,
      model_prefix_path: list[str] | None = None,
      name_mapping: Mapping[str, str] | None = None,
      skip_regex: str | None = None,
  ) -> None:
    """(Partially) initializes the train state. First unreplicates."""
    existing_train_state = self.model.unreplicate_state(self.train_state)
    del self.train_state
    new_train_state = init_from_pretrain_state(
        existing_train_state,
        pretrain_state,
        ckpt_prefix_path=ckpt_prefix_path,
        model_prefix_path=model_prefix_path,
        name_mapping=name_mapping,
        skip_regex=skip_regex,
    )
    del existing_train_state
    self.train_state = self.model.replicate_state(new_train_state)
    del new_train_state
    jax.clear_caches()

  def train_step(
      self, batch: types.Batch, shard: bool = True, filter_batch: bool = True
  ) -> tuple[
      types.AggregatedMetricsDict,
      types.AggregatedMetricsDict,
  ]:
    """Runs a single step of training and updates the internal state."""
    if filter_batch:
      batch = misc.filter_batch_for_jit(batch)
    if shard:
      batch = dataset_utils.shard(batch)
    self.train_state, metrics, training_logs = self.train_step_pmapped(
        self.train_state, batch
    )

    return metrics, training_logs

  def eval_step(
      self,
      batch: types.Batch,
      shard: bool = True,
      filter_batch: bool = True,
      to_cpu: bool = True,
  ) -> tuple[types.Predictions, types.AggregatedMetricsDict]:
    """Runs a single step of evaluation."""
    if filter_batch:
      batch = misc.filter_batch_for_jit(batch)
    if shard:
      batch = dataset_utils.shard(batch)
    pred, metrics = self.eval_step_pmapped(
        self.train_state,
        batch,
    )
    if to_cpu:
      # Move the predictions to CPU to avoid OOMs.
      pred = jax.device_get(pred)
    if shard:
      pred = dataset_utils.unshard(pred)
    return pred, metrics

  def sync_model_state_across_replicas(self):
    """Sync the model_state across replicas."""
    if jax.tree_util.tree_leaves(self.train_state.model_state):
      self.train_state = self.train_state.replace(  # pytype: disable=attribute-error
          model_state=self.model.sync_model_state_across_replicas(
              self.train_state
          )
      )

  def handle_checkpointing(
      self,
      workdir: str,
      max_checkpoints_to_keep: int = 10,
      overwrite: bool = False,
  ):
    """Adaptation of train_utils.handle_checkpointing."""
    self.sync_model_state_across_replicas()
    unrep_train_state = self.model.unreplicate_state(self.train_state)
    metadata = unrep_train_state.metadata or {}
    metadata['chrono'] = self.chrono.save()
    # pytype: disable=attribute-error
    logging.info('Removing metadata from train state...')
    unrep_train_state = unrep_train_state.replace(metadata=metadata)
    logging.info('Saving checkpoint...')
    # pytype: enable=attribute-error
    gc.collect()
    checkpoints.save_checkpoint_multiprocess(
        workdir,
        unrep_train_state,
        int(unrep_train_state.global_step),
        keep=max_checkpoints_to_keep,
        overwrite=overwrite,
    )
    del unrep_train_state
    logging.info('Done saving checkpoint. Now saving dataset meta data...')
    if jax.process_index() == 0:
      # Save dataset meta data.
      with gfile.Open(
          os.path.join(workdir, 'dataset_meta_data.npz'), 'wb'
      ) as f:
        np.save(
            f,
            {
                k: v
                for k, v in self.model.dataset_meta_data.items()
                if k not in ['dummy_batch_fn']
            },
            allow_pickle=True,
        )

  def eval_loop(
      self,
      valid_iter: Iterator[types.Batch],
      num_eval_steps: int,
      hooks: Sequence[Callable[[int], Any]] = (),
      num_visualizations: int = 0,
      **eval_step_kwargs: Any,
  ) -> tuple[
      Sequence[types.AggregatedMetricsDict], Sequence[types.VisualizationDict]
  ]:
    """Runs the evaluation loop.

    Args:
      valid_iter: An iterator over the validation set.
      num_eval_steps: The number of steps to run the evaluation loop for.
      hooks: A list of hooks to run at each step.
      num_visualizations: The number of batches to visualize.
      **eval_step_kwargs: Additional arguments to pass to the eval step.

    Returns:
      A tuple of evaluation metrics and visualizations.
    """
    eval_metrics = []
    # Sync model state across replicas.
    self.sync_model_state_across_replicas()
    visualizations = []
    with debug_utils.StepTraceContextHelper('eval', 0) as trace_context:
      for step in range(num_eval_steps):
        if (step + 1) % 10 == 0:
          logging.info('Running eval step %d / %d', step + 1, num_eval_steps)
        eval_batch = next(valid_iter)
        eval_batch = dataset_utils.shard(misc.filter_batch_for_jit(eval_batch))
        pred, e_metrics = self.eval_step(
            eval_batch,
            to_cpu=False,
            filter_batch=False,
            shard=False,
            **eval_step_kwargs,
        )
        if len(visualizations) < num_visualizations:
          viz_batch = jax.tree.map(lambda x: x[0], eval_batch)
          viz_pred = jax.tree.map(lambda x: x[0], pred)
          viz_i = self.model.visualize(
              jax.device_get(viz_pred), jax.device_get(viz_batch)
          )
          visualizations.append(viz_i)

        eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))
        for h in hooks:
          h(step)
        trace_context.next_step()
    jax.clear_caches()
    return eval_metrics, visualizations

  def metrics_to_dict(
      self, metrics: Sequence[types.AggregatedMetricsDict]
  ) -> dict[str, float]:
    """Converts metrics to a dictionary."""
    metrics = train_utils.stack_forest(metrics)
    metrics_summary = jax.tree_util.tree_map(lambda x: x.sum(), metrics)
    metrics_summary = train_utils.normalize_metrics_summary(
        metrics_summary, 'eval'
    )
    metrics_summary = jax.tree_util.tree_map(
        lambda x: x.item(), metrics_summary
    )
    return metrics_summary

  def train_loop(
      self,
      dataset: types.Dataset,
      writer: metric_writers.MetricWriter,
      workdir: str | None = None,
  ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the training loop on the provided dataset.

    Args:
      dataset: The dataset that has train_iter, eval_iter, meta_data, and
        optionally, test_iter.
      writer: CLU metrics writer instance.
      workdir: Optional working directory to store the checkpoints and logs.

    Returns:
      train_summary and eval_summary which are dict of metrics.
      These outputs are used for regression testing.
    """
    lead_host = jax.process_index() == 0

    start_step = int(jax_utils.unreplicate(self.train_state.global_step))

    # Calculate the total number of training steps.
    total_steps, steps_per_epoch = train_utils.get_num_training_steps(
        self.config, dataset.meta_data
    )

    log_eval_interval = self.config.get('log_eval_interval') or steps_per_epoch
    if not log_eval_interval:
      raise ValueError("'log_eval_interval' should be specified in the config.")
    log_train_viz_interval = (
        self.config.get('log_train_viz_interval') or log_eval_interval
    )
    checkpoint_interval = (
        self.config.get('checkpoint_interval') or log_eval_interval
    )
    log_summary_interval = (
        self.config.get('log_summary_interval') or log_eval_interval
    )

    mining_interval = self.config.get('mining_interval', None)

    # Ceil rounding such that we include the last incomplete batch.
    eval_batch_size = (
        self.config.get('eval_batch_size') or self.config.batch_size
    )
    total_eval_steps = int(
        np.ceil(dataset.meta_data['num_valid_examples'] / eval_batch_size)
    )
    num_eval_steps = self.config.get('num_eval_steps') or total_eval_steps
    num_eval_steps = min(num_eval_steps, total_eval_steps)

    train_metrics, extra_training_logs = [], []
    train_summary, eval_summary = None, None

    self.chrono.inform(
        start_step, total_steps, self.config.batch_size, steps_per_epoch
    )
    logging.info('Starting training loop at step %d.', start_step + 1)
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=total_steps,
        writer=writer,
        every_secs=None,
        every_steps=self.config.get(
            'report_progress_step', log_summary_interval
        ),
    )

    def write_note(note):
      if lead_host:
        platform.work_unit().set_notes(note)

    hooks = []
    if lead_host:
      hooks.append(report_progress)
    if self.config.get('xprof', True) and lead_host and workdir is not None:
      hooks.append(
          periodic_actions.Profile(
              logdir=workdir,
              num_profile_steps=5,
              first_profile=self.config.xprof_num_steps_after_validation,
              every_steps=None,
              every_secs=None,
              on_steps=range(
                  self.config.xprof_num_steps_after_validation,
                  total_steps,
                  log_eval_interval,
              ),
          )
      )

    mining_data = {}
    if isinstance(dataset.train_iter, (Iterator, Generator)):
      train_iter = dataset.train_iter
    else:
      train_iter = dataset.train_iter(**mining_data)

    valid_init_state = None
    if dataset.valid_iter is None or self.config.get('skip_eval'):
      valid_iter = None
    elif isinstance(dataset.valid_iter, (Iterator, Generator)):
      valid_iter = dataset.valid_iter
    else:
      valid_iter = dataset.valid_iter()
      if isinstance(valid_iter, grain.PyGrainDatasetIterator):
        valid_init_state = valid_iter.get_state()  # pytype: disable=attribute-error
    do_eval = valid_iter is not None
    write_note(f'First step compilations...\n{self.chrono.note}')
    for step in range(start_step + 1, total_steps + 1):

      ##################### MINING ###################

      if (
          mining_interval
          and ((step % mining_interval) == 1 or not mining_data)
          and step > self.config.get('skip_first_mining_steps', 0)
      ):
        logging.info('Starting mining at step %d...', step)
        self.chrono.pause(wait_for=(self.train_state.params))
        with report_progress.timed('mining'):
          del train_iter
          mining_data = self.model.mine_examples(
              self.train_state,
              dataset.train_ds,
              **self.config.get(
                  'mining', ml_collections.ConfigDict()
              ).to_dict(),
          )
          train_iter = dataset.train_iter(**mining_data)
        jax.clear_caches()
        self.chrono.resume()
        logging.info('Finished mining at step %d.', step)

      with jax.profiler.StepTraceAnnotation('train', step_num=step):
        train_batch = next(train_iter)
        train_batch = resize_batch(
            train_batch,
            step,
            self.config.model.get('resize_batch_sequence', []),
        )
        t_metrics, t_logs = self.train_step(
            train_batch,
            shard=self.config.get('shard_train_batch', True),
            filter_batch=self.config.get('filter_train_batch', True),
        )
        train_metrics.append(t_metrics)
        # Additional training logs: learning rate:
        t_logs = jax.tree.map(jax_utils.unreplicate, t_logs)
        extra_training_logs.append(t_logs)
      for h in hooks:
        h(step)
      # Below are once-in-a-while ops -> pause.
      ###################### LOG TRAIN SUMMARY ########################
      if (
          (step % log_summary_interval == 1)
          or (step == total_steps)
          # Don't log during the warmup steps after restoring a checkpoint.
          or (lead_host and self.chrono.warmup and not start_step)
      ):
        self.chrono.pause(wait_for=(train_metrics))
        if lead_host:
          try:
            self.chrono.tick(step, writer, write_note)
          except ValueError as exc:
            logging.error('Step: %d', step)
            logging.error(
                'chrono.accum_examples_seen: %d',
                self.chrono.accum_examples_seen,
            )
            logging.error('chrono.global_bs: %d', self.chrono.global_bs)
            raise exc
        extra_training_logs = jax.tree_util.tree_map(
            jax.device_get, extra_training_logs
        )
        extra_training_logs = [
            t for t in extra_training_logs if t.get('is_finite', True)
        ]
        train_summary = train_utils.log_train_summary(
            step=step,
            train_metrics=jax.tree.map(
                train_utils.unreplicate_and_get, train_metrics
            ),
            extra_training_logs=extra_training_logs,
            writer=writer,
            key_separator='/',
        )

        # Reset metric accumulation for next evaluation cycle.
        train_metrics, extra_training_logs = [], []
        self.chrono.resume()

      ################### TRAIN VISUALIZATION #######################
      if step % log_train_viz_interval == 1 and not self.config.get(
          'skip_train_viz'
      ):
        self.chrono.pause(wait_for=(self.train_state.params))
        with report_progress.timed('train_viz'):
          eval_batch = misc.filter_batch_for_jit(next(train_iter))
          if self.config.get('shard_train_batch', True):
            eval_batch = dataset_utils.shard(eval_batch)
          eval_pred, _ = self.eval_step(
              eval_batch,
              shard=False,
              filter_batch=False,
              # Force to CPU to avoid OOMs.
              to_cpu=False,
          )
          eval_batch = jax.device_get(jax.tree.map(lambda x: x[0], eval_batch))
          eval_pred = jax.device_get(jax.tree.map(lambda x: x[0], eval_pred))
          visualizations = self.model.visualize(
              eval_pred, jax.device_get(eval_batch)
          )
          for name, imgs in visualizations.items():
            if jnp.ndim(imgs) == 3:
              imgs = [imgs]
            for i, img in enumerate(imgs):
              writer.write_images(
                  step,
                  {f'train_viz/{name}/{i}': img},
              )
          del eval_batch, eval_pred, visualizations
        writer.flush()
        self.chrono.resume()

      ################### EVALUATION #######################
      if do_eval and (
          (step % log_eval_interval == 1 and step > self.config.skip_first_eval)
          or (step == total_steps)
      ):
        self.chrono.pause(wait_for=(self.train_state.params))
        if valid_init_state is not None:
          # Reset validation iterator to initial state for reproducibility.
          logging.info('Resetting valid_iter to initial state.')
          valid_iter.set_state(valid_init_state)  # pytype: disable=attribute-error
        with report_progress.timed('eval'):
          eval_metrics, visualizations = self.eval_loop(
              valid_iter,
              num_eval_steps,
              num_visualizations=self.config.get('num_eval_visualizations', 0),
          )
          eval_summary = train_utils.log_eval_summary(
              step=step,
              eval_metrics=eval_metrics,
              writer=writer,
              prefix='eval',
              key_separator='/',
          )
          if visualizations:
            # Concatenate batch visualizations and log them.
            visualizations = jax.tree_util.tree_map(
                lambda *x: jnp.concatenate(
                    x,
                    axis=-3
                    if self.config.stack_visualizations_vertically
                    else -4,
                ),
                *visualizations,
            )
            for name, imgs in visualizations.items():
              if jnp.ndim(imgs) == 3:
                imgs = [imgs]
              for i, img in enumerate(imgs):
                writer.write_images(
                    step,
                    {f'eval_viz/{name}/{i}': img},
                )

        writer.flush()
        del eval_metrics
        del visualizations
        jax.clear_caches()
        self.chrono.resume()

      ################### TESTING #######################
      for eval_name, evaluator in self.test_evals.items():
        if (
            step % evaluator.config.eval_interval == 1
            and step > self.config.skip_first_eval
        ) or step == total_steps:
          with report_progress.timed(eval_name):
            if lead_host:
              logging.info('Running test eval: %s', eval_name)
            _, test_summary, test_visualizations = evaluator.evaluate(
                self.train_state, writer, workdir
            )
            if lead_host:
              logging.info('Finished test eval: %s', eval_name)
            train_utils.log_eval_summary(
                step=step,
                eval_metrics=[],
                extra_eval_summary=test_summary,
                writer=writer,
                prefix=f'test_{eval_name}',
                key_separator='/',
            )

            for name, imgs in test_visualizations.items():
              if jnp.ndim(imgs) == 3:
                imgs = [imgs]
              for i, img in enumerate(imgs):
                writer.write_images(
                    step,
                    {f'test_viz_{eval_name}/{name}/{i}': img},
                )
          del test_summary
          del test_visualizations
          jax.clear_caches()

      ##################### CHECKPOINTING ###################
      if (
          (
              (step % checkpoint_interval == 1 and step > 1)
              or (step == total_steps)
          )
          and self.config.checkpoint
          and workdir is not None
      ):
        self.chrono.pause(
            wait_for=(self.train_state.params, self.train_state.opt_state)
        )
        train_iter_state = None
        if self.config.get('safe_checkpointing', True):
          logging.info('Deleting worker pool for safe checkpointing.')
          # pytype: disable=attribute-error
          train_iter_state = train_iter.get_state()
          # pytype: enable=attribute-error
          del train_iter
          train_iter = None
          gc.collect()
          logging.info('Sleeping for 10 seconds before checkpointing.')
          time.sleep(10)
        logging.info('DOING CHECKPOINTING')
        with report_progress.timed('checkpoint'):
          self.handle_checkpointing(
              workdir, max_checkpoints_to_keep=1, overwrite=True
          )
        logging.info('SAVED CHECKPOINT.')
        jax.clear_caches()
        if train_iter is None:
          logging.info('Recreating train iterator with worker pool.')
          train_iter = dataset.train_iter(**mining_data)
          # pytype: disable=attribute-error
          train_iter.set_state(train_iter_state)
          # pytype: enable=attribute-error
          time.sleep(10)  # Wait for 10 seconds to allow gc collect.
          logging.info('DONE CHECKPOINTING')
        self.chrono.resume()
    # Wait until computations are done before exiting.
    train_utils.barrier_across_hosts()
    # Return the train and eval summary after last step for regression testing.
    if eval_summary is None:
      eval_summary = {}
    assert self.train_state is not None
    assert train_summary is not None
    assert eval_summary is not None
    return train_summary, eval_summary
