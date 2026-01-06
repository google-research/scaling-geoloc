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

"""Training Script."""

import copy
import functools
import os
from typing import Any, Callable, Dict, Optional, Tuple, Type

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
import flax
from flax import jax_utils
from flax import struct
import flax.linen as nn
from flax.training import checkpoints
import jax
from jax.example_libraries.optimizers import clip_grads  # pylint: disable=g-importing-member
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
import optax
from scenic.model_lib.base_models import base_model
from scenic.projects.dino.datasets import knn_imagenet_dataset  # pylint: disable=unused-import
from scenic.projects.dino.evaluators import eval_knn
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import train_utils
from tensorflow.io import gfile

from scaling_geoloc.scenic import load_params
from scaling_geoloc.scenic import types

DinoPatch = Dict[str, jnp.ndarray]
DinoLossFn = Callable[[jnp.ndarray, jnp.ndarray], float]


@flax.struct.dataclass
class DinoTrainState:
  """Dataclass to keep track of state of training.

  The state of training is structured as a flax.struct.dataclass, which enables
  instances of this class to be passed into jax transformations like tree_map
  and pmap.
  """

  tx: Optional[optax.GradientTransformation] = struct.field(pytree_node=False)
  opt_state: Optional[optax.OptState] = None
  teacher_weights: Optional[Any] = None
  student_weights: Optional[Any] = None
  center: Optional[Any] = None
  student_state: Optional[Any] = None
  teacher_state: Optional[Any] = None
  global_step: Optional[int] = 0
  rng: Optional[jnp.ndarray] = None
  accum_train_time: Optional[int] = 0

  def __getitem__(self, item):
    """Make TrainState a subscriptable object."""
    return getattr(self, item)

  def get(self, keyname: str, default: Optional[Any] = None) -> Any:
    """Return the value for key if it exists otherwise the default."""
    try:
      return self[keyname]
    except KeyError:
      return default


def get_representation(
    train_state: DinoTrainState,
    batch: Dict[str, jnp.ndarray],
    *,
    flax_model: nn.Module,
    gather_to_host: bool = True,
    use_teacher: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Feeds the inputs to the model and returns their representations.

  Args:
    train_state: Dict, the state of training including the current. global_step,
      model_state, rng, and optimizer. The buffer of this argument can be
      donated to the computation.
    batch: A single batch of data from the dataset.
    flax_model: A Flax model.
    gather_to_host: Whether to gather results from all devices to the host,
      rather than leaving them distributed.
    use_teacher: Whether to use teacher (True) or student (False) weights.

  Returns:
    Representation learned by the model for the given inputs.
    If `gather_to_host` is True, these are collected from all hosts.
  """
  if use_teacher:
    variables = {
        'params': train_state['teacher_weights'],
        **train_state['teacher_state'],
    }
  else:
    variables = {
        'params': train_state['student_weights'],
        **train_state['student_state'],
    }

  _, representation = flax_model.apply(
      variables, batch['inputs'], train=False, debug=False
  )
  representation = representation[..., 0, :]
  if gather_to_host:
    representation = jax.lax.all_gather(representation, 'batch')
    batch = jax.lax.all_gather(batch, 'batch')
  return representation, batch['label'], batch['batch_mask']


def prepare_multicrop_input(
    inputs: Dict[str, jnp.ndarray],
    config: ml_collections.ConfigDict,
    concat_global_views: bool = True,
) -> Dict[str, jnp.ndarray]:
  """Prepare the crops for DINO training."""
  if concat_global_views:
    returned_view_dict = dict(
        global_views=jnp.concatenate(
            (inputs['global_view0'], inputs['global_view1'])
        )
    )
  else:
    returned_view_dict = dict(
        anchor=inputs['global_view0'], target=inputs['global_view1']
    )

  n_local_views = config.dataset_configs.num_local_views
  if n_local_views:
    returned_view_dict['local_views'] = jnp.concatenate(
        [inputs['local_view' + str(i)] for i in range(n_local_views)]
    )
  return returned_view_dict


def save_checkpoint(
    workdir: str,
    train_state: DinoTrainState,
    max_to_keep: int = 10,
    overwrite: bool = False,
    keep_every_n_steps: int = 50000,
):
  """Saves a checkpoint.

  First syncs the model state across replicas, then it unreplicates it by taking
  the train state of the first replica and saves it as a checkpoint.

  Args:
    workdir: Experiment directory for saving the checkpoint.
    train_state: An instance of TrainState that holds the state of training.
    max_to_keep: The number of checkpoints to keep.
    overwrite: Overwrite existing checkpoint  if a checkpoint at the current or
      a later step already exits (default: False).
    keep_every_n_steps: Keep every checkpoints every n steps.
  """
  if jax.process_index() == 0:
    # Get train state from the first replica.
    checkpoint_state = jax.device_get(jax_utils.unreplicate(train_state))
    checkpoints.save_checkpoint(
        workdir,
        checkpoint_state,
        int(checkpoint_state.global_step),
        overwrite=overwrite,
        keep=max_to_keep,
        keep_every_n_steps=keep_every_n_steps,
    )


def sync_states_across_replicas(ds: DinoTrainState) -> DinoTrainState:
  """Sync the model_state (like batch statistics) across replicas.

  Args:
    ds: DinoTrainState; Current state of training.

  Returns:
    Updated state of training in which {student,teacher}_state are synced
    across replicas.
  """
  new_teacher_state = None
  if jax.tree_util.tree_leaves(ds.teacher_state):
    new_teacher_state = flax.core.copy(
        ds.teacher_state,
        {'batch_stats': train_utils.pmap_mean(ds.teacher_state['batch_stats'])},
    )
  if jax.tree_util.tree_leaves(ds.student_state):
    new_student_state = flax.core.copy(
        ds.student_state,
        {'batch_stats': train_utils.pmap_mean(ds.student_state['batch_stats'])},
    )
    return ds.replace(  # pytype: disable=attribute-error
        student_state=new_student_state, teacher_state=new_teacher_state
    )
  else:
    return ds


def restore_checkpoint(
    checkpoint_path: str,
    train_state: Optional[DinoTrainState] = None,
    assert_exist: bool = False,
    step: Optional[int] = None,
) -> Tuple[DinoTrainState, int]:
  """Restores the last checkpoint.

  First restores the checkpoint, which is an instance of TrainState that holds
  the state of training.

  Args:
    checkpoint_path: Directory to restore the checkpoint.
    train_state: An instance of TrainState that holds the state of training.
    assert_exist: Assert that there is at least one checkpoint exists in the
      given path.
    step: Step number to load or None to load latest. If specified,
      checkpoint_path must be a directory.

  Returns:
    training state and an int which is the current step.
  """
  if assert_exist:
    glob_path = os.path.join(checkpoint_path, 'checkpoint_*')
    if not gfile.glob(glob_path):
      raise ValueError(
          'No checkpoint for the pretrained model is found in: '
          f'{checkpoint_path}'
      )
  if train_state is None:
    raise ValueError(
        'Please use `restore_pretrained_checkpoint` for loading'
        'a checkpoint without providing a Scenic TrainState.'
    )
  train_state = checkpoints.restore_checkpoint(
      checkpoint_path, train_state, step
  )
  return train_state, int(train_state.global_step)


def train_step(
    dino_train_state: DinoTrainState,
    batch: types.Batch,
    *,
    flax_model: nn.Module,
    teacher_temperature_scheduler: Callable[[int], float],
    momentum_parameter_scheduler: Callable[[int], float],
    lr_fn: Callable[[int], float],
    loss_fn: DinoLossFn,
    config: ml_collections.ConfigDict,
) -> Tuple[DinoTrainState, Dict[str, Tuple[float, int]]]:
  """Runs a single step of training.

  Given the state of the training and a batch of data, computes
  the loss and updates the parameters of the model.

  Note that in this code, the buffers of the first (train_state) and second
  (batch) arguments are donated to the computation.

  Args:
    dino_train_state: The state of training including the current global_step,
      model_state, rng, and optimizer. The buffer of this argument can be
      donated to the computation.
    batch: A single batch of data. The buffer of this argument can be donated to
      the computation.
    flax_model: A Flax model.
    teacher_temperature_scheduler: Teacher temperature scheduler.
    momentum_parameter_scheduler: Momentum parameter scheduler.
    lr_fn: The learning rate function (for logging).
    loss_fn: Given scaled logits for student and teacher returns the DINO loss
      function.
    config: Configurations of the experiment.

  Returns:
    Updated state of training.
  """
  new_rng, rng = jax.random.split(dino_train_state.rng)
  # Bind the rng to the host/device we are on.
  dropout_rng = train_utils.bind_rng_to_host_device(
      rng, axis_name='batch', bind_to='device'
  )
  step = dino_train_state.global_step
  teacher_temperature = teacher_temperature_scheduler(step)
  momentum_parameter = momentum_parameter_scheduler(step)
  lr = lr_fn(step)

  # Create patch of pixels
  dino_batch = prepare_multicrop_input(batch, config)

  def training_loss_fn(params):
    variables_student = {'params': params, **dino_train_state.student_state}
    (logits_student, _), new_student_state = flax_model.apply(
        variables_student,
        dino_batch['global_views'],
        mutable=['batch_stats'],
        train=True,
        rngs={'dropout': dropout_rng},
    )
    if 'local_views' in dino_batch:
      (logits_local_views, _), _ = flax_model.apply(
          variables_student,
          dino_batch['local_views'],
          mutable=['batch_stats'],
          train=True,
          rngs={'dropout': dropout_rng},
      )
      logits_student = jnp.concatenate(
          (logits_student, logits_local_views), axis=0
      )
    variables_teacher = {
        'params': dino_train_state.teacher_weights,
        **dino_train_state.teacher_state,
    }
    (logits_teacher, _), new_teacher_state = flax_model.apply(
        variables_teacher,
        dino_batch['global_views'],
        mutable=['batch_stats'],
        train=False,  # no stochastic depth on the teacher
    )
    logits_teacher = jax.lax.stop_gradient(logits_teacher)
    dino_loss = loss_fn(
        logits_student / config.model.temperature,
        (logits_teacher - dino_train_state.center) / teacher_temperature,
    )
    return (dino_loss, (logits_teacher, new_student_state, new_teacher_state))

  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
  (
      (dino_loss, (logits_teacher, new_student_state, new_teacher_state)),
      grad,
  ) = compute_gradient_fn(dino_train_state.student_weights)
  bs = logits_teacher.shape[0] // 2
  metrics = {'dino_loss': (dino_loss * bs, bs), 'lr': (lr * bs, bs)}

  # Update the student weights
  grad = jax.lax.pmean(grad, axis_name='batch')
  if config.get('max_grad_norm', None) is not None:
    grad = clip_grads(grad, config.max_grad_norm)
  # freeze the prototype layer
  keep_last = step > config.freeze_prototypes_steps
  grad = optimizers.tree_map_with_names(
      lambda param: param * keep_last, grad, lambda name: 'prototypes' in name
  )
  updates, new_opt_state = dino_train_state.tx.update(  # pytype: disable=attribute-error
      grad, dino_train_state.opt_state, dino_train_state.student_weights
  )
  new_student_weights = optax.apply_updates(
      dino_train_state.student_weights, updates
  )

  # update the teacher weights
  new_teacher_weights = jax.tree_util.tree_map(
      lambda s, t: momentum_parameter * t + (1 - momentum_parameter) * s,
      new_student_weights,
      dino_train_state.teacher_weights,
  )

  # update the center
  batch_center = jnp.mean(logits_teacher, axis=0)
  batch_center = jax.lax.pmean(batch_center, axis_name='batch')
  cm = config.model.center_momentum
  center = dino_train_state.center * cm + batch_center * (1 - cm)

  new_dino_train_state = dino_train_state.replace(  # pytype: disable=attribute-error
      global_step=step + 1,
      opt_state=new_opt_state,
      student_state=new_student_state,
      teacher_state=new_teacher_state,
      student_weights=new_student_weights,
      teacher_weights=new_teacher_weights,
      center=center,
      rng=new_rng,
  )
  return new_dino_train_state, metrics


def initialize(
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model: Any,
    meta_data: Any,
    pretrained: str | None = None,
) -> tuple[DinoTrainState, Any]:
  """Initialize model and train state."""
  # Initialize models.
  rng, init_rng = jax.random.split(rng)
  (student_params, student_state, _, _) = train_utils.initialize_model(
      model_def=model.flax_model,
      input_spec=[(
          meta_data['input_shape'],
          meta_data.get('input_dtype', jnp.float32),
      )],
      config=config,
      rngs=init_rng,
  )

  if pretrained:
    student_params = load_params.load_params(pretrained, '', student_params)

  # Only one model function and two states/sets of parameters
  teacher_params = copy.deepcopy(student_params)
  teacher_state = copy.deepcopy(student_state)

  # Get learning rate, weight decay, teacher temp and momentum schedulers.
  learning_rate_fn = lr_schedules.get_learning_rate_fn(config)
  weight_decay_fn = lr_schedules.compound_lr_scheduler(config.weight_decay)
  teacher_temperature_scheduler = lr_schedules.compound_lr_scheduler(
      config.teacher_temperature
  )
  momentum_parameter_scheduler = lr_schedules.compound_lr_scheduler(
      config.momentum_rate
  )

  # Create optimizer.
  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  if config.model.get('reg_wn_scale', True):
    # Apply regularization to weight norm scale
    # pylint: disable=g-long-lambda
    mask_func = (
        lambda x, s: (x.ndim != 1 and 'bias' not in s and 'scale' not in s)
        or 'weightnorm_scale' in s
    )
    # pylint: enable=g-long-lambda
  else:
    # Do not regularize bias and norm parameters
    mask_func = (
        lambda x, s: x.ndim != 1 and 'bias' not in s and 'scale' not in s
    )

  weight_decay_mask = optimizers.tree_map_with_names_values(
      mask_func, student_params
  )

  tx = optax.inject_hyperparams(optax.adamw)(
      learning_rate=learning_rate_fn,
      weight_decay=weight_decay_fn,
      mask=weight_decay_mask,
  )
  opt_state = jax.jit(tx.init, backend='cpu')(student_params)

  _, train_rng = jax.random.split(rng)
  dino_train_state = DinoTrainState(
      global_step=0,
      opt_state=opt_state,
      tx=tx,
      student_state=student_state,
      teacher_state=teacher_state,
      student_weights=student_params,
      teacher_weights=teacher_params,
      center=np.zeros(config.model.head_output_dim),
      rng=train_rng,
      accum_train_time=0,
  )

  # Replicate the optimizer, states, and rng.
  del student_params, teacher_params

  train_step_pmapped = jax.pmap(
      functools.partial(
          train_step,
          flax_model=model.flax_model,
          teacher_temperature_scheduler=teacher_temperature_scheduler,
          momentum_parameter_scheduler=momentum_parameter_scheduler,
          lr_fn=learning_rate_fn,
          loss_fn=model.loss_function,
          config=config,
      ),
      axis_name='batch',
      # We can donate both buffers of train_state and train_batch.
      donate_argnums=(0, 1),
  )

  return dino_train_state, train_step_pmapped


def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Type[base_model.BaseModel],
    dummy_input: types.Batch,
    dataset: types.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> Tuple[Any, Any]:
  """Main training loop lives in this function.

  Given the model class and dataset, it prepares the items needed to run the
  training, including the DinoTrainState.

  Args:
    rng: Jax rng key.
    config: Configurations of the experiment.
    model_cls: Model class; A model has a flax_module and a loss_fn.
    dataset: The dataset that has train_iter, eval_iter, meta_data, and
      optionally, test_iter.
    workdir: Directory for checkpointing.
    writer: CLU metrics writer instance.

  Returns:
    dino_train_state that has the state of training.
  """

  # Build the loss_fn, metrics, and flax_model.
  if 'input_shape' not in dataset.meta_data:
    dataset.meta_data['input_shape'] = dummy_input['global_view0'].shape
  model = model_cls(config, dataset.meta_data)

  dino_train_state, train_step_pmapped = initialize(
      rng, config, model, dataset.meta_data
  )

  lead_host = jax.process_index() == 0

  start_step = dino_train_state.global_step

  if config.checkpoint:
    dino_train_state, start_step = restore_checkpoint(workdir, dino_train_state)

  dino_train_state = jax_utils.replicate(dino_train_state)

  # Calculate the total number of training steps.
  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset.meta_data
  )

  checkpoint_steps = config.get('checkpoint_steps')
  log_summary_steps = config.get('log_summary_steps')

  train_metrics = []
  train_summary = None
  chrono = train_utils.Chrono()

  chrono.inform(start_step, total_steps, config.batch_size, steps_per_epoch)

  logging.info('Starting training loop at step %d.', start_step + 1)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps, writer=writer
  )

  def write_note(note):
    if lead_host:
      platform.work_unit().set_notes(note)

  representation_fn = functools.partial(
      get_representation,
      flax_model=model.flax_model,
  )
  knn_classifier = eval_knn.KnnEvaluator(representation_fn, config.knn)

  hooks = []
  if lead_host:
    hooks.append(report_progress)
  if config.get('xprof', True) and lead_host:
    hooks.append(periodic_actions.Profile(num_profile_steps=15, logdir=workdir))

  train_iter = dataset.train_iter
  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch = next(train_iter)
      dino_train_state, tm = train_step_pmapped(dino_train_state, train_batch)
      train_metrics.append(tm)
      # Additional training logs.
    for h in hooks:
      h(step)
    ###################### LOG TRAIN SUMMARY ########################
    if (step % log_summary_steps == 1) or (step == total_steps):
      chrono.pause(wait_for=(train_metrics))
      if lead_host:
        chrono.tick(step, writer, write_note)
      train_summary = train_utils.log_train_summary(
          step=step,
          train_metrics=jax.tree_util.tree_map(
              train_utils.unreplicate_and_get, train_metrics
          ),
          writer=writer,
      )
      # Reset metric accumulation for next evaluation cycle.
      train_metrics = []
      chrono.resume()

    if step % config.knn.eval_interval == 1:
      test_summary = knn_classifier.run(dino_train_state)
      train_utils.log_eval_summary(
          step=step,
          eval_metrics=[],
          extra_eval_summary=test_summary,
          writer=writer,
          prefix='knn',
          key_separator='/',
      )
    ##################### CHECKPOINTING ###################
    if (
        (step % checkpoint_steps == 0 and step > 0) or (step == total_steps)
    ) and config.checkpoint:
      chrono.pause()
      with report_progress.timed('checkpoint'):
        # Sync model state across replicas.
        dino_train_state = sync_states_across_replicas(dino_train_state)
        if lead_host:
          dino_train_state.replace(  # pytype: disable=attribute-error
              accum_train_time=chrono.accum_train_time
          )
          save_checkpoint(workdir, dino_train_state)
      chrono.resume()  # un-pause now
  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  # Return the train summary after last step.
  return dino_train_state, train_summary
