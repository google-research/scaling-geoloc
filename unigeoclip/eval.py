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

"""Evaluation script."""

import functools
import os

from absl import flags
from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.train_lib import train_utils

from google3.learning.brain.python.debug_server import debug_server  # pylint: disable=unused-import
from unigeoclip.scenic import datasets
from unigeoclip.scenic import load
from unigeoclip.scenic import trainer
from google3.pyglib import gfile
from google3.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
_FLAGS = flags.FLAGS

XMExperiment = tuple[int, int, ml_collections.ConfigDict | None]


def write_note(
    note: str,
):
  """Writes a note to the workdir."""
  if jax.process_index() == 0:
    platform.work_unit().set_notes(note)


def evaluate(
    rng: jnp.ndarray,
    eval_name: str,
    eval_config: ml_collections.ConfigDict,
    experiment_config: ml_collections.ConfigDict | XMExperiment,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> None:
  """Evaluate one experiment on one eval."""
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=1,
      writer=writer,
      every_secs=None,
  )
  with report_progress.timed(eval_name):
    if isinstance(experiment_config, tuple):
      _, model, train_state = trainer.load_from_xm(*experiment_config)
      train_state = model.replicate_state(train_state)
    else:
      model_cls = load.get_model(experiment_config.model_name)
      dummy_batch = datasets.build_dataset(
          eval_config.data,
          eval_config.batch_size,
          eval_config.batch_size,
          experiment_config.rng_seed,
      ).meta_data['dummy_batch_fn']()

      experiment = trainer.Experiment.init(
          rng=rng,
          config=experiment_config,
          model=model_cls(
              experiment_config.model,
              {},
              getattr(jnp, experiment_config.dtype_str),
          ),
          dummy_input=dummy_batch,
          init_train=experiment_config.get('init_train', True),
      )
      experiment.maybe_init_from_checkpoint(workdir)
      train_state = experiment.train_state
      model = experiment.model
      del experiment

    evaluator = load.get_eval(eval_config.get('eval_name', eval_name))(
        model,
        eval_config,
    )

    _, test_summary, test_visualizations = evaluator.evaluate(
        train_state, writer, workdir
    )
    train_utils.log_eval_summary(
        step=0,
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
            0,
            {f'test_viz_{eval_name}/{name}/{i}': img},
        )
  del test_summary
  del test_visualizations
  del evaluator
  del train_state
  jax.clear_caches()


def main(
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> None:
  """Main function for Scenic."""
  del writer
  _, rng = jax.random.split(rng)

  config.device_count = jax.device_count()
  config = config.copy_and_resolve_references()
  lead_host = jax.process_index() == 0
  for eval_name, eval_config in config.test_evals.items():
    if lead_host:
      logging.info('Running eval: %s', eval_name)
    for experiment_name, experiment_config in config.experiments.items():
      if lead_host:
        logging.info(
            'Evaluating experiment %s on %s', experiment_name, eval_name
        )
        write_note(f'Evaluating experiment {experiment_name} on {eval_name}')
      experiment_workdir = os.path.join(workdir, experiment_name)
      experiment_writer = metric_writers.create_default_writer(
          experiment_workdir,
          just_logging=jax.process_index() > 0,
          asynchronous=True,
      )
      experiment_eval_workdir = os.path.join(experiment_workdir, eval_name)
      if gfile.Exists(experiment_eval_workdir) and not config.get(
          'overwrite_existing_evals', False
      ):
        if lead_host:
          logging.info(
              'Skipping already evaluated experiment %s on %s',
              experiment_name,
              eval_name,
          )
        continue
      evaluate(
          rng,
          eval_name,
          eval_config,
          experiment_config,
          experiment_eval_workdir,
          experiment_writer,
      )
      if lead_host:
        logging.info('Finished experiment %s on %s', experiment_name, eval_name)
    if lead_host:
      logging.info('Finished eval: %s', eval_name)


def _g3_multiprocessing_hook(*_):
  _FLAGS.jax_allow_unused_tpus = True
  _FLAGS.jax_allow_unused_gpus = True
  _FLAGS.jax_port = 0
  _FLAGS.jax_controller_address = ''


if __name__ == '__main__':
  g3_multiprocessing.g3_mp_lib.set_pre_run_hook(_g3_multiprocessing_hook)
  jax.config.config_with_absl()
  scenic_main = functools.partial(
      app._run_main, main=main  #  pylint: disable=protected-access
  )
  g3_multiprocessing.handle_main(scenic_main)
