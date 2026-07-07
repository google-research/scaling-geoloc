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

"""Evaluation entry point for UniGeoCLIP downstream evals.

Runs every eval in ``config.test_evals`` against a single trained checkpoint
(``config.experiment``). Each eval builds its own dataset and extracts features
from the model through the documented encoder interface (see README), so the
same script evaluates UniGeoCLIP checkpoints or any follow-up encoder that
implements that interface.
"""

import os

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.train_lib import train_utils

from unigeoclip import load
from unigeoclip import trainer


def write_note(note: str):
  if jax.process_index() == 0:
    platform.work_unit().set_notes(note)


def evaluate(
    eval_name: str,
    eval_config: ml_collections.ConfigDict,
    experiment_path: str,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> None:
  """Evaluate one experiment on one eval."""
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=1, writer=writer, every_secs=None
  )
  with report_progress.timed(eval_name):
    _, model, train_state = trainer.reload_experiment(experiment_path)
    train_state = model.replicate_state(train_state)

    evaluator = load.get_eval(eval_config.get('eval_name', eval_name))(
        model, eval_config
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
        writer.write_images(0, {f'test_viz_{eval_name}/{name}/{i}': img})
  del test_summary, test_visualizations, evaluator, train_state
  jax.clear_caches()


def main(
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> None:
  """Main function for Scenic."""
  del rng, writer

  config.device_count = jax.device_count()
  config = config.copy_and_resolve_references()
  lead_host = jax.process_index() == 0
  for eval_name, eval_config in config.test_evals.items():
    if lead_host:
      write_note(f'Running eval: {eval_name}')
    eval_workdir = os.path.join(workdir, eval_name)
    if os.path.exists(eval_workdir) and not config.get(
        'overwrite_existing_evals', False
    ):
      if lead_host:
        logging.info('Skipping already evaluated: %s', eval_name)
      continue
    eval_writer = metric_writers.create_default_writer(
        eval_workdir,
        just_logging=jax.process_index() > 0,
        asynchronous=True,
    )
    evaluate(
        eval_name,
        eval_config,
        config.experiment,
        eval_workdir,
        eval_writer,
    )
    if lead_host:
      logging.info('Finished eval: %s', eval_name)


if __name__ == '__main__':
  app.run(main)
