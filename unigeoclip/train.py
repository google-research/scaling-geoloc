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

"""Training entry point."""

import functools
import os

from absl import flags
from absl import logging
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app

from google3.learning.brain.python.debug_server import debug_server  # pylint: disable=unused-import
import google3.learning.deepmind.xmanager2.client.google as xm

from unigeoclip.scenic import datasets
from unigeoclip.scenic import load
from unigeoclip.scenic import trainer

from google3.pyglib.contrib.g3_multiprocessing import g3_multiprocessing


os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
_FLAGS = flags.FLAGS


def main(
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> None:
  """Main function for Scenic."""
  # Call xm.setup_work_unit() to ensure the "Parameter Config" tab is available
  # in the XManager UI (see go/xm-setup-binaries). Scenic's main function
  # doesn't do this.
  xm.setup_work_unit()

  _, rng = jax.random.split(rng)

  config.device_count = jax.device_count()

  if config.batch_size_eval is None:
    config.batch_size_eval = config.batch_size

  logging.info(
      'batch_size: %d, batch_size_eval: %d, device_count: %d, lr: %f',
      config.batch_size,
      config.batch_size_eval,
      config.device_count,
      config.lr.base_learning_rate,
  )

  config = config.copy_and_resolve_references()

  pretrained_meta_data = None
  pretrained_train_state = None
  if config.get('pretrained_xm') is not None:
    if isinstance(config.pretrained_xm, ml_collections.ConfigDict):
      xid, wid = config.pretrained_xm.xid, config.pretrained_xm.wid
    else:
      xid, wid = config.pretrained_xm
    logging.info('Loading pretrained model from XM at (%d, %d).', xid, wid)
    pretrained_config, pretrained_model, pretrained_train_state = (
        trainer.load_from_xm(xid, wid)
    )
    if config.get('pretrained_load_meta_data'):
      logging.info('Loading pretrained meta data.')
      pretrained_meta_data = pretrained_model.dataset_meta_data
    if config.get('pretrained_load_model_config'):
      logging.info('Loading pretrained model config (overwrite).')
      config.model.update(pretrained_config.model)
      logging.info('Updated model config: %s', config.model)
    del pretrained_model

  dataset = datasets.build_dataset(
      config.data,
      config.batch_size,
      config.batch_size_eval,
      config.rng_seed,
      meta_data=pretrained_meta_data,
  )

  logging.info('Created dataset.')

  model_cls = load.get_model(config.model_name)
  experiment = trainer.Experiment.init(
      rng=rng,
      config=config,
      model=model_cls(
          config.model, dataset.meta_data, getattr(jnp, config.dtype_str)
      ),
      dummy_input=dataset.meta_data['dummy_batch_fn'](),
      init_train=config.get('init_train', True),
      pretrained_train_state=pretrained_train_state,
      checkpoint_dir=workdir,
  )

  if pretrained_train_state is not None:
    del pretrained_train_state
    jax.clear_caches()

  logging.info('Created experiment.')

  logging.info('Starting training loop.')

  experiment.train_loop(dataset, writer, workdir)


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
