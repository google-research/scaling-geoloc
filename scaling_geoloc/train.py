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

"""Training entry point."""

from absl import logging
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scaling_geoloc.data import datasets
from scaling_geoloc import load
from scaling_geoloc import trainer


def main(
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> None:
  """Main function for Scenic."""

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

  if config.overfit:
    logging.info('Create overfit dataset.')
    dummy_batch = dataset.meta_data['dummy_batch_fn']()
    dataset = datasets.create_overfit_dataset(
        lambda: dummy_batch,
        num_train_examples=dataset.meta_data['num_train_examples'],
        num_valid_examples=dataset.meta_data['num_valid_examples'],
        meta_data=dataset.meta_data,
    )

  logging.info('Starting training loop.')

  experiment.train_loop(dataset, writer, workdir)


if __name__ == '__main__':
  app.run(main)
