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

"""Trainer for UniGeoCLIP models.

The heavy lifting (train loop, train/eval step, checkpointing, optimizer setup)
is shared with the public ``scaling_geoloc.trainer``. This module only rebinds
the two places that look up models/evals by name so they resolve against the
UniGeoCLIP registry (``unigeoclip.scenic.load``) instead of the geo-localization
one.
"""

from collections.abc import Mapping
import os

from absl import logging
import jax.numpy as jnp
import ml_collections
from scenic.train_lib import pretrain_utils as scenic_pretrain_utils
from scenic.train_lib import train_utils

from scaling_geoloc import trainer as _base
from scaling_geoloc.evals import base_eval
from scaling_geoloc.models import base
from scaling_geoloc.utils import io
from unigeoclip.scenic import load

# Re-export the shared, registry-independent helpers so callers can keep using
# ``trainer.<symbol>`` regardless of which trainer module they import.
initialize_model = _base.initialize_model
reduce_metrics = _base.reduce_metrics
train_step = _base.train_step
eval_step = _base.eval_step
optimizer_from_config = _base.optimizer_from_config
init_from_pretrain_state = _base.init_from_pretrain_state


class Experiment(_base.Experiment):
  """Public ``scaling_geoloc`` Experiment bound to the UniGeoCLIP registry."""

  def create_test_evals(
      self, eval_configs: Mapping[str, ml_collections.ConfigDict]
  ) -> Mapping[str, base_eval.BaseEval]:
    """Creates in-process test evals using the UniGeoCLIP eval registry."""
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


def reload_experiment(
    checkpoint_path: str,
    config: ml_collections.ConfigDict | None = None,
) -> tuple[ml_collections.ConfigDict, base.BaseModel, train_utils.TrainState]:
  """Loads model, config and train state from a workdir (UniGeoCLIP registry)."""
  reload_config = io.config_load(checkpoint_path)
  if config is None:
    config = reload_config
  else:
    config.update(reload_config)

  dataset_meta_data = io.npload(
      os.path.join(checkpoint_path, 'dataset_meta_data.npz'),
      allow_pickle=True,
  )

  model_cls = load.get_model(config.model_name)
  model = model_cls(
      config.model,
      dataset_meta_data,
      getattr(jnp, config.dtype_str),
  )

  restored_train_state = scenic_pretrain_utils.restore_pretrained_checkpoint(
      checkpoint_path, assert_exist=True
  )
  restored_train_state = restored_train_state.replace(metadata={})
  return config, model, restored_train_state
