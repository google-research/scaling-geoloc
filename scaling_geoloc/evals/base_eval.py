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

"""Interface for evaluation."""

from clu import metric_writers
from clu import platform
import jax
import ml_collections
from scenic.train_lib import pretrain_utils as scenic_pretrain_utils
from scenic.train_lib import train_utils

from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.configs import defaults
from scaling_geoloc.scenic.models import base


class BaseEval:
  """Base class for evals."""

  def __init__(self, model: base.BaseModel, config: ml_collections.ConfigDict):
    self.model = model
    self.config = config

  @classmethod
  def default_config(cls, batch_size: int) -> ml_collections.ConfigDict:
    return defaults.eval_base(batch_size)

  def evaluate(
      self,
      train_state: train_utils.TrainState,
      writer: metric_writers.MetricWriter | None = None,
      workdir: str | None = None,
  ) -> types.EvalResults:
    raise NotImplementedError()

  def evaluate_checkpoint(
      self,
      workdir: str,
      step: int | None = None,
      metric_writer: metric_writers.MetricWriter | None = None,
  ) -> types.EvalResults:
    train_state = scenic_pretrain_utils.restore_pretrained_checkpoint(
        workdir, step=step, assert_exist=True
    )

    return self.evaluate(train_state, metric_writer, workdir)

  def write_note(self, note: str):
    if jax.process_index() == 0:
      platform.work_unit().set_notes(note)
