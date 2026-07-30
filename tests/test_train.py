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

"""Tests for the training loop: Experiment.init and train_step."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scenic.dataset_lib import dataset_utils

from scaling_geoloc.trainer import Experiment
from tests.conftest import BATCH_SIZE


class TestExperimentInit:
    def test_init_succeeds(self, geo_model, dummy_batch, train_config):
        """Experiment.init runs without error and returns an Experiment."""
        rng = jax.random.PRNGKey(0)
        experiment = Experiment.init(
            rng=rng,
            config=train_config,
            model=geo_model,
            dummy_input=dummy_batch,
        )
        assert experiment is not None
        assert experiment.train_state is not None

    def test_params_initialized(self, geo_model, dummy_batch, train_config):
        """Model params are non-empty after init."""
        rng = jax.random.PRNGKey(0)
        experiment = Experiment.init(
            rng=rng,
            config=train_config,
            model=geo_model,
            dummy_input=dummy_batch,
        )
        params = jax.tree.leaves(experiment.train_state.params)
        assert len(params) > 0


class TestTrainStep:
    @pytest.fixture(scope='class')
    def experiment(self, geo_model, dummy_batch, train_config):
        rng = jax.random.PRNGKey(0)
        return Experiment.init(
            rng=rng,
            config=train_config,
            model=geo_model,
            dummy_input=dummy_batch,
        )

    def test_train_step_runs(self, experiment, dummy_batch):
        """A single train_step completes without error."""
        batch = dataset_utils.shard(dummy_batch)
        metrics, logs = experiment.train_step(batch, shard=False)
        assert metrics is not None

    def test_train_step_returns_total_loss(self, experiment, dummy_batch):
        """train_step metrics include 'total' loss."""
        batch = dataset_utils.shard(dummy_batch)
        metrics, _ = experiment.train_step(batch, shard=False)
        assert 'loss/total' in metrics

    def test_params_change_after_step(self, geo_model, dummy_batch, train_config):
        """Parameters are updated after a train step."""
        rng = jax.random.PRNGKey(1)
        experiment = Experiment.init(
            rng=rng,
            config=train_config,
            model=geo_model,
            dummy_input=dummy_batch,
        )
        params_before = jax.tree.map(np.array, jax.device_get(
            jax.tree.map(lambda x: x[0], experiment.train_state.params)
        ))

        batch = dataset_utils.shard(dummy_batch)
        experiment.train_step(batch, shard=False)

        params_after = jax.tree.map(np.array, jax.device_get(
            jax.tree.map(lambda x: x[0], experiment.train_state.params)
        ))

        any_changed = any(
            not np.array_equal(b, a)
            for b, a in zip(
                jax.tree.leaves(params_before),
                jax.tree.leaves(params_after),
            )
        )
        assert any_changed, 'No parameters changed after a train step'

    def test_step_counter_increments(self, geo_model, dummy_batch, train_config):
        """Global step increments after each train step."""
        rng = jax.random.PRNGKey(2)
        experiment = Experiment.init(
            rng=rng,
            config=train_config,
            model=geo_model,
            dummy_input=dummy_batch,
        )
        batch = dataset_utils.shard(dummy_batch)
        experiment.train_step(batch, shard=False)
        step = int(jax.device_get(experiment.train_state.global_step[0]))
        assert step == 1
