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

"""Tests for distributed execution: shard_prototypes parity."""

import jax
import numpy as np
import pytest
from scenic.dataset_lib import dataset_utils
from scenic.train_lib import train_utils

from scaling_geoloc.trainer import initialize_model
from scaling_geoloc.models.hybrid_geolocalization import GeoLocalizerModel
from tests.conftest import NUM_CELLS, _make_model_config


def _pmap_forward(model, train_state, dummy_batch):
    """Run a pmap forward pass and unshard [D, B//D, ...] → [B, ...] outputs."""
    sharded_state = model.replicate_state(train_state)
    sharded_vars = {'params': sharded_state.params, **sharded_state.model_state}
    sharded_batch = dataset_utils.shard(dummy_batch)

    def _forward(vars_, batch):
        return model.flax_model.apply(vars_, batch, train=False, axis_name='batch')

    jax.config.update('jax_disable_jit', False)
    try:
        sharded_pred = jax.pmap(_forward, axis_name='batch')(sharded_vars, sharded_batch)
    finally:
        jax.config.update('jax_disable_jit', True)

    return jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), sharded_pred)


@pytest.mark.skipif(jax.device_count() < 2, reason="requires ≥2 devices")
class TestShardPrototypes:
    """Verify that shard_prototypes=True gives identical outputs to the default path."""

    @pytest.fixture(scope='class')
    def sharded_geo_model(self, dataset_meta_data) -> GeoLocalizerModel:
        cfg = _make_model_config()
        cfg.shard_prototypes = True
        cfg.prototype_dim = 64  # must match vit output_dim
        return GeoLocalizerModel(cfg, dataset_meta_data)

    def _init_train_state(self, geo_model, dummy_batch) -> train_utils.TrainState:
        rng = jax.random.PRNGKey(0)
        params, model_state, _ = initialize_model(
            model_def=geo_model.flax_model,
            dummy_input=dummy_batch,
            rng=rng,
            model_rngs=geo_model.required_rngs,
            freeze=True,
            train=False,
        )
        return train_utils.TrainState(params=params, model_state=model_state)

    def test_prototype_shapes(self, sharded_geo_model, geo_model, dummy_batch):
        """replicate_state shards prototypes across devices; unreplicate_state restores them."""
        D = jax.device_count()
        assert D > 1, f"expected >1 devices, got {D}"
        train_state = self._init_train_state(geo_model, dummy_batch)

        # Before replication: prototypes are (N, D) and all other params are unsharded
        assert train_state.params['prototypes'].shape == (NUM_CELLS, 64)

        sharded_state = sharded_geo_model.replicate_state(train_state)
        # Prototypes are sharded across devices; everything else is replicated
        assert sharded_state.params['prototypes'].shape == (D, NUM_CELLS // D, 64)
        for leaf in jax.tree_util.tree_leaves(sharded_state.params):
            assert leaf.shape[0] == D, f"leaf {leaf.shape} missing device axis"

        unsharded_state = sharded_geo_model.unreplicate_state(sharded_state)
        # Roundtrip: prototype values are restored exactly
        assert unsharded_state.params['prototypes'].shape == (NUM_CELLS, 64)
        np.testing.assert_array_equal(
            unsharded_state.params['prototypes'],
            train_state.params['prototypes'],
        )

    def test_outputs_and_losses_match(self, geo_model, sharded_geo_model, dummy_batch):
        """Forward pass and losses are numerically identical with and without prototype sharding."""
        train_state = self._init_train_state(geo_model, dummy_batch)
        variables = {'params': train_state.params, **train_state.model_state}

        # Reference: unsharded forward pass + loss
        ref_pred = geo_model.flax_model.apply(variables, dummy_batch, train=False, axis_name=None)
        ref_losses, _ = geo_model.loss_metrics_function(ref_pred, dummy_batch)

        pred = _pmap_forward(sharded_geo_model, train_state, dummy_batch)

        np.testing.assert_allclose(pred['query_emb'], ref_pred['query_emb'], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(pred['aerial_emb'], ref_pred['aerial_emb'], rtol=1e-5, atol=1e-5)

        losses, _ = sharded_geo_model.loss_metrics_function(pred, dummy_batch)
        np.testing.assert_allclose(
            np.asarray(losses['total']),
            np.asarray(ref_losses['total']),
            rtol=1e-5, atol=1e-5,
        )


@pytest.mark.skipif(jax.device_count() < 2, reason="requires ≥2 devices")
class TestMaskedExpsum:
    """Verify masked_expsum=True (distributed) matches local computation."""

    @pytest.fixture(scope='class')
    def ref_model(self, dataset_meta_data) -> GeoLocalizerModel:
        cfg = _make_model_config()
        cfg.masked_expsum = False
        cfg.shard_prototypes = False
        cfg.prototype_dim = 64
        cfg.compose_loss = 'query_aerial_p_multi_similarity'
        return GeoLocalizerModel(cfg, dataset_meta_data)

    @pytest.fixture(scope='class')
    def masked_model(self, dataset_meta_data) -> GeoLocalizerModel:
        cfg = _make_model_config()
        cfg.masked_expsum = True
        cfg.shard_prototypes = True
        cfg.prototype_dim = 64
        cfg.compose_loss = 'query_aerial_p_multi_similarity'
        return GeoLocalizerModel(cfg, dataset_meta_data)

    def test_expsum_and_loss_match(self, ref_model, masked_model, dummy_batch):
        """masked_expsum via pmap matches local computation; total loss is identical."""
        assert jax.device_count() > 1

        rng = jax.random.PRNGKey(0)
        params, model_state, _ = initialize_model(
            model_def=ref_model.flax_model,
            dummy_input=dummy_batch,
            rng=rng,
            model_rngs=ref_model.required_rngs,
            freeze=True,
            train=False,
        )
        train_state = train_utils.TrainState(params=params, model_state=model_state)
        variables = {'params': params, **model_state}

        # Reference: non-sharded forward pass + loss
        ref_pred = ref_model.flax_model.apply(
            variables, dummy_batch, train=False, axis_name=None
        )
        ref_losses, _ = ref_model.loss_metrics_function(ref_pred, dummy_batch)

        # Sharded forward pass with masked_expsum=True
        pred = _pmap_forward(masked_model, train_state, dummy_batch)

        # Distributed gather/scatter introduces fp32 reordering; 1e-4 is appropriate
        np.testing.assert_allclose(
            pred['query_masked_expsum'], ref_pred['query_masked_expsum'],
            rtol=1e-4, atol=1e-4,
        )
        np.testing.assert_allclose(
            pred['aerial_masked_expsum'], ref_pred['aerial_masked_expsum'],
            rtol=1e-4, atol=1e-4,
        )

        losses, _ = masked_model.loss_metrics_function(pred, dummy_batch)
        np.testing.assert_allclose(
            np.asarray(losses['total']),
            np.asarray(ref_losses['total']),
            rtol=1e-4, atol=1e-4,
        )
