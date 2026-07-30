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

"""Tests for model inference: encode_query, encode_aerial, and full forward pass."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaling_geoloc.trainer import initialize_model
from tests.conftest import BATCH_SIZE


def _init_params(geo_model, dummy_batch):
    """Initialize model params from a dummy batch."""
    rng = jax.random.PRNGKey(0)
    params, model_state, _ = initialize_model(
        model_def=geo_model.flax_model,
        dummy_input=dummy_batch,
        rng=rng,
        model_rngs=geo_model.required_rngs,
        freeze=True,
        train=False,
    )
    return {'params': params, **model_state}


class TestEncodeQuery:
    def test_output_shape(self, geo_model, dummy_batch):
        """encode_query returns (embedding [B D], tokens [B L D])."""
        variables = _init_params(geo_model, dummy_batch)
        query_pixels = dummy_batch['query']['image']  # float32 [B H W C]

        logits, tokens = geo_model.flax_model.apply(
            variables,
            {'image':query_pixels},
            train=False,
            method=geo_model.flax_model.encode_query,
        )

        assert logits.ndim == 2
        assert logits.shape[0] == BATCH_SIZE
        assert logits.shape[-1] == 64  # output_dim from conftest

    def test_output_dtype(self, geo_model, dummy_batch):
        variables = _init_params(geo_model, dummy_batch)
        logits, _ = geo_model.flax_model.apply(
            variables,
            {'image':dummy_batch['query']['image']},
            train=False,
            method=geo_model.flax_model.encode_query,
        )
        assert logits.dtype == jnp.float32

    def test_deterministic(self, geo_model, dummy_batch):
        """Inference is deterministic (no dropout at train=False)."""
        variables = _init_params(geo_model, dummy_batch)
        kwargs = dict(
            variables=variables,
            batch={'image':dummy_batch['query']['image']},
            train=False,
            method=geo_model.flax_model.encode_query,
        )
        logits_a, _ = geo_model.flax_model.apply(**kwargs)
        logits_b, _ = geo_model.flax_model.apply(**kwargs)
        np.testing.assert_array_equal(logits_a, logits_b)


class TestEncodeAerial:
    def test_output_shape(self, geo_model, dummy_batch):
        """encode_aerial returns (embeddings [B N D], tokens [B N L D])."""
        variables = _init_params(geo_model, dummy_batch)
        aerial_pixels = dummy_batch['aerial']['image']  # float32 [B H W C]

        logits, tokens = geo_model.flax_model.apply(
            variables,
            {'image':aerial_pixels},
            train=False,
            method=geo_model.flax_model.encode_aerial,
        )

        assert logits.ndim == 2
        assert logits.shape[0] == BATCH_SIZE
        assert logits.shape[-1] == 64

    def test_output_dtype(self, geo_model, dummy_batch):
        variables = _init_params(geo_model, dummy_batch)
        logits, _ = geo_model.flax_model.apply(
            variables,
            {'image':dummy_batch['aerial']['image']},
            train=False,
            method=geo_model.flax_model.encode_aerial,
        )
        assert logits.dtype == jnp.float32

    def test_different_from_query(self, geo_model, dummy_batch):
        """Query and aerial embeddings come from separate backbones."""
        variables = _init_params(geo_model, dummy_batch)
        same_pixels = dummy_batch['query']['image']

        q_logits, _ = geo_model.flax_model.apply(
            variables,
            {'image':same_pixels},
            train=False,
            method=geo_model.flax_model.encode_query,
        )
        a_logits, _ = geo_model.flax_model.apply(
            variables,
            {'image':same_pixels},
            train=False,
            method=geo_model.flax_model.encode_aerial,
        )
        # Same shape, but values differ because backbones are independently initialized.
        assert q_logits.shape == a_logits.shape
        assert not jnp.allclose(q_logits, a_logits)


class TestForwardPass:
    def test_forward_and_loss(self, geo_model, dummy_batch):
        """Full forward pass + loss are consistent and finite."""
        variables = _init_params(geo_model, dummy_batch)
        pred = geo_model.flax_model.apply(
            variables, dummy_batch, train=False, axis_name=None
        )

        assert pred['query_emb'].shape == (BATCH_SIZE, 64)
        assert pred['aerial_emb'].shape == (BATCH_SIZE, 64)

        losses, metrics = geo_model.loss_metrics_function(pred, dummy_batch)

        assert 'total' in losses
        assert 'query_sim_l2' in losses
        assert 'aerial_sim_l2' in losses
        assert jnp.isfinite(losses['total']).all()
