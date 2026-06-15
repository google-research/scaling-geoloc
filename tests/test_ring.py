"""Slow equivalence tests for max_and_masked_expsum_ring vs max_and_masked_expsum.

Not included in the default test run — invoke explicitly:
    pytest tests/test_ring.py
"""

import jax
import numpy as np
import pytest

from scaling_geoloc.utils import distributed


@pytest.mark.skipif(jax.device_count() < 2, reason="requires ≥2 devices")
class TestMaskedExpsumRing:
    """Verify max_and_masked_expsum_ring matches max_and_masked_expsum."""

    def _run_pmap(self, fn, batch_logits, batch_latlng, sharded_logits,
                  global_latlng, required_idx, distance_threshold):
        def _forward(b_logits, b_latlng, s_logits, g_latlng, req_idx):
            return fn(
                b_logits, b_latlng, s_logits, g_latlng,
                beta=2.0, base=0.5,
                required_idx=req_idx,
                distance_threshold=distance_threshold,
            )

        jax.config.update('jax_disable_jit', False)
        try:
            return jax.pmap(_forward, axis_name='batch', in_axes=(0, 0, 0, None, 0))(
                batch_logits, batch_latlng, sharded_logits, global_latlng, required_idx,
            )
        finally:
            jax.config.update('jax_disable_jit', True)

    @pytest.fixture(scope='class')
    def inputs(self):
        nd = jax.device_count()
        B, D, K = 4, 16, 3
        N = nd * 8
        rng = np.random.default_rng(42)
        return dict(
            batch_logits=rng.standard_normal((nd, B, D)).astype(np.float32),
            sharded_logits=rng.standard_normal((nd, N // nd, D)).astype(np.float32),
            batch_latlng=rng.uniform([-90, -180], [90, 180], (nd, B, 2)).astype(np.float32),
            global_latlng=rng.uniform([-90, -180], [90, 180], (N, 2)).astype(np.float32),
            required_idx=rng.integers(0, N, (nd, B, K)).astype(np.int32),
        )

    def _compare(self, orig, ring):
        np.testing.assert_allclose(
            np.asarray(orig['expsum']).ravel(),
            np.asarray(ring['expsum']).ravel(),
            rtol=1e-4, atol=1e-4, err_msg='expsum mismatch',
        )
        np.testing.assert_array_equal(
            np.asarray(orig['best_idx']).ravel(),
            np.asarray(ring['best_idx']).ravel(),
        )
        np.testing.assert_allclose(
            np.asarray(orig['pos_sim']).ravel(),
            np.asarray(ring['pos_sim']).ravel(),
            rtol=1e-4, atol=1e-4, err_msg='pos_sim mismatch',
        )

    def test_equivalence(self, inputs):
        orig = self._run_pmap(distributed.max_and_masked_expsum, **inputs, distance_threshold=None)
        ring = self._run_pmap(distributed.max_and_masked_expsum_ring, **inputs, distance_threshold=None)
        self._compare(orig, ring)

    def test_equivalence_with_distance_filter(self, inputs):
        orig = self._run_pmap(distributed.max_and_masked_expsum, **inputs, distance_threshold=500_000.0)
        ring = self._run_pmap(distributed.max_and_masked_expsum_ring, **inputs, distance_threshold=500_000.0)
        self._compare(orig, ring)
