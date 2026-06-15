"""Tests for retrieval_v2: build_index and evaluate_retrieval."""

import dataclasses

import grain.python as grain
import jax
import numpy as np
import pytest

from scaling_geoloc.configs import defaults
from scaling_geoloc.evals.retrieval import build_index, evaluate_retrieval, join_databases

N_ITEMS = 8
DIM = 16


@dataclasses.dataclass
class FlatDummyDataSource(grain.RandomAccessDataSource):
    """Flat data source matching the build_index interface."""
    num_examples: int = N_ITEMS
    dim: int = DIM

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng(index)
        return {
            'image': rng.integers(0, 256, (32, 32, 3), dtype=np.uint8),
            'latlng': rng.uniform([-90., -180.], [90., 180.]).astype(np.float32),
            'cell_idx': np.int32(index % 32),
        }


@pytest.fixture(scope='module')
def flat_source():
    return FlatDummyDataSource(num_examples=N_ITEMS)


@pytest.fixture(scope='module')
def index(flat_source):
    rng = np.random.default_rng(0)

    def representation_fn(batch):
        b = batch['image'].shape[0]
        return rng.standard_normal((b, DIM)).astype(np.float32)
    data_config = defaults.data('').unlock()
    data_config.grain_num_workers_valid = 0
    return build_index(representation_fn, flat_source, data_config, batch_size=4)


class TestBuildIndex:
    def test_keys(self, index):
        assert set(index.keys()) == {'embs', 'latlng', 'dataset_index', 'cell_idx'}

    def test_shapes(self, index):
        assert index['embs'].shape == (N_ITEMS, DIM)
        assert index['latlng'].shape == (N_ITEMS, 2)
        assert index['dataset_index'].shape == (N_ITEMS,)
        assert index['cell_idx'].shape == (N_ITEMS,)

    def test_dataset_index_is_arange(self, index):
        np.testing.assert_array_equal(index['dataset_index'], np.arange(N_ITEMS))

    def test_cell_idx_matches_source(self, index, flat_source):
        expected = np.array([flat_source[i]['cell_idx'] for i in range(N_ITEMS)])
        np.testing.assert_array_equal(index['cell_idx'], expected)

    def test_latlng_matches_source(self, index, flat_source):
        expected = np.stack([flat_source[i]['latlng'] for i in range(N_ITEMS)])
        np.testing.assert_array_equal(index['latlng'], expected)


def _make_retrieval_inputs(n_query=4, n_db=8, dim=DIM, seed=0):
    rng = np.random.default_rng(seed)
    latlng = rng.uniform([-90., -180.], [90., 180.], (n_query + n_db, 2)).astype(np.float32)
    query = {
        'embs': rng.standard_normal((n_query, dim)).astype(np.float32),
        'latlng': latlng[:n_query],
        'dataset_index': np.arange(n_query),
    }
    db = {
        'embs': rng.standard_normal((n_db, dim)).astype(np.float32),
        'latlng': latlng[n_query:],
        'dataset_index': np.arange(n_db),
    }
    return query, db


class TestEvaluateRetrieval:
    @pytest.fixture(scope='class')
    def results(self):
        query, db = _make_retrieval_inputs()
        return evaluate_retrieval(query, db, ks=(1, 4))

    def test_summary_keys(self, results):
        _, summaries = results
        for k in (1, 4):
            assert f'distance@{k}' in summaries
            assert f'recall@{k}@50m' in summaries
            assert f'recall@{k}@1km' in summaries

    def test_summary_values_in_range(self, results):
        _, summaries = results
        for key, val in summaries.items():
            assert np.isfinite(val), f'{key} is not finite'
            if key.startswith('recall'):
                assert 0.0 <= val <= 1.0, f'{key}={val} out of [0,1]'
            else:
                assert val >= 0.0, f'{key}={val} is negative'

    def test_per_query_keys(self, results):
        per_query, _ = results
        assert 'latlng' in per_query
        assert 'dataset_index' in per_query
        assert 'topk_idx' in per_query
        assert 'topk_dist' in per_query
        assert 'topk_dataset_index' in per_query

    def test_per_query_shapes(self, results):
        per_query, _ = results
        n_devices = jax.local_device_count()
        n_db = (8 // n_devices) * n_devices
        max_k = min(4, n_db // n_devices)
        assert per_query['topk_idx'].shape == (4, max_k)
        assert per_query['topk_dist'].shape == (4, max_k)

    def test_exact_match_retrieves_itself(self):
        """When query embs are identical to db entries, top-1 distance is 0."""
        rng = np.random.default_rng(42)
        n = jax.local_device_count() * 2  # ensure divisible
        embs = rng.standard_normal((n, DIM)).astype(np.float32)
        latlng = rng.uniform([-90., -180.], [90., 180.], (n, 2)).astype(np.float32)
        query = {'embs': embs.copy(), 'latlng': latlng.copy()}
        db = {'embs': embs.copy(), 'latlng': latlng.copy(), 'dataset_index': np.arange(n)}
        per_query, summaries = evaluate_retrieval(query, db, ks=(1,))
        np.testing.assert_allclose(per_query['topk_dist'][:, 0], 0.0, atol=1e-3)
        assert summaries['recall@1@50m'] == 1.0

    def test_match_in_second_device_shard(self):
        """Dist. search finds the correct match even when it lives in the second device shard."""
        n_devices = jax.local_device_count()
        if n_devices < 2:
            pytest.skip('requires at least 2 devices')

        rng = np.random.default_rng(7)
        n_db = n_devices * 4
        db_embs = rng.standard_normal((n_db, DIM)).astype(np.float32)
        db_latlng = rng.uniform([-90., -180.], [90., 180.], (n_db, 2)).astype(np.float32)

        # Query is identical to an entry in the second shard (index n_devices * 2).
        target_idx = n_devices * 2
        query = {
            'embs': db_embs[target_idx:target_idx + n_devices],
            'latlng': db_latlng[target_idx:target_idx + n_devices],
        }
        db = {'embs': db_embs, 'latlng': db_latlng, 'dataset_index': np.arange(n_db)}

        per_query, _ = evaluate_retrieval(query, db, ks=(1,))
        np.testing.assert_allclose(per_query['topk_dist'][:, 0], 0.0, atol=1e-3)


class TestJoinDatabases:
    def _make_inputs(self, n=8, n_cells=32, dim=DIM, seed=0):
        rng = np.random.default_rng(seed)
        aerial_db = {
            'embs': rng.standard_normal((n, dim)).astype(np.float32),
            'latlng': rng.uniform([-90., -180.], [90., 180.], (n, 2)).astype(np.float32),
            'cell_idx': rng.integers(0, n_cells, size=n, dtype=np.int32),
            'dataset_index': np.arange(n),
        }
        proto_db = {
            'embs': rng.standard_normal((n_cells, dim)).astype(np.float32),
        }
        return aerial_db, proto_db

    def test_output_shape_unchanged(self):
        aerial_db, proto_db = self._make_inputs()
        result = join_databases(aerial_db, proto_db)
        assert result['embs'].shape == aerial_db['embs'].shape

    def test_other_fields_preserved(self):
        aerial_db, proto_db = self._make_inputs()
        result = join_databases(aerial_db, proto_db)
        for key in ('latlng', 'cell_idx', 'dataset_index'):
            np.testing.assert_array_equal(result[key], aerial_db[key])

    def test_blend_formula(self):
        aerial_db, proto_db = self._make_inputs()
        kappa = 2.0
        result = join_databases(aerial_db, proto_db, kappa=kappa)
        proto_embs = proto_db['embs'][aerial_db['cell_idx']]
        expected = (aerial_db['embs'] + proto_embs * kappa) / (1 + kappa)
        np.testing.assert_allclose(result['embs'], expected, atol=1e-6)

    def test_kappa_zero_returns_aerial(self):
        aerial_db, proto_db = self._make_inputs()
        result = join_databases(aerial_db, proto_db, kappa=0.0)
        np.testing.assert_array_equal(result['embs'], aerial_db['embs'])

    def test_kappa_one_is_mean(self):
        aerial_db, proto_db = self._make_inputs()
        result = join_databases(aerial_db, proto_db, kappa=1.0)
        proto_embs = proto_db['embs'][aerial_db['cell_idx']]
        expected = (aerial_db['embs'] + proto_embs) / 2
        np.testing.assert_allclose(result['embs'], expected, atol=1e-6)

    def test_missing_cell_idx_not_blended(self):
        aerial_db, proto_db = self._make_inputs()
        aerial_db['cell_idx'] = aerial_db['cell_idx'].copy()
        aerial_db['cell_idx'][0] = -1
        result = join_databases(aerial_db, proto_db, kappa=1.0)
        # Entry 0 should be unchanged.
        np.testing.assert_array_equal(result['embs'][0], aerial_db['embs'][0])
        # Other entries should be blended.
        proto_embs = proto_db['embs'][aerial_db['cell_idx'][1:]]
        expected = (aerial_db['embs'][1:] + proto_embs) / 2
        np.testing.assert_allclose(result['embs'][1:], expected, atol=1e-6)

    def test_oob_cell_idx_raises(self):
        aerial_db, proto_db = self._make_inputs()
        aerial_db['cell_idx'] = aerial_db['cell_idx'].copy()
        aerial_db['cell_idx'][0] = proto_db['embs'].shape[0]  # exactly n_cells → OOB
        with pytest.raises(IndexError):
            join_databases(aerial_db, proto_db)
