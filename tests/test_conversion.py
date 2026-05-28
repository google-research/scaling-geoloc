"""Numerical equivalence: DINOv2 PyTorch ↔ converted Flax ViTBackbone weights.

Assumes weights/dinov2_vitb14.npz already exists. Run the converter first:
    python convert_pretrained.py --model dinov2_vitb14

Then run the test:
    pytest tests/test_conversion.py

Requires torch.
"""

import os
import numpy as np
import pytest

import jax
import jax.numpy as jnp
import ml_collections

from scaling_geoloc.models.vit import ViTBackbone

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

_WEIGHTS_NAME = 'dinov2_vitb14'
_WEIGHTS_PATH = f'weights/{_WEIGHTS_NAME}.npz'

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason='requires torch')

_IMAGE_SIZE = 518   # DINOv2 ViT-B/14 training resolution — no pos-emb interpolation
_PATCH_SIZE = 14
_SIDE       = _IMAGE_SIZE // _PATCH_SIZE  # 37
_NUM_LAYERS = 12


def _vit_config() -> ml_collections.ConfigDict:
    """ViTBackbone config matching DINOv2 ViT-B/14."""
    return ml_collections.ConfigDict({
        'mlp_dim': 3072,
        'num_layers': _NUM_LAYERS,
        'num_heads': 12,
        'patches': {'size': [_PATCH_SIZE, _PATCH_SIZE]},
        'hidden_size': 768,
        'head_output_dim': 768,
        'pooling': 'tok',
        'pos_interpolation_method': 'bilinear',
        'dropout_rate': 0.0,
        'attention_dropout_rate': 0.0,
        'stochastic_depth': 0.0,
        'posembs': [_SIDE, _SIDE],
        'use_bias': True,
        'freeze_backbone': False,
        'normalization': 'none',   # image is pre-normalized before passing in
        'approximate_gelu': False, # DINOv2 uses exact GELU
        'conv_embedding': True,
        'head': 'linear',
        'add_class_token': True,
        'num_storage_tokens': 0,
    })


@pytest.fixture(scope='module')
def pt_model():
    import importlib
    model = torch.hub.load(
        'facebookresearch/dinov2', 'dinov2_vitb14',
        source='github', pretrained=True,
    )
    # xformers memory_efficient_attention has no CPU backend; disable it so
    # DINOv2's attention falls back to standard scaled dot-product attention.
    attn_mod = importlib.import_module('dinov2.layers.attention')
    attn_mod.XFORMERS_AVAILABLE = False
    return model.eval().cpu()


@pytest.fixture(scope='module')
def flax_params():
    if not os.path.exists(_WEIGHTS_PATH):
        pytest.skip(f'{_WEIGHTS_PATH} not found — run: python convert_pretrained.py --model dinov2_vitb14')

    model = ViTBackbone.from_config(_vit_config())
    dummy = jnp.zeros((1, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    jax.config.update('jax_disable_jit', False)
    try:
        init_params = model.init(jax.random.PRNGKey(0), dummy, train=False)['params']
    finally:
        jax.config.update('jax_disable_jit', True)

    return ViTBackbone.maybe_load_params(init_params, pretrained_weights=_WEIGHTS_NAME)


@pytest.fixture(scope='module')
def test_image() -> np.ndarray:
    """Random image pre-normalized with ImageNet stats, shape [1, H, W, C]."""
    rng  = np.random.default_rng(0)
    raw  = rng.random((_IMAGE_SIZE, _IMAGE_SIZE, 3)).astype(np.float32)[None]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (raw - mean) / std


class TestDINOv2Conversion:
    def _forward(self, pt_model, flax_params, test_image):
        """Run both models and return (pt_output_dict, flax_backbone_out)."""
        img_pt = torch.from_numpy(test_image.transpose(0, 3, 1, 2))
        with torch.no_grad():
            pt_out = pt_model.forward_features(img_pt)

        model = ViTBackbone.from_config(_vit_config())
        jax.config.update('jax_disable_jit', False)
        try:
            _, backbone_out = model.apply(
                {'params': flax_params},
                jnp.asarray(test_image),
                train=False,
            )
        finally:
            jax.config.update('jax_disable_jit', True)

        return pt_out, np.asarray(backbone_out)

    def test_cls_token_matches(self, pt_model, flax_params, test_image):
        """CLS token from converted Flax weights matches DINOv2 PyTorch output."""
        pt_out, backbone_out = self._forward(pt_model, flax_params, test_image)
        np.testing.assert_allclose(
            backbone_out[:, 0, :],
            pt_out['x_norm_clstoken'].cpu().numpy(),
            rtol=1e-4, atol=1e-4,
        )

    def test_patch_tokens_match(self, pt_model, flax_params, test_image):
        """Patch tokens from converted Flax weights match DINOv2 PyTorch output."""
        pt_out, backbone_out = self._forward(pt_model, flax_params, test_image)
        np.testing.assert_allclose(
            backbone_out[:, 1:, :],
            pt_out['x_norm_patchtokens'].cpu().numpy(),
            rtol=1e-4, atol=1e-4,
        )
