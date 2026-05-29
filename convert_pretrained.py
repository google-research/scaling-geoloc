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

"""Convert DINOv2 PyTorch weights to a .npz file loadable by ViTBackbone.
"""

import argparse
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Model registries
# ---------------------------------------------------------------------------

DINOV2_MODELS = {
    'dinov2_vits14': dict(hidden_size=384,  num_heads=6,  mlp_dim=1536, num_layers=12),
    'dinov2_vitb14': dict(hidden_size=768,  num_heads=12, mlp_dim=3072, num_layers=12),
    'dinov2_vitl14': dict(hidden_size=1024, num_heads=16, mlp_dim=4096, num_layers=24),
    'dinov2_vitg14': dict(hidden_size=1536, num_heads=24, mlp_dim=6144, num_layers=40),
}

ALL_MODELS = {
    **{k: ('dinov2', v) for k, v in DINOV2_MODELS.items()},
}


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def pt(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def _num_heads(hidden_size: int) -> int:
    """Return num_heads assuming head_dim=64."""
    return hidden_size // 64


def convert_backbone(pt_params: dict, num_layers: int) -> dict[str, np.ndarray]:
    """Convert a DINOv2-style ViT state dict to flat Flax parameter paths."""
    out = {}

    # Patch embedding (nn.Conv in Flax: kernel [P, P, C_in, C_out])
    w = pt(pt_params['patch_embed.proj.weight'])  # [C_out, C_in, P, P]
    out['ToTokenSequence_0/embedding/kernel'] = w.transpose(2, 3, 1, 0)
    out['ToTokenSequence_0/embedding/bias']   = pt(pt_params['patch_embed.proj.bias'])

    # Positional embedding: DINOv2 stores [1, 1+N, D].
    # Slot 0 is added to the cls token; slots 1: are added to patch tokens.
    # Our ToTokenSequence applies pos embeddings only to patches (before prepending
    # cls), so we fold slot 0 into the cls token and store the rest as posembed_input.
    posemb = pt(pt_params['pos_embed'])   # [1, 1+N, D]
    N, D_pos = posemb.shape[1] - 1, posemb.shape[2]
    side = int(round(N ** 0.5))
    assert side * side == N, f'Non-square patch grid: {N} patches'
    out['ToTokenSequence_0/cls'] = pt(pt_params['cls_token']) + posemb[:, :1, :]
    out['ToTokenSequence_0/posembed_input'] = posemb[:, 1:].reshape(1, side, side, D_pos)

    # Transformer blocks
    for i in range(num_layers):
        src = f'blocks.{i}'
        dst = f'encoderblock_{i}'

        # Pre-attention LayerNorm
        out[f'{dst}/LayerNorm_0/scale'] = pt(pt_params[f'{src}.norm1.weight'])
        out[f'{dst}/LayerNorm_0/bias']  = pt(pt_params[f'{src}.norm1.bias'])

        # Self-attention: DINOv2 fuses Q/K/V → split and reshape for Flax.
        # Flax MultiHeadDotProductAttention stores query/key/value kernels as
        # [in_dim, num_heads, head_dim].
        qkv_w = pt(pt_params[f'{src}.attn.qkv.weight'])  # [3*D, D]
        qkv_b = pt(pt_params[f'{src}.attn.qkv.bias'])    # [3*D]
        D = qkv_w.shape[1]
        num_heads = _num_heads(D)
        head_dim  = D // num_heads

        attn = f'{dst}/MultiHeadDotProductAttention_0'
        for j, name in enumerate(['query', 'key', 'value']):
            w_flat = qkv_w[j * D:(j + 1) * D].T           # [D, D]
            b_flat = qkv_b[j * D:(j + 1) * D]             # [D]
            out[f'{attn}/{name}/kernel'] = w_flat.reshape(D, num_heads, head_dim)
            out[f'{attn}/{name}/bias']   = b_flat.reshape(num_heads, head_dim)

        # DINOv2 uses LayerScale: x = x + ls(f(x)) where ls is a per-channel
        # scalar. Fold ls1.gamma into the attention output projection and
        # ls2.gamma into mlp.fc2 so our model (which has no LayerScale) matches.
        ls1 = pt(pt_params[f'{src}.ls1.gamma'])  # [D]
        ls2 = pt(pt_params[f'{src}.ls2.gamma'])  # [D]

        proj_w = pt(pt_params[f'{src}.attn.proj.weight'])  # [D, D]
        proj_b = pt(pt_params[f'{src}.attn.proj.bias'])
        # Fold ls1: new_proj_w[i, :] = ls1[i] * proj_w[i, :]
        proj_w = ls1[:, None] * proj_w
        proj_b = ls1 * proj_b
        # Flax out-projection kernel: [num_heads, head_dim, D]
        out[f'{attn}/out/kernel'] = proj_w.T.reshape(num_heads, head_dim, D)
        out[f'{attn}/out/bias']   = proj_b

        # Pre-MLP LayerNorm
        out[f'{dst}/LayerNorm_1/scale'] = pt(pt_params[f'{src}.norm2.weight'])
        out[f'{dst}/LayerNorm_1/bias']  = pt(pt_params[f'{src}.norm2.bias'])

        # MLP (Flax Dense kernel: [in, out])
        out[f'{dst}/MlpBlock_0/Dense_0/kernel'] = pt(pt_params[f'{src}.mlp.fc1.weight']).T
        out[f'{dst}/MlpBlock_0/Dense_0/bias']   = pt(pt_params[f'{src}.mlp.fc1.bias'])
        fc2_w = pt(pt_params[f'{src}.mlp.fc2.weight'])  # [D, mlp_dim]
        fc2_b = pt(pt_params[f'{src}.mlp.fc2.bias'])
        # Fold ls2: new_fc2_w[i, :] = ls2[i] * fc2_w[i, :]
        out[f'{dst}/MlpBlock_0/Dense_1/kernel'] = (ls2[:, None] * fc2_w).T
        out[f'{dst}/MlpBlock_0/Dense_1/bias']   = ls2 * fc2_b

    # Final layer norm
    out['encoder_norm/scale'] = pt(pt_params['norm.weight'])
    out['encoder_norm/bias']  = pt(pt_params['norm.bias'])

    return out


# ---------------------------------------------------------------------------
# Per-family loaders
# ---------------------------------------------------------------------------

def load_dinov2(model_name: str, hub_dir: str | None) -> tuple[dict, dict]:
    kwargs = dict(hub_dir=hub_dir) if hub_dir else {}
    model = torch.hub.load(
        'facebookresearch/dinov2', model_name,
        source='github', pretrained=True, **kwargs,
    )
    model.eval()
    return dict(model.named_parameters()), DINOV2_MODELS[model_name]


LOADERS = {
    'dinov2': load_dinov2,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Convert DINOv2 PyTorch weights to Flax .npz'
    )
    parser.add_argument(
        '--model', required=True, choices=list(ALL_MODELS),
        help='Model name, e.g. dinov2_vitb14',
    )
    parser.add_argument('--output', default=None,
                        help='Output path (default: weights/<model>.npz)')
    parser.add_argument('--hub_dir', default=None, help='Optional torch.hub cache dir')
    args = parser.parse_args()

    family, meta = ALL_MODELS[args.model]
    loader = LOADERS[family]

    output = args.output or f'weights/{args.model}.npz'

    print(f'Loading {args.model} ...')
    pt_params, meta = loader(args.model, args.hub_dir)
    print(f'  {len(pt_params)} tensors loaded')

    print('Converting ...')
    flax_params = convert_backbone(pt_params, num_layers=meta['num_layers'])

    np.savez(output, **flax_params)
    print(f'Saved {len(flax_params)} arrays → {output}')
    print()
    for k, v in sorted(flax_params.items()):
        print(f'  {k:<70s} {str(v.shape):<20s} {v.dtype}')


if __name__ == '__main__':
    main()
