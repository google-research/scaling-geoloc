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

"""Loading pretrained params utilities. Adapted from scenic/projects/dino/."""

import re
from typing import Any, Final, Mapping

from absl import logging
from big_vision import utils as bv_utils
from big_vision.models import vit as bv_vit
import flax
from flax.training import checkpoints
import jax
import jax.profiler
import numpy as np

# pylint: disable=line-too-long
PYTORCH_STYLE_WEIGHTS: Final[Mapping[str, str]] = {
    'ibot_b16': 'ibot_b16_1k.npz',
    'dinov2_vits14': 'dinov2_vits14.npz',
    'dinov2_vitb14': 'dinov2_vitb14.npz',
    'dinov2_vitl14': 'dinov2_vitl14.npz',
    'dinov3_vits16': 'dinov3_vits16.npz',
    'dinov3_vitb16': 'dinov3_vitb16.npz',
    'dinov3_vitl16': 'dinov3_vitl16.npz',
    'dinov3_satellite_vitl16': 'dinov3_satellite_vitl16.npz',
    'ibot_l16': (
        'pretrained_weights/ibot_l16.npz'
    ),
    'dinov3_vits16': (
        'pretrained_weights/dinov3_vits16.npz'
    ),
    'dinov3_vitb16': (
        'pretrained_weights/dinov3_vitb16.npz'
    ),
    'dinov3_vitl16': (
        'pretrained_weights/dinov3_vitl16.npz'
    ),
    'dinov3_satellite_vitl16': (
        'pretrained_weights/dinov3_satellite_vitl16.npz'
    ),
    'dinov2_salad_b14': (
        'pretrained_weights/dino_salad.npz'
    ),
}

SOURCE_ROOT_CONVNEXT: Final[str] = 'pretrained_weights/'
MODEL_NAMES_CONVNEXT: Final[frozenset[str]] = frozenset({
    'convnext_tiny_1k',
    'convnext_small_1k',
    'convnext_base_1k',
    'convnext_large_1k',
    'convnext_tiny_in22ft1k',
    'convnext_small_in22ft1k',
    'convnext_base_in22ft1k',
    'convnext_large_in22ft1k',
    'convnext_xlarge_in22ft1k',
    'convnext_tiny_384_in22ft1k',
    'convnext_small_384_in22ft1k',
    'convnext_base_384_in22ft1k',
    'convnext_large_384_in22ft1k',
    'convnext_xlarge_384_in22ft1k',
    'convnext_tiny_in22k',
    'convnext_small_in22k',
    'convnext_base_in22k',
    'convnext_large_in22k',
    'convnext_xlarge_in22k',
})


def split_qkv_w(qkv, nheads):
  # qkv in pytorch is for example [2304, 768], in bv it's 3x [768, 12, 64]
  win, wout = qkv.shape[1], qkv.shape[0] // 3
  qkv = qkv.T.reshape(win, 3, nheads, wout // nheads)
  return qkv[:, 0], qkv[:, 1], qkv[:, 2]


def split_qkv_b(qkv, nheads):
  qkv = qkv.reshape(3, nheads, -1)
  return qkv[0], qkv[1], qkv[2]


def split_swiglu_w(swiglu):
  # swiglu = (8192,  1536)
  win, wout = swiglu.shape[1], swiglu.shape[0] // 2
  swiglu = swiglu.T.reshape(win, 2, wout)
  return swiglu[:, 0], swiglu[:, 1]


def split_swiglu_b(swiglu):
  # swiglu = (8192, )
  swiglu = swiglu.reshape(2, -1)
  return swiglu[0], swiglu[1]



def get_param_from_pytorch_style_dict(
    dino: Mapping[str, np.ndarray],
    scenic_name: str,
    nheads: int,
    absorbe_ls: bool = True,
) -> np.ndarray | None:
  """Converts key name from dino official weights to this codebase name."""
  easy_mappings = {
      'ToTokenSequence_0/embedding/kernel': (
          dino['patch_embed.proj.weight'].transpose(2, 3, 1, 0)
      ),  # OIHW -> HWIO
      'ToTokenSequence_0/embedding/bias': dino['patch_embed.proj.bias'],
      'encoder_norm/scale': dino.get('norm.weight', dino.get('fc_norm.weight')),
      'encoder_norm/bias': dino.get('norm.bias', dino.get('fc_norm.bias')),
  }
  if 'pos_embed' in dino:
    n = int((dino['pos_embed'].shape[1]) ** 0.5)
    easy_mappings['ToTokenSequence_0/cls'] = (
        dino['cls_token'] + dino['pos_embed'][:, 0, :]
    )
    easy_mappings['ToTokenSequence_0/posembed_input'] = dino['pos_embed'][
        :, 1:, :
    ].reshape(1, n, n, -1)
  else:
    easy_mappings['ToTokenSequence_0/cls'] = dino['cls_token']
    easy_mappings['ToTokenSequence_0/storage'] = dino['storage_tokens']
  try:
    return easy_mappings[scenic_name]
  except KeyError:
    pass

  # Enumerated blocks.
  assert scenic_name.startswith(
      'encoderblock_'
  ), f'{scenic_name} does not start with encoderblock_'
  match = re.compile(r'encoderblock_(\d+)').match(scenic_name)
  assert match is not None, f'Regex match failed for {scenic_name}'
  iblock = int(match.group(1))

  prefix = f'encoderblock_{iblock}/'
  pt_prefix = f'blocks.{iblock}.'

  q, k, v = split_qkv_w(
      dino[pt_prefix + 'attn.qkv.weight'], nheads
  )
  block = {
      prefix + 'LayerNorm_0/scale': dino[pt_prefix + 'norm1.weight'],
      prefix + 'LayerNorm_0/bias': dino[pt_prefix + 'norm1.bias'],
      prefix + 'MultiHeadDotProductAttention_0/query/kernel': q,
      prefix + 'MultiHeadDotProductAttention_0/key/kernel': k,
      prefix + 'MultiHeadDotProductAttention_0/value/kernel': v,
      prefix
      + 'MultiHeadDotProductAttention_0/out/kernel': (
          dino[pt_prefix + 'attn.proj.weight'].T.reshape(
              nheads, -1, dino[pt_prefix + 'attn.proj.weight'].shape[-1]
          )
      ),
      prefix
      + 'MultiHeadDotProductAttention_0/out/bias': dino[
          pt_prefix + 'attn.proj.bias'
      ],
      prefix + 'LayerNorm_1/scale': dino[pt_prefix + 'norm2.weight'],
      prefix + 'LayerNorm_1/bias': dino[pt_prefix + 'norm2.bias'],
  }

  if pt_prefix + 'mlp.fc1.weight' in dino:
    block.update({
        prefix
        + 'MlpBlock_0/Dense_0/kernel': dino[pt_prefix + 'mlp.fc1.weight'].T,
        prefix + 'MlpBlock_0/Dense_0/bias': dino[pt_prefix + 'mlp.fc1.bias'],
        prefix
        + 'MlpBlock_0/Dense_1/kernel': dino[pt_prefix + 'mlp.fc2.weight'].T,
        prefix + 'MlpBlock_0/Dense_1/bias': dino[pt_prefix + 'mlp.fc2.bias'],
    })
  elif pt_prefix + 'mlp.w12.weight' in dino:
    w1, w2 = split_swiglu_w(dino[pt_prefix + 'mlp.w12.weight'])
    b1, b2 = split_swiglu_b(dino[pt_prefix + 'mlp.w12.bias'])
    block.update({
        prefix + 'FFNSwiGluFused_0/Dense_0/kernel': w1,
        prefix + 'FFNSwiGluFused_0/Dense_0/bias': b1,
        prefix + 'FFNSwiGluFused_0/Dense_1/kernel': w2,
        prefix + 'FFNSwiGluFused_0/Dense_1/bias': b2,
        prefix
        + 'FFNSwiGluFused_0/Dense_2/kernel': (
            dino[pt_prefix + 'mlp.w3.weight'].T
        ),
        prefix
        + 'FFNSwiGluFused_0/Dense_2/bias': dino[pt_prefix + 'mlp.w3.bias'],
    })

  if pt_prefix + 'attn.qkv.bias' in dino:
    q_b, k_b, v_b = split_qkv_b(
        dino[pt_prefix + 'attn.qkv.bias'], nheads
    )
    block.update({
        prefix + 'MultiHeadDotProductAttention_0/query/bias': q_b,
        prefix + 'MultiHeadDotProductAttention_0/key/bias': k_b,
        prefix + 'MultiHeadDotProductAttention_0/value/bias': v_b,
    })

  if pt_prefix + 'ls1.gamma' in dino:
    ls1 = dino[pt_prefix + 'ls1.gamma']
    ls2 = dino[pt_prefix + 'ls2.gamma']
    block.update({prefix + 'gamma_1': ls1, prefix + 'gamma_2': ls2})
    if absorbe_ls:
      k_key = prefix + 'MultiHeadDotProductAttention_0/out/kernel'
      b_key = prefix + 'MultiHeadDotProductAttention_0/out/bias'
      block[k_key] = block[k_key] * ls1
      block[b_key] = block[b_key] * ls1

      if pt_prefix + 'mlp.fc1.weight' in dino:
        k_key = prefix + 'MlpBlock_0/Dense_1/kernel'
        b_key = prefix + 'MlpBlock_0/Dense_1/bias'
      elif pt_prefix + 'mlp.w12.weight' in dino:
        k_key = prefix + 'FFNSwiGluFused_0/Dense_2/kernel'
        b_key = prefix + 'FFNSwiGluFused_0/Dense_2/bias'
      block[k_key] = block[k_key] * ls2
      block[b_key] = block[b_key] * ls2

  return block[scenic_name]


def restore_params(checkpoint_name: str, checkpoint_path: str, params: Any,
                   params_key: str = 'teacher_weights',) -> Any:
  """Returns restored params."""
  # Option 1: We load weights from a pytorch like dict.
  if checkpoint_name in PYTORCH_STYLE_WEIGHTS.keys():
    params_key = ''
    checkpoint_path = PYTORCH_STYLE_WEIGHTS[checkpoint_name]
    logging.info('Loading pytorch style weights from %s', checkpoint_path)
    pt_weights = bv_utils.npload(checkpoint_path)

    logging.info('Available elements in %s pytorch weights:', checkpoint_name)
    for k, v in pt_weights.items():  # pylint: disable=attribute-error
      logging.info('%s: %s', k, str(v.shape))
    # Pytorch param keys might have extra prefix 'module.'.
    pt_weights = {k.replace('module.', ''): v for k, v in pt_weights.items()}  # pylint: disable=attribute-error
    # Number of head.
    nh = params['encoderblock_0']['MultiHeadDotProductAttention_0']['key']['kernel'].shape[1]  # pylint: disable=line-too-long
    get_param = get_param_from_pytorch_style_dict
    if checkpoint_name.startswith('beitv2'):
      # We load only the encoder.
      pt_weights = {k.replace('encoder.', ''): v for (
          k, v) in pt_weights.items() if k.startswith('encoder.')}
    def dontload(key):
      if 'projecti' in key or 'UperNet' in key or 'fpn' in key:
        return True
      if (
          key.startswith('pixel_')
          or key.startswith('Dense_0')
          or key.startswith('ToTokenSequence_0/mask_tok')
          or key.startswith('ToTokenSequence_0/pos_embed_')
          or key.startswith('extra_tokens_projection')
          or 'shared_mask_token' in key
      ):
        return True
      return False
    restored_params = bv_utils.tree_map_with_names(
        lambda k, v: None if dontload(k) else get_param(pt_weights, k, nh),
        params)

  # Option 4: We load an official big_vision checkpoint.
  elif checkpoint_name in bv_vit.VANITY_NAMES.keys():

    restored_params = bv_vit.load(None, checkpoint_name, {})
    target_shape = params['bv_vit']['pos_embedding'].shape

    if restored_params['pos_embedding'].shape != target_shape:
      logging.info('Resampling patch positional embedding to %s', target_shape)
      pos_patches = jax.image.resize(
          restored_params['pos_embedding'],
          target_shape,
          'bilinear',
      )
      restored_params['pos_embedding'] = pos_patches
    restored_params = {'bv_vit': restored_params}

    # No further parameter surgery is needed for big_vision checkpoints
    return restored_params

  # Option 5: We load from a convnext checkpoint.
  elif checkpoint_name in MODEL_NAMES_CONVNEXT:
    params_key = 'ConvNeXt_0'
    path = SOURCE_ROOT_CONVNEXT + checkpoint_name

    restored_params = checkpoints.restore_checkpoint(ckpt_dir=path, target=None)
    restored_params = restored_params['params']

    restored_params = flax.core.unfreeze(restored_params)
    params[params_key] = restored_params

  # Option 6: We load directly from the provided checkpoint_path.
  else:
    logging.info('Loading weights from %s', checkpoint_path)
    restored_params = checkpoints.restore_checkpoint(checkpoint_path, None)

  if restored_params is None:
    return None

  # In the remaining of this function we do param surgery.
  restored_params = flax.core.unfreeze(restored_params)
  if params_key in restored_params:
    logging.info('Taking "%s" in restored checkpoint dict...', params_key)
    restored_params = restored_params[params_key]

  if 'Transformer' in restored_params:
    logging.info('There is key `Transformer` in `restored_params`.')
    pos_emb, cls_token = None, None
    if 'cls' in restored_params:
      cls_token = restored_params['cls']
    if 'posembed_input' in restored_params:
      pos_emb = restored_params['posembed_input']['pos_embedding']
    elif 'pos_embedding' in restored_params:
      pos_emb = restored_params['pos_embedding']

    restored_params = {**restored_params['Transformer'],
                       'embedding': restored_params['embedding']}
    if pos_emb is not None:
      if pos_emb.shape[1] == 197:
        pos_emb = pos_emb[:, 1:]
        cls_token = cls_token + pos_emb[:, 0]
      n = int(pos_emb.shape[1]**0.5)
      pos_emb = pos_emb.reshape(1, n, n, pos_emb.shape[-1])
      restored_params['posembed_input'] = pos_emb
    if cls_token is not None:
      restored_params['cls'] = cls_token

  # Backward compatibility.
  tks = 'ToTokenSequence_0'
  if tks not in restored_params:
    logging.info('Backward compatibility.')
    keys = ['cls', 'embedding', 'posembed_input']
    to_token = {k: restored_params[k] for k in keys if k in restored_params}
    restored_params[tks] = to_token
    for k in keys:
      if k in restored_params: del restored_params[k]

  # Resampling of patch positional embeddings (for scenic checkpoints)
  if 'posembed_input' in restored_params[tks]:
    target_shape = params[tks]['posembed_input'].shape
    if restored_params[tks]['posembed_input'].shape != target_shape:
      logging.info('Resampling patch positional embedding to %s', target_shape)
      pos_patches = jax.image.resize(
          restored_params[tks]['posembed_input'], target_shape, 'bilinear'
      )
      restored_params[tks]['posembed_input'] = pos_patches
  return restored_params


def load_params(
    checkpoint_name: str,
    checkpoint_path: str,
    params: Any,
    params_key: str = 'teacher_weights',
    force_random_init: str | None = None,
) -> Any:
  """Load parameters.

  Args:
    checkpoint_name: Name of the checkpoint.
    checkpoint_path: Path to checkpoint.
    params: A PyTree containing the random parameters.
    params_key: What key to fetch in restored checkpoint dict.
    force_random_init: We randomly initialized even if the param is in restored.

  Returns:
    restored parameters: PyTree
  """
  restored_params = restore_params(checkpoint_name, checkpoint_path, params,
                                   params_key)
  if restored_params is None:
    logging.info('Loading weights failed -> we use random weights...')
    return params

  # Flatten the tree.
  restored_params = flax.traverse_util.flatten_dict(restored_params, sep='/')
  params = flax.traverse_util.flatten_dict(params, sep='/')

  # Random init of the params not found in the restored dict.
  for k, v in params.items():
    if k not in restored_params or restored_params[k] is None:
      logging.info('%s not found in restored param -> random init.', k)
      restored_params[k] = v
    elif force_random_init and force_random_init in k:
      logging.info('We use random init for %s even if present in restored.', k)
      restored_params[k] = v
    else:
      logging.info('%s loaded from restored param.', k)

  # Removes unused loaded params.
  to_remove = []
  for k in restored_params:
    if k not in params:
      to_remove.append(k)
  for k in to_remove:
    logging.info('Removing unused %s from the restored params.', k)
    del restored_params[k]

  # Unflatten the tree.
  restored_params = flax.traverse_util.unflatten_dict(restored_params, sep='/')
  restored_params = flax.core.freeze(restored_params)
  return restored_params
