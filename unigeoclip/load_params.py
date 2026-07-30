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

"""Pretrained-parameter loading for the public release.

Loads a pretrained ViT checkpoint for ``unigeoclip/models/vit.py`` from the
flat-``.npz`` convention used by
``scaling_geoloc.models.vit.ViTBackbone.maybe_load_params`` (one entry per
parameter, slash-separated keys, produced by ``convert_pretrained.py``).

Only keys present in the ``.npz`` overwrite the corresponding initialized
parameter; every other parameter keeps its initialized value, so the returned
tree has exactly the same structure as ``model_params``.
"""

from typing import Any

from absl import logging
import flax.core
import flax.traverse_util
import jax.numpy as jnp
import numpy as np

from scaling_geoloc import types


def load_params(
    pretrained_weights: str | None,
    pretrained_checkpoint: str | None,
    model_params: types.ModelParams,
) -> Any:
  """Loads pretrained parameters from a ``.npz`` file.

  Args:
    pretrained_weights: Name (resolved to ``weights/<name>.npz``) or a path
      ending in ``.npz``.
    pretrained_checkpoint: Alternative path to a ``.npz`` file. Used only if
      ``pretrained_weights`` is ``None``.
    model_params: The freshly initialized parameter tree to overwrite.

  Returns:
    A parameter tree with the same structure as ``model_params`` where keys
    present in the ``.npz`` have been replaced by the pretrained values.
  """
  path = pretrained_weights or pretrained_checkpoint
  if path is None:
    return model_params
  if not path.endswith('.npz'):
    path = f'weights/{path}.npz'
  logging.info('Loading pretrained weights: %s', path)

  flat_pretrained = dict(np.load(path))
  flat_model = flax.traverse_util.flatten_dict(
      flax.core.unfreeze(model_params), sep='/'
  )

  for key, value in flat_pretrained.items():
    if key not in flat_model:
      logging.warning('Pretrained key not found in model, skipping: %s', key)
      continue
    if value.shape != flat_model[key].shape:
      raise ValueError(
          f'Shape mismatch for {key}: '
          f'pretrained {value.shape} vs model {flat_model[key].shape}'
      )
    flat_model[key] = jnp.asarray(value)

  return flax.traverse_util.unflatten_dict(flat_model, sep='/')
