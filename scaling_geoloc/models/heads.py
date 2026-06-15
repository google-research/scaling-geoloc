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

"""Heads for the ViT backbone."""

import functools
from typing import Any

from etils.array_types import FloatArray
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt
import ml_collections
import numpy as np
from scenic.model_lib.layers import nn_layers

from scaling_geoloc.utils import misc


def backbone_head(
    backbone_output: jt.Float[jt.Array, 'B L D'],
    x: jt.Float[jt.Array, 'B D'],
    head_output_dim: int | None,
    head: str,
    head_config: ml_collections.ConfigDict,
    train: bool,
    normalize_per_head: bool = False,
    dtype: Any = jnp.float32,
) -> jt.Float[jt.Array, 'B D']:
  """Applies the head to the backbone output."""
  if not head_output_dim:
    return x

  if head == 'projection':
    x = ProjectionHead(
        hidden_dim=head_config['head_hidden_dim'],
        bottleneck_dim=head_config['head_bottleneck_dim'],
        output_dim=head_output_dim,
        normalize_last_layer=head_config['normalize_last_layer'],
        use_bn=head_config['use_bn'],
        approximate_gelu=head_config['approximate_gelu'],
        name='output_projection',
    )(x, train)
  elif head == 'linear':
    x = nn.Dense(head_output_dim, dtype=dtype, name='output_projection')(x)
  elif head.startswith('salad'):
    salad = SALAD.default(
        head_output_dim,
        dtype=dtype,
        name='output_projection',
        l2_normalize='nol2' not in head,
    )
    x = salad(backbone_output, train)
  else:
    raise ValueError(f'Unknown head {head}')

  if normalize_per_head:
    x = misc.l2_normalize_per_head_vit(x, head)
  return x


class SALAD(nn.Module):
  """Module for aggregating features with SALAD."""

  hidden_dim: int
  token_dim: int
  cluster_dim: int
  num_clusters: int
  dtype: jnp.dtype = jnp.float32
  l2_normalize: bool = True
  pretrained: bool = False

  def setup(self):
    self.token_mlp = nn.Sequential([
        nn.Dense(self.hidden_dim),
        nn.relu,
        nn.Dense(self.token_dim),
    ])

    self.cluster_mlp = nn.Sequential([
        nn.Dense(self.hidden_dim),
        nn.relu,
        *([nn.relu] if self.pretrained else []),
        nn.Dense(self.cluster_dim),
    ])

    self.score_mlp = nn.Sequential([
        nn.Dense(self.hidden_dim),
        nn.relu,
        *([nn.relu] if self.pretrained else []),
        nn.Dense(self.num_clusters),
    ])

    self.dust_bin = self.param(
        'dust_bin',
        nn.initializers.ones_init(),
        (),
        self.dtype,
    )

  @classmethod
  def default(
      cls,
      output_dim: int,
      dtype: jnp.dtype = jnp.float32,
      name: str = 'salad',
      l2_normalize: bool = True,
  ) -> 'SALAD':
    if output_dim == 8192 + 256:
      return SALAD(
          hidden_dim=512, token_dim=256, cluster_dim=128, num_clusters=64,
          dtype=dtype, name=name, l2_normalize=l2_normalize,
      )
    elif output_dim == 4096 + 128:
      return SALAD(
          hidden_dim=512, token_dim=128, cluster_dim=64, num_clusters=64,
          dtype=dtype, name=name, l2_normalize=l2_normalize,
      )
    elif output_dim == 2048 + 128:
      return SALAD(
          hidden_dim=512, token_dim=128, cluster_dim=64, num_clusters=32,
          dtype=dtype, name=name, l2_normalize=l2_normalize,
      )
    elif output_dim == 1024 + 64:
      return SALAD(
          hidden_dim=512, token_dim=64, cluster_dim=64, num_clusters=16,
          dtype=dtype, name=name, l2_normalize=l2_normalize,
      )
    elif output_dim == 512 + 64:
      return SALAD(
          hidden_dim=512, token_dim=64, cluster_dim=32, num_clusters=16,
          dtype=dtype, name=name, l2_normalize=l2_normalize,
      )
    elif output_dim == 512 + 32:
      return SALAD(
          hidden_dim=512, token_dim=32, cluster_dim=32, num_clusters=16,
          dtype=dtype, name=name, l2_normalize=l2_normalize,
      )
    else:
      raise ValueError(f'Unknown SALAD output_dim: {output_dim}')

  @classmethod
  def from_config(
      cls, config: ml_collections.ConfigDict, dtype: jnp.dtype = jnp.float32
  ) -> 'SALAD':
    return SALAD(
        config.hidden_dim,
        config.token_dim,
        config.cluster_dim,
        config.num_clusters,
        dtype,
        pretrained=config.pretrained,
    )

  def output_dim(self) -> int:
    return self.cluster_dim * self.num_clusters + self.token_dim

  @nn.compact
  def __call__(
      self, features: FloatArray['B M+1 D'], train: bool = False
  ) -> FloatArray['B NC*C+T']:
    """Aggregates features with SALAD."""
    cls_tok, patch_tokens = features[..., 0, :], features[..., 1:, :]
    f = self.cluster_mlp(patch_tokens)  # B x M x C
    p = self.score_mlp(patch_tokens)  # B x M x NC
    t = self.token_mlp(cls_tok)  # B x T
    log_prob = misc.log_optimal_transport(p, self.dust_bin, 3)  # B x M x NC+1
    prob = jnp.exp(log_prob)[..., :-1]  # B x M x NC
    wf = (f[..., None, :] * prob[..., None]).sum(-3)  # B x NC x C
    if self.l2_normalize:
      wf = misc.l2_normalize(wf)
      t = misc.l2_normalize(t)
    wf = wf.reshape(cls_tok.shape[0], -1)  # CxNC -> C*NC
    f = jnp.concatenate([t, wf], axis=-1)
    if self.l2_normalize:
      f = misc.l2_normalize(f)
    return f

  @classmethod
  def load_pretrained_params(cls) -> dict[str, jnp.ndarray]:
    pt_weights: dict[str, jt.Float[jt.Array, '...']] = np.load(
        'pretrained_weights/dino_salad.npz'
    )
    pt_weights = {
        k.replace('aggregator.', ''): v
        for k, v in pt_weights.items()
        if 'aggregator' in k
    }
    pt_weights = misc.map_dict(
        lambda k, v: (v.T if 'weight' in k[0] else v).squeeze(), pt_weights
    )
    pt_weights = {
        k.replace('_features.', '_mlp.layers_'): v
        for k, v in pt_weights.items()
    }
    pt_weights = {
        k.replace('score.', 'score_mlp.layers_'): v
        for k, v in pt_weights.items()
    }
    pt_weights = {
        k.replace('weight', 'kernel'): v for k, v in pt_weights.items()
    }
    restored_params = flax.traverse_util.unflatten_dict(pt_weights, sep='.')
    return jax.tree.map(jnp.asarray, restored_params)


class ProjectionHead(nn.Module):
  """Projection head (DINO-style MLP with weight-normalized prototype layer).

  Attributes:
    hidden_dim: Dimension of the hidden layer in the projection mlp.
    bottleneck_dim: Dimension of the bottleneck.
    output_dim: Dimension of the output ("number of prototypes").
    normalize_last_layer: Normalize the last layer of prototypes.
    use_bn: Use batch normalizations.
    approximate_gelu: approximate parameter to GELU activation.
    n_layers: Depth of the projection head.
  """

  hidden_dim: int = 2048
  bottleneck_dim: int = 256
  output_dim: int = 65536
  normalize_last_layer: bool = False
  use_bn: bool = False
  approximate_gelu: bool = True
  n_layers: int = 2

  @nn.compact
  def __call__(
      self, x: jt.Float[jt.Array, 'B L C'], train: bool
  ) -> jt.Float[jt.Array, 'B L D']:
    batch_norm = nn_layers.IdentityLayer
    if self.use_bn:
      batch_norm = functools.partial(
          nn.BatchNorm,
          use_running_average=not train,
          momentum=0.9,
          epsilon=1e-5,
      )
    for i in range(self.n_layers):
      x = nn.Dense(self.hidden_dim)(x)
      x = batch_norm()(x)
      x = nn.gelu(x, approximate=self.approximate_gelu)
      x = nn_layers.IdentityLayer(name=f'mlp_{i}')(x)
    x = nn.Dense(self.bottleneck_dim)(x)
    x = misc.l2_normalize(x)
    x = WeightNormDense(
        self.output_dim,
        use_bias=False,
        name='prototypes',
        kernel_init=norm_kernel_init_fn,
        apply_rescale=not self.normalize_last_layer,
    )(x)
    x = nn_layers.IdentityLayer(name='pre_logits')(x)
    return x


def norm_kernel_init_fn(rng, shape, dtype):
  """Initialize kernel with l2 normalized columns."""
  param = nn.linear.default_kernel_init(rng, shape, dtype)
  param = misc.l2_normalize(param, axis=0)
  return param


class WeightNormDense(nn.Dense):
  """Linear layer with weight normalized kernel."""

  apply_rescale: bool = False

  def param(self, name: str, *args, **kwargs) -> jnp.ndarray:
    param = super().param(name, *args, **kwargs)
    if name == 'kernel':
      param = misc.l2_normalize(param, axis=0)
      if self.apply_rescale:
        scale = super().param(
            'weightnorm_scale',
            jax.nn.initializers.ones,
            (1, param.shape[-1]),
            self.param_dtype,
        )
        param *= scale
    return param
