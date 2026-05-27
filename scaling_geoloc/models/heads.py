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
import numpy as np
import jaxtyping as jt
import ml_collections
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers

from scaling_geoloc.utils import misc


def backbone_head(
    backbone_output: jt.Float[jt.Array, 'B L D'],
    x: jt.Float[jt.Array, 'B D'],
    head_output_dim: int | None,
    head: str,
    head_config: ml_collections.ConfigDict,
    train: bool,
    pooling: None | str = None,
    sow_attention: bool = False,
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
    x = nn.Dense(
        head_output_dim,
        dtype=dtype,
        name='output_projection',
    )(x)
  elif head == 'geoclip':
    x = nn.Sequential(
        [
            nn.Dense(768, dtype=dtype, name='geoclip_projection_0'),
            nn.relu,
            nn.Dense(
                head_output_dim,
                dtype=dtype,
                name='geoclip_projection_1',
            ),
        ],
        name='output_projection',
    )(x)
  elif head == 'spatial_softmax':
    x = SpatialSoftmax(
        channels=head_output_dim,
        name='output_projection',
    )(backbone_output)
  elif head == 'fervers':
    pass  # Aggregate in encode_multiview.
  elif head.startswith('salad'):
    salad = SALAD.default(
        head_output_dim,
        dtype=dtype,
        name='output_projection',
        l2_normalize='nol2' not in head,
    )
    if pooling != 'tok' and pooling != 'gap':
      raise ValueError(
          f'Pooling {pooling} is not supported for SALAD head (need tok or'
          ' gap).'
      )
    x = salad(backbone_output, train)
  elif head.startswith('query'):
    if pooling != 'tok':
      raise ValueError(
          f'Pooling {pooling} is not supported for query head (need tok).'
      )
    if head == 'query':
      num_queries = 1
    else:
      num_queries = int(head.split('_')[1])
    head = QueryHead(
        output_dim=head_output_dim,
        num_queries=num_queries,
        mlp_dim=head_config['mlp_dim'],
        num_layers=head_config['num_layers'],
        num_heads=head_config['num_heads'],
        dropout_rate=head_config['dropout_rate'],
        attention_dropout_rate=head_config['attention_dropout_rate'],
        pooling=head_config['pooling'],
        l2_normalize=head_config['l2_normalize'],
        num_dustbins=head_config['num_dustbins'],
        dtype=dtype,
        name='output_projection',
    )
    x = head(backbone_output, train, sow_attention=sow_attention)
  else:
    raise ValueError(f'Unknown head {head}')

  if normalize_per_head:
    x = misc.l2_normalize_per_head_vit(x, head)
  return x


class SpatialSoftmax(nn.Module):
  """Spatial softmax."""

  channels: int
  heads: int = 64
  bias: bool = False

  @nn.compact
  def __call__(
      self, x: jt.Float[jt.Array, 'B N C']
  ) -> jt.Float[jt.Array, 'B C']:
    """Applies softmax to the spatial dimension."""
    attn = nn.Dense(self.heads, use_bias=self.bias)(x)
    attn = jax.nn.softmax(attn, axis=-2)
    value = nn.Dense(self.channels)(x)
    value = value.reshape(*value.shape[:-1], self.heads, -1)
    return (value * attn[..., None]).sum(axis=-3).reshape(*x.shape[:-2], -1)


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
          hidden_dim=512,
          token_dim=256,
          cluster_dim=128,
          num_clusters=64,
          dtype=dtype,
          name=name,
          l2_normalize=l2_normalize,
      )
    elif output_dim == 4096 + 128:
      return SALAD(
          hidden_dim=512,
          token_dim=128,
          cluster_dim=64,
          num_clusters=64,
          dtype=dtype,
          name=name,
          l2_normalize=l2_normalize,
      )
    elif output_dim == 2048 + 128:
      return SALAD(
          hidden_dim=512,
          token_dim=128,
          cluster_dim=64,
          num_clusters=32,
          dtype=dtype,
          name=name,
          l2_normalize=l2_normalize,
      )
    elif output_dim == 1024 + 64:
      return SALAD(
          hidden_dim=512,
          token_dim=64,
          cluster_dim=64,
          num_clusters=16,
          dtype=dtype,
          name=name,
          l2_normalize=l2_normalize,
      )
    elif output_dim == 512 + 64:
      return SALAD(
          hidden_dim=512,
          token_dim=64,
          cluster_dim=32,
          num_clusters=16,
          dtype=dtype,
          name=name,
          l2_normalize=l2_normalize,
      )
    elif output_dim == 512 + 32:
      return SALAD(
          hidden_dim=512,
          token_dim=32,
          cluster_dim=32,
          num_clusters=16,
          dtype=dtype,
          name=name,
          l2_normalize=l2_normalize,
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
    # cls tok <-> patch: get where info is from
    f = self.cluster_mlp(patch_tokens)  # B x M x C
    p = self.score_mlp(patch_tokens)  # B x M x NC --> probs
    t = self.token_mlp(cls_tok)  # B x T
    log_prob = misc.log_optimal_transport(p, self.dust_bin, 3)  # B x M x NC+1
    prob = jnp.exp(log_prob)[..., :-1]  # B x M x NC
    wf = (f[..., None, :] * prob[..., None]).sum(
        -3
    )  # sum over M --> B x NC x C
    if self.l2_normalize:
      wf = misc.l2_normalize(wf)  # normalize over C
      t = misc.l2_normalize(t)
    wf = wf.reshape(cls_tok.shape[0], -1)  # reshape CxNC -> C*NC
    f = jnp.concatenate([t, wf], axis=-1)  # concat channels
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


class CrossDecoder1DBlock(nn.Module):
  """Transformer cross decoder layer.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of self-attention heads.
    dtype: The dtype of the computation (default: float32).
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
      the provided value.
    approximate_gelu: approximate parameter to GELU activation.

  Returns:
    output after transformer decoder block.
  """

  mlp_dim: int
  num_heads: int
  dtype: Any = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  use_bias: bool = True
  approximate_gelu: bool = True

  @nn.compact
  def __call__(
      self,
      query: jt.Float[jt.Array, 'B L C'],
      source: jt.Float[jt.Array, 'B N C'],
      deterministic: bool,
      sow_attention_weights: bool = False,
      attention_mask: jt.Bool[jt.Array, 'B H L N'] | None = None,
  ) -> jt.Bool[jt.Array, 'B L C']:
    """Applies Decoder1DBlock module.

    Args:
      query: Input data.
      source: Input data.
      deterministic: Deterministic or not (to apply dropout).
      sow_attention_weights: Whether attention weights should be sowed.
      attention_mask: Ignore attention weights where mask is ``False``.

    Returns:
      Output after transformer encoder block.
    """
    # Attention block.
    assert query.ndim == 3
    x = nn.LayerNorm(dtype=self.dtype)(query)
    kv = nn.LayerNorm(dtype=self.dtype)(source)
    if x.shape[0] != kv.shape[0] and kv.shape[0] == 1:
      kv = jnp.repeat(kv, x.shape[0], axis=0)
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        use_bias=self.use_bias,
    )(x, kv, sow_weights=sow_attention_weights, mask=attention_mask)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = nn_layers.StochasticDepth(rate=self.stochastic_depth)(x, deterministic)
    x = x + query

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = attention_layers.MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=functools.partial(
            nn.gelu, approximate=self.approximate_gelu
        ),
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )(y, deterministic=deterministic)
    y = nn_layers.StochasticDepth(rate=self.stochastic_depth)(y, deterministic)
    return y + x


class CrossDecoder(nn.Module):
  """ViT cross decoder.

  Note: No position embeddings are added here, they are handled directly
  in the MaskedAutoEncoder module.
  """

  mlp_dim: int
  num_layers: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_droplayer_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      kv: jnp.ndarray,
      *,
      train: bool,
      sow_attention: bool = False,
  ):
    """Applies Transformer model on the inputs."""
    assert x.ndim == 3
    dtype = jax.dtypes.canonicalize_dtype(self.dtype)

    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
    for lyr in range(self.num_layers):
      droplayer_p = (
          lyr / max(self.num_layers - 1, 1)
      ) * self.stochastic_droplayer_rate
      x = CrossDecoder1DBlock(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          stochastic_depth=droplayer_p,
          name=f'encoderblock_{lyr}',
          dtype=dtype,
      )(
          x,
          kv,
          deterministic=not train,
          sow_attention_weights=(lyr == self.num_layers - 1) and sow_attention,
      )
    x = nn.LayerNorm(name='decoder_norm')(x)
    return x


class QueryHead(nn.Module):
  """Query head."""

  output_dim: int
  num_queries: int
  num_layers: int
  mlp_dim: int
  num_heads: int
  dtype: jnp.dtype = jnp.float32
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  stochastic_droplayer_rate: float = 0.0
  pooling: str = 'concat'
  l2_normalize: bool = False
  num_dustbins: int = 0

  def setup(self):
    self.decoder = CrossDecoder(
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_droplayer_rate=self.stochastic_droplayer_rate,
        dtype=self.dtype,
    )

    if self.pooling == 'learned':
      self.pooling_layer = CrossDecoder(
          mlp_dim=self.mlp_dim,
          num_layers=self.num_layers,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          stochastic_droplayer_rate=self.stochastic_droplayer_rate,
          dtype=self.dtype,
          name='pooling_decoder',
      )

  @nn.compact
  def __call__(
      self,
      features: FloatArray['B M+1 D'],
      train: bool = False,
      sow_attention: bool = False,
  ) -> FloatArray['B K']:
    """Aggregates features with SALAD."""
    queries = self.param(
        'query_tokens',
        nn.initializers.normal(stddev=0.02),
        (1, self.num_queries, features.shape[-1]),
    )
    queries = queries.repeat(features.shape[0], axis=0)
    if self.num_dustbins > 0:
      dustbins = self.param(
          'dust_bins',
          nn.initializers.normal(stddev=0.02),
          (1, self.num_dustbins, features.shape[-1]),
      )
      features = jnp.concatenate(
          [features, dustbins.repeat(features.shape[0], axis=0)], axis=-2
      )
    queries = self.decoder(
        queries, features, train=train, sow_attention=sow_attention
    )
    output_dim = self.output_dim
    if self.pooling == 'concat':
      output_dim = self.output_dim // self.num_queries
    elif self.pooling == 'learned':
      pool_tokens = self.param(
          'pool_tokens',
          nn.initializers.normal(stddev=0.02),
          (1, 1, queries.shape[-1]),
      ).repeat(queries.shape[0], axis=0)
      queries = self.pooling_layer(pool_tokens, queries, train=train)
    queries = nn.Dense(output_dim, name='query_projection')(queries)

    if self.pooling == 'concat' and self.l2_normalize:
      queries = misc.l2_normalize(queries)

    logits = {
        'max': functools.partial(jnp.max, axis=-2),
        'mean': functools.partial(jnp.mean, axis=-2),
        'sum': functools.partial(jnp.sum, axis=-2),
        'concat': lambda x: jnp.concatenate(jnp.swapaxes(x, -3, -2), axis=-1),
        'learned': lambda x: x[..., 0, :],
    }[self.pooling](queries)

    if self.l2_normalize:
      logits = misc.l2_normalize(logits)
    return logits


class ProjectionHead(nn.Module):
  """Projection head.

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
    # Normalize.
    x = misc.l2_normalize(x)
    x = WeightNormDense(
        self.output_dim,
        use_bias=False,
        name='prototypes',
        kernel_init=norm_kernel_init_fn,
        # no normalize now means apply rescale
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
