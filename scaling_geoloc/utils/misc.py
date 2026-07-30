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

"""Various convenience functions."""

from collections.abc import Mapping
import functools
from typing import Any, Callable

from etils.array_types import Array
from etils.array_types import BoolArray
from etils.array_types import FloatArray
from etils.array_types import IntArray
import flax
import jax
import jax.numpy as jnp
import numpy as np

from scaling_geoloc.utils import geo
from scaling_geoloc.utils.image import crop_to_multiple


def filter_dict(
    filter_fn: Callable[[str, Any], bool],
    d: dict[str, Any],
    unflatten: bool = True,
) -> dict[str, Any]:
  """Filter out elements of a nested dictionary."""
  d = flax.traverse_util.flatten_dict(d)
  d = dict(filter(lambda kv: filter_fn(*kv), d.items()))
  return flax.traverse_util.unflatten_dict(d) if unflatten else d


def concat_all(name: str, d: dict[str, Any], axis: int = -1) -> jnp.ndarray:
  """Concatenates all values of a dict where the final key is `name`."""
  filtered_d = filter_dict(lambda k, v: k[-1] == name, d, unflatten=False)
  return jnp.concatenate(
      [filtered_d[k] for k in sorted(filtered_d.keys())],
      axis=axis,
  )


def map_dict(
    map_fn: Callable[[str, Any], Any],
    d: dict[str, Any],
    sep: str | None = None,
) -> dict[str, Any]:
  """Filter out elements of a nested dictionary."""
  d = flax.traverse_util.flatten_dict(d, sep=sep)
  return flax.traverse_util.unflatten_dict(
      {k: map_fn(k, v) for k, v in d.items()}, sep=sep
  )


def filter_batch_for_jit(batch: dict[str, Any]) -> dict[str, Any]:
  """Exclude string and bytes arrays from a batch nested dictionary."""

  def is_string_or_bytes_array(x: Any) -> bool:
    return isinstance(x, np.ndarray) and x.dtype.type in [np.str_, np.bytes_]

  filtered = filter_dict(lambda _, v: not is_string_or_bytes_array(v), batch)
  return filter_dict(lambda _, v: v.size > 0, filtered)


def masked_mean(
    x: jnp.ndarray, mask: jnp.ndarray, axis: int | tuple[int, ...] | None = -1
) -> jnp.ndarray:
  x, mask = jnp.broadcast_arrays(x, mask)
  return jnp.sum(x * mask, axis=axis) / jnp.sum(mask, axis=axis).clip(
      min=1.0e-10
  )


def masked_bmean(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
  """Convenience function for batch masked mean."""
  x, mask = jnp.broadcast_arrays(x, mask)
  x = jax.lax.collapse(x, 1)
  mask = jax.lax.collapse(mask, 1)
  return jnp.sum(x * mask, axis=-1) / jnp.sum(mask, axis=-1).clip(min=1.0e-10)


def to_flat_batch(
    batch: Mapping[str, Any] | dict[str, Any], stride: int | None = None
) -> dict[str, jnp.ndarray]:
  """Converts a batch to a flat batch, and optionally crops to multiple."""
  batch = filter_batch_for_jit(batch)
  image = batch['query']['image']
  if stride is not None:
    image = crop_to_multiple(image, stride) if image.ndim == 3 else image
  return batch | {'image': image}


# source: third_party/py/scenic/model_lib/base_models/box_utils.py
def l2_normalize(
    x: FloatArray['... C'],
    grad_eps: float | None = 1.0e-8,
    axis: int = -1,
    numpy: bool = False,
) -> FloatArray['... C']:
  """Normalize x to unit length along last axis.

  Normalizing vectors is surprisingly tricky, because you have to address the
  case where the denominator in the normalization is tiny or zero, in which case
  gradients will explode. For this reason, we perform two normalizations: in the
  forward pass, we clamp the denominator with ~1e-40, but in the backward pass
  we clamp with `grad_eps`, which defaults to ~1e-7. This guarantees that the
  output of this function is unit norm (unless x is very very small) while
  preventing exploding gradients.

  Args:
    x: The array of values to normalize.
    grad_eps: The value to clip the squared norm by before division in the
      backward pass. None disables safe scaling. None is simple normalization .
    axis: The axis to normalize along.
    numpy: If True, use numpy instead of jax.

  Returns:
    A normalized array x / ||x||, normalized along the last axis.
  """

  def _override_gradient(fval: jnp.ndarray, bval: jnp.ndarray) -> jnp.ndarray:
    """Use `fval` in the forward pass but `bval` in the backward pass."""
    # Note that the parentheses are needed to avoid catastrophic cancellation.
    return jax.lax.stop_gradient(fval) + (bval - jax.lax.stop_gradient(bval))
  if grad_eps is None or isinstance(x, np.ndarray) or numpy:
    xnp = np if numpy else jnp
    return x / (
        xnp.linalg.norm(x, axis=axis, keepdims=True) + np.finfo(x.dtype).eps
    )
  else:
    tiny = jnp.finfo(jnp.float32).tiny
    grad_eps = jnp.maximum(tiny, grad_eps)
    denom_sq = jnp.sum(x**2, axis=axis, keepdims=True)
    normal_val = x / jnp.sqrt(jnp.maximum(tiny, denom_sq))
    normal_grad = x / jnp.sqrt(jnp.maximum(grad_eps, denom_sq))
    # Use `normal_val` in the forward pass,
    # but `normal_grad` in the backward pass.
    normal = _override_gradient(normal_val, normal_grad)
    return jnp.where(denom_sq < tiny, jnp.zeros_like(normal), normal)


def sample(
    arr: Array,
    max_num_samples: int,
    axis: int = 0,
    rng: np.random.Generator | None = None,
) -> Array:
  if max_num_samples >= arr.shape[axis]:
    return arr
  elif rng is None:
    return arr.take(np.arange(max_num_samples), axis=axis)
  else:
    return arr[rng.choice(arr.shape[axis], max_num_samples, replace=False), ...]


def to_homogeneous(x: Array['... C']) -> Array['... C+1']:
  return jnp.concatenate([x, jnp.ones_like(x[..., :1])], axis=-1)


def cross_entropy(
    source_logits: FloatArray['... D'], target_logits: FloatArray['... D']
) -> FloatArray['... 1']:
  target_probs = jax.nn.softmax(target_logits, axis=-1)
  log_source_probs = jax.nn.log_softmax(source_logits, axis=-1)
  return -(target_probs * log_source_probs).sum(axis=-1, keepdims=True)


def shuffle_and_partition(
    n_batch: int, n_tokens: int, n_masked: int, rng: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Implements random shuffling and partitioning necessary for MAE (adapted from FOMO)."""
  ids = jnp.tile(jnp.arange(n_tokens), n_batch).reshape((n_batch, n_tokens))
  n_other = n_tokens - n_masked
  rng_keys = jax.random.split(rng, n_batch)
  ids = jax.vmap(
      lambda seq, rng: jax.random.permutation(rng, seq, independent=True)
  )(ids, rng_keys)
  masked = jax.lax.dynamic_slice(ids, (0, 0), (n_batch, n_masked))
  others = jax.lax.dynamic_slice(ids, (0, n_masked), (n_batch, n_other))
  return others, masked


def indices_to_mask(
    i_keep: IntArray['... M'],
    seqlen: int,
) -> BoolArray['... S']:
  """Creates a binary mask s.t. binary_mask[i_keep] = True."""
  i_keep_flat = jax.lax.collapse(i_keep, 0, -1)
  batch_size = i_keep_flat.shape[0]
  mask = jnp.zeros((batch_size, seqlen), dtype=bool)
  all_rows = jnp.arange(batch_size).reshape(batch_size, 1)
  mask = mask.at[all_rows, i_keep_flat].set(True)
  return mask.reshape(i_keep.shape[:-1] + (seqlen,))


def batch_sample_indices(
    rng: jnp.ndarray, probs: FloatArray['B N'], k: int
) -> IntArray['B K']:
  """Sample k indices from a probability distribution."""

  def sample_indices(rng_keys: jnp.ndarray, probs: FloatArray['N']):
    return jax.random.choice(
        key=rng_keys, a=probs.shape[0], shape=(k,), p=probs, replace=False
    )

  keys = jax.random.split(rng, probs.shape[0])
  return jax.vmap(functools.partial(sample_indices))(keys, probs)


def compute_similarity(
    x: FloatArray['... D'], y: FloatArray['... D'], normalize: bool = False
) -> FloatArray['... 1']:
  """Computes similarity between two arrays."""
  if normalize:
    x = l2_normalize(x)
    y = l2_normalize(y)
  # return jnp.einsum('...d,...d->...', x, y)[..., None]
  # return jnp.dot(x, y)[..., None]
  return jnp.vecdot(x, y, axis=-1)[..., None]


def log_safe(x: FloatArray['...'], eps: float = 1.0e-8) -> FloatArray['...']:
  return jnp.log(jnp.where(x > eps, x, x + eps))


# Source: https://arxiv.org/pdf/1904.06627
class MultiSimilarityLoss:
  """Multi-similarity loss."""

  def __init__(
      self,
      alpha: float = 1.0,
      beta: float = 50.0,
      base: float = 0.0,
      margin: float = 0.0,
      skip_positives: bool = False,
      negative_weight: int = -1,  # Default no scaling.
      relative_to_pos: bool = False,
      split_terms: bool = False,
  ):
    self.alpha = alpha
    self.beta = beta
    self.base = base
    self.margin = margin
    self.pos = []
    self.neg = []
    self.num_positives = 0
    self.num_negatives = 0
    self.negative_weight = negative_weight
    self.relative_to_pos = relative_to_pos
    self.split_terms = split_terms
    self.skip_positives = skip_positives

  def pos_scale(self, element: FloatArray) -> FloatArray:
    return jnp.exp(-self.alpha * (element - self.base - self.margin))

  def neg_scale(self, element: FloatArray) -> FloatArray:
    base = -self.base if self.relative_to_pos else self.base
    return jnp.exp(self.beta * (element + (self.margin - base)))

  def add_with_mask(
      self,
      similarity: FloatArray['B N'],
      valid: IntArray['B N'],  # 1 is positive, 0 is negative, -1 is padding.
      weights: FloatArray['B N'] | None = None,
      add_positives: bool = True,
      add_negatives: bool = True,
  ) -> tuple[FloatArray['B'], FloatArray['B']]:
    """Adds the loss to the running sum."""
    positives = valid == 1
    negatives = valid == 0
    self.num_negatives += negatives.sum(-1)
    pos_sim = jnp.where(positives, similarity, jnp.inf)
    if weights is not None:
      positives = positives * weights
      negatives = negatives * weights
    pos_sum = masked_mean(self.pos_scale(pos_sim), positives)
    if add_positives:
      self.pos.append(pos_sum)
      self.num_positives += positives.sum(-1)
    if self.relative_to_pos:
      similarity = similarity - pos_sim.min(-1, keepdims=True)
      if not self.skip_positives:
        similarity = similarity.clip(min=self.base)
    return pos_sum, (
        self.add_negatives(similarity, negatives)
        if add_negatives
        else jnp.zeros_like(pos_sum)
    )

  def add_with_indices(
      self,
      similarity: FloatArray['B N'],
      positive: IntArray['B K'],
      where: BoolArray['B N'] | None = None,
      pos_weights: FloatArray['B K'] | None = None,
      add_positives: bool = True,
      add_negatives: bool = True,
  ) -> tuple[FloatArray['B'], FloatArray['B']]:
    """Adds the loss to the running sum."""
    pos_mask = positive > -1
    # Positives of -1 are ignored -> set the index to OOB, and use fill_value.
    positive_oob = jnp.where(pos_mask, positive, similarity.shape[-1])
    # OOB value is jnp.inf here, as then exp(-alpha * jnp.inf) = 0.
    pos_sim = jnp.take_along_axis(
        similarity, positive_oob, axis=-1, fill_value=jnp.inf
    )
    if where is not None:
      similarity = jnp.where(where, similarity, -jnp.inf)
    # OOB value is -jnp.inf here, as then exp(beta * -jnp.inf) = 0.
    neg_sim = jnp.put_along_axis(
        similarity, positive_oob, -jnp.inf, axis=-1, inplace=False, mode='drop'
    )

    self.num_negatives += (neg_sim > -jnp.inf).sum(-1)
    if pos_weights is not None:
      pos_weights = pos_mask * pos_weights
    else:
      pos_weights = pos_mask
    pos_sum = masked_mean(self.pos_scale(pos_sim), pos_weights)

    if add_positives:
      self.pos.append(pos_sum)
      self.num_positives += pos_mask.sum(-1)
    if self.relative_to_pos:
      neg_sim = neg_sim - pos_sim.min(-1, keepdims=True)
      if not self.skip_positives:
        neg_sim = neg_sim.clip(min=self.base)
    return pos_sum, (
        self.add_negatives(neg_sim, neg_sim > -jnp.inf)
        if add_negatives
        else jnp.zeros_like(pos_sum)
    )

  def compute(self) -> tuple[FloatArray['B'], FloatArray['B']]:
    """Computes the loss."""
    if self.skip_positives:
      pos = [jnp.zeros_like(p) for p in self.pos]
    else:
      pos = self.pos
    if self.split_terms:
      pos = jnp.stack(pos, axis=0)
      neg = jnp.stack(self.neg, axis=0)
      return (
          log_safe(1 + pos.sum(-1)).mean(0) / self.alpha,
          log_safe(1 + neg.sum(-1)).mean(0) / self.beta,
      )
    else:
      return (
          log_safe(1 + sum(pos)) / self.alpha,
          log_safe(1 + sum(self.neg)) / self.beta,
      )

  def add_negatives(
      self,
      similarity: FloatArray['B N'],
      weights: FloatArray['B N'] | BoolArray['B N'],
  ) -> FloatArray['B']:
    """Adds negative similarities to the loss."""
    if weights.dtype != jnp.bool_:
      valid = weights > 0.0
    else:
      valid = weights
    self.num_negatives += valid.sum(-1)
    similarity = jnp.where(valid, similarity, -jnp.inf)
    neg_exp = self.neg_scale(similarity)
    if self.negative_weight < 1.0e-6:
      neg_sum = (neg_exp * weights).sum(-1)
    else:
      neg_sum = masked_mean(neg_exp, weights) * self.negative_weight
    self.neg.append(neg_sum)
    return neg_sum


class CosFace:
  """CosFace loss."""

  def __init__(
      self,
      *,
      scale: float = 64.0,
      margin: float = 0.35,
  ):
    self.scale = scale
    self.margin = margin

  def __call__(
      self,
      similarity: FloatArray['B C'],
      identities: IntArray['B'],
  ) -> FloatArray['']:
    """Computes the CosFace loss from precomputed similarity."""

    batch_size, num_classes = similarity.shape

    one_hot_labels: FloatArray['B C'] = jax.nn.one_hot(
        identities, num_classes=num_classes, dtype=jnp.float32
    )

    margin_similarity = similarity - one_hot_labels * self.margin
    scaled_margin_similarity = self.scale * margin_similarity

    return (
        -jnp.sum(jax.nn.log_softmax(scaled_margin_similarity) * one_hot_labels)
        / batch_size
    )


def find_topk(
    qy: FloatArray['Q D'],
    db: FloatArray['DB D'],
    k: int,
    db_offset: IntArray['N'] | int = 0,
    recall_target: float | None = None,
    similarity_fn: Callable[[FloatArray, FloatArray], FloatArray] | str = 'dot',
    ascending: bool = False,
    where: BoolArray['DB'] | None = None,
) -> tuple[FloatArray['Q K'], IntArray['DB K']]:
  """Pmap nearest neighbor search (top-k)."""
  if isinstance(similarity_fn, str):
    similarity_fns = {
        'dot': lambda x, y: compute_similarity(x, y, normalize=False)[..., 0],
        'cosine_similarity': lambda x, y: (
            compute_similarity(x, y, normalize=True)[..., 0]
        ),
        'haversine': lambda x, y: geo.haversine_distance(x, y)[..., 0],
        'euclidean': lambda x, y: jnp.linalg.norm(x - y, axis=-1),
    }
    similarity_fn = similarity_fns[similarity_fn]
  sim = similarity_fn(qy[:, None], db[None])
  if ascending:
    sim = -sim
  if where is not None:
    sim = jnp.where(where[None], sim, -jnp.inf)
  db_size = db.shape[0] * db.shape[1]
  if recall_target is not None and recall_target < 0.99999:
    sim, neighbors = jax.lax.approx_max_k(
        sim,
        k=k,
        recall_target=recall_target,
        reduction_input_size_override=db_size,
    )
  else:
    sim, neighbors = jax.lax.top_k(sim, k=k)
  if ascending:
    sim = -sim
  if where is not None:
    neighbors = jnp.where(
        jnp.take_along_axis(where[None], neighbors, axis=-1),
        neighbors + db_offset,
        -1,
    )
  else:
    neighbors = neighbors + db_offset
  return sim, neighbors


def l2_norm(
    x: FloatArray['... D'], y: FloatArray['... D']
) -> FloatArray['... 1']:
  return ((x - y) ** 2).sum(-1, keepdims=True)


def log_sinkhorn_iterations(
    log_source: FloatArray['B|1 M'],
    log_target: FloatArray['B|1 N'],
    log_cost_matrix: FloatArray['B M N'],
    num_iters: int = 20,
) -> FloatArray['B M N']:
  """Runs the sinkhorn algorithm for a fixed number of iterations."""
  u, v = jnp.zeros_like(log_source), jnp.zeros_like(log_target)
  for _ in range(num_iters):
    u = log_source - jax.scipy.special.logsumexp(
        log_cost_matrix + v[:, None], axis=2
    )
    v = log_target - jax.scipy.special.logsumexp(
        log_cost_matrix + u[..., None], axis=1
    )
  return log_cost_matrix + u[..., None] + v[:, None]


def log_optimal_transport(
    scores: FloatArray['B N M'],
    dustbin_score: FloatArray[' '] | float,
    num_iters: int = 20,
):
  """Computes the optimal transport between a source and a target distribution."""

  b, n, m = scores.shape

  scores_aug = jnp.empty_like(scores, shape=(b, n, m + 1))
  scores_aug = scores_aug.at[:, :n, :m].set(scores)
  scores_aug = scores_aug.at[:, :, m].set(dustbin_score)
  norm = -jnp.log(m + n)
  log_source = jnp.repeat(norm, n)
  log_target = jnp.repeat(norm, m + 1)
  log_target = log_target.at[-1].set(log_target[-1] + jnp.log(n - m))
  log_probs = log_sinkhorn_iterations(
      log_source[None],
      log_target[None],
      scores_aug,
      num_iters=num_iters,
  )
  return log_probs - norm


def l2_normalize_per_head(
    x: FloatArray['... D'],
    num_heads: int,
    offset: int = 0,
    numpy: bool = False,
) -> FloatArray['... D']:
  """Normalizes the last dimension of the input per head."""
  xh = x[..., offset:]
  xh = l2_normalize(xh.reshape(x.shape[:-1] + (num_heads, -1)), numpy=numpy)
  xh = xh.reshape(x.shape[:-1] + (-1,))
  xnp = np if numpy else jnp
  if offset != 0:
    return xnp.concatenate(
        [l2_normalize(x[..., :offset], numpy=numpy), xh], axis=-1
    )
  else:
    return xh


def num_heads_and_offset_vit(output_dim: int, head: str) -> tuple[int, int]:
  """Returns the number of heads and negative offset from a ViT head."""
  if head.startswith('query'):
    return int(head.split('_')[1]), -0
  elif head.startswith('salad'):
    return {
        8192 + 256: (64, 256),
        2048 + 128: (32, 128),
        512 + 64: (16, 64),
        512 + 32: (16, 32),
    }[output_dim]
  else:
    return 1, 0


def l2_normalize_per_head_vit(
    x: FloatArray['B L D'],
    head: str,
    numpy: bool = False,
) -> FloatArray['B L D']:
  """Normalizes the last dimension of the input per head, from ViT configs."""

  num_heads, offset = num_heads_and_offset_vit(x.shape[-1], head)
  return l2_normalize_per_head(x, num_heads, offset=offset, numpy=numpy)


def weighted_softmax(
    x: FloatArray['...'], weights: FloatArray['...'], axis: int
):
  exps = jnp.exp(x - x.max(axis=axis, keepdims=True))
  return exps * weights / (exps * weights).sum(axis, keepdims=True)


