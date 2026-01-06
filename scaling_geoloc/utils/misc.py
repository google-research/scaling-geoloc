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

from collections.abc import Mapping, Sequence
import functools
import time
from typing import Any, Callable
from absl import logging
import dm_pix
from etils.array_types import Array
from etils.array_types import BoolArray
from etils.array_types import FloatArray
from etils.array_types import IntArray
import flax
from flax import jax_utils
import jax
from jax import nn
import jax.numpy as jnp
import jax.scipy as jsp
import jaxcam
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.train_lib import train_utils
import scipy
from skimage import draw as skdraw
from sklearn import decomposition
import tqdm
from google3.location.visualmapping.experimental.lemming.utils import geometry


def default_pmap_sharding(
    shape: tuple[int, ...],
    sharded_dim: int | None = 0,
    devices: list[jax.Device] | None = None,
) -> jax.sharding.Sharding:
  """Creates a sharding for pmap-style parallelism using only public JAX APIs."""
  if not jax.config.jax_pmap_shmap_merge:
    return jax.sharding.PmapSharding.default(
        shape, sharded_dim=sharded_dim, devices=devices
    )
  if sharded_dim is None:
    if devices is None:
      raise ValueError('One of sharded_dim or devices must be set.')
    mesh = jax.sharding.Mesh(np.array(devices), ('_pmap',))
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  if not shape:
    raise ValueError('shape must be non-empty for sharded_dim != None')
  num_ways_sharded = shape[sharded_dim]
  if devices is None:
    pmap_devices = np.array(jax.local_devices()[:num_ways_sharded])
  else:
    pmap_devices = np.array(devices)
  mesh = jax.sharding.Mesh(pmap_devices, ('_pmap',))
  spec_list = [None] * len(shape)
  spec_list[sharded_dim] = '_pmap'
  return jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec(*spec_list)
  )


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


def to_device_safe(data: Any, device: jax.Device) -> Any:
  def maybe_put_to_device(x):
    try:
      return jax.device_put(x, device)
    except TypeError:
      return x

  return jax.tree.map(maybe_put_to_device, data)


def sequence_to_image(
    sequence: FloatArray['... HW C'],
    img_height: int,
    img_width: int,
) -> FloatArray['... H W C']:
  """Converts sequence dimension to height width."""
  return jnp.reshape(
      sequence,
      sequence.shape[:-2] + (img_height, img_width) + (-1,),
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


def crop_to_multiple(
    images: Array['... H W N'], stride: int
) -> Array['... H2 W2 N']:
  """Crop a batch of images such as their size is a multiple of stride."""
  h, w = images.shape[-3:-1]
  ch = min((h // stride) * stride, h)
  cw = min((w // stride) * stride, w)
  return images[..., :ch, :cw, :]


def resize_to_multiple(
    images: Array['... H W N'],
    stride: int,
    method: str = 'bicubic',
    antialias: bool = True,
) -> Array['... H2 W2 N']:
  """Resize a batch of images such as their size is a multiple of stride."""
  h, w = images.shape[-3:-1]
  ch = (h // stride + 1) * stride
  cw = (w // stride + 1) * stride
  return jax.image.resize(
      images,
      (images.shape[:-3], ch, cw, images.shape[-1]),
      method=method,
      antialias=antialias,
  )


def to_flat_batch(
    batch: Mapping[str, Any] | dict[str, Any], stride: int | None = None
) -> dict[str, jnp.ndarray]:
  """Converts a batch to a flat batch, and optionally crops to multiple."""
  batch = filter_batch_for_jit(batch)
  batch_views = jax.tree.map(
      lambda x, y: jnp.concatenate([x, y], axis=1),
      batch['map']['views'],
      batch['query']['views'],
  )

  if stride is not None:
    batch_views = {
        k: (
            crop_to_multiple(v, stride)
            if (isinstance(v, jnp.ndarray) and v.ndim == 5)
            else v
        )
        for k, v in batch_views.items()
    }
  return batch | batch_views


def token_mask(
    rng: Any,
    image_shape: tuple[int, ...],
    patch_size: tuple[int, int],
    mask_ratio: float,
    mask_prob: float = 1.0,
) -> BoolArray['... NH NW 1']:
  """Generate a patch-mask for a batch of images."""
  h, w = image_shape[-3:-1]
  rng, prob_rng = jax.random.split(rng, 2)
  nh, nw = h // patch_size[0], w // patch_size[1]
  bs = image_shape[:-3]
  do_mask = jax.random.uniform(prob_rng, shape=bs + (1, 1)) < mask_prob
  is_mask = jax.random.uniform(rng, shape=bs + (nh * nw, 1)) < mask_ratio
  return do_mask & is_mask


def pad_to_multiple(
    images: Array['B H W N'], stride: int, fill_value: int = 0
) -> Array['B H2 W2 N']:
  """Pad a batch of images such as their size is a multiple of stride."""
  shape = images.shape[-3:-1]
  pad = stride - np.array(shape) % stride
  padded = jnp.pad(
      images,
      [(0, 0), (0, pad[0]), (0, pad[1]), (0, 0)],
      constant_values=fill_value,
  )
  return padded


def pad_like(
    inp: FloatArray['... Hi Wi N'], target: FloatArray['... H W N']
) -> FloatArray['... H W N']:
  out = jnp.zeros_like(target)
  out = out.at[..., : inp.shape[-3], : inp.shape[-2], :].set(inp)
  return out


def alpha_blend(
    data_i: FloatArray['...'],
    data_j: FloatArray['...'],
    alpha: FloatArray['...'] | float,
) -> FloatArray['...']:
  """Alpha blends two data sources."""
  return data_i * (1.0 - alpha) + data_j * alpha


def to_grayscale(image: FloatArray['... H W C']) -> FloatArray['... H W 1']:
  """Converts an RGB image to grayscale."""
  return jnp.sum(
      image * jnp.array([0.2989, 0.5870, 0.1140]), axis=-1, keepdims=True
  ).repeat(axis=-1, repeats=3)


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


def bbox_norm(
    points: FloatArray['N d'],
    axis: int | tuple[int, ...] = 0,
    eps: float = 1.0e-8,
    center: bool = True,
) -> np.ndarray:
  """Returns point coordinates normalized to cube of size [0, 1]."""
  lower_bound = points.min(axis=axis, keepdims=True)
  upper_bound = points.max(axis=axis, keepdims=True)
  points_norm = (points - lower_bound) / (upper_bound - lower_bound).clip(
      min=eps
  )
  if center:
    points_norm = (points_norm - 0.5) * 2.0
  return points_norm


def side_by_side(
    images: Sequence[FloatArray['... H W 3']],
) -> FloatArray['... H W 3']:
  """Concatenates images side by side."""
  return jnp.concatenate(images, axis=-3)


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


def features_to_rgb(
    features: FloatArray['... C'],
    normalize: bool = True,
    mask: BoolArray['...'] | None = None,
    offset: int = 0,
    renormalize: bool = True,
) -> FloatArray['... 3']:
  """Project a c-dimensional feature map to RGB colors using PCA."""
  pca = decomposition.PCA(n_components=offset + 3)
  features_flat = features.reshape([-1, features.shape[-1]])
  if mask is not None:
    features_fit = features[mask]
  else:
    features_fit = features_flat
  if normalize:
    features_fit = l2_normalize(features_fit)
  features_pca = (
      pca.fit(features_fit)
      .transform(features_flat)
      .reshape(features.shape[:-1] + (pca.n_components,))
  )[..., -3:]

  # Convert to range 0-1
  if renormalize:
    features_pca = l2_normalize(features_pca)
    features_pca = (features_pca + 1) / 2
  else:
    features_pca = (features_pca - features_pca.min()) / (
        jnp.ptp(features_pca) + jnp.finfo(jnp.float32).eps
    )

  return features_pca


def float_to_byte(image: FloatArray['...']) -> FloatArray['...']:
  """Converts a float image to a byte image."""
  return (image * 255.0).clip(0.0, 255.0).astype(jnp.uint8)


def byte_to_float(image: FloatArray['...']) -> FloatArray['...']:
  """Converts a byte image to a float image."""
  return image.astype(jnp.float32) / 255.0


# Adapted from location/visualmapping/experimental/fomo/scenic/trainer.py
def reshape_patches_to_image(
    array: Array['... (NH*NW) PH PW C'] | Array['... NH NW PH PW C'],
    num_rows: int,
    num_cols: int,
    image_like: bool = False,
) -> Array['... H W C']:
  """Reshapes an array of patched tokens into an image."""

  orig_shape = array.shape[: (-5 if image_like else -4)]
  patch_height, patch_width, input_channels = array.shape[-3:]
  array = array.reshape(
      *orig_shape,
      num_rows,
      num_cols,
      patch_height,
      patch_width,
      input_channels,
  )
  array = array.swapaxes(-4, -3)
  return array.reshape(
      *orig_shape,
      num_rows * patch_height,
      num_cols * patch_width,
      input_channels,
  )


def patchify(x: Array['... H W C'], ph: int, pw: int) -> Array['... L C']:
  """Converts an image to a sequence of patches."""
  return jax.lax.collapse(
      reshape_image_to_patches(crop_to_multiple(x, ph), ph, pw),
      -3,
      -1,
  )


def to_homogeneous(x: Array['... C']) -> Array['... C+1']:
  return jnp.concatenate([x, jnp.ones_like(x[..., :1])], axis=-1)


# Adapted from location/visualmapping/experimental/fomo/scenic/trainer.py
def reshape_image_to_patches(
    array: Array['... H W C'],
    patch_height: int,
    patch_width: int,
    autopad: bool = False,
    image_like: bool = False,
) -> Array['... (NH*NW) PH PW C'] | Array['... NH NW PH PW C']:
  """Reshapes an image into an array of patched tokens."""
  orig_shape = array.shape[:-3]
  array = jax.lax.collapse(array, 0, -3)
  if autopad:
    if patch_height != patch_width:
      raise ValueError('Patch height and width must be equal for padding.')
    array = pad_to_multiple(array, patch_height)
  _, input_height, input_width, input_channels = array.shape
  num_rows, num_cols = input_height // patch_height, input_width // patch_width
  array = array.reshape(
      *orig_shape,
      num_rows,
      patch_height,
      num_cols,
      patch_width,
      input_channels,
  )
  array = array.swapaxes(-4, -3)

  if image_like:
    return array
  else:
    return array.reshape(
        *orig_shape,
        num_rows * num_cols,
        patch_height,
        patch_width,
        input_channels,
    )


def cross_entropy(
    source_logits: FloatArray['... D'], target_logits: FloatArray['... D']
) -> FloatArray['... 1']:
  target_probs = jax.nn.softmax(target_logits, axis=-1)
  log_source_probs = jax.nn.log_softmax(source_logits, axis=-1)
  return -(target_probs * log_source_probs).sum(axis=-1, keepdims=True)


def batched_cov(
    w_points: FloatArray['... N d'],
    weights: FloatArray['... N'] | None = None,
    bias: bool = False,
) -> FloatArray['... d d']:
  """Computes the covariance over N samples, batcher over arbitrary dims."""
  if weights is None:
    cov_fn = functools.partial(jnp.cov, bias=bias, rowvar=False)
    cov = jax.vmap(cov_fn)(jax.lax.collapse(w_points, 0, -2))
  else:
    cov_fn = lambda p, w: jnp.cov(p, bias=bias, aweights=w, rowvar=False)
    cov = jax.vmap(cov_fn)(
        jax.lax.collapse(w_points, 0, -2),
        jax.lax.collapse(weights, 0, -1),
    )
  return cov.reshape(w_points.shape[:-2] + cov.shape[1:])


def matches_from_similarity(
    similarity: FloatArray['... M N'],
    threshold: float = 0.0,
    mutual_check: bool = False,
) -> Mapping[str, Array]:
  """Compute directional matches from similarity matrix with mutual check."""
  m, n = similarity.shape[-2:]

  num_batch_dims = jnp.ndim(similarity) - 2
  ii = jnp.expand_dims(jnp.arange(0, m), axis=range(num_batch_dims))
  jj = jnp.expand_dims(jnp.arange(0, n), axis=range(num_batch_dims))

  mi = similarity.argmax(axis=-1)  # ... M
  mj = similarity.argmax(axis=-2)  # ... N

  mscoresi = similarity.max(axis=-1)
  mscoresj = similarity.max(axis=-2)
  if mutual_check:
    mutuali = ii == jnp.take_along_axis(mj, mi, axis=-1)
    mutualj = jj == jnp.take_along_axis(mi, mj, axis=-1)
    mscoresi = mscoresi * mutuali
    mscoresj = mscoresj * mutualj

  mi = jnp.where(mscoresi > threshold, mi, -1)
  mj = jnp.where(mscoresj > threshold, mj, -1)
  return {
      'matches_i': mi[..., None],
      'matches_j': mj[..., None],
      'matching_scores_i': mscoresi[..., None],
      'matching_scores_j': mscoresj[..., None],
  }


# Source: third_party/py/dm_pix/_src/augmentation.py
def get_interpolate_function(
    mode: str,
    order: int,
    cval: float = 0.0,
    numpy: bool = False,
) -> Callable[[FloatArray, FloatArray], FloatArray]:
  """Selects the interpolation function to use based on the given parameters.

  PIX interpolations are preferred given they are faster on accelerators. For
  the cases where such interpolation is not implemented by PIX we really on
  jax.scipy.ndimage.map_coordinates. See specifics below.

  Args:
    mode: the mode parameter determines how the input array is extended beyond
      its boundaries. Modes 'nearest and 'constant' use PIX interpolation, which
      is very fast on accelerators (especially on TPUs). For all other modes,
      'wrap', 'mirror' and 'reflect', we rely on
      `jax.scipy.ndimage.map_coordinates`, which however is slow on
      accelerators, so use it with care.
    order: the order of the spline interpolation. The order has to be in the
      range [0, 1]. Note that PIX interpolation will only be used for order=1,
      for other values we use `jax.scipy.ndimage.map_coordinates`.
    cval: value to fill past edges of input if mode is 'constant'.
    numpy: If true, return a numpy interpolationfunction.

  Returns:
    The selected interpolation function.
  """
  if numpy:
    interpolate_function = functools.partial(
        scipy.ndimage.map_coordinates, mode=mode, order=order, cval=cval
    )
  else:
    if mode == 'nearest' and order == 1:
      interpolate_function = dm_pix.flat_nd_linear_interpolate
    elif mode == 'constant' and order == 1:
      interpolate_function = functools.partial(
          dm_pix.flat_nd_linear_interpolate_constant, cval=cval
      )
    else:
      interpolate_function = functools.partial(
          jax.scipy.ndimage.map_coordinates, mode=mode, order=order, cval=cval
      )
  return interpolate_function


def grid_indices(
    h: int, w: int, batch_dims: tuple[int, ...] | None = None
) -> IntArray['*b h w 2']:
  xy = jnp.stack(
      jnp.meshgrid(jnp.arange(0, w), jnp.arange(0, h), indexing='xy'), axis=-1
  )
  if batch_dims is not None:
    xy = jnp.expand_dims(xy, axis=range(len(batch_dims)))
    xy = jnp.tile(xy, batch_dims + (1, 1, 1))
  return xy


def indices_to_coords(indices: IntArray) -> FloatArray:
  return indices + 0.5


def coords_to_indices(coords: FloatArray) -> IntArray:
  return np.rint(coords - 0.5).astype(np.int32)


def resize_image_like(
    data: Mapping[str, Any],
    resize_factor: float = 2.0,
    antialias: bool = True,
) -> Mapping[str, Any]:
  """Resize all image-like entries by a factor."""
  for scene in ['query', 'map']:
    if scene not in data:
      continue
    image_pixels = data[scene]['views']['image_pixels']
    batch_dims = image_pixels.shape[:-3]
    h, w = image_pixels.shape[-3:-1]
    hr, wr = int(h * resize_factor), int(w * resize_factor)

    for k, v in data[scene]['views'].items():
      if not isinstance(v, (np.ndarray, jnp.ndarray)):
        continue
      if v.shape[:-1] == image_pixels.shape[:-1]:
        data[scene]['views'][k] = jax.image.resize(
            v,
            shape=batch_dims + (hr, wr, v.shape[-1]),
            method=('bilinear' if k == 'image_pixels' else 'nearest'),
            antialias=antialias,
        )
  return data


def match_features(
    features_i: FloatArray['... MH MW C'],
    features_j: FloatArray['... NH NW C'],
    **kwargs,
) -> Mapping[str, Array]:
  """Computes matches between features_i and features_j.

  Assumes features to be on a grid.

  Args:
    features_i: Features for each patch in image i.
    features_j: Features for each patch in image j.
    **kwargs: Arguments for matches_from_similarity.

  Returns:
    Dictionary with similarity, keypoints_i, keypoints_j, matches_i, matches_j,
    matching_scores_i, matching_scores_j.
  """
  mh, mw = features_i.shape[-3:-1]
  nh, nw = features_j.shape[-3:-1]
  batch_dims = features_j.shape[:-3]
  xy_i = jax.lax.collapse(grid_indices(mh, mw, batch_dims), -3, -1)
  xy_j = jax.lax.collapse(grid_indices(nh, nw, batch_dims), -3, -1)

  similarity = jnp.einsum(
      '...md,...nd->...mn',
      jax.lax.collapse(features_i, -3, -1),
      jax.lax.collapse(features_j, -3, -1),
  )

  matches = matches_from_similarity(similarity, **kwargs)

  return {
      'similarity': similarity[..., None],
      'keypoints_i': indices_to_coords(xy_i),
      'keypoints_j': indices_to_coords(xy_j),
      **{k: v for k, v in matches.items()},
  }


def xy_to_rgb(x: float, y: float) -> FloatArray['3']:
  """Converts a 2D point (between 0 and 1) to an RGB color."""
  return jnp.where(x > y, jnp.stack([x, 1 - x, y]), jnp.stack([x, 1 - y, y]))


def cmap_2d(xy_norm: FloatArray['... 2']) -> FloatArray['... 3']:
  """Converts normalized 2D coordinates to RGB colors."""

  def to_rgb(xy):
    x, y = xy
    return xy_to_rgb(x, y)

  return jnp.apply_along_axis(to_rgb, -1, xy_norm)


def draw_correspondences(
    image_ij: FloatArray['H W C'],
    mpoints_i: FloatArray['L 2'],
    mpoints_j: FloatArray['L 2'],
    mscores: FloatArray['L 1'],
    valid: BoolArray['L 1'],
    offset_b: tuple[float, float] | None = None,  # (r, c), default (0, w/2)
    color_by_coords: bool = True,
    draw: str = 'lines',
) -> FloatArray['H W C']:
  """Draw lines between points_a and points_b into image_ab."""
  h, w = image_ij.shape[-3:-1]
  opacity = 1.0
  if offset_b is None:
    offset_b = (0, int(w // 2))
  mpoints_i = coords_to_indices(mpoints_i)
  mpoints_j = coords_to_indices(mpoints_j)

  for (c0, r0), (c1, r1), (sc,), (isv,) in zip(
      mpoints_i, mpoints_j, mscores, valid
  ):
    if not isv:
      continue
    or1, oc1 = offset_b
    if draw == 'lines':
      rr, cc, _ = skdraw.line_aa(r0, c0, r1 + or1, c1 + oc1)
    elif draw == 'points' or sc < 0:
      rr, cc = zip(
          skdraw.disk((r0, c0), 2), skdraw.disk((r1 + or1, c1 + oc1), 2)
      )
      rr, cc = np.concatenate(rr), np.concatenate(cc)
    elif draw == 'points+lines':
      rr, cc = zip(
          skdraw.line_aa(r0, c0, r1 + or1, c1 + oc1),
          skdraw.disk((r0, c0), 3),
          skdraw.disk((r1 + or1, c1 + oc1), 3),
      )
      rr, cc = np.concatenate(rr), np.concatenate(cc)
    else:
      raise ValueError(f'Unknown draw method: {draw}')
    if color_by_coords:
      c = xy_to_rgb(c0 / (w // 2), r0 / h)
      opc = sc * opacity
    else:
      c = np.array((0.0, 1.0, 0.0)) * sc + np.array((1.0, 0.0, 0.0)) * (1 - sc)
      opc = opacity
    image_ij[rr, cc] = (
        image_ij[rr, cc] * (1 - opc) + opc * opacity * np.array(c)[None]
    )
  return image_ij


def draw_matches(
    image_ij: FloatArray['H W C'],
    points_i: FloatArray['M 2'],
    points_j: FloatArray['N 2'],
    matches_i: IntArray['M 1'],
    scores_i: FloatArray['M 1'],
    offset_b: tuple[float, float],
    color_by_coords: bool,
    draw: str,
) -> FloatArray['H W C']:
  """Draw matches between two images."""
  return draw_correspondences(
      image_ij,
      points_i,
      np.take_along_axis(points_j, matches_i, axis=-2),
      jnp.where(matches_i > -1, scores_i, -1),
      matches_i > -1,
      offset_b=offset_b,
      color_by_coords=color_by_coords,
      draw=draw,
  )


def batch_draw_matches(
    image_ij: FloatArray['... H W C'],
    points_i: FloatArray['... M 2'],
    points_j: FloatArray['... N 2'],
    matches_i: IntArray['... M 1'],
    scores_i: FloatArray['... M 1'],
    offset_b: tuple[float, float] | None = None,  # (r, c), default (0, w/2)
    color_by_coords: bool = True,
    draw: str = 'lines',
) -> FloatArray['... H W C']:
  """Draw matches over arbitrary amount of batch dimensions."""
  batch_dims = image_ij.shape[:-3]
  nb = len(batch_dims)
  args = (
      jax.lax.collapse(image_ij, 0, nb),
      jax.lax.collapse(points_i, 0, nb),
      jax.lax.collapse(points_j, 0, nb),
      jax.lax.collapse(matches_i, 0, nb),
      jax.lax.collapse(scores_i, 0, nb),
  )
  b = args[0].shape[0]
  images = []
  for i in range(b):
    images.append(
        draw_matches(
            *[np.array(arg[i]) for arg in args],  # To numpy.
            offset_b,
            color_by_coords,
            draw,
        )
    )
  return jnp.stack(images).reshape(*batch_dims, *image_ij.shape[-3:])


def image_resize_host(
    image: FloatArray['... H W C'],
    shape: tuple[int, ...],
    method: str,
    antialias: bool = True,
    jit: bool = False,
) -> FloatArray['... H W C']:
  """Resize image to a certain shape on host. Disable jit by default."""
  # Even though we manually transfer, there are still some hidden transfers...
  host = jax.devices('cpu')[0]
  with jax.disable_jit(disable=not jit):
    image = jax.device_put(image, host)
    with jax.default_device(host):
      with jax.transfer_guard('allow'):
        return jax.device_get(
            jax.image.resize(
                image,
                shape=shape,
                method=method,
                antialias=antialias,
            )
        )


def inv_pose(
    y_t_x: FloatArray['... 4 4'], check: bool = False
) -> FloatArray['... 4 4']:
  """Invert SE(3) pose."""
  y_r_x, y_p_x = y_t_x[..., :3, :3], y_t_x[..., :3, -1:]
  x_r_y = jnp.moveaxis(y_r_x, -2, -1)
  x_p_y = -jnp.einsum('...ij,...jk->...ik', x_r_y, y_p_x)
  x_t_y = jnp.concatenate([x_r_y, x_p_y], axis=-1)
  x_t_y = jnp.concatenate([x_t_y, y_t_x[..., -1:, :]], -2)
  if check:
    r = np.abs(
        x_t_y.astype(np.float64)
        - np.linalg.inv(np.array(y_t_x).astype(np.float64))
    ).max()
    if r > 1.0e-6:
      raise OverflowError(
          f'Numerical instability in inv_pose detected. Delta={r}.'
      )
  return x_t_y


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


def info_nce(
    pos_sim: FloatArray['... 1'],
    neg_sim: FloatArray['... N'],
    neg_valid: BoolArray['... N'] | None = None,
    temperature: float = 1.0,
) -> FloatArray['...']:
  """Computes the information NCE loss."""
  sample_sim = jnp.concatenate([pos_sim, neg_sim], axis=-1)
  if neg_valid is None:
    sample_valid = None
  else:
    sample_valid = jnp.concatenate([jnp.ones_like(pos_sim), neg_valid], axis=-1)
  sample_log_smax = nn.log_softmax(
      sample_sim / temperature, axis=-1, where=sample_valid
  )
  sample_log_p = sample_log_smax[..., 0]
  return -sample_log_p


def info_dce(
    pos_sim: FloatArray['... 1'],
    neg_sim: FloatArray['... N'],
    neg_valid: BoolArray['... N'] | None = None,
    axis: int = -1,
    temperature: float = 1.0,
) -> FloatArray['...']:
  """Computes the decoupled information NCE loss."""
  return -pos_sim / temperature + jsp.special.logsumexp(
      jnp.where(neg_valid, neg_sim, -jnp.inf) / temperature,
      axis=axis,
      where=neg_valid,
  )


def lidar_depth_to_xyz(
    depth: FloatArray['V H W 1'],
    camera: jaxcam.Camera,
) -> FloatArray['V H W 3']:
  """Converts lidar depth to 3D points."""
  h, w = depth.shape[-3:-1]
  pixels = grid_indices(h, w, depth.shape[:-3]) + 0.5
  w_points = jax.vmap(jaxcam.pixels_to_points)(camera, pixels, depth[..., 0])
  w_points = w_points[..., :3] / w_points[..., -1:]
  return w_points


def distort_image_coordinates(
    xy: FloatArray['H W 2'],
    camera: jaxcam.Camera,
) -> tuple[FloatArray['H W 2'], jaxcam.Camera]:
  """Distort image coordinates and return the pinhole camera."""
  assert not camera.use_inverted_distortion
  distorted_camera = camera.replace(  # pytype: disable=attribute-error
      position=jnp.zeros_like(camera.position),
      orientation=jnp.eye(3),
  )
  pinhole_camera = distorted_camera.replace(  # pytype: disable=attribute-error
      radial_distortion=None,
      tangential_distortion=None,
      projection_type='perspective',
  )
  uv = jaxcam.pixels_to_local_rays(pinhole_camera, xy)
  xy_distorted = jaxcam.project(distorted_camera, uv)[..., :2]
  return xy_distorted, pinhole_camera.replace(
      position=camera.position, orientation=camera.orientation
  )


def interpolate_image(
    image: FloatArray['H W C'],
    coords: FloatArray['H W 2'],
    order: int = 1,  # 0 for nearest, 1 for linear
    cval: float = 0.0,
    mode: str = 'constant',
    numpy: bool = False,
) -> FloatArray['H W C']:
  """Interpolates an image at given coordinates."""
  interp_function = get_interpolate_function(mode, order, cval, numpy)
  if numpy:
    image = np.array(image)
    coords = np.array(coords)
    return np.stack(
        [
            interp_function(layer, np.moveaxis(coords, -1, 0))
            for layer in np.permute_dims(image, (2, 0, 1))
        ],
        axis=-1,
    )
  else:
    return jax.vmap(interp_function, in_axes=(-1, None), out_axes=-1)(
        image, jnp.moveaxis(coords, -1, 0)
    )


def undistort_image(
    image: FloatArray['H W C'],
    camera: jaxcam.Camera,
    **interp_kwargs,
) -> tuple[FloatArray['H W C'], jaxcam.Camera]:
  """Convenience function to undistort an image."""
  h, w = image.shape[-3:-1]
  xy = grid_indices(h, w) + 0.5
  xy_d, pinhole_camera = distort_image_coordinates(xy, camera)
  ij_d = jnp.flip(xy_d, -1)
  return interpolate_image(image, ij_d - 0.5, **interp_kwargs), pinhole_camera


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
def multi_similarity_loss(
    similarity_l2: FloatArray['B x N'],
    positive: IntArray[
        'B x K'
    ],  # all others are negative. -1 is also negative.
    alpha: float = 1.0,
    beta: float = 50.0,
    base: float = 0.0,
    where: BoolArray['B x N'] | None = None,  # on negatives only.
    pos_weights: FloatArray['B x K'] | None = None,
) -> tuple[FloatArray['B'], FloatArray['B']]:
  """Computes the multi-similarity loss."""
  # Positives of -1 are ignored -> set the index to OOB, and use fill_value.
  positive = jnp.where(positive == -1, similarity_l2.shape[-1], positive)
  # OOB value is jnp.inf here, as then exp(-alpha * jnp.inf) = 0.
  pos_sim = jnp.take_along_axis(
      similarity_l2, positive, axis=-1, fill_value=jnp.inf
  )
  if where is not None:
    similarity_l2 = jnp.where(where, similarity_l2, -jnp.inf)
  # OOB value is -jnp.inf here, as then exp(beta * -jnp.inf) = 0.
  neg_sim = jnp.put_along_axis(
      similarity_l2, positive, -jnp.inf, axis=-1, inplace=False, mode='drop'
  )
  pos_exp = jnp.exp(-alpha * (pos_sim - base))
  if pos_weights is not None:
    pos_exp = pos_exp * pos_weights
  ms_pos = log_safe(1 + pos_exp.sum(-1))
  ms_neg = log_safe(1 + jnp.exp(beta * (neg_sim - base)).sum(-1))
  return ms_pos / alpha, ms_neg / beta


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


# Source: https://arxiv.org/pdf/2003.13911
def proxy_anchor_loss(
    proxy_sim: FloatArray['B N'],
    proxy_mask: IntArray['B N'],  # -1 is ignore / padding.
    margin: float = 0.1,
    beta: float = 1.0,
    alpha: float | None = None,
    axis_name: str | None = None,
) -> tuple[FloatArray['1'], FloatArray['1']]:
  """Computes the proxy anchor loss."""
  # Positives have the same label, simple as that.
  alpha = alpha if alpha is not None else beta
  pos_exponent = jnp.exp(-alpha * (proxy_sim - margin))
  neg_exponent = jnp.exp(beta * (proxy_sim + margin))

  pos_lse = pos_exponent * (proxy_mask == 1)
  neg_lse = neg_exponent * (proxy_mask == 0)

  if axis_name is not None:
    # Assume that sharded over batch dimension
    pos_lse = jax.lax.psum(pos_lse, axis_name=axis_name)
    neg_lse = jax.lax.psum(neg_lse, axis_name=axis_name)
    proxy_mask = (
        jax.lax.psum(proxy_mask == 1, axis_name=axis_name) > 0
    ).astype(jnp.int32)
  pos_proxies = (proxy_mask == 1).any(-2)
  pos_term = masked_mean(log_safe(1 + pos_lse).sum(-2), pos_proxies)
  neg_term = log_safe(1 + neg_lse).sum(-2).mean()

  w_alpha = beta / (alpha + beta)
  w_beta = 1 - w_alpha
  return pos_term[None] * w_alpha, neg_term[None] * w_beta


def proxy_anchor_loss_from_labels(
    proxy_sim: FloatArray['B N'],
    proxy_labels: IntArray['B'],  # -1 is ignore / padding.
    margin: float = 0.1,
    alpha: float = 1.0,
    axis_name: str | None = None,
) -> tuple[FloatArray['1'], FloatArray['1']]:
  """Computes the proxy anchor loss from labels."""
  # Positives have the same label, simple as that.
  proxy_mask = jnp.zeros_like(proxy_sim, dtype=jnp.int32)
  proxy_mask = proxy_mask.at[jnp.arange(proxy_sim.shape[0]), proxy_labels].set(
      1
  )
  return proxy_anchor_loss(
      proxy_sim, proxy_mask, margin=margin, alpha=alpha, axis_name=axis_name
  )


batched_lidar_depth_to_xyz = jax.vmap(lidar_depth_to_xyz)


def draw_image_border(
    image: FloatArray['... H W C'],
    border_color: FloatArray['... 3'],
    border_width: int = 1,
) -> FloatArray['... H W C']:
  """Draws a border around points in an image."""
  h, w = image.shape[-3:-1]
  coords = grid_indices(h, w, image.shape[:-3])
  border_mask = (
      (coords < border_width).any(-1)
      | (coords[..., 0] >= w - border_width)
      | (coords[..., 1] >= h - border_width)
  )[..., None]
  return jnp.where(border_mask, border_color, image)


def label_image_border(
    image: FloatArray['... H W C'],
    labels: IntArray['...'],
    colors_per_label: FloatArray['NL 3'],
    border_width: int = 1,
) -> FloatArray['H W C']:
  """Labels an image with a border."""
  for label, label_color in enumerate(colors_per_label):
    image = jnp.where(
        labels[..., None, None, None] == label,
        draw_image_border(
            image,
            label_color,
            border_width=border_width,
        ),
        image,
    )
  return image


def spherical_project(
    direction: FloatArray['3'], f_horizontal: float, f_vertical: float
) -> FloatArray['2']:
  """Projects a 3D direction into pixel coordinates of a spherical image."""
  unit_ray = direction / jnp.linalg.norm(direction).clip(min=1.0e-6)
  w_r_cam = jnp.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
  unit_ray = (unit_ray[None] @ w_r_cam.T)[0]
  phi = jnp.acos(unit_ray[2])
  y = phi * f_vertical
  theta = jnp.arctan2(unit_ray[1], unit_ray[0]) + jnp.pi
  x = theta * f_horizontal
  return jnp.stack([x, y])


def deg2rad(deg: float) -> float:
  return deg / 180 * jnp.pi


def rotation_from_rpy(
    roll: float, pitch: float, yaw: float, degree: bool = False
) -> FloatArray['3 3']:
  """Creates a rotation matrix from roll, pitch, yaw."""
  if degree:
    roll = deg2rad(roll)
    pitch = deg2rad(pitch)
    yaw = deg2rad(yaw)
  rot_x = jnp.zeros((3, 3))
  rot_x = rot_x.at[0, 0].set(1)
  rot_x = rot_x.at[1, 1].set(jnp.cos(pitch))
  rot_x = rot_x.at[1, 2].set(jnp.sin(pitch))
  rot_x = rot_x.at[2, 1].set(-jnp.sin(pitch))
  rot_x = rot_x.at[2, 2].set(jnp.cos(pitch))

  rot_y = jnp.zeros((3, 3))
  rot_y = rot_y.at[0, 0].set(jnp.cos(yaw))
  rot_y = rot_y.at[0, 2].set(-jnp.sin(yaw))
  rot_y = rot_y.at[1, 1].set(1)
  rot_y = rot_y.at[2, 0].set(jnp.sin(yaw))
  rot_y = rot_y.at[2, 2].set(jnp.cos(yaw))

  rot_z = jnp.zeros((3, 3))
  rot_z = rot_z.at[0, 0].set(jnp.cos(roll))
  rot_z = rot_z.at[0, 1].set(jnp.sin(roll))
  rot_z = rot_z.at[1, 0].set(-jnp.sin(roll))
  rot_z = rot_z.at[1, 1].set(jnp.cos(roll))
  rot_z = rot_z.at[2, 2].set(1)
  return rot_z @ rot_x @ rot_y


def camera_from_rpyf(
    height: int, width: int, roll: float, pitch: float, yaw: float, hfov: float
) -> jaxcam.Camera:
  """Creates a camera from roll, pitch, yaw, hfov, height, width."""
  cam_r_identity = rotation_from_rpy(roll, pitch, yaw, degree=True)
  focal_length = width / 2 / jnp.tan(deg2rad(hfov / 2))
  principal_point = jnp.array([width / 2, height / 2]) + 0.5
  image_size = jnp.array([width, height])
  return jaxcam.Camera.create(
      orientation=cam_r_identity,
      position=jnp.zeros(3),
      focal_length=focal_length,
      principal_point=principal_point,
      image_size=image_size,
      is_fisheye=False,
      radial_distortion=np.zeros(3),
      tangential_distortion=np.zeros(2),
  )


def border_in_pano(
    pano_img: FloatArray['H W 3'], cam: jaxcam.Camera
) -> FloatArray['6 2']:
  """Converts the border of an image into pano image coordinates."""
  ph, pw = pano_img.shape[:2]
  f_v = ph / jnp.pi
  f_h = pw / (2 * jnp.pi)
  w, h = cam.image_size
  principal_point = jnp.array([w / 2, h / 2])
  xy = grid_indices(h, w) + 0.5
  border_xy = jnp.concatenate(
      [
          principal_point[None],
          xy[0],
          xy[:, -1],
          jnp.flip(xy[-1], 0),
          jnp.flip(xy[:, 0], 0),
      ],
      axis=0,
  )
  border_rays = jaxcam.pixels_to_rays(cam, border_xy)
  pano_xy = jax.vmap(
      functools.partial(spherical_project, f_horizontal=f_h, f_vertical=f_v)
  )(border_rays)
  return pano_xy - 0.5


def img_from_pano(
    pano_img: FloatArray['H W 3'], cam: jaxcam.Camera, **interp_kwargs
) -> FloatArray['Hi Wi 3']:
  """Extracts pixels of a camera from a pano."""
  w, h = cam.image_size
  xy = grid_indices(h, w) + 0.5
  rays = jaxcam.pixels_to_rays(cam, xy)
  ph, pw = pano_img.shape[:2]
  f_v = ph / jnp.pi
  f_h = pw / (2 * jnp.pi)
  pano_xy = jax.vmap(
      jax.vmap(
          functools.partial(spherical_project, f_horizontal=f_h, f_vertical=f_v)
      )
  )(rays)
  pano_ij = jnp.flip(pano_xy, -1)
  return interpolate_image(pano_img, pano_ij - 0.5, **interp_kwargs)


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
        'haversine': lambda x, y: haversine_distance(x, y)[..., 0],
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


pmap_topk = jax.pmap(
    find_topk,
    static_broadcasted_argnums=[2, 4, 5, 6],
    out_axes=(1, 1),
    axis_name='batch',
)


def parallel_topk(
    qy: FloatArray['Q D'],  # replicated
    db: FloatArray['N d_size//N D'],
    k: int,
    recall_target: float | None = None,  # None for exact search
    similarity_fn: Callable[[FloatArray, FloatArray], FloatArray] | str = 'dot',
    ascending: bool = False,
) -> tuple[FloatArray['Q K'], IntArray['Q K']]:
  """Parallel nearest neighbor search (top-k)."""
  sim, neighbors = pmap_topk(
      jax.device_put_replicated(qy, jax.local_devices()),
      db,
      k,
      jnp.arange(db.shape[0]) * db.shape[1],  # db_offset
      recall_target,
      similarity_fn,
      ascending,
  )
  # Collapse the device dimension.
  sim, neighbors = jax.tree.map(
      lambda x: jax.lax.collapse(x, start_dimension=1), (sim, neighbors)
  )  # Q x NK
  # Get top-k over all devices.
  sim, resorted_idx = jax.lax.top_k(sim if not ascending else -sim, k=k)
  sim = sim if not ascending else -sim
  neighbors = jnp.take_along_axis(neighbors, resorted_idx, axis=-1)

  return sim, neighbors


def distributed_topk(
    qy: FloatArray['Q D'],  # sharded
    db: FloatArray['d_size//N D'],  # Assumes db is spread across ALL devices.
    k: int,
    recall_target: float | None = None,  # None for exact search
    similarity_fn: Callable[[FloatArray, FloatArray], FloatArray] | str = 'dot',
    ascending: bool = False,
    where: BoolArray['d_size//N'] | None = None,  # Optional mask (True is keep)
    final_topk: bool = True,
) -> tuple[FloatArray['Q K'], IntArray['Q K']]:
  """Distributed nearest neighbor search with pmap. Gather over ALL devices."""
  # Get all queries across devices.
  a_idx = jax.lax.axis_index('batch')
  qy = jax.lax.all_gather(qy, axis_name='batch', tiled=True)
  sim, neighbors = find_topk(
      qy,
      db,
      k,
      db_offset=a_idx * db.shape[0],
      recall_target=recall_target,
      similarity_fn=similarity_fn,
      ascending=ascending,
      where=where,
  )

  sim, neighbors = jax.tree.map(
      lambda x: jax.lax.all_to_all(
          x,
          axis_name='batch',
          split_axis=0,  # ND*B --> B
          concat_axis=len(x.shape) - 1,  # Does not support -1 index.
          tiled=True,  # This chunks the batch dimension: ND*B --> B.
      ),
      (sim, neighbors),
  )

  # Get top-k over all devices.
  if final_topk:
    sim, resorted_idx = jax.lax.top_k(sim if not ascending else -sim, k=k)
    sim = sim if not ascending else -sim
    neighbors = jnp.take_along_axis(neighbors, resorted_idx, axis=-1)

  return sim, neighbors


def pmap_self_topk(
    db: FloatArray['S DB D'],
    k: int,
    batch_size: int,
    show_progress: bool = False,
    where: BoolArray['S DB'] | None = None,
    **kwargs,
) -> tuple[FloatArray['S DB K'], IntArray['S DB K']]:
  """Pmap nearest neighbor search (top-k) with mask."""
  if where is None:
    where = jax_utils.replicate(jnp.ones((db.shape[1],), dtype=jnp.bool_))

  def topk_fn(
      query_shard: FloatArray['Qi D'],
      db: FloatArray['DB D'],
      where: BoolArray['DB'],  # Valid mask.
  ):
    return distributed_topk(
        query_shard,
        db,
        k,
        where=where,
        **kwargs,
    )

  pmapped_topk = jax.pmap(
      topk_fn,
      axis_name='batch',
      donate_argnums=(0,),
  )
  all_sim = []
  all_neighbors = []

  for i in tqdm.tqdm(
      range(db.shape[1] // batch_size + 1), disable=not show_progress
  ):
    start = i * batch_size
    if start >= db.shape[1]:
      break
    end = min((i + 1) * batch_size, db.shape[1])
    sim, neighbors = pmapped_topk(db[:, start:end], db, where)
    all_sim.append(sim)
    all_neighbors.append(neighbors)
  return jnp.concatenate(all_sim, axis=1), jnp.concatenate(
      all_neighbors, axis=1
  )


def gather_global_fn(
    axis: int = 0,
    tiled: bool = True,
    donate: bool = False,
    start_end: tuple[int, int] | None = None,
) -> Callable[
    [FloatArray['S ...'], tuple[int, int] | None], FloatArray['S ...']
]:
  """Gathers a tensor over all devices."""
  donate_argnums = (0,) if donate else ()
  if axis != 0 and start_end is not None:
    raise ValueError('start_end is not supported for axis != 0.')

  def all_gather_fn(x, start_end):
    if start_end is not None:
      start, end = start_end
      x = x[start:end]
    return jax.lax.all_gather(x, axis_name='batch', axis=axis, tiled=tiled)

  return jax.pmap(
      all_gather_fn,
      axis_name='batch',
      donate_argnums=donate_argnums,
      static_broadcasted_argnums=(1,),
  )


def gather_globally(
    x: FloatArray['S ...'],
    axis: int = 0,
    tiled: bool = True,
    donate: bool = False,
    start_end: tuple[int, int] | None = None,
) -> FloatArray['S ...']:
  """Gathers a tensor over all devices."""
  return gather_global_fn(axis, tiled, donate)(x, start_end)


def gather_locally(x: FloatArray['NL N//NG ...']) -> FloatArray['N//NP ...']:
  """Gathers a tensor over all local devices."""
  return dataset_utils.unshard(x)


def gather_globally_safe(
    x: FloatArray['S ...'],
    tiled: bool = True,
    donate: bool = False,
) -> FloatArray['...']:
  """Gathers a tensor over all devices safely on cpu."""
  n_elements = x.shape[1]
  n_slice = max(n_elements // jax.device_count(), 1)

  host_data = []
  gather_fn = gather_global_fn(tiled=False, donate=donate)
  offset = 0
  while offset < n_elements:
    end = min(offset + n_slice, n_elements)
    data_i = gather_fn(x, (offset, end))  # NG x end-offset
    host_data.append(np.array(jax.device_get(data_i[0])))
    del data_i
    offset = end
  host_data = np.concatenate(host_data, axis=1)
  if tiled:
    return np.concatenate(host_data, axis=0)
  else:
    return host_data


def slice_globally(
    x_replicated: FloatArray['DL ...'],  # replicated
) -> FloatArray['S ...']:
  """Slices a replicated tensor s.t. each device has a different slice."""

  def _slice(x):
    axis_idx = jax.lax.axis_index('batch')
    size_per_device = x.shape[0] // jax.device_count()
    indices = jnp.arange(size_per_device) + size_per_device * axis_idx
    return x[indices]

  return jax.pmap(_slice, axis_name='batch')(x_replicated)


def gather_from_replicated(
    x_replicated: FloatArray['DL ...'],  # replicated
    indices: IntArray['DL K ...'],  # sharded
) -> FloatArray['S ...']:
  """Slices a replicated tensor s.t. each device has a different slice."""
  return jax.pmap(lambda x, i: x[i], axis_name='batch')(x_replicated, indices)


def get_process_slice(
    x: FloatArray['N ...'],  # numpy
) -> FloatArray['N//NP ...']:
  """Returns a slice of a tensor for a single process."""
  size_per_process = x.shape[0] // jax.process_count()
  start = jax.process_index() * size_per_process
  end = (jax.process_index() + 1) * size_per_process
  return x[start:end]


def distribute_globally(
    x: FloatArray['...'], shard_over_processes: bool = True
) -> FloatArray['S ...']:
  """Shards a tensor over all devices."""

  if shard_over_processes:
    x = get_process_slice(x)

  def pmap_shard(x):
    x = dataset_utils.shard(x)
    return jax.device_put(
        x,
        default_pmap_sharding(x.shape),
    )

  return jax.tree.map(pmap_shard, x)


def l2_norm(
    x: FloatArray['... D'], y: FloatArray['... D']
) -> FloatArray['... 1']:
  return ((x - y) ** 2).sum(-1, keepdims=True)


# Source: third_party/py/big_vision/models/xp/fmr3d/pct.py
def farthest_point_sampling(
    pointcloud: jnp.ndarray,
    k: int,
    pc_mask: jnp.ndarray | None = None,
    closest_first: bool = False,
    rng: jnp.ndarray | int | None = None,
    residuals_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = l2_norm,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Gets k farthest points for each example.

  The first point selected is the farthest point from the origin.  The remaining
  points are selected such that they are the farthest from the previously
  selected points.

  Warning: Any points with numerically identical coordinates are considered
  to be the same point and will only be selected once, so please add a small
  amount of noise to the pointcloud if duplicates should be selected
  separately.

  Args:
    pointcloud: [batch, num_points, 3] pointcloud.
    k: number of tokens to sample.
    pc_mask: optional[batch, num_points] mask for valid points in the
      pointcloud, with `1` for valid points and `0` for padding.
    closest_first: if True, the first point selected will be the closest to the
      origin.
    rng: optional jax random key for sampling. If None, the points are selected
      in a deterministic order (farthest next).
    residuals_fn: function to compute the residuals between two points (should
      be positive).

  Returns:
    Tuple of:
      - [batch, k, 3] sampled pointcloud.  Coordinates for padding points have
        0.0 values.
      - [batch, k] mask for valid points in the sampled pointcloud.
      - [batch, k] indices of the sampled points in the original pointcloud.
        Indices for paddings points are -1.
    If k is greater than `num_points`, the original pointcloud and mask
    are returned.
  """
  k = min(k, pointcloud.shape[1])

  if pc_mask is None:
    pc_mask = jnp.ones_like(pointcloud[..., 0], dtype=jnp.bool)

  pc_mask = pc_mask.astype(jnp.bool)

  # Set all padding point coordinates to 0 to remove any NaNs.
  pointcloud = jnp.where(pc_mask[..., jnp.newaxis], pointcloud, 0.0)
  init_point = jnp.zeros_like(pointcloud[:1, :1])
  # Arrange for the first selected point to be the one farthest from the origin.
  residuals = residuals_fn(pointcloud, init_point)
  squared_norm = residuals.sum(-1)
  if closest_first:
    squared_norm = jnp.where(pc_mask, squared_norm, jnp.inf)
    next_selected_index = jnp.argmin(squared_norm, axis=-1)
  else:
    squared_norm = jnp.where(pc_mask, squared_norm, -jnp.inf)
    next_selected_index = jnp.argmax(squared_norm, axis=-1)

  # As "carry" state we maintain the minimum squared distance to the set of
  # selected points so far.  All padding points are initialized to 0.0, and
  # each selected point will have their corresponding distance set to 0.0 also,
  # so we'll know that all valid points have been selected when
  # max(min_squared_dist) is 0.0.
  min_squared_residuals = jnp.where(
      pc_mask[..., None].repeat(residuals.shape[-1], axis=-1), jnp.inf, 0.0
  )

  if isinstance(rng, int):
    rng = jax.random.PRNGKey(rng)

  carry_init = dict(
      next_selected_index=next_selected_index,
      min_squared_residuals=min_squared_residuals,
      rng=rng,
  )

  def inner_fn(carry: dict[str, jnp.ndarray | None], unused_x):
    next_selected_index = carry['next_selected_index']
    min_squared_residuals = carry['min_squared_residuals']
    rng = carry['rng']

    # Gather point for selected index.
    # Shape: [batch, 1, 3]
    current_point = jnp.take_along_axis(
        pointcloud, next_selected_index[:, jnp.newaxis, jnp.newaxis], axis=1
    )

    # Determine if the selected point is valid or if we've already selected all
    # valid points since there are no remaining distances > 0.0.
    current_mask = (
        jnp.max(jnp.sum(min_squared_residuals, axis=-1), axis=-1) > 0.0
    )

    # Set sentinel values if the current point is padding.
    current_point = jnp.where(
        current_mask[:, jnp.newaxis, jnp.newaxis], current_point, 0.0
    )
    current_index = jnp.where(current_mask, next_selected_index, -1)

    # Find the distance from every point to the currently selected point
    # and updated `min_squared_dist` accordingly.
    squared_residuals_to_curr = residuals_fn(pointcloud, current_point)
    min_squared_residuals = jnp.minimum(
        min_squared_residuals, squared_residuals_to_curr
    )
    min_squared_dist = jnp.sum(min_squared_residuals, axis=-1)

    # Find next point to sample.
    if rng is None:
      next_selected_index = jnp.argmax(min_squared_dist, axis=-1)
    else:
      rngs = jax.random.split(rng, num=pointcloud.shape[0] + 1)
      rngs, rng = rngs[:-1], rngs[-1]
      next_selected_index = jax.vmap(
          lambda r, p: jax.random.choice(r, jnp.arange(p.shape[0]), p=p)
      )(rngs, jnp.sqrt(min_squared_dist))

    return dict(
        next_selected_index=next_selected_index,
        min_squared_residuals=min_squared_residuals,
        rng=rng,
    ), dict(
        selected_points=jnp.squeeze(current_point, axis=1),
        selected_masks=current_mask,
        selected_indices=current_index,
    )

  unused_carry, out = jax.lax.scan(inner_fn, carry_init, length=k)

  selected_points = out['selected_points'].swapaxes(0, 1)  # pylint: disable=attribute-error
  selected_masks = out['selected_masks'].swapaxes(0, 1)  # pylint: disable=attribute-error
  selected_indices = out['selected_indices'].swapaxes(0, 1)  # pylint: disable=attribute-error

  return selected_points, selected_masks, selected_indices


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


def haversine_distance(
    p1_deg: FloatArray['... 2'],
    p2_deg: FloatArray['... 2'],
    numpy: bool = False,
) -> FloatArray['... 1']:
  """Computes the haversine distance between two points (in meters)."""
  xnp = np if numpy else jnp
  d_lat = xnp.radians(p2_deg[..., 0] - p1_deg[..., 0])
  d_lng = xnp.radians(p2_deg[..., 1] - p1_deg[..., 1])
  a = (
      xnp.sin(d_lat / 2) ** 2
      + xnp.cos(xnp.radians(p1_deg[..., 0]))
      * xnp.cos(xnp.radians(p2_deg[..., 0]))
      * xnp.sin(d_lng / 2) ** 2
  )
  c = 2 * xnp.atan2(xnp.sqrt(a), xnp.sqrt(1 - a))
  return c[..., None] * 6371 * 1000


def distance_metrics(
    distance_to_gt: FloatArray['B'], prefix: str = ''
) -> dict[str, Any]:
  """Computes distance metrics."""
  thresholds = [
      50,
      100,
      200,
      500,
      1_000,
      2_000,
      5_000,
      10_000,
      25_000,
      50_000,
      100_000,
      200_000,
      500_000,
      750_000,
      1_000_000,
  ]
  key_to_threshold = {
      f'{prefix}within_{t if t < 1000 else t // 1000}{"m" if t < 1000 else "km"}': (
          t
      )
      for t in thresholds
  }
  metrics = {f'{prefix}distance_to_gt': distance_to_gt}
  for key, threshold in key_to_threshold.items():
    metrics[key] = (distance_to_gt < threshold).astype(jnp.float32)
  return metrics


def distances_to_weights(
    scaled_distances: FloatArray['... N'],
    clip_min: float | None = None,
) -> FloatArray['... N']:
  """Converts distances to inverse weights (large distance -> small weight)."""
  haversine_weights = jnp.exp(-scaled_distances)
  if clip_min:
    haversine_weights = jnp.where(
        haversine_weights > clip_min, haversine_weights, 0.0
    )
  return haversine_weights


def weighted_sigmoid_binary_cross_entropy(
    logits: FloatArray['... N'],
    labels: FloatArray['... N'],
    temperature: float = 1.0,
) -> FloatArray['...']:
  """Computes the weighted sigmoid binary cross entropy."""
  positives = labels * nn.log_sigmoid(logits / temperature)
  negatives = (1 - labels) * nn.log_sigmoid(-logits / temperature)
  loss_positive = -positives.sum(-1) / labels.sum(-1).clip(min=1.0)
  loss_negative = -negatives.sum(-1) / (1 - labels).sum(-1).clip(min=1.0)
  return loss_positive + loss_negative


def tree_mask(
    tree_idx: FloatArray['N H'],
    gt_idxs: FloatArray['B K'],  # -1 is ignore / padding.
) -> FloatArray['B N H']:
  """Computes the tree mask from leaf indices."""
  gt_tree_idxs = jnp.take_along_axis(
      tree_idx[None],
      gt_idxs[..., None],
      axis=-2,
  )  # B x K x H
  gt_tree_idxs = jnp.where(gt_idxs[..., None] > -1, gt_tree_idxs, -1)
  if gt_idxs.shape[-1] == 1:
    gt_tree_mask = tree_idx[None] == gt_tree_idxs
  else:
    gt_tree_mask = jax.vmap(
        jax.vmap(
            lambda x, y: (jnp.isin(x, y)) & (x > -1),
            in_axes=-1,
            out_axes=-1,
        )
    )(
        tree_idx[None].repeat(gt_idxs.shape[0], axis=0), gt_tree_idxs
    )  # B X N x H (vmap over B and H)
  return gt_tree_mask


def weights_from_tree(
    tree_prob: FloatArray['B H'],
) -> FloatArray['B H']:
  """Computes the weights based on previous level probabilities."""
  weights = jnp.roll(tree_prob, 1, axis=-1)
  weights = weights.at[:, 0].set(1.0)
  return weights


def hierarchical_loss(
    cell_probs: FloatArray['B N H'],
    gt_idxs: FloatArray['B K'],
    tree_idx: FloatArray['N H'],
    weighted: bool = False,
) -> FloatArray['B H']:
  """Computes the hierarchical loss."""
  gt_mask_per_level = tree_mask(tree_idx, gt_idxs)
  gt_prob_tree = (gt_mask_per_level * cell_probs[..., None]).sum(-2)
  gt_log_prob_tree = log_safe(gt_prob_tree)
  if weighted:
    weights = weights_from_tree(gt_prob_tree)
    return -(jax.lax.stop_gradient(weights) * gt_log_prob_tree)
  else:
    return -gt_log_prob_tree


def masked_distances_loss(
    distances_th: Sequence[float],
    probs: FloatArray['B N'],
    distances: FloatArray['B N'],
    weighted: bool = False,
) -> FloatArray['B']:
  """Computes the masked distance loss."""
  temperatures = jnp.array(distances_th)[None, :, None]
  probs = probs[:, None]
  tree_weights = distances_to_weights(
      distances[:, None] / temperatures, clip_min=1.0e-3
  )
  haversine_mask = distances[:, None] < temperatures
  haversine_mask_probs = (haversine_mask * probs).sum(-1)
  if weighted:
    tree_prob = (tree_weights * probs).sum(-1)  # B x H
    weights = weights_from_tree(tree_prob)
    return -(jax.lax.stop_gradient(weights) * log_safe(tree_prob)).sum(-1)
  else:
    return -log_safe(haversine_mask_probs).sum(-1)


def crop_aerial_image(
    image: FloatArray['H W 3'],  # (x, y, c)
    resolution_meters_source: float,  # meters / pixel
    height_width: tuple[int, int],  # (height, width)
    dy_dx: FloatArray['2'],  # from center, (dy, dx)
    rotation_angle_degrees: float,  # degrees
    resolution_meters_crop: float,  # meters / pixel
    scale_offset_by_extent: bool = True,
    numpy: bool = False,
    order: int = 1,
    make_upright: bool = True,
) -> tuple[FloatArray['4 4'], FloatArray['H W 3'], FloatArray['H W 1']]:
  """Samples a crop from an aerial image."""
  xnp = np if numpy else jnp
  if scale_offset_by_extent:
    offset_meters = (
        dy_dx * xnp.array(image.shape[:-1]) * resolution_meters_source
    )
  else:
    offset_meters = dy_dx
  center_source = resolution_meters_source * xnp.array(image.shape[:-1]) / 2
  center_crop = resolution_meters_crop * xnp.array(height_width[::-1]) / 2
  grid_crop = grid_indices(*height_width) * resolution_meters_crop
  grid_crop = xnp.flip(grid_crop, -1)
  source_t2d_crop = (
      geometry.Transform2D.from_radians(angle=0, t=center_source)
      @ geometry.Transform2D.from_radians(
          angle=xnp.deg2rad(rotation_angle_degrees), t=offset_meters
      )
      @ geometry.Transform2D.from_radians(angle=0, t=-center_crop)
  )
  grid_transformed = (
      source_t2d_crop.transform(grid_crop)
  ) / resolution_meters_source
  valid = (grid_transformed >= 0.0).all(-1) & (
      grid_transformed < xnp.array(image.shape[:-1])[None, None]
  ).all(-1)
  if not numpy:
    image = byte_to_float(image)
  image_crop = interpolate_image(
      image, grid_transformed + 0.5, order=order, numpy=numpy
  )
  if not numpy:
    image_crop = float_to_byte(image_crop)
  source_t_crop = xnp.eye(4)
  if make_upright:
    image_crop = xnp.swapaxes(image_crop, 0, 1)[::-1]
  if numpy:
    source_t_crop[:2, :2] = source_t2d_crop.R
    source_t_crop[:2, -1] = source_t2d_crop.t
  else:
    source_t_crop = source_t_crop.at[:2, :2].set(source_t2d_crop.R)
    source_t_crop = source_t_crop.at[:2, -1].set(source_t2d_crop.t)
  return source_t_crop, image_crop, valid[..., None]


def unravel_global_dataset_index(
    global_dataset_index: int,
    num_workers: int,
    batch_size: int,
    num_shards: int,
) -> tuple[int, int]:
  """Unravels the global dataset index to a batch and item index."""
  unsharded_index = global_dataset_index // num_shards
  step_index = unsharded_index // (batch_size * num_workers)
  local_index = unsharded_index % (batch_size * num_workers)
  worker_index = unsharded_index % num_workers
  batch_index = step_index * num_workers + worker_index
  item_index = (local_index - worker_index) // num_workers
  return batch_index, item_index


def sharded_similarity(
    batch_logits: FloatArray['B ... 1 D'],  # total K dimensions
    sharded_logits: FloatArray['1 ... N//ND D'],  # total K dimensions
    normalize: bool = False,
) -> FloatArray['B ... N']:
  """Computes the similarity between batch logits and sharded logits."""
  # Gather batch logits across devices.
  all_logits = jax.lax.all_gather(
      batch_logits, axis_name='batch', tiled=True  # Concat in axis 0.
  )  # ND*B x ... x D
  # Compute similarity to sharded cell codes per device.
  similarity = compute_similarity(
      all_logits,  # ND*B x ... x 1 x D
      sharded_logits,  # Add device dim: 1 x ... x N // ND x D
      normalize=normalize,
  )[..., 0]
  # Similarity is ND*B x ... x N // ND

  # All-to-all to get similarity across devices.
  return jax.lax.all_to_all(
      similarity,
      axis_name='batch',
      split_axis=0,  # ND*B --> B
      concat_axis=len(similarity.shape) - 1,  # Does not support -1 index.
      tiled=True,  # This chunks the batch dimension: ND*B --> B.
  )


def sharded_batched_gather(
    x: FloatArray['N//DG D'],
    indices: IntArray['B K'],
    axis_name: str = 'batch',
) -> FloatArray['B K D']:
  """Gathers x[indices]."""
  # Still quite memory intensive --> make cyclic.
  size_per_shard = x.shape[0]
  device_offset = jax.lax.axis_index(axis_name) * size_per_shard
  indices = jax.lax.all_gather(indices, axis_name=axis_name) - device_offset
  # DG x B x K
  valid = (indices >= 0) & (indices < size_per_shard)

  # DG x B x K x D
  x_gathered = jnp.take_along_axis(
      x[None, None], indices[..., None], axis=-2, fill_value=0.0
  )
  x_gathered = jnp.where(valid[..., None], x_gathered, 0.0)
  return jax.lax.psum_scatter(x_gathered, axis_name=axis_name)


def kernel_interpolate(
    x: FloatArray['N 2'],
    y: FloatArray['K 2'],
    z: FloatArray['N D'],
    inv_scale: float,
    normalize=True,
) -> tuple[FloatArray['K'], FloatArray['K D']]:
  """Computes the kernel interpolation of y in x & z."""
  dist = haversine_distance(x[None], y[:, None])
  scaled_dist = distances_to_weights(dist / inv_scale)
  weights = scaled_dist
  if normalize:
    weights = weights / weights.sum(-2, keepdims=True).clip(min=1.0e-6)
  return dist[..., 0].min(-1), (weights * z[None]).sum(-2)


def rbf_interpolate(
    x: FloatArray['N 2'],
    y: FloatArray['K 2'],
    z: FloatArray['N D'],
    inv_scale: float,
) -> tuple[FloatArray['K'], FloatArray['K D']]:
  """Computes the Radial Basis Function interpolation between of y in x & z."""
  xx_dist = haversine_distance(x[None], x[:, None])[..., 0]
  xx_scaled_dist = distances_to_weights(xx_dist / inv_scale)
  weights = jax.vmap(jnp.linalg.solve, in_axes=(None, -1), out_axes=-1)(
      xx_scaled_dist, z
  )
  yx_dist = haversine_distance(y[:, None], x[None])[..., 0]
  yx_scaled_dist = distances_to_weights(yx_dist / inv_scale)
  min_dist = yx_dist.min(-1)
  return min_dist, yx_scaled_dist @ weights


def unshard_or_unreplicate_from_shape_fn(
    target_shape: tuple[int, ...] | Sequence[tuple[int, ...]],
    half_precision: bool = False,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Unshards a tensor if it has the target shape, otherwise unreplicates."""
  if half_precision:
    logging.info('Unshard or unreplicate with half precision.')
  if isinstance(target_shape[0], int):
    target_shape = [target_shape]
  def unshard_or_unreplicate(x):
    if x.shape in target_shape:
      # Carefully gather sharded cell codes to CPU.
      x = gather_globally_safe(x)
      if half_precision:
        x = x.astype(np.float16)
      return x
    else:
      # All other parameters are just transferred to CPU.
      return jax.device_get(x[0])

  return unshard_or_unreplicate


def shard_or_replicate_from_shape_fn(
    target_shape: tuple[int, ...] | Sequence[tuple[int, ...]],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Shards a tensor if it has the target shape, otherwise replicates."""
  if isinstance(target_shape[0], int):
    target_shape = [target_shape]
  def shard_or_replicate(x):
    if x.shape in target_shape:
      # Cell codes should be sharded over all devices.
      if x.dtype == np.float16:
        x = x.astype(np.float32)
      return distribute_globally(x)
    else:
      # All other parameters should be replicated.
      return jax_utils.replicate(x)

  return shard_or_replicate


def pmean_or_divide_from_shape_fn(
    target_shape: tuple[int, ...] | Sequence[tuple[int, ...]],
    axis_name: str = 'batch',
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Computes the pmean or divide."""
  if isinstance(target_shape[0], int):
    target_shape = [target_shape]

  def pmean_or_divide(x):
    if x.shape in target_shape:
      # pmean divides by the number of devices, so we also do this here.
      return x / jax.device_count()
    else:
      # All other gradients are pmean'ed.
      return jax.lax.pmean(x, axis_name=axis_name)

  return pmean_or_divide


def unshard_or_unreplicate_train_state(
    train_state: train_utils.TrainState,
    unshard_or_unreplicate_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> train_utils.TrainState:
  """Unshard or unreplicate a train state my mapping a function over params and opt_state."""
  logging.info('Starting unreplicate_state....')
  start = time.time()
  # pytype: disable=attribute-error
  train_state = train_state.replace(
      params=jax.tree.map(unshard_or_unreplicate_fn, train_state.params),
      opt_state=jax.tree.map(unshard_or_unreplicate_fn, train_state.opt_state),
      global_step=jax.device_get(
          jax_utils.unreplicate(train_state.global_step)
      ),
      rng=jax.device_get(jax_utils.unreplicate(train_state.rng)),
      model_state=jax.tree.map(
          unshard_or_unreplicate_fn, train_state.model_state
      ),
      metadata=jax.device_get(jax_utils.unreplicate(train_state.metadata)),
  )
  # pytype: enable=attribute-error
  logging.info('unreplicate_state took %f seconds.', time.time() - start)
  return train_state


def shard_or_replicate_train_state(
    train_state: train_utils.TrainState,
    shard_or_replicate_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> train_utils.TrainState:
  """Shard or replicate a train state my mapping a function over params and opt_state."""
  logging.info('Starting replicate_state....')
  start = time.time()
  # pytype: disable=attribute-error
  train_state = train_state.replace(
      params=jax.tree.map(shard_or_replicate_fn, train_state.params),
      opt_state=jax.tree.map(shard_or_replicate_fn, train_state.opt_state),
      global_step=jax_utils.replicate(train_state.global_step),
      rng=jax_utils.replicate(train_state.rng),
      model_state=jax.tree.map(shard_or_replicate_fn, train_state.model_state),
      metadata=jax_utils.replicate(train_state.metadata),
  )
  logging.info('replicate_state took %f seconds.', time.time() - start)
  return train_state
  # pytype: enable=attribute-error#


def filter_hard_negatives(
    similarity: FloatArray['B N'],
    keep: IntArray['B K'],
    num_negatives: int | None = None,
    where: BoolArray['B N'] | None = None,  # Only for negatives.
) -> tuple[FloatArray['B K'], FloatArray['B k']]:
  """Extracts hard negatives excluding a set of 'keep' indices."""
  if num_negatives is None:
    num_negatives = keep.shape[-1]
  similarity_keep = jnp.take_along_axis(similarity, keep, axis=-1)  # B x K
  neg_similarity = jnp.put_along_axis(
      similarity,
      keep,
      -jnp.ones_like(keep) * jnp.inf,
      axis=-1,
      inplace=False,
  )
  if where is not None:
    neg_similarity = jnp.where(where, neg_similarity, -jnp.inf)
  _, topk_neg = jax.lax.approx_max_k(neg_similarity, k=num_negatives)
  return similarity_keep, jnp.take_along_axis(similarity, topk_neg, axis=-1)


def combine_similarities(
    similarities: Sequence[FloatArray['B Ni']],
    gt_idxs: Sequence[IntArray['B Ki']],
    valid: Sequence[IntArray['B Ni']] | None = None,
    axis: int = -1,
) -> (
    tuple[FloatArray['B N'], IntArray['B K']]
    | tuple[FloatArray['B N'], IntArray['B K'], BoolArray['B N']]
):
  """Combines similarities, and apply offsets to gt_idxs."""
  lengths = jnp.array(
      [0] + [x.shape[axis] for x in similarities[:-1]], dtype=jnp.int32
  )
  offsets = jnp.cumsum(lengths)
  gt_idxs = jnp.concatenate(
      [
          jnp.where(gt_idx > -1, gt_idx + offsets[i], -1)
          for i, gt_idx in enumerate(gt_idxs)
      ],
      axis=axis,
  )
  similarity = jnp.concatenate(similarities, axis=axis)
  if valid is None:
    return similarity, gt_idxs
  else:
    return similarity, gt_idxs, jnp.concatenate(valid, axis=axis)


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


def multihead_similarity(
    source_f: FloatArray['... H*D+NO'],
    target_f: FloatArray['... H*D+NO'],
    num_heads: int,
    offset: int = 0,
    normalize: bool = False,
) -> FloatArray['B N H|H+1 1']:
  """Computes multi-head similarity."""

  def unflatten(x):
    return x.reshape(x.shape[:-1] + (num_heads, x.shape[-1] // num_heads))

  sim = compute_similarity(
      unflatten(source_f[..., offset:]),
      unflatten(target_f[..., offset:]),
      normalize=normalize,
  )
  if offset != 0:
    sim = jnp.concatenate(
        [
            compute_similarity(
                source_f[..., None, :offset],
                target_f[..., None, :offset],
                normalize=normalize,
            ),
            sim,
        ],
        axis=-2,
    )
  return sim


def weighted_softmax(
    x: FloatArray['...'], weights: FloatArray['...'], axis: int
):
  exps = jnp.exp(x - x.max(axis=axis, keepdims=True))
  return exps * weights / (exps * weights).sum(axis, keepdims=True)


def heterogeneous_similarity(
    multihead_sim: FloatArray['B N H'],
    weights: FloatArray['B N'] | None = None,
    temperature: float = 0.1,
    smoothing: float = 1.0,
) -> tuple[FloatArray['B H'], FloatArray['B N H']]:
  """Combines the multi-head similarity into a single similarity via smax."""
  if weights is None:
    weights = jnp.ones_like(multihead_sim[..., 0], dtype=jnp.float32)
  weights = weights / weights.sum(axis=-1, keepdims=True).clip(min=1.0e-8)
  probs = weighted_softmax(
      multihead_sim / temperature, weights[..., None], axis=-2
  )
  probs_scaled = (probs + smoothing * weights[..., None]) / (1 + smoothing)
  sim_per_head = (probs_scaled * multihead_sim).sum(-2)
  return sim_per_head, probs_scaled


def compute_heterogeneous_similarity(
    source_f: FloatArray['B D'],
    target_f: FloatArray['B N D'],
    weights: FloatArray['B N'],
    temperature: float = 0.1,
    smoothing: float = 1.0,  # Can also be negative to sparsify results!
    num_heads_and_offset: tuple[int, int] | str = (1, 0),
    normalize: bool = False,
) -> tuple[FloatArray['B H'], FloatArray['B N H'], FloatArray['B N H']]:
  """Computes the heterogeneous similarity from features."""
  if isinstance(num_heads_and_offset, str):
    num_heads_and_offset = num_heads_and_offset_vit(
        source_f.shape[-1], num_heads_and_offset
    )
  mh_pos_sim = multihead_similarity(
      source_f[:, None],
      target_f,
      num_heads=num_heads_and_offset[0],
      offset=num_heads_and_offset[1],
      normalize=normalize,
  )[..., 0]
  mh_het_sim, mh_probs = heterogeneous_similarity(
      mh_pos_sim,
      weights=weights,
      temperature=temperature,
      smoothing=smoothing,
  )
  het_sim = mh_het_sim.mean(-1, keepdims=True)
  return het_sim, mh_pos_sim, mh_probs


def ecef_to_latlngalt(
    ecef_coords: FloatArray['... 3'],
    max_iter: int = 2,
    tolerance: float | None = None,
) -> FloatArray['... 3']:
  """ECEF to LatLngAlt in Jax.

  Converts Earth-Centered, Earth-Fixed (ECEF) coordinates (x, y, z) in meters
  to Latitude (degrees), Longitude (degrees), and Altitude (meters) using
  Bowring's iterative method in JAX for improved latitude accuracy.

  Args:
      ecef_coords: A JAX NumPy array of shape (3,) or (n, 3) representing ECEF
        coordinates [x, y, z].
      max_iter: Maximum number of iterations for the latitude and altitude
        refinement.
      tolerance: Tolerance for the convergence of latitude and altitude.

  Returns:
      A JAX NumPy array of shape (3,) or (n, 3) representing
      [latitude (degrees), longitude (degrees), altitude (meters)].
  """
  x = ecef_coords[..., 0]
  y = ecef_coords[..., 1]
  z = ecef_coords[..., 2]

  # WGS 84 ellipsoid parameters
  a = 6378137.0  # Semi-major axis (meters)
  b = 6356752.314245179  # Semi-minor axis (meters)
  e_sq = 1 - (b / a) ** 2  # First eccentricity squared

  p = jnp.sqrt(x**2 + y**2)
  lon_rad = jnp.arctan2(y, x)

  # Initial guess for latitude (geocentric latitude)
  lat_rad = jnp.arctan2(z, p * jnp.sqrt(1 - e_sq))
  alt = jnp.zeros_like(lat_rad)

  for _ in range(max_iter):
    sin_lat = jnp.sin(lat_rad)
    # Radius of curvature in the prime vertical
    n = a / jnp.sqrt(1 - e_sq * sin_lat**2)
    h_prev = alt
    alt = p / jnp.cos(lat_rad) - n
    lat_rad_prev = lat_rad
    # Bowring's formula
    lat_rad = jnp.arctan2(z / (p * (1 - e_sq * n / (n + alt))), 1)
    if tolerance is not None:
      if jnp.all(jnp.abs(lat_rad - lat_rad_prev) < tolerance) and jnp.all(
          jnp.abs(alt - h_prev) < tolerance
      ):
        break

  lat_deg = jnp.degrees(lat_rad)
  lon_deg = jnp.degrees(lon_rad)

  return jnp.stack([lat_deg, lon_deg, alt], axis=-1)


def transform_points(
    y_t_x: FloatArray['4 4'],
    x_points: FloatArray['N 3'],
) -> FloatArray['N 3']:
  """3D transform points from coordinate system x to y."""
  return x_points @ y_t_x[:3, :3].T + y_t_x[:3, -1]


def aerial_pixel_coords(
    rgb: FloatArray['H W 3'],
    resolution_meters: float | jnp.ndarray,
    w_t_grid: FloatArray['4 4'],
    upright: bool | jnp.ndarray = True,
) -> FloatArray['H W 3']:
  """Returns the world coordinates of each pixel in the aerial image."""
  grid_ij = grid_indices(*rgb.shape[-3:-1]).astype(jnp.float32)
  grid_xy = jnp.flip(grid_ij + 0.5, axis=-1) * resolution_meters
  grid_xyz = jnp.concatenate(
      [grid_xy, jnp.zeros_like(grid_xy[..., :1])], axis=-1
  )
  w_xyz = jax.vmap(transform_points, in_axes=(None, 0))(w_t_grid, grid_xyz)
  w_xyz = jnp.where(upright, jnp.swapaxes(w_xyz, -3, -2)[::-1], w_xyz)
  return w_xyz


def dropout_token(
    rng: jnp.ndarray,
    x: FloatArray['... D'],
    rate: float,
    fill_value: float = 0.0,
) -> FloatArray['... D']:
  """Drops out a fraction of the input."""
  return jnp.where(
      jax.random.uniform(rng, x.shape[:-1])[..., None] < rate,
      x,
      fill_value,
  )


def softclip(
    x: FloatArray['...'],
    min_value: float | None = None,
    max_value: float | None = None,
) -> FloatArray['...']:
  """Softclip the input."""
  if min_value is not None:
    x = jnp.where(
        x < min_value,
        x + (min_value - jax.lax.stop_gradient(x)),
        x,
    )
  if max_value is not None:
    x = jnp.where(
        x > max_value,
        x - (jax.lax.stop_gradient(x) - max_value),
        x,
    )
  return x
