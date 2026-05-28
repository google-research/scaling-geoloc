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
from typing import Any, Callable
import dm_pix
from etils.array_types import Array
from etils.array_types import BoolArray
from etils.array_types import FloatArray
from etils.array_types import IntArray
import flax
import jax
import jax.numpy as jnp
import jaxcam
import numpy as np
import scipy
from sklearn import decomposition

from scaling_geoloc.utils import geometry


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
  image = batch['query']['image']
  if stride is not None:
    image = crop_to_multiple(image, stride) if image.ndim == 3 else image
  return batch | {'image': image}


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
    image = data[scene]['image']
    batch_dims = image.shape[:-3]
    h, w = image.shape[-3:-1]
    hr, wr = int(h * resize_factor), int(w * resize_factor)

    for k, v in data[scene].items():
      if not isinstance(v, (np.ndarray, jnp.ndarray)):
        continue
      if v.shape[:-1] == image.shape[:-1]:
        data[scene][k] = jax.image.resize(
            v,
            shape=batch_dims + (hr, wr, v.shape[-1]),
            method=('bilinear' if k == 'image' else 'nearest'),
            antialias=antialias,
        )
  return data


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
