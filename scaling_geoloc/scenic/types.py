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

"""Types for the training pipeline."""

from collections.abc import Sequence
import dataclasses
import enum
from typing import Any, Callable, Mapping

from etils.array_types import Array
from etils.array_types import FloatArray
import grain.python as grain
import jax
import jax.numpy as jnp
import jaxcam
from scenic.dataset_lib import dataset_utils
from scenic.train_lib import train_utils

from scaling_geoloc.data import types as data_types


DataDict = dict[str, Any]
Batch = dict[str, Any]
Predictions = dict[str, Any]
LossDict = MetricsDict = dict[str, FloatArray['batch']]
LossMetricsTuple = tuple[LossDict, MetricsDict]
ModelParams = jnp.ndarray
ModelState = Any
LearningRates = Mapping[str, float | jnp.ndarray]
Grad = jnp.ndarray
LossMetricsFn = Callable[..., LossMetricsTuple]
UpdateFn = Callable[
    [Predictions, ModelParams, ModelState, LearningRates],
    tuple[ModelParams, ModelState],
]
AggregateGradsFn = Callable[[Grad], Grad]
ScaleGradFn = Callable[[Grad, LearningRates], Grad]
LrFn = Callable[[jnp.ndarray], jnp.ndarray]
PyTree = train_utils.PyTree
AggregatedMetricsDict = dict[str, tuple[float, int]]
VisualizationDict = Mapping[str, Array['H W 3'] | Array['B H W 3']]
DataDict = dict[str, Any]
EvalResults = tuple[dict[str, Any], dict[str, Any], VisualizationDict]

SEMANTIC_INDEX_TO_NAME = data_types.SEMANTIC_INDEX_TO_NAME
SEMANTIC_NAME_TO_INDEX = {v: k for k, v in SEMANTIC_INDEX_TO_NAME.items()}


@dataclasses.dataclass(frozen=True)
class Dataset:
  train_ds: grain.RandomAccessDataSource | None = None
  valid_ds: grain.RandomAccessDataSource | None = None
  test_ds: grain.RandomAccessDataSource | None = None

  train_iter: dataset_utils.DatasetIteratorType | None = None
  valid_iter: dataset_utils.DatasetIteratorType | None = None
  test_iter: dataset_utils.DatasetIteratorType | None = None

  meta_data: DataDict = dataclasses.field(default_factory=dict)


class MaskType(enum.Enum):
  IGNORE = -1
  INVALID = 0
  VALID = 1


def weighted_loss(
    losses: LossDict, loss_weights: Sequence[tuple[str | float, str | float]]
) -> FloatArray['batch']:
  """Extracts losses by name, applies weights and sums them."""
  total_loss = 0.0
  for v1, v2 in loss_weights:
    v1 = v1 if not isinstance(v1, str) else losses[v1]
    v2 = v2 if not isinstance(v2, str) else losses[v2]
    total_loss += v1 * v2
  return total_loss


def compose_loss(losses: LossDict, compose_str: str) -> FloatArray['batch']:
  """Combines losses from loss dict by a string.

  Example:
  mse + 0.1 * regularization = losses["mse"] + 0.1 * losses["regularization"]

  Args:
    losses: Dictionary of losses.
    compose_str: The string for how to combine the losses.

  Returns:
    Composed loss.
  """

  def isfloat(s: str) -> bool:
    try:
      float(s)
      return True
    except ValueError:
      return False

  loss_elements = compose_str.replace(' ', '').split('+')
  losses_and_weights = []
  for loss in loss_elements:
    loss_terms = loss.split('*')
    if len(loss_terms) > 2 or not loss_terms:
      print(loss_terms)
      raise ValueError(f'Invalid loss term: {loss}')
    elif len(loss_terms) == 1:
      loss_and_weight = (loss_terms[0], 1.0)
    else:
      loss_and_weight = (v if not isfloat(v) else float(v) for v in loss_terms)
    losses_and_weights.append(tuple(loss_and_weight))
  return weighted_loss(losses, losses_and_weights)


def to_cpu(batch: Any) -> Any:
  """Transfers array (replicated on multiple hosts) to a single host.

  Args:
    batch: Replicated array of shape [num_hosts, num_devices, local_batch_size,
      ...].

  Returns:
    array of shape [global_batch_size, ...] where
      global_batch_size = num_devices * local_batch_size
  """
  return dataset_utils.unshard(jax.device_get(batch))


def create_jaxcam(
    cam_t_w: FloatArray['... 4 4'],
    intrinsics: DataDict | jnp.ndarray,
    is_fisheye: bool = True,
    use_inverted_distortion: bool = False,
) -> jaxcam.Camera:
  """Create a jaxcam.Camera from a 4x4 pose matrix and intrinsics."""
  cam_r_w = cam_t_w[..., :3, :3]
  w_p_cam = jnp.linalg.inv(cam_t_w)[..., :3, -1]
  cam = jaxcam.Camera.create(
      orientation=cam_r_w,
      position=w_p_cam,
      focal_length=intrinsics['K'][..., 0, 0],
      principal_point=intrinsics['K'][..., :2, -1],
      image_size=jnp.stack(
          [intrinsics['width'], intrinsics['height']], axis=-1
      ),
      radial_distortion=None,
      tangential_distortion=(
          None
          if use_inverted_distortion
          else intrinsics['tangential_distortion']
      ),
      skew=jnp.zeros_like(intrinsics['K'][..., 0, 1]),
      pixel_aspect_ratio=intrinsics['K'][..., 1, 1]
      / intrinsics['K'][..., 0, 0],
      is_fisheye=is_fisheye,
  )
  radial_distortion = jnp.concatenate(
      [
          intrinsics['radial_distortion'],
          jnp.zeros_like(intrinsics['radial_distortion'][..., -1:]),
      ],
      axis=-1,
  )
  if use_inverted_distortion:
    radial_distortion = jaxcam.invert_radial_distortion_coefficients(
        radial_distortion
    )

  # Manually set the radial distortion to avoid hidden padding.
  return dataclasses.replace(
      cam,
      radial_distortion=radial_distortion,
      use_inverted_distortion=use_inverted_distortion,
  )
