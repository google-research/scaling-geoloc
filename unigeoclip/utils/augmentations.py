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

"""Image augmentation utils add to lama ones."""

import functools

from etils.array_types import FloatArray
import jax
from jax import random

from scaling_geoloc.utils import augmentations

maybe_apply = augmentations.maybe_apply
per_image = augmentations.per_image
sequential = augmentations.sequential
random_gaussian_blur = augmentations.random_gaussian_blur
random_solarize = augmentations.random_solarize
get_crop_fn = augmentations.get_crop_fn
AugmentationFn = augmentations.AugmentationFn

# Define what is part of the public API
__all__ = [
    "maybe_apply",
    "per_image",
    "sequential",
    "random_gaussian_blur",
    "random_solarize",
    "get_crop_fn",
    "AugmentationFn",
    "random_90_rotation",
]


def random_90_rotation(
    probability: float,
) -> AugmentationFn:
  """Create a function for random 90-degree rotation."""

  def _rotate(x: FloatArray["H W 3"], rng: jax.Array) -> FloatArray["H W 3"]:
    num_rotations = random.randint(rng, shape=(), minval=0, maxval=4)

    rot_fns = [functools.partial(jax.numpy.rot90, k=i) for i in range(4)]
    # Use jax.lax.switch to apply the rotation based on num_rotations
    return jax.lax.switch(num_rotations, rot_fns, x)

  return maybe_apply(_rotate, probability)
