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

"""Code for normalization layers."""

import flax.linen as nn
import jax.numpy as jnp


class DynamicTanhNorm(nn.Module):
  """DyT normalization. Implementation of https://arxiv.org/abs/2503.10622 norm.

  DyT(x) = tanh(alpha * x) * gamma + beta

  Attributes:
    alpha_init_value: Initial value of the alpha parameter.
    dtype: The dtype of the computation (default: float32).
  """

  alpha_init_value: float = 0.5
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x):
    """Applies DyT normalization to the input.

    Args:
      x: Input tensor of shape [..., features].

    Returns:
      Normalized tensor.
    """
    num_features = x.shape[-1]
    alpha = self.param(
        'alpha',
        nn.initializers.constant(self.alpha_init_value),
        (),
        self.dtype,
    )
    beta = self.param(
        'beta', nn.initializers.zeros, (num_features,), self.dtype
    )
    gamma = self.param(
        'gamma', nn.initializers.ones, (num_features,), self.dtype
    )
    x = jnp.tanh(alpha * x)
    x = x * gamma + beta
    return x
