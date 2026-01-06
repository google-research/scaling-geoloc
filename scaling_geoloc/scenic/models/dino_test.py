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

from collections.abc import Mapping
from typing import Any, Final

import jax.numpy as jnp
import ml_collections
import numpy as np

from scaling_geoloc.scenic.configs import dino_base
from scaling_geoloc.scenic.models import dino
from scaling_geoloc.scenic.models import vit
from google3.pyglib.contrib.g3_multiprocessing import googletest_launcher
from google3.testing.pybase import googletest

RTOL: Final[float] = 1e-6
ATOL: Final[float] = 1e-6


def generate_dummy_state(
    b: int, n: int, h: int, w: int, ng: int, nl: int, l: int, d: int, head: int
) -> Mapping[str, Any]:
  predictions = {
      'student_cls_logits': np.random.rand(b, n, ng, head),
      'student_local_cls_logits': np.random.rand(b, n, nl, head),
      'student_patch_logits': np.random.rand(b, n, ng, l, head),
      'student_tokens': np.random.rand(b, n, ng, 1 + l, d),
      'teacher_cls_logits': np.random.rand(b, n, ng, head),
      'teacher_patch_logits': np.random.rand(b, n, ng, l, head),
      'teacher_tokens': np.random.rand(b, n, ng, 1 + l, d),
      'student_views': np.random.rand(b, n, ng, h, w, d),
  }

  learning_rates = {
      'student_temperature': 1.0,
      'teacher_temperature': 1.0,
  }

  center = jnp.zeros(head)
  model_state = {
      'model_stats': {
          'center': center,
          'patch_center': jnp.zeros(head),
      }
  }

  return {
      'predictions': predictions,
      'learning_rates': learning_rates,
      'model_state': model_state,
      'batch': {},
  }


def compute_baseline_loss(
    dummy_state: Mapping[str, Any],
    config: ml_collections.ConfigDict,
) -> float:

  predictions = dummy_state['predictions']
  b, n, ng, head = predictions['student_cls_logits'].shape
  nl = predictions['student_local_cls_logits'].shape[-2]
  learning_rates = dummy_state['learning_rates']
  center = dummy_state['model_state']['model_stats']['center']

  # Reshape to (b * n * ng, head)
  student_cls_logits_flat = (
      np.concatenate(
          [
              predictions['student_cls_logits'],
              predictions['student_local_cls_logits'],
          ],
          axis=2,
      )
      .swapaxes(0, 2)
      .reshape(b * n * (ng + nl), head)
  )

  teacher_cls_logits_flat = (
      predictions['teacher_cls_logits'].swapaxes(0, 2).reshape(b * n * ng, head)
  )

  baseline_config = config.copy_and_resolve_references().unlock()
  baseline_config.model = baseline_config.model.vit

  baseline_config.dataset_configs = ml_collections.ConfigDict()
  baseline_config.dataset_configs.num_local_views = nl
  baseline_model = vit.ViTDINOModel(baseline_config, {})
  return baseline_model.loss_function(
      student_cls_logits_flat / learning_rates['student_temperature'],
      (teacher_cls_logits_flat - center)
      / learning_rates['teacher_temperature'],
  )


class DINOTest(googletest.TestCase):

  def test_dino_loss_identical_single_image(self):
    b, n, d = 1, 1, 32
    nh, nw = 3, 4
    l = nh * nw
    head = 64

    ng = 2  # Number of global views.
    nl = 0  # Number of local views.

    config = dino_base.get_config()
    config.model.compose_loss = 'cls_all'
    model = dino.ViTDINOModel(config.model, {})
    ph, pw = config.model.vit.patches.size
    h, w = nh * ph, nw * pw

    dummy_state = generate_dummy_state(b, n, h, w, ng, nl, l, d, head)

    loss, _ = model.loss_metrics_function(
        dummy_state['predictions'],
        dummy_state['batch'],
        model_state=dummy_state['model_state'],
        learning_rates=dummy_state['learning_rates'],
    )

    baseline_loss = compute_baseline_loss(dummy_state, config)

    np.testing.assert_allclose(
        loss['total'].mean(), baseline_loss, rtol=RTOL, atol=ATOL
    )

  def test_dino_loss_identical_batched(self):
    b, n, d = 2, 1, 32
    nh, nw = 3, 4
    l = nh * nw
    head = 64

    ng = 2  # Number of global views.
    nl = 0  # Number of local views.

    config = dino_base.get_config()
    config.model.compose_loss = 'cls_all'
    model = dino.ViTDINOModel(config.model, {})
    ph, pw = config.model.vit.patches.size
    h, w = nh * ph, nw * pw

    dummy_state = generate_dummy_state(b, n, h, w, ng, nl, l, d, head)

    loss, _ = model.loss_metrics_function(
        dummy_state['predictions'],
        dummy_state['batch'],
        model_state=dummy_state['model_state'],
        learning_rates=dummy_state['learning_rates'],
    )

    baseline_loss = compute_baseline_loss(dummy_state, config)

    np.testing.assert_allclose(
        loss['total'].mean(), baseline_loss, rtol=RTOL, atol=ATOL
    )

  def test_dino_multicrop_loss_identical_batched(self):
    b, n, d = 2, 1, 32
    nh, nw = 3, 4
    l = nh * nw
    head = 64

    ng = 2  # Number of global views.
    nl = 6  # Number of local views.

    config = dino_base.get_config()
    config.model.compose_loss = 'cls_all'
    model = dino.ViTDINOModel(config.model, {})
    ph, pw = config.model.vit.patches.size
    h, w = nh * ph, nw * pw

    dummy_state = generate_dummy_state(b, n, h, w, ng, nl, l, d, head)

    loss, _ = model.loss_metrics_function(
        dummy_state['predictions'],
        dummy_state['batch'],
        model_state=dummy_state['model_state'],
        learning_rates=dummy_state['learning_rates'],
    )

    baseline_loss = compute_baseline_loss(dummy_state, config)

    np.testing.assert_allclose(
        loss['total'].mean(), baseline_loss, rtol=RTOL, atol=ATOL
    )


if __name__ == '__main__':
  googletest_launcher.main()
