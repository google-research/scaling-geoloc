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

"""Dino with 3D contrastive modelling."""

from collections.abc import Mapping
import functools
from typing import Any
from etils.array_types import FloatArray
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxcam
import jaxtyping as jt
import numpy as np
from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.models import dino
from scaling_geoloc.scenic.models import vit
from scaling_geoloc.utils import gaussians
from scaling_geoloc.utils import misc
from scaling_geoloc.utils import plot


class ViTDINO3D(dino.ViTDINO):
  """DINO with 3D contrastive modelling."""

  def encode_multiview(
      self,
      images: FloatArray['... H W C'],
      *args,
      train: bool = False,
      **kwargs,
  ) -> tuple[jt.Float[jt.Array, '... D'], jt.Float[jt.Array, '... L D']]:
    """Interface to backbone encoder."""
    logits, bb_output = self.student.encode_multiview(
        images, *args, train=train, **kwargs
    )
    if self.config.decode:
      bb_output = self.decode(bb_output, train=train)
    return logits, bb_output

  def encode_image(
      self, image: jt.Float[jt.Array, 'B H W C'], *, train: bool
  ) -> tuple[jt.Float[jt.Array, 'B h w c'], jt.Float[jt.Array, 'B c']]:
    """Interface to FOMO evals."""
    _, bb_output = self.encode_multiview(image, train=train)
    cls_tok = bb_output[..., 0, :]
    h, w = image.shape[-3:-1]
    ph, pw = self.config.vit.patches.size
    nh, nw = h // ph, w // pw
    patch_tok = misc.sequence_to_image(bb_output[..., 1:, :], nh, nw)
    return patch_tok, cls_tok

  @nn.compact
  def decode(
      self,
      tokens: FloatArray['... H W C'],
      train: bool = False,
  ) -> tuple[jt.Float[jt.Array, '... D'], jt.Float[jt.Array, '... L D']]:
    """Interface to backbone encoder."""
    decoder_config = self.config.decoder
    mv_encoder = vit.MultiViewEncoder(
        mlp_dim=decoder_config.mlp_dim,
        num_layers=decoder_config.num_layers,
        num_heads=decoder_config.num_heads,
        dropout_rate=decoder_config.dropout_rate,
        attention_dropout_rate=decoder_config.attention_dropout_rate,
        stochastic_droplayer_rate=decoder_config.stochastic_droplayer_rate,
        dtype=self.dtype,
    )
    student_tokens = mv_encoder(tokens, train=train)
    return student_tokens

  @nn.compact
  def __call__(
      self,
      batch: types.Batch,
      train: bool = False,
      debug: bool = False,
  ) -> types.Predictions:
    ph, _ = self.config.vit.patches.size
    batch = misc.filter_batch_for_jit(batch)
    batch = jax.tree.map(
        lambda x, y: jnp.concatenate([x, y], axis=1),
        batch['map']['views'],
        batch['query']['views'],
    )
    map_images = misc.crop_to_multiple(batch['image_pixels'], ph)

    if self.config.get('do_augment'):
      _, rng_s, rng_t = jax.random.split(self.make_rng('augment'), 3)
      student_images = dino.augment_global0(map_images, rng_s)
      teacher_images = dino.augment_global1(map_images, rng_t)
    else:
      student_images = map_images
      teacher_images = map_images

    mask_points = batch['lidar_mask']
    if 'w_lidar_points' in batch:
      w_points = batch['w_lidar_points']
    else:
      cameras = types.create_jaxcam(
          batch['cam_t_w'],
          batch['camera'],
          is_fisheye=True,
          use_inverted_distortion=True,
      )
      depth = batch['lidar_depth'].astype(map_images.dtype)
      h, w = depth.shape[-3:-1]
      pixels = misc.grid_indices(h, w, depth.shape[:-3]) + 0.5
      # Batch first two dimensions B x N
      w_points = jax.vmap(jax.vmap(jaxcam.pixels_to_points))(
          cameras, pixels, depth[..., 0]
      )
      w_points = w_points[..., :3] / w_points[..., -1:]
    w_points = misc.crop_to_multiple(w_points, ph)
    mask_points = misc.crop_to_multiple(mask_points, ph)

    map_gaussians = gaussians.gaussians_from_points(
        misc.patchify(w_points, *self.config.vit.patches.size),
        misc.patchify(mask_points, *self.config.vit.patches.size),
    )

    if self.config.mask_ratio > 0:
      student_logits, student_tokens, is_masked = self.student.encode_masked(
          student_images,
          self.config.mask_ratio,
          train=train,
          debug=debug,
      )
    else:
      student_logits, student_tokens = self.student.encode_multiview(
          student_images, train=train, debug=debug
      )
      is_masked = jnp.zeros_like(student_tokens[..., 1:, :1])

    if self.config.decode:
      student_tokens = self.decode(student_tokens, train=train)

    student_patch_tokens = student_tokens[..., 1:, :]
    student_patch_tokens = jax.lax.collapse(
        student_patch_tokens, 1, -1
    )  # b x nl x cin

    student_cloud = student_patch_tokens

    teacher_logits, teacher_tokens = jax.lax.stop_gradient(
        self.teacher.encode_multiview(teacher_images, train=False, debug=debug)
    )
    teacher_patch_tokens = teacher_tokens[..., 1:, :]
    student_cloud = student_cloud.reshape(*teacher_patch_tokens.shape)
    student_patch_tokens = student_patch_tokens.reshape(
        *teacher_patch_tokens.shape
    )

    pred = {
        'teacher_cls_logits': teacher_logits,
        'student_cls_logits': student_logits,
        'teacher_patch_tokens': teacher_patch_tokens,
        'student_patch_tokens': student_patch_tokens,
        'is_masked': is_masked,
        'student_cloud': student_cloud,
        'gaussians': map_gaussians,
        'student_views': student_images,
        'teacher_views': teacher_images,
        'views': map_images,
    }

    if self.config.vit.pooling == 'none':
      pred = {
          k: v[..., 0, :] if 'cls_logits' in k else v for k, v in pred.items()
      }
      pred['student_patch_logits'] = student_logits[..., 1:, :]
      pred['teacher_patch_logits'] = teacher_logits[..., 1:, :]

    if not train:
      del pred['teacher_patch_logits']
      del pred['student_patch_logits']
    return pred


class ViTDINO3DModel(dino.ViTDINOModel):
  """Trainer facing wrapper for ViTDINO."""

  required_rngs: tuple[str, ...] = (
      'droptok',
      'dropout',
      'masking_rng',
      'augment',
      'negative_sampling',
  )

  def build_flax_model(self) -> nn.Module:
    return ViTDINO3D(
        config=self.config,
        dtype=self.dtype,
    )

  def loss_metrics_function(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      model_params: jnp.ndarray | None = None,
      model_state: Any | None = None,
      rate_params: Mapping[str, float] | None = None,
  ) -> types.LossMetricsTuple:
    # assert rate_params is not None
    teacher_cls_logits = pred['teacher_cls_logits']
    teacher_cls_logits -= model_state['model_stats']['center']
    teacher_cls_logits /= rate_params['teacher_temperature']
    student_cls_logits = (
        pred['student_cls_logits'] / rate_params['student_temperature']
    )

    losses, metrics = {}, {}
    # b x m x d, b x m x d -> b x m x m x 1
    cls_global = misc.cross_entropy(
        jnp.expand_dims(student_cls_logits, -2),
        jnp.expand_dims(teacher_cls_logits, -3),
    )

    cls_weights = 1.0 - jnp.eye(cls_global.shape[-2])[None, ..., None]
    losses['cls_global'] = misc.masked_mean(
        cls_global, cls_weights, axis=(-1, -2, -3)
    )

    source_tokens = pred['student_cloud']
    # B x V X S x D
    source_tokens_l2 = misc.l2_normalize(source_tokens)
    target_tokens = pred['teacher_patch_tokens']
    target_tokens_l2 = misc.l2_normalize(target_tokens)

    alignments = {}
    alignments['similarity'] = (source_tokens * target_tokens).sum(
        -1, keepdims=True
    )
    alignments['similarity_l2'] = (source_tokens_l2 * target_tokens_l2).sum(
        -1, keepdims=True
    )

    alignments['mse_l2'] = ((source_tokens_l2 - target_tokens_l2) ** 2).sum(
        -1, keepdims=True
    )
    alignments['ce_patch'] = misc.cross_entropy(
        source_tokens / rate_params['student_temperature'],
        target_tokens / rate_params['teacher_temperature'],
    )

    # B x L x D
    source_tokens_l2_f = jax.lax.collapse(source_tokens_l2, 1, -1)
    source_tokens_f = jax.lax.collapse(source_tokens, 1, -1)
    target_tokens_l2_f = jax.lax.collapse(target_tokens_l2, 1, -1)
    target_tokens_f = jax.lax.collapse(target_tokens, 1, -1)

    # B x L x L x 1
    similarity_3d = (
        source_tokens_l2_f[:, None] * source_tokens_l2_f[:, :, None]
    ).sum(-1)

    teacher_similarity = (
        target_tokens_l2_f[:, None] * target_tokens_l2_f[:, :, None]
    ).sum(-1)

    # gt similarity
    map_gaussians = pred['gaussians']
    map_gaussians_f = jax.tree.map(
        lambda x: jax.lax.collapse(x, 1, 3),
        map_gaussians,
    )
    centers = map_gaussians_f[0]
    occ = map_gaussians_f[2]

    # distance matrix (meters)
    cdist_sq = (centers[:, None] - centers[:, :, None]) ** 2
    cdist_sq = jnp.sum(cdist_sq, axis=-1)
    cdist = jnp.sqrt(cdist_sq)

    valid = (occ[:, None] > 0) & (occ[:, :, None] > 0)
    cdist = jnp.where(valid, cdist, 0.0)
    cprobs = gaussians.batched_probabilities_between_gaussians(
        map_gaussians_f, map_gaussians_f
    )
    csw = self.config.get('clip_self_weight', 0.05)
    cprobs = jax.vmap(functools.partial(jnp.fill_diagonal, inplace=False))(
        cprobs, cprobs.diagonal(axis1=-2, axis2=-1).clip(max=csw)
    )
    coccupancy = cprobs.sum(-1)
    cvalid = coccupancy[..., None] > csw
    tfeat_space = misc.masked_mean(
        target_tokens_f[:, None],
        cprobs[..., None],
        axis=-2,
    )

    # Sample k points ~ distance
    k = self.config.get('num_negative_samples', 100)
    rng = rate_params['negative_sampling']
    sample_fn = functools.partial(misc.batch_sample_indices, k=k)
    weights = 1.0 - cprobs
    weights = jax.vmap(
        functools.partial(jnp.fill_diagonal, val=0.0, inplace=False)
    )(weights.clip(min=1.0e-6))
    neg_sample_indices = jax.vmap(sample_fn, in_axes=(None, 0))(rng, weights)
    # B X L x k
    _, neg_sim, teacher_neg_sim, neg_valid = [
        jnp.take_along_axis(x, neg_sample_indices, axis=-1)
        for x in [cdist, similarity_3d, teacher_similarity, valid]
    ]
    alignment_sim_l2 = (
        misc.l2_normalize(tfeat_space) * misc.l2_normalize(source_tokens_f)
    ).sum(-1, keepdims=True)
    pos_sim = alignment_sim_l2

    # Contrastive loss terms.
    t = self.config.get('nce_temperature', 0.1)
    info_nce = misc.info_nce(
        pos_sim, neg_sim, neg_valid=neg_valid, temperature=t
    )
    info_nce_teacher = misc.info_nce(
        jnp.ones_like(pos_sim),
        teacher_neg_sim,
        neg_valid=neg_valid,
        temperature=t,
    )
    sample_p = jnp.exp(-info_nce)
    sample_p_teacher = jnp.exp(-info_nce_teacher)

    losses['info_nce'] = misc.masked_bmean(info_nce, neg_valid.any(-1))
    losses['info_nce_teacher'] = misc.masked_bmean(
        info_nce_teacher, neg_valid.any(-1)
    )
    losses['info_prob'] = misc.masked_mean(sample_p, neg_valid.any(-1))
    losses['info_prob_teacher'] = misc.masked_bmean(
        sample_p_teacher, neg_valid.any(-1)
    )

    # Decoupled contrastive loss terms.
    info_dce = -pos_sim[..., 0] / t + jsp.special.logsumexp(
        neg_sim / t, axis=-1
    )
    losses['info_dce_all'] = misc.masked_bmean(info_dce, cvalid[..., 0])
    losses['info_dce'] = misc.masked_bmean(info_dce, neg_valid.any(-1))

    # Alignment loss terms.
    losses['alignment_similarity'] = misc.masked_bmean(alignment_sim_l2, cvalid)
    losses['alignment_mse_l2'] = misc.masked_bmean(
        0.5 * (1 - alignment_sim_l2), cvalid
    )

    # Matching losses.
    source_feats = source_tokens_f
    source_feats_space = misc.masked_mean(
        source_feats[:, None],
        cprobs[..., None],
        axis=-2,
    )

    def outer_product(x, y, normalize=False):
      if normalize:
        x = misc.l2_normalize(x)
        y = misc.l2_normalize(y)
      return (x[:, :, None] * y[:, None, :]).sum(-1)

    similarity_2d3d = outer_product(source_feats, source_feats_space)
    similarity_2d3d_l2 = outer_product(
        source_feats, source_feats_space, normalize=True
    )
    similarity_valid = outer_product(cvalid, cvalid).astype(jnp.bool)
    nll_2d3d = -nn.log_softmax(
        similarity_2d3d,
        axis=-1,
        where=similarity_valid,
    ).diagonal(axis1=-2, axis2=-1)
    losses['matching_nll_2d3d'] = misc.masked_bmean(nll_2d3d, cvalid[..., 0])

    # pair-wise matching loss
    # b x n x n x l x l
    similarity_all = jnp.einsum(
        'bmkd,bnld->bmnkl', source_tokens, source_tokens
    )

    valid_all = jnp.einsum(
        'bmkd,bnld->bmnkl',
        map_gaussians[2][..., None] > 0,
        map_gaussians[2][..., None] > 0,
    )
    probs_all = gaussians.batched_probabilities_between_gaussians(
        map_gaussians, map_gaussians
    ).swapaxes(-3, -2)

    best = probs_all.argmax(-1, keepdims=True)
    log_softmax_all = nn.log_softmax(similarity_all, axis=-1, where=valid_all)
    nll = -jnp.take_along_axis(log_softmax_all, best, axis=-1)[..., 0]
    valid = (valid_all * (probs_all > 0.2)).sum(-1) > 0

    total_nll = jnp.where(valid, nll, 0.0).sum(-1) / valid.sum(-1).clip(
        min=1.0e-6
    )

    n = best.shape[1]
    is_not_same = 1 - jnp.eye(n)[None, :, :]
    # Semantics: Image-contrastive OR MAE
    # Correspondences: Patch-contrastive
    losses['pairwise_matching'] = misc.masked_bmean(total_nll, is_not_same)
    metrics['pairwise_matching_prob'] = jnp.exp(-losses['pairwise_matching'])
    metrics['pairwise_num_gt_matches'] = misc.masked_bmean(
        valid.sum(-1), is_not_same
    )

    # Proportionality loss.
    losses['sim_dist_prop'] = misc.masked_bmean(
        (2 * cprobs * similarity_valid - 1) * (-similarity_2d3d_l2),
        similarity_valid,
    )

    # Pool alignment losses.
    for k, v in alignments.items():
      losses[k] = jnp.mean(v, axis=(-1, -2, -3))
      losses[f'{k}_masked'] = misc.masked_bmean(v, pred['is_masked'])
      # Valid are patches which have depth.
      losses[f'{k}_valid'] = misc.masked_bmean(
          v, pred['gaussians'][2][..., None] > 0
      )
    losses['total'] = types.compose_loss(losses, self.config.compose_loss)
    metrics['masking_rate'] = jax.lax.collapse(pred['is_masked'], 1).mean(-1)
    return losses, metrics

  def eval_loss_metrics_function(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      model_params: jnp.ndarray | None = None,
  ) -> types.LossMetricsTuple:
    return {}, {}

  def visualize(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      max_num_plots: int = 64,
      match_threshold: float = 0,
      stack_axis: int = -3,
      plot_attention: bool = False,
      **kwargs,
  ) -> Mapping[str, FloatArray['B H W 3']]:
    visualizations = {}

    h, w = pred['views'].shape[-3:-1]
    ph, pw = self.config.vit.patches.size

    def stack(x):
      return jnp.concat(
          jax.lax.collapse(x, 1, -3).swapaxes(-5, -4),
          axis=stack_axis,
      )

    def pstack(x):
      x_img = misc.sequence_to_image(x, h // ph, w // pw)
      x_img = jax.image.resize(
          x_img,
          x_img.shape[:-3] + (h, w, 3),
          'nearest',
      )
      return stack(x_img)

    teacher_views = stack(pred['views'])

    student_patch_pca = misc.features_to_rgb(pred['student_patch_tokens'])
    student_pca = misc.features_to_rgb(pred['student_cloud'])
    teacher_pca = misc.features_to_rgb(pred['teacher_patch_tokens'])

    teacher_patch_logits = pred.get(
        'teacher_patch_logits', pred['teacher_patch_tokens']
    )
    student_patch_logits = pred.get(
        'student_cloud_logits', pred['student_cloud']
    )
    ce_patch_global = misc.cross_entropy(
        student_patch_logits / 0.1,
        teacher_patch_logits / 0.04,
    )  # b x n x 2 x L x 1

    ce_image = plot.apply_colormap(
        ce_patch_global[..., 0], cmap='turbo', vmin=0.0, vmax=10.0
    )

    token_similarity_l2 = (
        misc.l2_normalize(pred['student_cloud'])
        * misc.l2_normalize(pred['teacher_patch_tokens'])
    ).sum(-1)

    token_sim_image = plot.apply_colormap(
        token_similarity_l2, cmap='turbo_r', vmin=0.0, vmax=1.0
    )

    student_patch_pca = pstack(student_patch_pca)
    student_pca = pstack(student_pca)
    teacher_pca = pstack(teacher_pca)
    ce_image = pstack(ce_image)
    token_sim_image = pstack(token_sim_image)

    visualizations['features'] = jnp.concatenate(
        [
            teacher_views,
            student_patch_pca,
            student_pca,
            ce_image,
            token_sim_image,
            teacher_pca,
        ],
        axis=-2,
    )

    features = misc.sequence_to_image(pred['student_cloud'], h // ph, w // pw)
    n_match_viz = min(features.shape[0], 32)
    matches = misc.match_features(
        misc.l2_normalize(features[:n_match_viz, -1:]),
        misc.l2_normalize(features[:n_match_viz, :-1]),
        threshold=0.0,
        mutual_check=False,
    )
    all_gaussians = pred['gaussians']
    source_gaussians = tuple(x[:n_match_viz, -1:] for x in all_gaussians)
    target_gaussians = tuple(x[:n_match_viz, :-1] for x in all_gaussians)
    gt_similarity = jax.vmap(gaussians.batched_probabilities_between_gaussians)(
        source_gaussians, target_gaussians
    )

    gt_matches = misc.matches_from_similarity(
        gt_similarity, threshold=0.001, mutual_check=False
    )

    m_images = pred['teacher_views'][:n_match_viz].swapaxes(1, 0)
    n = m_images.shape[0]
    m_images = jnp.concatenate(
        [jnp.repeat(m_images[-1:], n - 1, 0), m_images[:-1]], axis=-2
    ).swapaxes(1, 0)
    matching_images = misc.batch_draw_matches(
        m_images,
        matches['keypoints_i'] * ph,
        matches['keypoints_j'] * pw,
        gt_matches['matches_i'],
        np.where(
            matches['matches_i'] == gt_matches['matches_i'],
            1.0,
            -1.0,
        )
        * (gt_matches['matches_i'] > -1),
        draw='points+lines',
        color_by_coords=False,
    )

    visualizations['correspondences'] = stack(
        jnp.concatenate(matching_images.swapaxes(1, 0), axis=-2)
    )

    visualizations = {
        k: v[: min(v.shape[0], max_num_plots)]
        for k, v in visualizations.items()
    }
    return visualizations
