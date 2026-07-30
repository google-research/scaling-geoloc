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

"""Model file for training backbones on multiples modalities and train them contrastively."""

import itertools
from typing import Any, Mapping, Sequence

from etils.array_types import FloatArray
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxtyping as jt
import ml_collections

from unigeoclip.models import location_encoder
from unigeoclip.models import siglip
from unigeoclip.models import siglip_text
from unigeoclip.models import vit
from unigeoclip.utils import augmentations
from scaling_geoloc import types
from scaling_geoloc.models import base
from scaling_geoloc.utils import geo
from scaling_geoloc.utils import misc

augment: augmentations.AugmentationFn = augmentations.per_image(
    augmentations.sequential([
        augmentations.random_90_rotation(1.0),
    ])
)


def add_gaussian_noise_to_latlng(
    latlngs: jnp.ndarray, key: jax.Array, meters: float = 10.0
) -> jnp.ndarray:
  """Adds Gaussian noise to a JAX array of (latitude, longitude) coordinates.

  The noise is scaled such that its standard deviation corresponds to a
  given distance in meters.

  Args:
      latlngs (jnp.ndarray): A JAX array of shape (n, 2) with lat/lng in
        degrees.
      key (jax.random.PRNGKey): The JAX pseudo-random number generator key.
      meters (float): The standard deviation of the noise in meters.

  Returns:
      jnp.ndarray: A new JAX array with the added noise.
  """
  # Constant for meters per degree of latitude at the equator.
  meters_per_degree_lat = 111319.9
  std_dev_lat = meters / meters_per_degree_lat
  lats_rad = jnp.radians(latlngs[:, 0])
  std_dev_lon = meters / (meters_per_degree_lat * jnp.cos(lats_rad))
  std_devs_deg = jnp.stack(
      [jnp.full_like(lats_rad, std_dev_lat), std_dev_lon], axis=1
  )

  noise = jax.random.normal(key, latlngs.shape)
  noisy_latlngs = latlngs + noise * std_devs_deg

  return noisy_latlngs


def normalize_features(
    features: jt.Float[jt.Array, 'B C'],
    eps: float = 1e-5,
) -> jt.Float[jt.Array, 'B C']:
  """Normalizes features across the batch dimension."""
  invalid = jnp.linalg.norm(features, axis=-1, keepdims=True) < eps
  # We use the double-where trick to avoid NaN gradients if the norm is zero.
  y = jnp.where(invalid, eps, features)
  z = features / jnp.linalg.norm(y, axis=-1, keepdims=True)
  return jnp.where(invalid, 0, z.astype(features.dtype))


def check_multiple(
    batch: types.Batch,
    strides: Mapping[str, int],
) -> types.Batch:
  """Checks that all images in the batch are multiples of the patch size."""
  modalities = list(strides.keys())
  for modality in modalities:
    h, w = batch[modality].shape[-3:-1]
    stride = strides[modality]
    ch = min((h // stride) * stride, h)
    cw = min((w // stride) * stride, w)
    batch[modality] = batch[modality][:, :ch, :cw, :]
  return batch


def flatten_batch(batch: types.Batch, stride: int | None = None) -> types.Batch:
  """Lifts the street-view pixels to the top level for the encoders."""
  batch = misc.filter_batch_for_jit(batch)
  sv_pixels = {'image_pixels': batch['query']['image']}

  if stride is not None:
    sv_pixels = {
        k: (
            misc.crop_to_multiple(v, stride)
            if (isinstance(v, jnp.ndarray) and v.ndim == 4)
            else v
        )
        for k, v in sv_pixels.items()
    }
  return {**batch, **sv_pixels}


def process_batch(
    batch: types.Batch, modalities: Sequence[str], text_precomputed: bool = True
) -> types.Batch:
  """Process the batch."""
  batch = flatten_batch(batch, stride=None)
  batch = batch.copy()
  if 'sv' in modalities:
    batch['sv'] = batch['image_pixels']
  if 'aerial' in modalities:
    batch['aerial_metadata'] = batch['aerial'].copy()
    batch['aerial'] = batch['aerial_metadata']['image']
  if 'dsm' in modalities:
    batch['dsm_metadata'] = batch['dsm'].copy()
    batch['dsm'] = jnp.expand_dims(batch['dsm_metadata']['image'], axis=-1)
  if 'text' in modalities:
    if text_precomputed and 'area_embeddings' in batch:
      batch['text'] = batch['area_embeddings']
    elif isinstance(batch.get('text'), dict) and 'siglip_embeddings' in batch['text']:
      batch['text'] = batch['text']['siglip_embeddings']
  return batch


class BackbonesContrastive(nn.Module):
  """Extract info from backbones and perform contrastive learning."""

  config: ml_collections.ConfigDict
  dtype: jnp.dtype

  def setup(self):
    self.modalities = sorted(self.config.get('modalities', []))
    self.text_backbone = None

    for modality in self.modalities:
      if modality == 'latlng':
        self.latlng_backbone = location_encoder.from_config(
            self.config.get(f'{modality}_encoder', None)
        )
        self.hidden_size = self.config.latlng_encoder.output_dim
      elif modality == 'text':
        if self.config.get(f'{modality}_encoder', None):
          self.text_backbone = siglip_text.from_config(
              self.config.get(f'{modality}_encoder', None)
          )
      else:
        encoder_config = self.config.get(f'{modality}_encoder', None)
        if encoder_config.get('name', None) == 'vit':
          setattr(
              self,
              f'{modality}_backbone',
              vit.ViTBackbone.from_config(
                  self.config.get(f'{modality}_encoder', None), self.dtype
              ),
          )
        elif encoder_config.get('name', None) == 'siglip':
          setattr(
              self,
              f'{modality}_backbone',
              siglip.from_config(self.config.get(f'{modality}_encoder', None)),
          )
        else:
          raise ValueError(
              f'Unsupported backbone: {encoder_config.get("name", None)}'
          )

    self.crop_fn = augmentations.per_image(
        augmentations.get_crop_fn(
            self.config.get('crop', 'none'),
            self.config.get('crop_size', (224, 224)),
        )
    )

    if self.config.latlng_proj and ('latlng' not in self.modalities):
      raise ValueError(
          'latlng_proj is set to True but latlng is not in the list of'
          ' modalities.'
      )
    self.do_all2all_proj = self.config.get('all2all_proj', False)
    if self.do_all2all_proj:
      for modality in self.modalities:
        for modality_out in self.modalities:
          if modality_out != modality:
            setattr(
                self,
                f'{modality}-{modality_out}_proj',
                nn.Dense(
                    self.hidden_size,
                    name=f'{modality}-{modality_out}_projs',
                ),
            )

    self.latlng_proj = (
        self.config.latlng_proj
        and ('latlng' in self.modalities)
        and not self.do_all2all_proj
    )

    self.text_proj = (
        self.config.get('text_proj', False)
        and ('text' in self.modalities)
        and not self.do_all2all_proj
    )

    self.add_latlng_noise = self.config.latlng_noise_meters != 0.0

    temperature = self.config.get('temperature', 0.1)
    if self.config.get('learned_temperature', False):
      self.temperature = self.param(
          'temperature',
          nn.initializers.constant(temperature),
          (),
          self.dtype,
      )
    else:
      self.temperature = jnp.array(temperature)

  def encode(
      self,
      x: jt.Float[jt.Array, '...'],
      modality: str,
      *args,
      train: bool = False,
      **kwargs,
  ) -> jt.Float[jt.Array, '... D']:
    """Interface to backbone encoder."""
    kwargs['normalize'] = self.config.get('l2_normalize', False)
    kwargs['normalize_per_head'] = self.config.get(
        'l2_normalize_per_head', False
    )
    if modality == 'text':
      if self.text_backbone is None:
        tokens = nn.Dense(self.hidden_size, name='text_proj')(x)
      else:
        tokens = getattr(self, f'{modality}_backbone')(x, train=train)
    elif (
        self.config.get(f'{modality}_encoder', None).get('name', None) == 'vit'
    ):
      tokens = getattr(self, f'{modality}_backbone').encode_multiview(
          x,
          train=train,
          *args,
          **kwargs,
      )
    else:
      tokens = getattr(self, f'{modality}_backbone').encode_multiview(
          x, train=train
      )

    return tokens

  def get_image_pixels(self, batch: types.Batch, *, train: bool) -> types.Batch:
    """Preprocess the image pixels (crop, augment)."""
    batch = process_batch(
        batch,
        modalities=self.modalities,
        text_precomputed=self.text_backbone is not None,
    )

    # Crop images to multiple of patch size to avoid boundary effects.
    strides = {}
    for modality in self.modalities:
      encoder_config = self.config.get(f'{modality}_encoder', None)
      if encoder_config is not None and modality not in ('latlng', 'text'):
        strides[modality] = encoder_config.patches.size[0]
    batch = check_multiple(batch, strides)

    if train and 'latlng' in self.modalities and self.add_latlng_noise:
      rng_latlng = self.make_rng('augment_latlng')
      batch['latlng'] = add_gaussian_noise_to_latlng(
          batch['latlng'],
          rng_latlng,
          meters=self.config.latlng_noise_meters,
      )

    if (
        train
        and self.config.get('do_augment', False)
        and 'aerial' in self.modalities
    ):
      # Augment aerial images
      rng_a = self.make_rng('augment_aerial')
      batch['aerial'] = augment(batch['aerial'], rng_a)

    if train and self.config.get('crop', 'none') != 'none':
      batch['sv'] = self.crop_fn(batch['sv'], self.make_rng('crop'))
    return batch

  @nn.compact
  def __call__(
      self,
      batch: types.Batch,
      *,
      train: bool,
      debug: bool = False,
  ) -> types.Predictions:
    batch = self.get_image_pixels(batch, train=train)

    pred = {
        'is_initializing': self.is_initializing(),
        'temperature': self.temperature,
    }
    for modality in self.modalities:
      if modality in batch:
        features = self.encode(
            batch[modality],
            modality=modality,
            train=train,
        )
        pred[f'{modality}_preds'] = features
        if self.do_all2all_proj:
          for modality_out in self.modalities:
            if modality_out != modality:
              pred[f'{modality}-{modality_out}_preds'] = getattr(
                  self, f'{modality}-{modality_out}_proj'
              )(features)
        elif modality != 'latlng' and self.latlng_proj:
          pred[f'{modality}_latlngproj'] = nn.Dense(
              self.config.latlng_encoder.output_dim,
              name=f'{modality}_latlngproj',
          )(features)
        elif modality != 'text' and self.text_proj:
          pred[f'{modality}_textproj'] = nn.Dense(
              self.config.latlng_encoder.output_dim, name=f'{modality}_textproj'
          )(features)

    return pred

  def get_location_embeddings(
      self, latlngs: FloatArray['N 2'], proj: bool = True
  ) -> FloatArray['N D']:
    """Generates representations for an array of latlngs."""
    if 'latlng' not in self.modalities:
      raise ValueError('latlng is not in the list of modalities.')
    features = self.latlng_backbone(latlngs)
    if self.do_all2all_proj and proj:
      features = getattr(self, 'latlng-sv_proj')(features)
    return features

  def get_sv_embeddings(self, sv: FloatArray['N H W 3']) -> FloatArray['N D']:
    """Generates representations for an array of sv."""
    if 'sv' not in self.modalities:
      raise ValueError('sv is not in the list of modalities.')
    return self.sv_backbone.encode_multiview(sv, train=False)

  def get_aerial_embeddings(
      self, aerial: FloatArray['N H W 3']
  ) -> FloatArray['N D']:
    """Generates representations for an array of aerials."""
    if 'aerial' not in self.modalities:
      raise ValueError('aerial is not in the list of modalities.')
    return self.aerial_backbone.encode_multiview(aerial, train=False)

  def get_text_embeddings(
      self, text: FloatArray['N L D']
  ) -> FloatArray['N L D']:
    """Generates representations for an array of text."""
    if 'text' not in self.modalities:
      raise ValueError('text is not in the list of modalities.')
    return self.text_backbone(text)

  def get_dsm_embeddings(
      self, dsm: FloatArray['N H W 1']
  ) -> FloatArray['N L D']:
    """Generates representations for an array of dsms."""
    if 'dsm' not in self.modalities:
      raise ValueError('dsm is not in the list of modalities.')
    dsm = jnp.expand_dims(dsm, axis=-1)
    return self.dsm_backbone.encode_multiview(dsm, train=False)

  def get_dense_aerial_embeddings(
      self, aerial: FloatArray['N H W 3'], get_depths: Sequence[int] = ()
  ) -> FloatArray['N D']:
    """Generates representations for an array of aerials."""
    if 'aerial' not in self.modalities:
      raise ValueError('aerial is not in the list of modalities.')
    return self.aerial_backbone.encode_dense(aerial, get_depths)

  def get_dense_dsm_embeddings(
      self, dsm: FloatArray['N H W 1']
  ) -> FloatArray['N L D']:
    """Generates representations for an array of dsms."""
    if 'dsm' not in self.modalities:
      raise ValueError('dsm is not in the list of modalities.')
    dsm_expand = jnp.expand_dims(dsm, axis=-1)
    return self.dsm_backbone.encode_dense(dsm_expand)


class ContrastiveModel(base.BaseModel):
  """Trainer-facing wrapper for the cell code retrieval model."""

  required_rngs: tuple[str, ...] = (
      'dropout',
      'droptok',
      'crop',
      'augment_aerial',
      'augment_latlng',
  )

  def __init__(
      self,
      config: ml_collections.ConfigDict,
      dataset_meta_data: dict[str, Any],
      dtype: jnp.dtype[Any] = jnp.float32,
  ):
    super().__init__(config, dataset_meta_data, dtype)
    config = self.config.copy_and_resolve_references().unlock()
    self.config = config
    self.modalities = sorted(self.config.get('modalities', []))
    self.flax_model = self.build_flax_model()
    self.do_all2all_proj = self.config.get('all2all_proj', False)
    self.latlng_proj = (
        self.config.latlng_proj
        and ('latlng' in self.modalities)
        and not self.do_all2all_proj
    )
    self.text_proj = (
        self.config.get('text_proj', False)
        and ('text' in self.modalities)
        and not self.do_all2all_proj
    )

  def build_flax_model(self) -> nn.Module:
    return BackbonesContrastive(self.config, self.dtype)

  def init_state(
      self, model_params: types.ModelParams, model_state: types.ModelState
  ) -> tuple[types.ModelParams, types.ModelState]:
    for modality in self.modalities:
      encoder_config = self.config.get(f'{modality}_encoder', None)
      if encoder_config is None:
        continue
      if encoder_config.get('name', None) == 'vit':
        model_params[f'{modality}_backbone'] = (
            vit.ViTBackbone.maybe_load_params(
                model_params[f'{modality}_backbone'],
                encoder_config.get('pretrained_weights', None),
                encoder_config.get('pretrained_checkpoint', None),
            )
        )
      elif encoder_config.get('name', None) == 'siglip':
        # ``load_pretrained=False`` keeps the random init (used by CPU tests that
        # have no SigLIP2 checkpoint on disk); the default loads the weights.
        if encoder_config.get('load_pretrained', True):
          model_params[f'{modality}_backbone'] = (
              siglip.PreTrainedSigLip2Model.load_pretrained_params(
                  encoder_config.model_name,
              )
          )
      elif encoder_config.get('name', None) == 'siglip_text':
        model_params[f'{modality}_backbone'] = (
            siglip_text.PreTrainedSigLip2TextModel.load_pretrained_params(
                model_params,
            )
        )
    return model_params, model_state

  def contrast_embeddings_loss(
      self,
      pred_a: FloatArray['... D'],
      pred_b: FloatArray['... D'],
      batch: types.Batch,
      is_init: bool = True,
      t: float = 1.0,
      prefix: str = '',
      axis_name: str = 'batch',
  ) -> tuple[dict[str, FloatArray['...']], dict[str, FloatArray['...']]]:
    """Contrastive loss between two embeddings."""
    losses = {}
    modality_a, modality_b = prefix.split('-')

    pred_a_in_batch = pred_a

    # Gather embeddings across devices.
    if not is_init:
      pred_b_in_batch = jax.lax.all_gather(
          pred_b, axis_name=axis_name, tiled=True
      )  # NB x D
    else:
      pred_b_in_batch = pred_b

    if self.config.normalize_output_embeddings:
      pred_a_in_batch = normalize_features(pred_a_in_batch)
      pred_b_in_batch = normalize_features(pred_b_in_batch)

    # Compute similarities
    similarities = jnp.einsum('bd,cd->bc', pred_a_in_batch, pred_b_in_batch) / t

    local_batch_size = similarities.shape[0]
    device_id = jax.lax.axis_index(axis_name)
    labels = jnp.arange(local_batch_size) + device_id * local_batch_size
    positive_logits = similarities[jnp.arange(local_batch_size), labels]

    # Compute InfoNCE loss.
    loss = -positive_logits + jsp.special.logsumexp(similarities, axis=-1)
    losses[f'{prefix}_InfoNCE'] = jnp.mean(loss)

    metrics = {}

    log_probs = nn.log_softmax(
        similarities,
        axis=-1,
    )
    probs = jnp.exp(log_probs)

    max_k = min(probs.shape[-1], 100)
    pred_assignment = jax.lax.top_k(probs, k=max_k)[1]

    is_correct = pred_assignment == labels[:, None]

    for k in [1, 5, 20, max_k]:
      metrics[f'{prefix}_recall@{k:03d}'] = jnp.any(is_correct[:, :k], axis=-1)

    if modality_a == 'sv':
      if modality_b == 'latlng':
        latlngs = batch['latlng']
      elif modality_b == 'aerial':
        latlngs = batch['aerial']['latlng']
      elif modality_b == 'dsm':
        latlngs = batch['dsm']['latlng']
      elif modality_b == 'text':
        latlngs = batch['latlng']
      else:
        raise ValueError(f'Unsupported modality: {modality_b}')

      if not is_init:
        latlngs_gathered = jax.lax.all_gather(
            latlngs, axis_name=axis_name, tiled=True
        )
        latlngs_predicted = latlngs_gathered[pred_assignment]
      else:
        latlngs_predicted = latlngs[pred_assignment]

      distance_meters = geo.haversine_distance(
          latlngs[:, None, :], latlngs_predicted
      )[..., 0]

      for k in [10, 50, 100, 200, 500, 1000]:
        acc_k = (distance_meters[:, 0] < k).astype(jnp.float32)
        metrics[f'{modality_b}_within_{k}m'] = jnp.mean(acc_k)

      for k in [1, 5, 20, max_k]:
        distance_meters_k = distance_meters[:, :k].min(axis=-1)
        metrics[f'{modality_b}_Haversine@{k:03d}'] = distance_meters_k.mean()

    return losses, metrics

  def ensemble_predictions(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      is_init: bool = True,
      t: float = 1.0,
      axis_name: str = 'batch',
  ) -> dict[str, FloatArray['...']]:
    """Ensemble predictions from different modalities."""
    metrics = {}
    pred_sv = pred['sv_preds']
    if self.latlng_proj:
      pred_sv_latlng = pred['sv_latlngproj']
    else:
      pred_sv_latlng = pred_sv

    if self.config.normalize_output_embeddings:
      pred_sv = normalize_features(pred_sv)
      pred_sv_latlng = normalize_features(pred_sv_latlng)

    similarities = []
    for modality in self.modalities:
      if modality != 'sv':
        if self.do_all2all_proj:
          pred_a = pred[f'sv-{modality}_preds']
          pred_b = pred[f'{modality}-sv_preds']
          if self.config.normalize_output_embeddings:
            pred_a = normalize_features(pred_a)
        elif modality == 'latlng':
          pred_a = pred_sv_latlng
          pred_b = pred[f'{modality}_preds']
        else:
          pred_a = pred_sv
          pred_b = pred[f'{modality}_preds']

        if not is_init:
          pred_b = jax.lax.all_gather(
              pred_b, axis_name=axis_name, tiled=True
          )  # NB x D

        if self.config.normalize_output_embeddings:
          pred_b = normalize_features(pred_b)

        similarities.append(jnp.einsum('bd,cd->bc', pred_a, pred_b) / t)

    similarities = jnp.stack(similarities, axis=0).mean(axis=0)

    log_probs = nn.log_softmax(
        similarities,
        axis=-1,
    )
    probs = jnp.exp(log_probs)

    max_k = min(probs.shape[-1], 100)
    pred_assignment = jax.lax.top_k(probs, k=max_k)[1]

    latlngs = batch['latlng']

    if not is_init:
      latlngs_gathered = jax.lax.all_gather(
          latlngs, axis_name=axis_name, tiled=True
      )
      latlngs_predicted = latlngs_gathered[pred_assignment]
    else:
      latlngs_predicted = latlngs[pred_assignment]

    distance_meters = geo.haversine_distance(
        latlngs[:, None, :], latlngs_predicted
    )[..., 0]

    for k in [10, 50, 100, 200, 500, 1000]:
      acc_k = (distance_meters[:, 0] < k).astype(jnp.float32)
      metrics[f'Ensemble_within_{k}m'] = jnp.mean(acc_k)

    for k in [1, 5, 20, max_k]:
      distance_meters_k = distance_meters[:, :k].min(axis=-1)
      metrics[f'Ensemble_Haversine@{k:03d}'] = distance_meters_k.mean()
    return metrics

  def contrast_mil_loss(
      self,
      pred_a: FloatArray['... D'],
      pred_b: FloatArray['... D'],
      batch: types.Batch,
      latlngs: FloatArray['... 2'],
      t: float = 1.0,
      prefix: str = '',
      axis_name: str = 'batch',
  ) -> tuple[dict[str, FloatArray['...']], dict[str, FloatArray['...']]]:
    """Contrastive loss between two embeddings."""
    losses = {}
    pred_a_in_batch = pred_a
    pred_b_in_batch = pred_b
    n_modalities = len(self.modalities)

    # Compute similarities
    similarities = jnp.einsum('bd,cd->bc', pred_a_in_batch, pred_b_in_batch) / t

    local_batch_size = similarities.shape[0]
    device_id = jax.lax.axis_index(axis_name)
    modality_id = self.modalities.index(prefix)
    labels = (
        jnp.arange(local_batch_size)
        + modality_id * local_batch_size
        + device_id * local_batch_size * n_modalities
    )
    similarities = similarities.at[jnp.arange(local_batch_size), labels].set(
        -jnp.inf
    )
    labels_mask = jnp.zeros_like(similarities, dtype=bool)
    for i in range(n_modalities):
      if i != modality_id:
        labels_mask = labels_mask.at[
            jnp.arange(local_batch_size),
            jnp.arange(local_batch_size)
            + i * local_batch_size
            + device_id * local_batch_size * n_modalities,
        ].set(True)

    loss = 0
    for i in range(n_modalities):
      if i != modality_id:
        labels = (
            jnp.arange(local_batch_size)
            + i * local_batch_size
            + device_id * local_batch_size * n_modalities
        )
        positive_logits = similarities[jnp.arange(local_batch_size), labels]

        labels_mask2 = labels_mask.at[jnp.arange(local_batch_size), labels].set(
            False
        )
        loss += -positive_logits + jsp.special.logsumexp(
            jnp.where(~labels_mask2, similarities, -jnp.inf), axis=-1
        )
    losses[f'{prefix}_multisim'] = jnp.mean(loss)

    metrics = {}

    log_probs = nn.log_softmax(
        similarities,
        axis=-1,
    )
    probs = jnp.exp(log_probs)

    max_k = min(probs.shape[-1] // n_modalities, 100)
    pred_assignment = jax.lax.top_k(probs, k=max_k)[1]

    is_correct = jnp.take_along_axis(labels_mask, pred_assignment, axis=-1)

    for k in [1, 5, 20, max_k]:
      num_pos = jnp.sum(is_correct[:, : (n_modalities * k)], axis=-1)
      metrics[f'{prefix}_recall@{k:03d}'] = jnp.mean(num_pos) / n_modalities

    if prefix == 'sv':
      latlng_query = batch['latlng']
      latlngs_target = latlngs

      latlngs_predicted = latlngs_target[pred_assignment]

      distance_meters = geo.haversine_distance(
          latlng_query[:, None, :], latlngs_predicted
      )[..., 0]

      for k in [10, 50, 100, 200, 500, 1000]:
        acc_k = (distance_meters[:, 0] < k).astype(jnp.float32)
        metrics[f'{prefix}_within_{k}m'] = jnp.mean(acc_k)

      for k in [1, 5, 20, max_k]:
        distance_meters_k = distance_meters[:, :k].min(axis=-1)
        metrics[f'{prefix}_Haversine@{k:03d}'] = distance_meters_k.mean()

    return losses, metrics

  def _loss_metrics_function(
      self,
      pred: types.Predictions,
      batch: types.Batch,
  ) -> types.LossMetricsTuple:
    batch = flatten_batch(batch, stride=None)
    losses, metrics = {}, {}
    if self.config.get('compose_loss', 'InfoNCE') == 'InfoNCE':
      pairs = itertools.combinations(self.modalities, 2)
      exclude_pairs = self.config.get('exclude_pairs', [])

      for modality_a, modality_b in pairs:
        if (modality_a, modality_b) in exclude_pairs:
          continue
        if modality_b == 'sv':
          modality_a, modality_b = modality_b, modality_a
        pred_b = pred[f'{modality_b}_preds']
        if self.do_all2all_proj:
          pred_a = pred[f'{modality_a}-{modality_b}_preds']
          pred_b = pred[f'{modality_b}-{modality_a}_preds']
        elif modality_b == 'latlng' and self.latlng_proj:
          pred_a = pred[f'{modality_a}_latlngproj']
        elif modality_b == 'text' and self.text_proj:
          pred_a = pred[f'{modality_a}_textproj']
        elif modality_a == 'text' and self.text_proj:
          pred_a = pred[f'{modality_a}_preds']
          pred_b = pred[f'{modality_b}_textproj']
        else:
          pred_a = pred[f'{modality_a}_preds']
        loss, metric = self.contrast_embeddings_loss(
            pred_a,
            pred_b,
            batch,
            is_init=pred['is_initializing'],
            t=pred['temperature'],
            prefix=f'{modality_a}-{modality_b}',
        )
        losses.update(loss)
        metrics.update(metric)

      if len(self.modalities) > 2:
        metric = self.ensemble_predictions(
            pred,
            batch,
            is_init=pred['is_initializing'],
            t=pred['temperature'],
        )
        metrics.update(metric)

    elif self.config.get('compose_loss', 'InfoNCE') == 'multisim':
      pred_b = []
      latlngs = []
      for modality in self.modalities:
        pred_b.append(pred[f'{modality}_preds'])
        if modality == 'latlng':
          ll = batch['latlng']
        elif modality == 'aerial':
          ll = batch['aerial']['latlng']
        elif modality == 'dsm':
          ll = batch['dsm']['latlng']
        else:
          ll = batch['latlng']
        latlngs.append(ll)
      pred_b = jnp.concatenate(pred_b, axis=0)
      latlngs = jnp.concatenate(latlngs, axis=0)
      if not pred['is_initializing']:
        pred_b = jax.lax.all_gather(pred_b, axis_name='batch', tiled=True)
        latlngs = jax.lax.all_gather(latlngs, axis_name='batch', tiled=True)
      if self.config.normalize_output_embeddings:
        pred_b = normalize_features(pred_b)
      for modality in self.modalities:
        pred_a = pred[f'{modality}_preds']
        if self.config.normalize_output_embeddings:
          pred_a = normalize_features(pred_a)

        loss, metric = self.contrast_mil_loss(
            pred_a,
            pred_b,
            {'latlng': batch['latlng']},
            latlngs,
            t=pred['temperature'],
            prefix=modality,
        )
        losses.update(loss)
        metrics.update(metric)

    return losses, metrics

  def loss_metrics_function(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      model_params: jnp.ndarray | None = None,
      model_state: Any | None = None,
      learning_rates: Mapping[str, float] | None = None,
  ) -> types.LossMetricsTuple:
    losses, metrics = self._loss_metrics_function(pred, batch)
    selected_losses = []
    for key in losses:
      if key.endswith(self.config.compose_loss):
        selected_losses.append(losses[key])
    losses['total'] = jnp.sum(jnp.array(selected_losses))
    return losses, metrics

  def visualize(
      self,
      pred: types.Predictions,
      batch: types.Batch,
      max_num_plots: int = 64,
      max_images_per_row: int = 8,
      plot_attention: bool = False,
      **kwargs,
  ) -> Mapping[str, FloatArray['B H W 3']]:
    return {}
