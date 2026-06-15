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

from typing import Any, Mapping, Sequence

from etils.array_types import FloatArray
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxtyping as jt
import ml_collections

from unigeoclip.scenic.models import contrastive_merge
from unigeoclip.scenic.models import location_encoder
from unigeoclip.scenic.models import modality_fusion
from unigeoclip.scenic.models import siglip
from unigeoclip.scenic.models import siglip_text
from unigeoclip.scenic.models import vit
from unigeoclip.utils import augmentations
from scaling_geoloc.scenic import types
from scaling_geoloc.scenic.models import base
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
  meters_per_degree_lat = 111319.9
  std_dev_lat = meters / meters_per_degree_lat
  lats_rad = jnp.radians(latlngs[:, 0])
  std_dev_lon = meters / (meters_per_degree_lat * jnp.cos(lats_rad))
  std_devs_deg = jnp.stack(
      [jnp.full_like(lats_rad, std_dev_lat), std_dev_lon], axis=1
  )

  noise = jax.random.normal(key, latlngs.shape)
  return latlngs + noise * std_devs_deg


def get_other_ll(
    latlngs: jnp.ndarray,
    key: jax.Array,
    min_meters: float,
    max_meters: float,
    *,
    n_samples: int = 100,
) -> jnp.ndarray:
  """Generates random lat/lng positions within a distance range."""
  latlngs = latlngs.repeat(n_samples, axis=0)
  n = latlngs.shape[0]
  k1, k2 = jax.random.split(key)
  dist = jax.random.uniform(k1, (n,), minval=min_meters, maxval=max_meters)
  bearing = jax.random.uniform(k2, (n,), minval=0, maxval=2 * jnp.pi)

  r = 6371e3  # Earth radius in meters
  dist_rad = dist / r

  lat1_rad = jnp.radians(latlngs[:, 0])
  lon1_rad = jnp.radians(latlngs[:, 1])

  lat2_rad = jnp.arcsin(
      jnp.sin(lat1_rad) * jnp.cos(dist_rad)
      + jnp.cos(lat1_rad) * jnp.sin(dist_rad) * jnp.cos(bearing)
  )
  lon2_rad = lon1_rad + jnp.arctan2(
      jnp.sin(bearing) * jnp.sin(dist_rad) * jnp.cos(lat1_rad),
      jnp.cos(dist_rad) - jnp.sin(lat1_rad) * jnp.sin(lat2_rad),
  )

  return jnp.stack([jnp.degrees(lat2_rad), jnp.degrees(lon2_rad)], axis=-1)


class BackbonesFusion(nn.Module):
  """Extract info from backbones and fuse them."""

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

    self.fusion_block = modality_fusion.FusionBlock(
        self.config.get('modality_fusion_block', None), self.dtype
    )
    self.config.hidden_size = self.fusion_block.config.mlp_dim
    self.prob = self.config.get('prob', 0.7)

    for modality in self.modalities:
      setattr(
          self,
          f'{modality}_mask',
          self.param(
              f'{modality}_mask',
              nn.initializers.normal(stddev=1.0),
              (1, self.config.hidden_size),
              self.dtype,
          ),
      )

    self.crop_fn = augmentations.per_image(
        augmentations.get_crop_fn(
            self.config.get('crop', 'none'),
            self.config.get('crop_size', (224, 224)),
        )
    )

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
      **kwargs,
  ) -> jt.Float[jt.Array, '... D']:
    """Interface to backbone encoder."""
    if modality == 'text':
      if self.text_backbone is None:
        tokens = x
      else:
        tokens = getattr(self, f'{modality}_backbone')(x)
    elif modality == 'latlng':
      tokens = getattr(self, f'{modality}_backbone').encode_multiview(x)
    else:
      tokens = getattr(self, f'{modality}_backbone').encode_multiview(
          x, *args, **kwargs
      )

    if self.config.two_stage_training:
      tokens = jax.lax.stop_gradient(tokens)
    return tokens

  def get_image_pixels(
      self, batch: types.Batch, train: bool, other_ll: bool = True
  ) -> types.Batch:
    """Preprocess the image pixels (crop, augment)."""
    batch = contrastive_merge.process_batch(
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
    batch = contrastive_merge.check_multiple(batch, strides)

    if (
        train
        and 'latlng' in self.modalities
        and self.config.get('add_latlng_noise', False)
    ):
      rng_latlng = self.make_rng('augment_latlng')
      if other_ll:
        rng_latlng, rng2 = jax.random.split(rng_latlng)
        batch['latlng2'] = add_gaussian_noise_to_latlng(
            batch['latlng'],
            rng2,
            meters=self.config.get('latlng_noise_meters', 50.0),
        )
      batch['latlng'] = add_gaussian_noise_to_latlng(
          batch['latlng'],
          rng_latlng,
          meters=self.config.get('latlng_noise_meters', 50.0),
      )
    else:
      batch['latlng2'] = batch['latlng'].copy()

    if train and self.config.get('do_augment', False):
      # Augment aerial images
      rng_a = self.make_rng('augment_aerial')
      batch['aerial'] = augment(batch['aerial'], rng_a)

    if train and self.config.get('crop', 'none') != 'none':
      batch['sv'] = self.crop_fn(batch['sv'], self.make_rng('crop'))
    return batch

  def _generate_modality_masks(
      self, batch_size: int, train: bool
  ) -> tuple[jt.Bool[jt.Array, 'B M'], jt.Bool[jt.Array, 'B M']]:
    """Generates masks for modality dropout and fusion.

    Args:
      batch_size: The current batch size.
      train: Whether the model is in training mode.

    Returns:
      A tuple containing two boolean masks, `mask_a` and `mask_b`.
      `mask_a`: Indicates which modalities are included in the first fused
        embedding.
      `mask_b`: Indicates which modalities are included in the second fused
        embedding (complement of `mask_a`).
    """
    num_modalities = len(self.modalities)
    mask_a = jnp.ones((batch_size, num_modalities), dtype=bool)
    mask_b = jnp.ones((batch_size, num_modalities), dtype=bool)

    if not self.is_initializing() and train:
      rng_modalities = self.make_rng('sample_modalities')

      # Step 1: Sample the number of modalities to include in `mask_a`.
      # We sample between 2 or 3 modalities for each sample in the batch.
      items = jax.random.choice(
          key=rng_modalities,
          a=jnp.array([2, 3]),
          shape=(batch_size,),
          p=jnp.array([self.prob, 1 - self.prob]),
      )

      # Create an unshuffled mask where the first `items` columns are True.
      unshuffled_mask = jnp.arange(num_modalities) < items[:, None]

      # Step 2: Randomly permute the modalities for each batch item.
      # This ensures that different combinations of modalities are sampled.
      noise = jax.random.uniform(
          rng_modalities, shape=(batch_size, num_modalities)
      )
      permutation_indices = jnp.argsort(noise, axis=1)
      mask_a = jnp.take_along_axis(unshuffled_mask, permutation_indices, axis=1)

      # Step 3: Apply additional dropout for the 'latlng' modality.
      # If 'latlng' is present, we randomly drop it out with a 50% probability,
      # regardless of whether it was selected in the previous steps.
      if 'latlng' in self.modalities:
        latlng_idx = self.modalities.index('latlng')
        rng_dropout_latlng = self.make_rng('dropout_latlng')
        dropout_mask = jax.random.bernoulli(
            rng_dropout_latlng, 0.5, (batch_size,)
        )
        # The latlng modality is included in mask_a only if it was selected
        # in the permutation AND it was not dropped out.
        mask_a = mask_a.at[:, latlng_idx].set(
            mask_a[:, latlng_idx] & dropout_mask
        )

      # Step 4: `mask_b` is the complement of `mask_a`.
      # This means `mask_b` contains all modalities NOT in `mask_a`.
      mask_b = ~mask_a

    return mask_a, mask_b

  @nn.compact
  def __call__(
      self,
      batch: types.Batch,
      *,
      train: bool,
      debug: bool = False,
  ) -> types.Predictions:
    batch = self.get_image_pixels(batch, train)
    batch_size = batch['latlng'].shape[0]

    pred = {
        'is_initializing': self.is_initializing(),
        'temperature': self.temperature,
    }

    for modality in self.modalities:
      features = self.encode(
          batch[modality],
          modality=modality,
          train=train,
      )
      embeddings = []
      for modality2 in self.modalities:
        if modality == modality2:
          embeddings.append(features)
        else:
          embeddings.append(
              getattr(self, f'{modality2}_mask').repeat(batch_size, axis=0)
          )

      pred[f'{modality}_features'] = self.fusion_block(
          jnp.stack(embeddings, axis=1)
      )

    features = self.encode(
        batch['latlng2'],
        modality='latlng',
        train=train,
    )

    embeddings = []
    for modality2 in self.modalities:
      if 'latlng' == modality2:
        embeddings.append(features)
      else:
        embeddings.append(
            getattr(self, f'{modality2}_mask').repeat(batch_size, axis=0)
        )

    pred['latlng2_features'] = self.fusion_block(jnp.stack(embeddings, axis=1))

    mask_a, mask_b = self._generate_modality_masks(batch_size, train)

    embeddings_a, embeddings_b = [], []
    for i, modality in enumerate(self.modalities):
      name = f'{modality}_features'
      if modality == 'latlng':
        name = 'latlng2_features'
      # For embeddings_a, use the modality feature if mask_a is True,
      # otherwise use the learned modality mask.
      embeddings_a.append(
          jnp.where(
              mask_a[:, i, None],
              pred[name],
              getattr(self, f'{modality}_mask'),
          )
      )
      # For embeddings_b, use the modality feature if mask_b is True,
      # otherwise use the learned modality mask.
      embeddings_b.append(
          jnp.where(
              mask_b[:, i, None],
              pred[f'{modality}_features'],
              getattr(self, f'{modality}_mask'),
          )
      )
    pred['fused_encoding_a'] = self.fusion_block(
        jnp.stack(embeddings_a, axis=1)
    )
    pred['fused_encoding_b'] = self.fusion_block(
        jnp.stack(embeddings_b, axis=1)
    )

    return pred

  @nn.compact
  def get_location_embeddings(
      self, latlngs: FloatArray['N 2']
  ) -> FloatArray['N D']:
    """Generates representations for an array of latlngs."""
    if 'latlng' not in self.modalities:
      raise ValueError('latlng is not in the list of modalities.')
    return self.get_fused_embeddings({'latlng': latlngs}, modalities=['latlng'])

  @nn.compact
  def get_aerial_embeddings(
      self, aerial: FloatArray['N H W 3']
  ) -> FloatArray['N D']:
    """Generates representations for an array of aerials."""
    if 'aerial' not in self.modalities:
      raise ValueError('aerial is not in the list of modalities.')
    return self.get_fused_embeddings(
        {'aerial': aerial}, modalities=['aerial'], process_b=False
    )

  @nn.compact
  def get_text_embeddings(self, text: FloatArray['N D']) -> FloatArray['N D']:
    """Generates representations for an array of text."""
    if 'text' not in self.modalities:
      raise ValueError('text is not in the list of modalities.')
    return self.get_fused_embeddings(
        {'text': text}, modalities=['text'], process_b=False
    )

  def get_dense_aerial_embeddings(
      self, aerial: FloatArray['N H W 3'], get_depths: Sequence[int] = ()
  ) -> FloatArray['N D']:
    """Generates representations for an array of aerials."""
    if 'aerial' not in self.modalities:
      raise ValueError('aerial is not in the list of modalities.')
    return self.aerial_backbone.encode_dense(aerial, get_depths)

  @nn.compact
  def get_dense_dsm_embeddings(
      self, dsm: FloatArray['N H W 1']
  ) -> FloatArray['N D']:
    """Generates representations for an array of dsms."""
    if 'dsm' not in self.modalities:
      raise ValueError('dsm is not in the list of modalities.')
    return self.dsm_backbone.encode_dense(jnp.expand_dims(dsm, axis=-1))

  @nn.compact
  def get_fused_embeddings(
      self,
      batch: types.Batch,
      modalities: Sequence[str],
      process_b: bool = True,
  ) -> FloatArray['N D']:
    """Generates fused representations for given modalities."""
    for modality in modalities:
      if modality not in self.modalities:
        raise ValueError(f'{modality} is not in the list of modalities.')

    if process_b:
      batch = contrastive_merge.process_batch(
          batch,
          modalities=modalities,
          text_precomputed=self.text_backbone is not None,
      )
    embeddings = []
    batch_size = batch[modalities[0]].shape[0]

    for modality in self.modalities:
      if modality in modalities:
        features = self.encode(
            batch[modality],
            modality=modality,
            train=False,
        )
      else:
        features = getattr(self, f'{modality}_mask').repeat(batch_size, axis=0)
      embeddings.append(features)
    return self.fusion_block(jnp.stack(embeddings, axis=1))


class ContrastiveModel(base.BaseModel):
  """Trainer-facing wrapper for the contrastive fusion model."""

  required_rngs: tuple[str, ...] = (
      'dropout',
      'droptok',
      'crop',
      'augment_aerial',
      'augment_latlng',
      'sample_modalities',
      'other_ll',
      'dropout_latlng',
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

  def build_flax_model(self) -> nn.Module:
    return BackbonesFusion(self.config, self.dtype)

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
      latlngs: FloatArray['... 2'],
      s2cells_a: FloatArray['...'] | None,
      s2cells_b: FloatArray['...'] | None,
      other_negatives: FloatArray['... D'] | None = None,
      t: float = 1.0,
      prefix: str = '',
      axis_name: str = 'batch',
  ) -> tuple[dict[str, FloatArray['...']], dict[str, FloatArray['...']]]:
    """Contrastive loss between two embeddings."""
    losses = {}
    pred_a_in_batch = pred_a
    pred_b_in_batch = pred_b
    n_modalities = len(self.modalities)

    if other_negatives is not None:
      other_negatives = contrastive_merge.normalize_features(other_negatives)
      pred_b_in_batch = jnp.concatenate(
          [pred_b_in_batch, other_negatives], axis=0
      )

    # Compute similarities
    similarities = jnp.einsum('bd,cd->bc', pred_a_in_batch, pred_b_in_batch) / t

    local_batch_size = similarities.shape[0]
    device_id = jax.lax.axis_index(axis_name)
    modality_id = 0
    if prefix != 'fused':
      modality_id = self.modalities.index(prefix) + 1
      labels = (
          jnp.arange(local_batch_size)
          + modality_id * local_batch_size
          + device_id * local_batch_size * (n_modalities + 1)
      )
      similarities = similarities.at[jnp.arange(local_batch_size), labels].set(
          -jnp.inf
      )
    labels_mask = jnp.zeros_like(similarities, dtype=bool)
    for i in range(n_modalities + 1):
      if i != modality_id or prefix == 'fused':
        labels_mask = labels_mask.at[
            jnp.arange(local_batch_size),
            jnp.arange(local_batch_size)
            + i * local_batch_size
            + device_id * local_batch_size * (n_modalities + 1),
        ].set(True)

    if (
        self.config.get('mask_negatives_in_same_s2cell', False)
        and s2cells_a is not None
        and s2cells_b is not None
    ):
      if prefix == 'text':
        same_s2cell = s2cells_a[:, None] == s2cells_b[None, :]
        # We mask out entries that are from same s2 cell but are not positives.
        similarities = jnp.where(
            same_s2cell & ~labels_mask, -jnp.inf, similarities
        )
      else:
        # Get the same s2cells
        same_s2cell = s2cells_a[:, None] == s2cells_b[None, :]
        # We need to mask out entries that are from same s2 cell but are not
        # positives. We do this in a tiled fashion over devices.
        num_devices = similarities.shape[1] // (
            (n_modalities + 1) * local_batch_size
        )
        labels_mask_device = jnp.zeros_like(similarities, dtype=bool)
        for device in range(num_devices):
          start_index = (
              modality_id * local_batch_size
              + device * local_batch_size * (n_modalities + 1)
          )
          labels_mask_device = labels_mask_device.at[
              :, start_index : start_index + local_batch_size
          ].set(True)
        # We mask out entries that are from same s2 cell but are not positives.
        similarities = jnp.where(
            same_s2cell & ~labels_mask & labels_mask_device,
            -jnp.inf,
            similarities,
        )

    loss = 0
    for i in range(n_modalities + 1):
      if i != modality_id or prefix == 'fused':
        labels = (
            jnp.arange(local_batch_size)
            + i * local_batch_size
            + device_id * local_batch_size * (n_modalities + 1)
        )
        positive_logits = similarities[jnp.arange(local_batch_size), labels]

        labels_mask2 = labels_mask.at[jnp.arange(local_batch_size), labels].set(
            False
        )
        loss += -positive_logits + jsp.special.logsumexp(
            jnp.where(~labels_mask2, similarities, -jnp.inf), axis=-1
        )
    losses[f'{prefix}_InfoNCE'] = jnp.mean(loss)

    if prefix == 'fused':
      n_modalities += 1

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
      if other_negatives is not None:
        latlngs_target = jnp.concatenate([latlngs, batch['other_ll']], axis=0)
      else:
        latlngs_target = latlngs

      latlngs_predicted = latlngs_target[pred_assignment]

      distance_meters = misc.haversine_distance(
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
    batch = contrastive_merge.flatten_batch(batch, stride=None)
    losses, metrics = {}, {}

    pred_b = [pred['fused_encoding_b']]
    latlngs = [batch['latlng']]
    s2cells_list = []
    s2cells_a = None
    if self.config.mask_negatives_in_same_s2cell and 's2cell_id' in batch:
      s2cells_list.append(batch['s2cell_id'])
      s2cells_a = jnp.array(batch['s2cell_id'])
    for modality in self.modalities:
      pred_b.append(pred[f'{modality}_features'])
      if self.config.mask_negatives_in_same_s2cell and 's2cell_id' in batch:
        s2cells_list.append(batch['s2cell_id'])
      if modality == 'latlng':
        latlngs.append(batch['latlng'])
      elif modality == 'aerial':
        latlngs.append(batch['aerial']['center_ll'])
      elif modality == 'dsm':
        latlngs.append(batch['dsm']['center_ll'])
      else:
        latlngs.append(batch['latlng'])
    pred_b = jnp.concatenate(pred_b, axis=0)
    latlngs = jnp.concatenate(latlngs, axis=0)
    s2cells_b = None
    if self.config.mask_negatives_in_same_s2cell and 's2cell_id' in batch:
      s2cells_b = jnp.concatenate(s2cells_list, axis=0)
    if not pred['is_initializing']:
      pred_b = jax.lax.all_gather(pred_b, axis_name='batch', tiled=True)
      latlngs = jax.lax.all_gather(latlngs, axis_name='batch', tiled=True)
      if s2cells_b is not None:
        s2cells_b = jax.lax.all_gather(s2cells_b, axis_name='batch', tiled=True)
    if self.config.normalize_output_embeddings:
      pred_b = contrastive_merge.normalize_features(pred_b)
    for modality in self.modalities:
      pred_a = pred[f'{modality}_features']
      if self.config.normalize_output_embeddings:
        pred_a = contrastive_merge.normalize_features(pred_a)
      loss, metric = self.contrast_embeddings_loss(
          pred_a,
          pred_b,
          {'latlng': batch['latlng']},
          latlngs,
          s2cells_a=s2cells_a,
          s2cells_b=s2cells_b,
          other_negatives=None,
          t=pred['temperature'],
          prefix=modality,
      )
      losses.update(loss)
      metrics.update(metric)

    pred_a = pred['fused_encoding_a']
    if self.config.normalize_output_embeddings:
      pred_a = contrastive_merge.normalize_features(pred_a)
    loss, metric = self.contrast_embeddings_loss(
        pred_a,
        pred_b,
        {'latlng': batch['latlng']},
        latlngs,
        s2cells_a=s2cells_a,
        s2cells_b=s2cells_b,
        other_negatives=None,
        t=pred['temperature'],
        prefix='fused',
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
