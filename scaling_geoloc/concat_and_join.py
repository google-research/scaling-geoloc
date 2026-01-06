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

"""Flume pipeline to extract labels from arrayrecord files."""

from collections.abc import Iterable, Sequence
import dataclasses
import functools
import logging

from absl import app
from absl import flags
import apache_beam as beam
import fancyflags as ff
import numpy as np
import tensorflow_datasets as tfds

from google3.location.visualmapping.experimental.fomo.data import data_utils
from scaling_geoloc.data import export
from scaling_geoloc.data import labels
from scaling_geoloc.data import pipeline_steps
from scaling_geoloc.data import reverse_geocoder
from google3.pipeline.flume.py import runner
from google3.pyglib import gfile
from google3.util.geometry.python import s2

_INPUT_PREFIX = flags.DEFINE_string(
    'input_prefix', None, 'input prefix.', required=True
)

_INPUT_LOCATIONS = flags.DEFINE_list(
    'input_locations',
    None,
    'The locations to concatenate.',
    required=True,
)

_OUTPUT_PREFIX = flags.DEFINE_string(
    'output_prefix', None, 'Output prefix.', required=True
)

_FILTER_COUNTRIES = flags.DEFINE_list(
    'filter_countries',
    None,
    'If not None, only process the given countries.',
)

# The target_level defines the 'join' level.
_DATA = ff.DEFINE_auto('data', export.AerialDataConfig)

_AERIAL_PREFIX = flags.DEFINE_string(
    'aerial_prefix', None, 'Aerial prefix.', required=True
)

_AERIAL_SPLIT_NAME = flags.DEFINE_string(
    'aerial_split_name',
    'aerial',
    'Aerial split name.',
)

_AERIAL_RE_CENTER = flags.DEFINE_bool(
    'aerial_re_center',
    True,
    'If true, re-center the aerial crop to the SV camera center.',
)

_SPLIT_NAMES = flags.DEFINE_list(
    'split_names',
    None,
    'The name of the splits to extract labels from (train, valid, test).',
    required=True,
)

_OVERWRITE = flags.DEFINE_bool(
    'overwrite',
    True,
    'If true, overwrite the output prefix if it already exists.',
)

_EXPORT_AERIAL_TEST_SET = flags.DEFINE_bool(
    'export_aerial_test_set',
    True,
    'If true, export a test set of aerial crops.',
)


_AERIAL_TARGET_LEVEL = flags.DEFINE_integer(
    'aerial_target_level',
    None,
    'The target level for the aerial crops in the test set.',
)


class AddAerialCropToScene(beam.DoFn):
  """Adds aerial crop to the example."""

  def __init__(
      self, aerial_data: export.AerialDataConfig, re_center: bool = True
  ):
    self.aerial_data = aerial_data
    self.re_center = re_center

    if aerial_data.num_samples > 1:
      raise ValueError(
          'num_samples > 1 is not supported in join. Use'
          ' aerial_postprocessing instead.'
      )

  def setup(self):
    self.rng = np.random.default_rng(self.aerial_data.seed + 1)
    self.crops_from_aerial = export.CropsFromAerial(self.aerial_data)
    self.crops_from_aerial.setup()

  def process(
      self,
      s2cell_and_data: tuple[bytes, dict[str, Iterable[export.Example]]],
  ) -> Iterable[tuple[str, export.Example]]:
    s2cell, data = s2cell_and_data
    sv_samples = list(data['sv'])
    aerial_samples = list(data['aerial'])
    if not aerial_samples:
      beam.metrics.Metrics.counter(self.__class__.__name__, 'no_aerial').inc()
      return
    if not sv_samples:
      beam.metrics.Metrics.counter(self.__class__.__name__, 'no_sv').inc()
      return

    beam.metrics.Metrics.distribution(
        self.__class__.__name__, 'aerial_samples_per_s2cell'
    ).update(len(aerial_samples))
    beam.metrics.Metrics.distribution(
        self.__class__.__name__, 'sv_samples_per_s2cell'
    ).update(len(sv_samples))

    for sv_index, sv_sample in enumerate(sv_samples):
      for aerial_sample_index, aerial_sample in enumerate(aerial_samples):
        for aerial_coll_index in range(len(aerial_sample['rgb'])):
          aerial_crop = self.crops_from_aerial.generate_sample(
              aerial_sample,
              image_idx=aerial_coll_index,
              at_ecef=export.center_ecef_from_example(sv_sample)
              if self.re_center
              else None,
          )
          aerial_crop['s2cell'] = sv_sample['s2cell']
          beam.metrics.Metrics.counter(
              self.__class__.__name__, 'sv_aerial_pairs'
          ).inc()
          key = f'{s2cell.decode("utf-8")}_sv{sv_index}_aerial{aerial_sample_index}.{aerial_coll_index}'
          yield key, {**sv_sample, 'aerial': aerial_crop}


def filter_by_country(
    example: export.Example, required_country_ids: Sequence[int]
) -> bool:
  """Filters examples by country."""
  if example['country'] in required_country_ids:
    beam.metrics.Metrics.counter(
        'FilterByCountry', f'examples_in_country_{example["country"]}'
    ).inc()
    beam.metrics.Metrics.counter(
        'FilterByCountry', 'examples_in_required_countries'
    ).inc()
    return True
  else:
    beam.metrics.Metrics.counter(
        'FilterByCountry', 'examples_not_in_required_countries'
    ).inc()
  return False


class FilterByCountry(beam.DoFn):
  """Filters examples by country."""

  def __init__(self, required_country_ids: Sequence[int]):
    self.required_country_ids = required_country_ids

  def setup(self):
    self.country_coder = reverse_geocoder.ReverseGeocoder('country')

  def process(self, example: export.Example) -> Iterable[export.Example]:
    if 'country' in example:
      country_id = example['country']
      country_label = self.country_coder.id_to_label(country_id)
    else:
      lat, lng = export.center_latlng_from_example(example).tolist()
      country_label = self.country_coder.get_label_for_latlng(lat, lng)
      country_id = self.country_coder.label_to_id(country_label)

    if country_id in self.required_country_ids:
      beam.metrics.Metrics.counter(
          'FilterByCountry', f'examples_in_country_{country_label}'
      ).inc()
      beam.metrics.Metrics.counter(
          'FilterByCountry', 'examples_in_required_countries'
      ).inc()
      yield example
    else:
      beam.metrics.Metrics.counter(
          'FilterByCountry', 'examples_not_in_required_countries'
      ).inc()
    return


def update_s2cell_of_example(
    example: export.Example, target_level: int
) -> export.Example:
  return {
      **example,
      's2cell': export.s2cell_token_at_level(example, target_level),
  }


def expand_s2cells_of_example(
    example: export.Example, target_level: int
) -> Iterable[export.Example]:
  """Expands the s2cell of the example to the target level."""
  parent = s2.S2CellId.from_token(example['s2cell'])
  if parent.level > target_level:
    # Target is coarser than the current level.
    parent = parent.parent_at_level(target_level)
  child = parent.child_begin_at_level(target_level)
  while child != parent.child_end_at_level(target_level):
    yield {
        **example,
        's2cell': child.token(),
    }
    child = child.next()


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if gfile.Exists(_OUTPUT_PREFIX.value) and not _OVERWRITE.value:
    raise ValueError(f'Output prefix already exists: {_OUTPUT_PREFIX.value}')
  else:
    gfile.MakeDirs(_OUTPUT_PREFIX.value)

  input_directories = []
  for location in _INPUT_LOCATIONS.value:
    input_directories.append(f'{_INPUT_PREFIX.value}/{location}')
  logging.info('input_directories: %s', ','.join(input_directories))

  builder = tfds.builder_from_directories(input_directories)
  features_dict = builder.info.features
  # Unfortunately this is broken in tfds, so we have to manually concatenate...
  builders = []
  for location in _INPUT_LOCATIONS.value:
    builders.append(
        tfds.builder_from_directory(f'{_INPUT_PREFIX.value}/{location}')
    )

  data_config = _DATA.value()
  target_level = data_config.target_level

  if _AERIAL_PREFIX.value is not None:
    aerial_builder = tfds.builder_from_directory(_AERIAL_PREFIX.value)
    features_dict = tfds.features.FeaturesDict(
        {**features_dict, 'aerial': _DATA.value().data_spec()}
    )
    if _EXPORT_AERIAL_TEST_SET.value:
      test_aerial_config = dataclasses.replace(
          data_config,
          num_samples=1,
          max_rotation_angle_degrees=0.0,
          max_offset=0.0,
      )
      test_aerial_prefix = f'{_OUTPUT_PREFIX.value}/aerial'
      if not gfile.Exists(test_aerial_prefix):
        gfile.MakeDirs(test_aerial_prefix)
        # Write the metadata for train, valid and test splits.
    else:
      test_aerial_config = None
      test_aerial_prefix = None
  else:
    aerial_builder = None
    test_aerial_config = None
    test_aerial_prefix = None

  if _FILTER_COUNTRIES.value is not None:
    logging.info('Filtering countries: %s', ','.join(_FILTER_COUNTRIES.value))
    country_coder = reverse_geocoder.ReverseGeocoder('country')
    required_country_ids = [
        country_coder.label_to_id(country)
        for country in _FILTER_COUNTRIES.value
    ]
  else:
    required_country_ids = None

  runner.program_started()

  # Extract labels from arrayrecord files.
  with beam.Pipeline(runner.FlumeRunner()) as pipeline:
    aerial_data = None
    if aerial_builder is not None:
      # Read in aerial data.
      split_name = _AERIAL_SPLIT_NAME.value
      aerial_data = (
          pipeline
          | f'ReadAerialFromTFDS_{split_name}'
          >> tfds.beam.ReadFromTFDS(aerial_builder, split=split_name)
          | f'ConvertAerialToNumpy_{split_name}' >> beam.Map(tfds.as_numpy)
      )

      if required_country_ids is not None:
        # Filter the aerial data by country.
        aerial_data = (
            aerial_data
            | f'FilterCountries_{split_name}'
            >> beam.ParDo(FilterByCountry(required_country_ids))
        )

      # Extract non-offsetted aerial crops for the test set.
      if test_aerial_config is not None:
        aerial_test_data = (
            aerial_data
            | f'ExpandS2Cells_{split_name}'
            >> beam.FlatMap(
                expand_s2cells_of_example,
                target_level=_AERIAL_TARGET_LEVEL.value or target_level,
            )
            | f'CropAerial_{split_name}'
            >> beam.ParDo(export.CropsFromAerial(data_config, re_center=True))
        )
        export.serialize_and_write_examples(
            aerial_test_data,
            tfds.features.FeaturesDict(test_aerial_config.data_spec()),
            test_aerial_prefix,
            split_name='test_aerial',
        )

    for split_name in _SPLIT_NAMES.value:
      input_datas = []
      # Read the data from each location.
      for location_i, builder_i in zip(_INPUT_LOCATIONS.value, builders):
        input_datas.append(
            pipeline
            | f'ReadFromTFDS_{location_i}_{split_name}'
            >> tfds.beam.ReadFromTFDS(builder_i, split=split_name)
            | f'ConvertToNumpy_{location_i}_{split_name}'
            >> beam.Map(tfds.as_numpy)
        )
      # Concatenate the data from each location.
      data = input_datas | f'Flatten_{split_name}' >> beam.Flatten()
      data = data | f'UpdateS2Cell_{split_name}' >> beam.Map(
          functools.partial(update_s2cell_of_example, target_level=target_level)
      )

      if required_country_ids is not None:
        # Filter the data by country.
        data = data | f'FilterCountries_{split_name}' >> beam.ParDo(
            FilterByCountry(required_country_ids)
        )

      if aerial_data is not None:
        # Join SV and aerial data.
        svaerial_data = {
            'sv': (
                data
                | f'KeySVByS2Cell_{split_name}'
                >> beam.Map(lambda x: (x['s2cell'], x))
            ),
            'aerial': (
                aerial_data
                | f'KeyAerialByS2Cell_{split_name}'
                >> beam.Map(
                    lambda x: (
                        export.s2cell_token_at_level(x, target_level),
                        x,
                    )
                )
            ),
        } | f'CombineSVAndAerial_{split_name}' >> beam.CoGroupByKey()
        data = (
            svaerial_data
            | f'AddAerialCropToScene_{split_name}'
            >> beam.ParDo(
                AddAerialCropToScene(
                    data_config, re_center=_AERIAL_RE_CENTER.value
                )
            )
        )

      export.serialize_and_write_examples(
          data,
          features_dict,
          _OUTPUT_PREFIX.value,
          split_name=split_name,
      )

  # Write the beam metrics to the output directory.
  pipeline_steps.write_metrics(pipeline.result, _OUTPUT_PREFIX.value)

  # Write the metadata for train, valid and test splits.
  data_utils.write_tfds_metadata(
      data_dir=_OUTPUT_PREFIX.value, features_dict=features_dict
  )

  if test_aerial_prefix is not None:
    # Write the metadata for the test aerial crops.
    assert test_aerial_config is not None
    data_utils.write_tfds_metadata(
        data_dir=test_aerial_prefix,
        features_dict=tfds.features.FeaturesDict(
            test_aerial_config.data_spec()
        ),
    )

  # Write labels
  required_keys = ['s2cell']
  with beam.Pipeline(runner.FlumeRunner()) as pipeline:
    for split_name in _SPLIT_NAMES.value:
      split_labels = labels.extract_labels(
          pipeline,
          _OUTPUT_PREFIX.value,
          split_name,
          required_keys=required_keys,
      )
      labels.write_labels_to_csv(
          split_labels,
          f'{_OUTPUT_PREFIX.value}/{split_name}_s2cells.csv',
          required_keys=required_keys,
          tag=split_name,
      )
      labels.write_grouped_labels_to_csv(
          split_labels,
          f'{_OUTPUT_PREFIX.value}/{split_name}_grouped_s2cells.csv',
          required_keys=required_keys,
          tag=split_name,
      )


if __name__ == '__main__':
  app.run(main)
