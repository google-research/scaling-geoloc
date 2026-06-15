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

"""Utility functions for models."""

import importlib
from typing import Any, Type

from scaling_geoloc.evals import base_eval
from scaling_geoloc.models import base


MODELS = {
    'hybrid_geolocalization': ('hybrid_geolocalization', 'GeoLocalizerModel'),
}

EVALS = {
    'retrieval': ('retrieval', 'RetrievalEvaluation'),
}


def get_class(basepath: str, modulename: str, classname: str) -> Any:
  """Import a given class in a given module."""
  modulepath = basepath.format(modulename)
  return getattr(importlib.import_module(modulepath), classname)


def get_model(name: str) -> Type[base.BaseModel]:
  """Get a top-level model by name."""
  basepath = 'scaling_geoloc.models.{}'
  return get_class(basepath, *MODELS[name])


def get_eval(name: str) -> Type[base_eval.BaseEval]:
  """Get a top-level eval by name."""
  basepath = 'scaling_geoloc.evals.{}'
  return get_class(basepath, *EVALS[name])
