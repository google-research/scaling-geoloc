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

from scaling_geoloc.scenic.evals import base_eval
from scaling_geoloc.scenic.models import base


MODELS = {
    'depth_unet': ('depth_unet', 'DepthUNetModel'),
    'depth_dpt': ('depth_dpt', 'DepthDPTModel'),
    'mae_fomo': ('mae_fomo', 'MaskedAutoEncoderModel'),
    'mae_multiview': ('mae_multiview', 'MultiViewMaskedAutoEncoderModel'),
    'dino': ('dino', 'ViTDINOModel'),
    'scenebot': ('scene_bot', 'SceneBotModel'),
    'multiview_distill': ('multiview_distill', 'ViTMultiviewModel'),
    'mae3d': ('mae3d', 'MAE3DModel'),
    'dino3d': ('dino3d', 'ViTDINO3DModel'),
    'ijepa': ('ijepa', 'ViTIJEPAModel'),
    'scene_net': ('scene_net', 'SceneNetModel'),
    'cell_code': ('cell_code', 'CellCodeRetrievalModel'),
    'hierarchical_cell_code': (
        'hierarchical_cell_code',
        'HierarchicalCellCodeRetrievalModel',
    ),
    'recurrent_cell_code': (
        'recurrent_cell_code',
        'RecurrentCellCodeRetrievalModel',
    ),
    'sv_aerial': ('sv_aerial', 'SVAerialRetrievalModel'),
    'sv_aerial_code': ('sv_aerial_code', 'SVAerialCellCodeRetrievalModel'),
    'sv_aerial_code_largescale': (
        'sv_aerial_code_largescale',
        'SVAerialCellCodeRetrievalModel',
    ),
    'sv_aerial_code_implicit': (
        'sv_aerial_code_implicit',
        'SVAerialCodeImplicitModel',
    ),
    'sv_aerial_hierarchical_code': (
        'sv_aerial_hierarchical_code',
        'SVAerialHierarchicalCellCodeRetrievalModel',
    ),
    'image_retrieval': ('image_retrieval', 'ImageRetrievalModel'),
    'location_encoder': ('location_encoder', 'LocationEncoderModelWrapper'),
    'geoclip': ('geoclip', 'GeoCLIPModel'),
}

EVALS = {
    'knn_imagenet': ('knn_imagenet', 'KNNImageNetEval'),
    'retrieval': ('retrieval', 'RetrievalEvaluation'),
    'correspondences': ('correspondences', 'CorrespondenceEvaluation'),
    'class_cooccurrence': ('class_cooccurrence', 'ClassCooccurrenceEvaluation'),
}


def get_class(basepath: str, modulename: str, classname: str) -> Any:
  """Import a given class in a given module."""
  modulepath = basepath.format(modulename)
  return getattr(importlib.import_module(modulepath), classname)


def get_model(name: str) -> Type[base.BaseModel]:
  """Get a top-level model by name."""
  basepath = 'scaling_geoloc.scenic.models.{}'
  return get_class(basepath, *MODELS[name])


def get_eval(name: str) -> Type[base_eval.BaseEval]:
  """Get a top-level eval by name."""
  basepath = 'scaling_geoloc.scenic.evals.{}'
  return get_class(basepath, *EVALS[name])
