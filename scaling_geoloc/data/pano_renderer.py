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

"""A helper class to render high-resolution perspective views from panoramas."""

from typing import TypedDict
import cv2
import numpy as np
from google3.cityblock.exposure.python import exposure_model
from google3.geo.imagery.streetview.store.client.python import client_request_options
from google3.geo.imagery.streetview.store.client.python import pixel_provider
from google3.geo.imagery.streetview.store.client.python import svms_client_util
from google3.geo.imagery.streetview.store.proto import image_type_pb2
from google3.geo.imagery.streetview.store.proto import pixel_options_pb2
from google3.geo.imagery.streetview.store.proto import resources_pb2
from google3.geo.imagery.streetview.store.shared.python import ids as gsv_ids
from scaling_geoloc.data import types


class CameraParams(TypedDict):
  width: int
  height: int
  K: np.ndarray
  type: bytes
  radial_distortion: np.ndarray
  tangential_distortion: np.ndarray
  fov_deg: float


def _get_pano_pose(exposure_proto) -> np.ndarray:
  """Computes the original pose of a stitched panorama."""
  exposure = exposure_model.ExposureModel(exposure_proto)
  capture_time = exposure.canonical_time()
  rosette_t_ecef = exposure.rosette_from_world(capture_time)

  # This fixed rotation makes the panorama's coordinate system forward-facing
  pano_t_rosette = np.eye(4)
  pano_t_rosette[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).T
  pano_t_ecef = pano_t_rosette @ rosette_t_ecef
  return pano_t_ecef


class PanoRenderer:
  """A helper class to render high-resolution perspective views from panoramas."""

  def __init__(self):
    self._svms_client = svms_client_util.CreateBatchSvmsClientFromTarget(
        'blade:street-view-metadata-service-staging'
    )
    self._image_request_options = (
        client_request_options.ClientRequestOptions().IncludeMetadata([
            resources_pb2.MetadataPiece.METADATA_IMAGE_ID,
            resources_pb2.MetadataPiece.METADATA_IMAGE_TYPE,
            resources_pb2.MetadataPiece.METADATA_CAPTURE_EXPOSURE,
        ])
    ).SetPixelOptions(
        pixel_options_pb2.PixelOptions(
            anonymization=pixel_options_pb2.PixelAnonymizationMode.SAFE
        )
    )
    self._provider = pixel_provider.CreatePixelProvider()

  def _get_pano_data(self, pano_id: gsv_ids.CaptureId, pano_stitch_height: int):
    """Fetches the full Capture object from SVMS."""
    batch_result = self._svms_client.BatchGetCapture(
        [pano_id], self._image_request_options
    )
    collects = batch_result.GetCollects()
    if len(collects) != 1:
      raise ValueError(collects)
    data_stream = collects[0].GetMainRosetteDataStream()

    captures = data_stream.GetCaptures()
    if len(captures) != 1:
      raise ValueError(captures)
    capture = captures[0]

    exposure_proto = capture.GetExposure()

    images = capture.GetImages()
    images = [
        i
        for i in images
        if i.GetImageType() == image_type_pb2.ImageType.STITCHED
    ]
    if len(images) != 1:
      raise ValueError([i.id() for i in images])
    pano_image = images[0]

    stitch_options = (
        pixel_options_pb2.ImagePixelOptions.GsvImageStitchingOptions(
            pano_height=pano_stitch_height
        )
    )
    image_pixel_options = pixel_options_pb2.ImagePixelOptions(
        gsv_image_stitching_options=stitch_options
    )

    pano_pixels_jpeg = self._provider.FromImage(pano_image).GetJpeg(
        image_pixel_options
    )
    pano_pixels = cv2.cvtColor(
        cv2.imdecode(
            np.frombuffer(pano_pixels_jpeg, np.uint8), cv2.IMREAD_ANYCOLOR
        ),
        cv2.COLOR_BGR2RGB,
    )
    return pano_pixels, exposure_proto

  def render_high_res_view(
      self,
      pano_name: bytes,
      ecef_t_sampled: np.ndarray,
      camera: CameraParams,
      target_height: int,
      pano_stitch_height: int | None = None,
  ) -> np.ndarray:
    """Renders a high-resolution perspective view from a StreetView panorama."""
    pano_id = gsv_ids.CaptureId.FromString(pano_name.decode())
    if pano_stitch_height is None:
      pano_stitch_height = target_height * 3
    pano_pixels, exposure_proto = self._get_pano_data(
        pano_id, pano_stitch_height
    )
    pano_t_ecef = _get_pano_pose(exposure_proto)
    pano_t_sampled = pano_t_ecef @ ecef_t_sampled
    intrinsics = types.Intrinsics(
        width=int(camera['width']),
        height=int(camera['height']),
        K=camera['K'],
        type=camera['type'].decode(),
        radial_distortion=camera['radial_distortion'],
        tangential_distortion=camera['tangential_distortion'],
        fov_deg=camera['fov_deg'],
        local_orientation=pano_t_sampled[:3, :3].T,
    )
    view = types.View(
        image_name='', cam_t_w=np.eye(4), camera=intrinsics, timestamp=0.0
    )
    scaled_view = view.scale_to_image_size(width=None, height=target_height)
    return scaled_view.sample_pixels_from_pano(pano_pixels)
