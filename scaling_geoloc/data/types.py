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

"""Type definitions for the data pipeline."""

from collections.abc import Sequence
import dataclasses
from typing import Any, Dict, Final, List, NamedTuple, Tuple, TypeAlias

from etils.array_types import Array
from etils.array_types import FloatArray
from etils.array_types import StrArray
import jax.numpy as jnp
import jaxcam
import numpy as np

from google3.cityblock.exposure import exposure_pb2
from google3.geo.imagery.streetview.store.proto import image_type_pb2
from google3.geo.imagery.streetview.store.shared.python import ids as gsv_ids
from scaling_geoloc.data import geo
from scaling_geoloc.utils import misc
from google3.location.visualmapping.experimental.utils.python import cbdb_utils
from google3.location.visualmapping.experimental.utils.python import svms_utils
from google3.location.visualmapping.ml.lentil.eval.common import image_resize
from google3.vision.sfm.wrappers.python import camera as vision_sfm

H1_IMAGE_HEIGHT: Final[int] = 5472
H1_IMAGE_WIDTH: Final[int] = 3648
STITCHED_CAMERA_INDEX: Final[int] = 999  # For stitched pano images.


SEMANTIC_INDEX_TO_NAME = {
    0: 'self',
    1: 'sky',
    2: 'building_exterior',
    3: 'tree',
    4: 'paved_road',
    5: 'unpaved_road',
    6: 'sidewalk',
    7: 'crosswalk',
    8: 'driveway',
    9: 'terrain',
    10: 'other_ground',
    11: 'other_structure',
    12: 'fence',
    13: 'wall',
    14: 'guard_rail',
    15: 'tunnel',
    16: 'bridge',
    17: 'bus_stop',
    18: 'phone_booth',
    19: 'traffic_sign',
    20: 'traffic_light',
    21: 'street_light',
    22: 'parking_meter',
    23: 'mailbox',
    24: 'fire_hydrant',
    25: 'pole',
    26: 'business_sign',
    27: 'billboard',
    28: 'other_permanent_object',
    29: 'other_temp_object',
    30: 'temp_traffic_sign',
    31: 'temp_cone',
    32: 'car',
    33: 'bus',
    34: 'truck',
    35: 'vehicle_on_rails',
    36: 'motorcycle',
    37: 'bicycle',
    38: 'other_vehicle',
    39: 'person',
    40: 'rider',
    41: 'animal',
    42: 'water',
    43: 'mountain',
    44: 'wire',
    45: 'building_interior',
    46: 'lidar_artifact',
    47: 'pedestrian_entrance',
    48: 'vehicle_entrance',
    255: 'unknown',
}


ROSETTE_CAMERA_NAMES: Final[Dict[str, List[str]]] = {
    'H1': [
        'front_right',
        'side_right',
        'rear_right',
        'rear_left',
        'side_left',
        'front_left',
    ],
    'R5': [
        'forward',
        'front_left',
        'side_left',
        'rear_left',
        'backward',
        'rear_right',
        'side_right',
        'front_right',
    ],
    'R7': [
        'forward',
        'front_left',
        'side_left',
        'rear_left',
        'backward',
        'rear_right',
        'side_right',
        'front_right',
        'slanted_camera1',
        'slanted_camera2',
    ],
}

# Source: https://g3doc.corp.google.com/util/geometry/g3doc/resources/s2cell_statistics.md?cl=head?  pytype: disable=line-too-long
CELL_EXTENTS: Final[Sequence[float]] = (
    7842000,
    5004000,
    2489000,
    1310000,
    636000,
    315000,
    156000,
    78000,
    39000,
    20000,
    10000,
    5000,
    2500,
    1225,
    613,
    306,
    160,
    80,
    40,
    20,
    10,
    5,
    2.5,
    1.25,
)


@dataclasses.dataclass(frozen=True)
class RunData:
  rosette_type: str
  vehicle_type: int
  scene_type: int

  @classmethod
  def from_run_metadata(
      cls, run_metadata: cbdb_utils.CbdbRunMetadata
  ) -> 'RunData':
    return cls(
        rosette_type=run_metadata.rosette_type,
        vehicle_type=run_metadata.vehicle_type,
        scene_type=run_metadata.scene,
    )


@dataclasses.dataclass
class Segment:
  """A segment is a sequence of panos from a single run.

  Panos have a name, timestamp, and ecef position.
  """

  run_name: str
  s2cell: str
  pano_names: StrArray['N']
  ecef_positions: FloatArray['N 3']
  timestamps: FloatArray['N']

  def asflatdict(self) -> Dict[str, Any]:
    return {
        'run_name': self.run_name,
        's2cell': self.s2cell,
        'pano_names': self.pano_names.tolist(),
        'ecef_positions': self.ecef_positions.tolist(),
        'timestamps': self.timestamps.tolist(),
    }


class ViewKey(NamedTuple):
  """Wrapper around frame names to extract information from parent levels.

  Inspired by:
  google3/geo/imagery/streetview/store/shared/ids.cc;l=287
  """

  image_name: str
  rosette_type: str
  sample_id: int = 0

  @property
  def camera_index(self) -> int:
    if not self.is_stitched:
      return int(self.image_name[-1])
    else:
      return STITCHED_CAMERA_INDEX

  @property
  def camera_name(self) -> str:
    return camera_index_to_name(self.rosette_type, self.camera_index)

  @property
  def run_name(self) -> str:
    return self.image_name.split(':')[1]

  @property
  def pano_name(self) -> str:
    return ':'.join(self.image_name.split(':')[:5])

  @property
  def image_id(self) -> gsv_ids.ImageId:
    return gsv_ids.ImageId.FromString(self.image_name)

  @property
  def pano_timestamp(self) -> float:
    return gsv_ids.CaptureId.FromString(self.pano_name).capture_timestamp_sec()

  @property
  def is_stitched(self) -> bool:
    return self.image_id.image_type() == image_type_pb2.STITCHED

  @property
  def unique_name(self) -> str:
    return self.image_name + ':' + str(self.sample_id)


@dataclasses.dataclass
class Intrinsics:
  """Camera intrinsics data."""

  width: int
  height: int
  type: str
  K: FloatArray['3 3']  # pylint: disable=invalid-name
  radial_distortion: FloatArray['3']
  tangential_distortion: FloatArray['2']
  fov_deg: float
  local_orientation: FloatArray['3 3'] | None = None

  @classmethod
  def from_exposure_pb2(
      cls, intrinsics: exposure_pb2.Intrinsics
  ) -> 'Intrinsics':
    """Extract intrinsics from the exposure intrinsics proto."""
    lens = intrinsics.lens
    width, height = lens.width, lens.height

    if lens.type == exposure_pb2.Lens.Type.PERSPECTIVE:
      lens_model = lens.perspective
    elif lens.type == exposure_pb2.Lens.Type.FISHEYE:
      lens_model = lens.fisheye
    elif lens.type == exposure_pb2.Lens.Type.FISHEYE_V2:
      lens_model = lens.fisheye_v2
    else:
      raise ValueError(f'Unsupported lens type: {lens.type}')

    # Focal length
    f = np.zeros(2)
    f[: len(lens_model.f)] = lens_model.f
    # Equal focal length
    f[1] = f[0] if len(lens_model.f) == 1 else f[1]

    # Center pixel
    c = np.zeros(2, dtype=int)
    c[: len(lens_model.c)] = lens_model.c
    # Convert center pixel coordinate frame
    c[1] = height - 1 - c[1]

    # Radial distortion
    k = np.zeros(3)
    k[: len(lens_model.k)] = lens_model.k

    # Tangential distortion
    p = np.zeros(2)
    if lens.type == exposure_pb2.Lens.Type.PERSPECTIVE:
      p[: len(lens.perspective.p)] = lens.perspective.p

    calibration_matrix = np.array([
        [f[0], 0, c[0]],
        [0, f[1], c[1]],
        [0, 0, 1],
    ])

    fov_deg = lens_model.fov_deg

    return cls(
        width=width,
        height=height,
        type=exposure_pb2.Lens.Type.Name(lens.type),
        K=calibration_matrix,
        radial_distortion=k,
        tangential_distortion=p,
        fov_deg=fov_deg,
    )

  @classmethod
  def spherical(cls, width: int, height: int) -> 'Intrinsics':
    """Create intrinsics for an equirectangular projection."""
    fx = width / np.pi
    fy = height / (2 * np.pi)
    K = np.eye(3, dtype=np.float32)  # pylint: disable=invalid-name
    K[0, 0] = fx
    K[1, 1] = fy
    K[:2, -1] = np.array([width / 2, height / 2]) + 0.5
    return cls(
        width=width,
        height=height,
        type=exposure_pb2.Lens.Type.Name(
            exposure_pb2.Lens.Type.EQUIRECTANGULAR
        ),
        K=K,
        radial_distortion=np.zeros(3),
        tangential_distortion=np.zeros(2),
        fov_deg=360,
    )

  @classmethod
  def from_jaxcam(cls, jcamera: jaxcam.Camera) -> 'Intrinsics':
    """Create intrinsics from a jaxcam camera."""
    intrinsic_matrix = np.array(jcamera.intrinsic_matrix, dtype=np.float64)
    width, height = np.array(jcamera.image_size, dtype=int).tolist()  # pylint: disable=unbalanced-tuple-unpacking
    return cls(
        width=width,
        height=height,
        K=intrinsic_matrix,
        type=exposure_pb2.Lens.Type.Name(exposure_pb2.Lens.Type.PERSPECTIVE),
        radial_distortion=np.array(jcamera.radial_distortion, dtype=np.float64)[
            :3
        ],
        tangential_distortion=np.array(
            jcamera.tangential_distortion, dtype=np.float64
        ),
        fov_deg=2
        * np.atan(width / (2 * intrinsic_matrix[0, 0])).item()
        * 180
        / np.pi,
        local_orientation=np.array(jcamera.orientation, dtype=np.float64),
    )

  def scale_to_image_size(
      self, width: int | None, height: int | None
  ) -> 'Intrinsics':
    """Scale the intrinsics to a new image size."""
    if not width and not height:
      raise ValueError('Either width or height must be specified.')
    elif not width:
      width = int(self.width / self.height * height)
    elif not height:
      height = int(self.height / self.width * width)

    scale_width = float(width) / self.width
    scale_height = float(height) / self.height

    K = np.copy(self.K)  # pylint: disable=invalid-name

    K[0, 0] *= scale_width
    K[1, 1] *= scale_height
    K[0, 2] *= scale_width
    K[1, 2] *= scale_height
    return dataclasses.replace(self, width=width, height=height, K=K)


@dataclasses.dataclass
class View:
  """A view is a single image from a pano, with optional RGB."""
  image_name: str
  cam_t_w: FloatArray['4 4']  # ecef coordinates
  camera: Intrinsics
  timestamp: float
  image_pixels: Array['H W 3'] | None = None

  def latlng_frustum(
      self, max_depth: float = 12.0, fov_deg: float = 90.0
  ) -> FloatArray['3 2']:
    cam_p_frustum = geo.camera_frustum_vertices(max_depth, fov_deg)
    w_p_frustum = geo.transform_points(
        np.linalg.inv(self.cam_t_w), cam_p_frustum
    )
    return geo.latlng_from_ecef(w_p_frustum)

  def latlng_position(self) -> FloatArray['1 2']:
    return geo.latlng_from_ecef(np.linalg.inv(self.cam_t_w)[None, :3, 3])

  def check_consistency(self) -> bool:
    """Check that the camera and pixels are consistent."""
    if self.image_pixels is not None:
      return (
          self.camera.width == self.image_pixels.shape[1]
          and self.camera.height == self.image_pixels.shape[0]
      )
    else:
      return True

  def is_pano_sample(self) -> bool:
    return (
        gsv_ids.ImageId.FromString(self.image_name).image_type()
        == image_type_pb2.STITCHED
    ) and self.camera.local_orientation is not None

  def sample_pixels_from_pano(
      self,
      pano_pixels: Array['HP WP 3'],
      mode: str = 'nearest',
      **interp_kwargs,
  ) -> Array['H W 3']:
    """Sample pixels from a pano."""
    jcamera = dataclasses.replace(
        self.to_jaxcam(),
        orientation=jnp.array(self.camera.local_orientation),
        position=jnp.zeros(3),
    )
    interp_kwargs['mode'] = mode
    if pano_pixels.dtype == np.uint8:
      pano_pixels = pano_pixels.astype(np.float32) / 255.0
      sampled_pixels = np.array(
          misc.img_from_pano(pano_pixels, jcamera, **interp_kwargs) * 255
      )
      return sampled_pixels.astype(np.uint8)
    else:
      return np.array(
          misc.img_from_pano(pano_pixels, jcamera, **interp_kwargs),
          dtype=pano_pixels.dtype,
      )

  def load_pixels(self) -> Array['H W 3']:
    """Convenience function to load the image pixels if not already loaded."""
    if self.image_pixels is None or True:
      image_id = gsv_ids.ImageId.FromString(self.image_name)
      image_pixels = svms_utils.GetImagePixels(image_id)
      if self.is_pano_sample():
        self.image_pixels = self.sample_pixels_from_pano(image_pixels)
      else:
        self.image_pixels = image_pixels

      self.image_pixels, _ = image_resize.resize_with_same_ratio(
          self.image_pixels, self.camera.height
      )
      if not self.check_consistency():
        raise ValueError(f'View {self.image_name} has inconsistent shapes.')
    return self.image_pixels

  def scale_to_image_size(
      self, width: int | None, height: int | None
  ) -> 'View':
    """Scale the intrinsics and image pixels to a new image size."""
    camera = self.camera.scale_to_image_size(width, height)
    if self.image_pixels is not None:
      image_pixels, _ = image_resize.resize_with_same_ratio(
          self.image_pixels, camera.height
      )
    else:
      image_pixels = None
    return dataclasses.replace(self, camera=camera, image_pixels=image_pixels)

  def to_visionsfm_camera(self) -> vision_sfm.Camera:
    """Convert to vision.sfm.Camera."""
    camera = vision_sfm.Camera(
        projection_type=vision_sfm.ProjectionType.FISHEYE
    )
    camera.SetCameraFromWorldTransform(self.cam_t_w)
    camera.SetRadialDistortion(
        self.camera.radial_distortion[0],
        self.camera.radial_distortion[1],
        self.camera.radial_distortion[2],
    )
    camera.SetTangentialDistortion(
        self.camera.tangential_distortion[0],
        self.camera.tangential_distortion[1],
    )

    camera.SetImageSize(self.camera.width, self.camera.height)
    camera.SetFocalLength(self.camera.K[0, 0])
    camera.SetPixelAspectRatio(self.camera.K[1, 1] / self.camera.K[0, 0])
    camera.SetPrincipalPoint(self.camera.K[0, 2], self.camera.K[1, 2])

    np.testing.assert_allclose(self.camera.K, camera.GetCalibrationMatrix())
    return camera

  def to_jaxcam(self) -> jaxcam.Camera:
    """Convert to jaxcam.Camera."""
    return jaxcam.Camera.create(
        image_size=jnp.array(
            [self.camera.width, self.camera.height], dtype=int
        ),
        focal_length=self.camera.K[0, 0],
        principal_point=self.camera.K[:2, -1],
        pixel_aspect_ratio=self.camera.K[1, 1] / self.camera.K[0, 0],
        radial_distortion=self.camera.radial_distortion,
        tangential_distortion=self.camera.tangential_distortion,
        orientation=self.camera.local_orientation,
        position=jnp.linalg.inv(self.cam_t_w)[:3, -1],
    )


@dataclasses.dataclass
class ViewCollectionMetadata:
  run_name: str
  s2cell: str
  rosette_type: str
  vehicle_type: int
  scene_type: int
  pano_split: str
  num_panos: int
  num_views: int


DataDict: TypeAlias = Dict[str, Any]
RunTuple: TypeAlias = Tuple[str, RunData]
SegmentTuple: TypeAlias = Tuple[str, RunData, Segment]
ViewDict: TypeAlias = Dict[ViewKey, View]
ViewCollection: TypeAlias = Tuple[str, ViewCollectionMetadata, ViewDict]


@dataclasses.dataclass
class Scene:
  s2cell: str
  query: ViewCollection
  map: Sequence[ViewCollection]
  overlap: FloatArray['M']  # Overlap from the query to all map collections
  ecef_t_w: FloatArray['4 4'] = dataclasses.field(
      default_factory=lambda: np.eye(4)
  )


def camera_index_to_name(rosette_type: str, camera_index: int) -> str:
  if camera_index == STITCHED_CAMERA_INDEX:
    return 'stitched'
  else:
    return ROSETTE_CAMERA_NAMES[rosette_type][camera_index]
