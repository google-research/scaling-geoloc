# generate_retrieval_data
target_level = 14  # The target level to perform sampling.
filter_runs = {
  scene_types = 'OUTDOOR'  # 'INDOOR,OUTDOOR'
  vehicle_types = 'CAR'  # 'CAR,TREKKER,...'
}
create_test_set_for_year = 2023  # The year to create the test set for.
regenerate_test_set_only = false  # If true, only regenerate test set. A file views_per_s2cell_after_sampling.json is required in the output_prefix directory.
min_spacing_meters = 40.0  # The minimum spacing between panos.
max_panos_per_s2cell = 120  # The maximum number of panos per s2cell.
spatial_sampling = 'farthest_point_sampling'  # FPS or uniform sampling.
spatial_margin_fraction = 0.1  # Remove points that are <10% from edge.
pano_splits = null  # Split into left, right, front, back.
views_per_pano = 4  # Number of views rendered per pano.
sample_views_from_stitched_pano = true  # Sample views from stitched pano.
efficient_pano_loading = true  # Use efficient pano loading (recommended).
pano_sampling = {
  height = 224  # The height of the pano sampling. Should be same as image_height.
  width = 224  # The width of the pano sampling.
  roll_range = '-10,10'  # Degrees.
  pitch_range = '-5,15'  # Degrees.
  yaw_range = '-180,180'  # Degrees.
  hfov_range = '40,75'  # Degrees.
  stratify_yaw = true  # Stratified sampling (recommended).
}
max_views_per_s2cell = 1000000
min_views_per_s2cell = 5  # Avoid undersampled cells. 5 or 10 are good.
num_valid_per_s2cell = 1  # Number of valid panos per cell (1 is enough).
num_test_per_s2cell = '1,1'  # 1 test pano and 1 valid pano per cell.
max_num_valid_global = 10_000_000  # Max number of valid examples globally.
data = {
  image_height = 224  # The height of the image. Should be same as pano_sampling.height.
  pano_stitch_height = 768  # The height of the pano stitch. 2-3x image height.
  extract_pixels = true  # Load image pixels.
  extract_lidar_pointcloud = false  # Load global point cloud.
  extract_lidar_per_image = false  # Load per image lidar points.
  extract_semantic_raster = false  # Load semantic raster per image.
  add_lidar_semantics = false  # Load lidar semantics.
  geo_meta_data = 'country,province,county,metro'  # Add geo meta data.
  semantic_raster_model = 'MT_SV45_BEST'
  transform_to_local_coordinate_system = true  # To local-level frame.
  projected_lidar_point_coordinate_system = null  # Not recommended here.
  add_aerial = false  # Add aerial images (gets a tile per example!).
  aerial_per_query = true  # Use same tile per pano or per view.
  encoding_format = 'jpeg'  # The format of the image encoding (jpeg is good).
  aerial = {
    image_size = 256  # The size of the aerial image.
    extent_factors = '1.0'  # The extent factors of the aerial image (can be multiple)
    max_rotation_angle_degrees = 90  # The maximum rotation angle of the aerial image.
    max_offset = 0.5  # The maximum offset of the aerial image (in meters).
    scale_offset_by_extent = false  # Scale offset by extent (not recommended).
    seed = 42
    resolution_meters = 0.6  # The resolution of the aerial image (in meters). Extent factors are relative to this.
    encoding_format = null  # overwritten by data.encoding_format
  }
}

# generate_aerial_data
target_level = 16  # The target level (one areal image per cell).
image_size = 1024  #  edge length of the image in pixels.
extent_factors = '1.0'  # Extent factors. Can use multiple for pyramid.
num_samples = 1  # Number of samples to generate per cell (just keep 1).
max_rotation_angle_degrees = 0  # Rotation angle.
max_offset = 0  # Offset in meters.
scale_offset_by_extent = false
seed = 42
encoding_format = 'jpeg'
make_upright = true  # Make the image upright (nicer for comp. to Maps).
max_num_collections = 1  # How many collections to use per cell (use >1!).
resolution_meters = 0.4  # Resolution of the image in meters / pixel.
min_ratio_valid = 0.8  # Minimum ratio of valid pixels in the image.
add_aerial = true  # Add aerial images.
add_satellite = true  # Add satellite images.

# concat_and_join
split_names = 'train,valid,test'  # Comma separated list of split names.
filter_countries = 'BE,DE,NL'  # Comma separated list of countries to include!
aerial_prefix = null  # The prefix of the aerial data.
aerial_split_name = null  # The split name of the aerial data.
aerial_re_center = true  # Re-center the aerial to the center of the example.
export_aerial_test_set = true  # Export a test set of aerial crops.
aerial_target_level = 16  # The target level of the aerial data (controls amount of aerials in test set).
data = {
  target_level = 16  # The target level of the data. Usually not changed.
  image_size = 256  # The size of the aerial crop (output).
  extent_factors = '1.0'  # The extent factors of the aerial crop (output).
  num_samples = 1  # The number of aerial crops to generate per example (keep 1).
  max_rotation_angle_degrees = 90.0  # The maximum rotation angle of the aerial crop (degrees).
  max_offset = 80.0  # The maximum offset of the aerial crop (meters).
  scale_offset_by_extent = false  # Scale offset by extent (not recommended).
  seed = 42
  encoding_format = 'jpeg'  # The format of the image encoding (jpeg is good).
  make_upright = true  # Make the aerial crop upright (changes coordinates!)
  resolution_meters = 0.6  # The resolution of the aerial crop (output, meters / pixel).
}
