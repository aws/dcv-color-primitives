dcp_deps = [
  dcv_color_primitives_dep,
]

unit = executable(
  'unit',
  'unit.c',
  include_directories: dcp_inc,
  c_args: common_cflags,
  dependencies: dcp_deps
)

test('init', unit, args : ['unit_init'])
test('get_buffers_size', unit, args : ['unit_get_buffers_size'])
test('convert_image_rgb_to_bgra_ok', unit, args : ['unit_image_convert_rgb_to_bgra_ok'])
test('convert_image_bgra_to_rgb_ok', unit, args : ['unit_image_convert_bgra_to_rgb_ok'])
test('convert_image_bgr_to_rgb_ok', unit, args : ['unit_image_convert_bgr_to_rgb_ok'])
test('convert_image_rgb_to_yuv_errors', unit, args : ['unit_convert_image_rgb_to_yuv_errors'])
test('convert_image_yuv_to_rgb_errors', unit, args : ['unit_convert_image_yuv_to_rgb_errors'])
test('convert_image_rgb_to_nv12_p1_ok', unit, args : ['unit_convert_image_rgb_to_nv12_ok', '1'])
test('convert_image_rgb_to_nv12_p2_ok', unit, args : ['unit_convert_image_rgb_to_nv12_ok', '2'])
test('convert_image_rgb_to_i420_ok', unit, args : ['unit_convert_image_rgb_to_i420_ok'])
test('convert_image_rgb_to_i444_ok', unit, args : ['unit_convert_image_rgb_to_i444_ok'])
test('convert_image_nv12_to_rgb_ok', unit, args : ['unit_convert_image_nv12_to_rgb_ok'])
test('convert_image_i420_to_rgb_ok', unit, args : ['unit_convert_image_i420_to_rgb_ok'])
test('convert_image_over_4gb_limit', unit, args : ['unit_convert_image_over_4gb_limit'], is_parallel : false)
