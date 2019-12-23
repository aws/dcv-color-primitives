project(
  'dcv-color-primitives-unit-tests', 'c',
  version : '0.1'
)

cc = meson.get_compiler('c')
host_system = host_machine.system()
arch = host_machine.cpu_family()
is_win = host_system == 'windows'

# Compiler flags
if cc.get_id() == 'msvc'
  # Make MSVC more pedantic, this is a recommended pragma list
  # from _Win32_Programming_ by Rector and Newcomer.  Taken from
  # glib's msvc_recommended_pragmas.h--please see that file for
  # the meaning of the warning codes used here
  test_cflags = [
    '-we4002', # too many actual parameters for macro
    '-we4003', # not enough actual parameters for macro
    '-w14010', # single-line comment contains line-continuation character
    '-we4013', # 'function' undefined; assuming extern returning int
    '-w14016', # no function return type; using int as default
    '-we4020', # too many actual parameters
    '-we4021', # too few actual parameters
    '-we4027', # function declared without formal parameter list
    '-we4029', # declared formal parameter list different from definition
    '-we4033', # 'function' must return a value
    '-we4035', # 'function' : no return value
    '-we4045', # array bounds overflow
    '-we4047', # different levels of indirection
    '-we4049', # terminating line number emission
    '-we4053', # an expression of type void was used as an operand
    '-we4071', # no function prototype given
    '-we4819', # the file contains a character that cannot be represented in the current code page
  ]
elif cc.get_id() == 'gcc' or cc.get_id() == 'clang'
  test_cflags = [
    '-Wpointer-arith',
    '-Wmissing-declarations',
    '-Wformat=2',
    '-Wstrict-prototypes',
    '-Wmissing-prototypes',
    '-Wnested-externs',
    '-Wold-style-definition',
    '-Wunused',
    '-Wno-uninitialized',
    '-Wshadow',
    '-Wcast-align',
    '-Wmissing-noreturn',
    '-Wmissing-format-attribute',
    '-Wlogical-op',
    '-Wno-discarded-qualifiers',
    '-Werror=implicit',
    '-Werror=nonnull',
    '-Werror=init-self',
    '-Werror=main',
    '-Werror=missing-braces',
    '-Werror=sequence-point',
    '-Werror=return-type',
    '-Werror=trigraphs',
    '-Werror=array-bounds',
    '-Werror=write-strings',
    '-Werror=address',
    '-Werror=int-to-pointer-cast',
    '-Werror=pointer-to-int-cast',
    '-Werror=empty-body',
    '-fno-strict-aliasing',
    '-Wno-int-conversion',
  ]
else
  test_cflags = []
endif

common_cflags = []
foreach cflag: test_cflags
  if cc.has_argument(cflag)
    common_cflags += [ cflag ]
  endif
endforeach

message('Building DCV Color Primitives')

if get_option('buildtype') == 'release'
  dcp_build_dir = 'release'
  cargo_target = 'build --release'
else
  dcp_build_dir = 'debug'
  cargo_target = 'build'
endif

run_command('cargo', cargo_target)

dcp_lib_path = meson.source_root() / 'target' / dcp_build_dir
dcp_deps = []

if is_win
  dcp_deps += [
    cc.find_library('dcv_color_primitives', dirs: [dcp_lib_path]),
    cc.find_library('ws2_32'),
    cc.find_library('userenv')
  ]
else
  dcp_deps += [
    cc.find_library('libdcv_color_primitives', dirs: [dcp_lib_path]),
    cc.find_library('pthread'),
    cc.find_library('dl'),
  ]
endif

unit = executable(
  'unit',
  'c_tests/unit.c',
  include_directories: [
    include_directories('include'),    
  ],
  c_args: common_cflags,
  dependencies: dcp_deps
)

test('init', unit, args : ['unit_init'])
test('get_buffers_size', unit, args : ['unit_get_buffers_size'])
test('convert_image_rgb_to_yuv_errors', unit, args : ['unit_convert_image_rgb_to_yuv_errors'])
test('convert_image_yuv_to_rgb_errors', unit, args : ['unit_convert_image_yuv_to_rgb_errors'])
test('convert_image_rgb_to_yuv_p1_ok', unit, args : ['unit_convert_image_rgb_to_yuv_ok', '1'])
test('convert_image_rgb_to_yuv_p2_ok', unit, args : ['unit_convert_image_rgb_to_yuv_ok', '2'])
test('convert_image_yuv_to_rgb_p1_ok', unit, args : ['unit_convert_image_yuv_to_rgb_ok', '1'])
test('convert_image_yuv_to_rgb_p2_ok', unit, args : ['unit_convert_image_yuv_to_rgb_ok', '2'])
test('convert_image_over_4gb_limit', unit, args : ['unit_convert_image_over_4gb_limit'], is_parallel : false)
