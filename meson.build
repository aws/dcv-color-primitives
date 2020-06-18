project(
  'dcv-color-primitives', 'c',
  license : 'MIT-0',
  version : run_command('build-aux' / 'getversion.py').stdout().strip(),
  meson_version: '>= 0.52.0'
)

cc = meson.get_compiler('c')
host_system = host_machine.system()
arch = host_machine.cpu_family()
is_win = host_system == 'windows'

prefix = get_option('prefix')
libdir = prefix / get_option('libdir')

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

common_cflags = cc.get_supported_arguments(test_cflags)

message('Building DCV Color Primitives')

if get_option('buildtype') == 'release'
  dcp_build_dir = 'release'
  cargo_target = ['build', '--release', ]
elif get_option('buildtype') == 'debugoptimized'
  dcp_build_dir = 'release'
  cargo_target = ['build', '--release', ]
else
  dcp_build_dir = 'debug'
  cargo_target = ['build', ]
endif

cargo_result = run_command('cargo', cargo_target)

message(cargo_result.stdout())
message(cargo_result.stderr())

if cargo_result.returncode() != 0
    error('-- cargo build failed')
endif

dcp_lib_path = meson.source_root() / 'target' / dcp_build_dir

if is_win
  static_lib_name = 'dcv_color_primitives.lib'
else
  static_lib_name = 'libdcv_color_primitives.a'
endif

install_data(
  dcp_lib_path / static_lib_name,
  install_dir: libdir
)

thread_dep = dependency('threads')
dcv_color_primitives_dep = cc.find_library('dcv_color_primitives', dirs: [dcp_lib_path])
libdl_dep = cc.find_library('dl', required: is_win == false)
ws2_32_dep = cc.find_library('ws2_32', required: is_win)
userenv_dep = cc.find_library('userenv', required: is_win)

clock_gettime_test_code = '''
  #include <time.h>
  struct timespec t;
  int main (int argc, char ** argv) {
    return clock_gettime(CLOCK_REALTIME, &t);
  }'''
librt_dep = []
if cc.links(clock_gettime_test_code, name : 'clock_gettime')
  # nothing to link to
elif cc.links(clock_gettime_test_code, args : '-lrt', name : 'clock_gettime in librt')
  librt_dep = cc.find_library('rt')
endif

subdir('include')
subdir('c_tests')

# Generate pc file
pkgg = import('pkgconfig')

pkgg.generate(libraries: '${libdir}' / static_lib_name,
              name: meson.project_name(),
              subdirs: meson.project_name(),
              description: 'DCV Color Primitives is a library to perform image color model conversion.')
