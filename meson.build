project('fil', 'c',
  version: '0.3.4',
  meson_version: '>=0.55.0',
  default_options: [
    'c_std=gnu11',
    'warning_level=2',
    'buildtype=release',
    'default_library=both',
    'b_staticpic=true',
  ]
)

pkg = import('pkgconfig')
cc = meson.get_compiler('c')

rt_dep = cc.find_library('rt', required: true)
if not cc.has_function('clock_gettime', prefix: '#include <time.h>', dependencies: rt_dep)
  error('clock_gettime not found')
endif

# CUDA gives the aisio-gpu and cufile backends and copy_to_gpu; without it the
# others build, and fil_init() rejects those
cuda_opt = get_option('cuda')
have_cuda = add_languages('cuda', required: cuda_opt, native: false)
cuda_dep = dependency('cuda', modules: ['cuda', 'cufile'], required: cuda_opt)
have_cuda = have_cuda and cuda_dep.found()
if have_cuda
  # Targets the local GPU architecture; change for cross-arch builds
  add_project_arguments('-arch=native', language: 'cuda')
  add_project_arguments('-DFIL_CUDA_ENABLED=1', language: ['c', 'cuda'])
endif

# The sub-times are of the aisio-gpu path, and are taken with CUDA events
if get_option('subtime')
  if not have_cuda
    error('subtime needs CUDA')
  endif
  add_project_arguments('-DFIL_SUBTIME=1', language: 'c')
endif

xnvme_dep = dependency(
  'xnvme',
  version: '0.7.5',
  required: true
)

xal_dep = dependency(
  'xal',
  version: '0.3.3',
  required: true
)

fil_deps = [
  rt_dep,
  xnvme_dep,
  xal_dep,
]
if have_cuda
  fil_deps += cuda_dep
endif

public_headers = files(
  'include/libfil.h',
)

sources = files(
  'src/iter.c',
  'src/io.c',
)
if have_cuda
  sources += files('src/gpu_io.cu')
endif

include_dirs = include_directories('include')

fil_library = library('fil',
  sources: sources,
  dependencies: fil_deps,
  include_directories: include_dirs,
  install: true
)

fil_cli = executable('filperf',
  sources: 'src/cli.c',
  link_with: fil_library,
  include_directories: include_dirs,
  install: true
)

pkg.generate(fil_library)

install_headers(public_headers)
