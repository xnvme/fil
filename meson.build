project('fil', 'c',
  version: '0.1.1',
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

cuda_dep = dependency('cuda', modules: ['cufile'])

xnvme_dep = dependency(
  'xnvme',
  version: '0.7.5',
  required: true
)

xal_dep = dependency(
  'xal',
  version: '0.2.0',
  required: true
)

fil_deps = [
  rt_dep,
  cuda_dep,
  xnvme_dep,
  xal_dep,
]

public_headers = files(
  'include/libfil.h',
)

sources = files(
  'src/iter.c',
  'src/io.c',
)

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
