from setuptools import Extension, setup
import numpy

setup(
    ext_modules=[
        Extension(
            name="fil",
            sources=["filmodule.c"],
            include_dirs=[numpy.get_include()],
            extra_link_args=["-lfil"],
            libraries=["fil"]
        ),
    ]
)
