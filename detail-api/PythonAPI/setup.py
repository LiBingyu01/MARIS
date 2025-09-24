from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

ext_modules = [
    Extension(
        'detail._mask',
        sources=['../common/maskApi.c', 'detail/_mask.pyx'],
        include_dirs = [np.get_include(), '../common'],
        extra_compile_args=[],#'-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    )
]

setup(name='detail',
      packages=['detail'],
      package_dir = {'detail': 'detail'},
      version='4.0',
      ext_modules=
          cythonize(ext_modules)
      )
