# setup.py
import numpy
from distutils.core import setup, Extension
from Cython.Build import cythonize


setup(ext_modules = cythonize(Extension(
    'CythonBFS',
    sources=['CythonBFS/non_recur_bfs.pyx', 'CythonBFS/queue.c'],
    language='c',
    include_dirs=[numpy.get_include(), "./cqueue.pxd"],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[]
)))