from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

#setup(
#    ext_modules = cythonize("profiling.py")
#)

setup(
    cmdclass = {"build_ext": build_ext},
    ext_modules = [Extension("testing",
                             ["profiling.py"],
                             extra_compile_args=["-fopenmp"],
                             extra_link_args=["-fopenmp"])]
)
