from setuptools import setup, Extension
import sys
import os
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Get Eigen path in current conda environment
eigen_include_dir = os.path.join(sys.prefix, "include", "eigen3")
nlopt_include_dir = os.path.join(sys.prefix, "include")
nlopt_library_dir = os.path.join(sys.prefix, "lib")

ext_modules = [
    Pybind11Extension(
        "opt_JSPqMT",
        sources=[
            "src/opt_JSPqMT.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            eigen_include_dir,
            nlopt_include_dir
        ],
        library_dirs=[nlopt_library_dir],
        libraries=["nlopt"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    ),
]

setup(
    name="opt_JSPqMT",
    version="0.1",
    description="Python binding for func_JSPqMT using pybind11 and Eigen",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)