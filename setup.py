from setuptools import setup, Extension
import sys
import os
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Get Eigen path in current conda environment
eigen_include_dir = os.path.join(sys.prefix, "include", "eigen3")

ext_modules = [
    Pybind11Extension(
        "_kernel_JSPqMT",
        ["src/_kernel_JSPqMT.cpp"],
        include_dirs=[
            pybind11.get_include(),
            eigen_include_dir
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    ),
]

setup(
    name="_kernel_JSPqMT",
    version="0.1",
    description="Python binding for func_JSPqMT using pybind11 and Eigen",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)