import os
from setuptools import find_packages, setup, Extension

import numpy as np
from numpy.distutils.system_info import get_info
from Cython.Build import cythonize


# libraries
mkl_info = get_info("mkl")
libs = ["mkl_rt"]

lib_dirs = mkl_info.get("library_dirs")
include_dirs = mkl_info.get("include_dirs")

# compiler option
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"
flags = ["-O3", "-fopenmp", "-xhost"]


print(mkl_info)
print(include_dirs)
print(lib_dirs)

# description
with open("README.md", "r") as f:
    long_description = f.read()

# C-extensions
c_extensions = [
    Extension(
        name="fastrna.utils",
        sources=["fastrna/utils.pyx"],
        extra_compile_args=flags,
        extra_link_args=flags,
    ),
    Extension(
        name="fastrna.mkl_funcs",
        sources=["fastrna/mkl_funcs.pyx"],
        include_dirs=include_dirs,
        libraries=libs,
        library_dirs=lib_dirs,
        extra_compile_args=flags,
        extra_link_args=flags,
    ),
    Extension(
        name="fastrna.core",
        sources=["fastrna/core.pyx"],
        include_dirs=include_dirs,
        libraries=libs,
        library_dirs=lib_dirs,
        extra_compile_args=flags,
        extra_link_args=flags,
		cython_directives = {"embedsignature": True},
    ),
]

setup(
    name="FastRNA",
    version="0.1.0",
    packages=find_packages(),
    author="Hanbin Lee",
	author_email="hanbin973@gmail.com",
    description="FastRNA for scalable scRNA-seq analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hanbin973/FastRNA",
    ext_modules=cythonize(c_extensions),
    include_dirs=np.get_include(),
    install_requires=[
        "numpy",
        "scipy",
        "mkl",
		"Cython",
    ],
)
