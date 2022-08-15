from setuptools import find_packages, setup, Extension

import numpy as np
from Cython.Build import cythonize


# description
with open("README.md", "r") as f:
    long_description = f.read()

# C-extensions
c_extensions = [
    Extension(
        name="fastrna.utils",
        sources=["fastrna/utils.pyx"],
    ),
    Extension(
        name="fastrna.mkl_funcs",
        sources=["fastrna/mkl_funcs.pyx"],
    ),
    Extension(
        name="fastrna.core",
        sources=["fastrna/core.pyx"],
    ),
]

setup(
    name="FastRNA",
    version="0.0.1",
    packages=find_packages(),
    author="Hanbin Lee",
    description="FastRNA for scalable scRNA-seq analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hanbin973/FastRNA",
    ext_modules=cythonize(c_extensions),
    include_dirs=np.get_include(),
    install_requires=[
        "numpy>=1.18.0",
        "scipy>=1.6.0",
		"mkl>=2021.3.0",
    ],
)
