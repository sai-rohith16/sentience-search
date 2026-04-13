from pathlib import Path

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "sentience_engine",
        ["engine/knn.cpp"],  # Path to your C++ file
        cxx_std=17,
    ),
]

setup(
    name="sentience_engine",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    # ADD THESE TWO LINES BELOW TO FIX THE ERROR:
    packages=[], 
    py_modules=[],
)
