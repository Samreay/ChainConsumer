#!/usr/bin/env python
import re
from setuptools import setup
import sys

# Synchronize version from code.
version = re.findall(r"__version__ = \"(.*?)\"", open("chainconsumer/chain.py").read())[0]

if "test" in sys.argv:
    version = "0.0.0"


setup(name="ChainConsumer",
      version=version,
      description="Consume chains and produce plots and tables",
      long_description="Package documentation: http://samreay.github.io/ChainConsumer",
      classifiers=["Development Status :: 4 - Beta",
                   "Programming Language :: Python :: 2",
                   "Programming Language :: Python :: 3",
                   "License :: OSI Approved :: MIT License",
                   "Topic :: Scientific/Engineering",
                   "Topic :: Scientific/Engineering :: Astronomy",
                   "Intended Audience :: Science/Research",
                   "Operating System :: OS Independent"],
      packages=["chainconsumer"],
      url="http://github.com/samreay/ChainConsumer",
      author="Samuel Hinton",
      author_email="samuelreay@gmail.com",
      requires=["numpy", "scipy", "matplotlib", "statsmodels"],
      install_requires=["numpy", "scipy", "matplotlib", "statsmodels"],
      tests_require=["pytest", "pytest-cov"]
)
