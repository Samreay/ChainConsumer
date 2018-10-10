#!/usr/bin/env python
import re
from setuptools import setup

# Synchronize version from code.
with open("chainconsumer/chainconsumer.py") as f:
    version = re.findall(r"__version__ = \"(.*?)\"", f.read())[0]

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
      install_requires=["numpy", "scipy", "matplotlib>1.6.0,!=2.1.*,!=2.2.*", "statsmodels>=0.7.0"])

