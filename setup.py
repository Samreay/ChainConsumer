#!/usr/bin/env python
import re
from setuptools import setup

# Synchronize version from code.
version = re.findall(r"__version__ = \"(.*?)\"", open("chain_consumer/chain.py").read())[0]

setup(name="ChainConsumer",
      version=version,
      description="ChainConsumer",
      long_description="Package documentation: http://samreay.github.io/ChainConsumer",
      classifiers=["Development Status :: 4 - Beta",
                   "Programming Language :: Python :: 2",
                   "Programming Language :: Python :: 3",
                   "License :: OSI Approved :: MIT License",
                   "Topic :: Scientific/Engineering",
                   "Topic :: Scientific/Engineering :: Astronomy",
                   "Intended Audience :: Science/Research"],
      packages=["chain_consumer"],
      include_package_data=True,
      url="http://github.com/samreay/ChainConsumer",
      author="Samuel Hinton",
      author_email="samuelreay@gmail.com",
      requires=['numpy', 'scipy', 'matplotlib', 'statsmodels'])
