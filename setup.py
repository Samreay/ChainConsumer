#!/usr/bin/env python
import re
from setuptools import setup
import sys
from setuptools.command.test import test

# Synchronize version from code.
version = re.findall(r"__version__ = \"(.*?)\"", open("chainconsumer/chain.py").read())[0]

if "test" in sys.argv:
    version = "0.0.0"


# Using framework from emcee. Pattern credit to Daniel Foreman-Mackey
class PyTest(test):
    user_options = [("pytest-args=", "a", "Arguments to pass to py.test")]

    def initialize_options(self):
        test.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        test.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(name="ChainConsumer",
      version=version,
      description="Consume chains and produce plots and tables",
      long_description="Package documentation: http://samreay.github.io/ChainConsumer",
      classifiers=["Development Status :: 5 - Production/Stable",
                   "Programming Language :: Python :: 2",
                   "Programming Language :: Python :: 3",
                   "License :: OSI Approved :: MIT License",
                   "Topic :: Scientific/Engineering",
                   "Topic :: Scientific/Engineering :: Astronomy",
                   "Intended Audience :: Science/Research"
                   "Operating System :: OS Independent"],
      packages=["chainconsumer"],
      url="http://github.com/samreay/ChainConsumer",
      author="Samuel Hinton",
      author_email="samuelreay@gmail.com",
      requires=requirements,
      tests_require=["pytest","pytest-cov"],
      cmdclass={"test": PyTest},
)
