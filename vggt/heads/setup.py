import glob
import os
import platform
import shutil
import sys
import warnings
from os import path as osp
from pkg_resources import DistributionNotFound, get_distribution
from setuptools import find_packages, setup


if __name__ == '__main__':
    setup(
        name='cubifyanything',
        version='0.0.1',
        description=("Public release of Cubify Anything"),
        author='Apple Inc.',
        author_email='jlazarow@apple.com',
        url='https://github.com/apple/ml-cubifyanything',
        packages=find_packages(),
        include_package_data=True,
        zip_safe=False)
