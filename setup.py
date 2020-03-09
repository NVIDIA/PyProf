#!/usr/bin/env python3

from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
	required = f.read().splitlines()

setup(
    name='pyprof',
    version='0.1',
    packages=find_packages(),
    author="Aditya Agrawal",
    author_email="aditya.iitb@gmail.com",
    maintainer="Elias Bermudez",
    maintainer_email="dbermudez13@gmail.com",
    url="https://github.com/NVIDIA/PyProf",
    license="BSD 3-Clause License",
    description='Pytorch profiler written by NVIDIA',
    install_requires=required,
)
