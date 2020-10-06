#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

with open('VERSION', 'r') as f:
    version = f.read()[:-4]

setup(
    name='nvidia-pyprof',
    version=version,
    packages=find_packages(),
    author="Aditya Agrawal,Marek Kolodziej",
    author_email="aditya.iitb@gmail.com,mkolod@gmail.com",
    maintainer="Elias Bermudez",
    maintainer_email="dbermudez13@gmail.com",
    url="https://github.com/NVIDIA/PyProf",
    download_url="https://github.com/NVIDIA/PyProf",
    license="BSD 3-Clause License",
    description='NVIDIA Pytorch Profiler',
    install_requires=required,
    classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Information Technology',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Utilities',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Environment :: Console',
            'Natural Language :: English',
            'Operating System :: OS Independent',
    ],
    keywords='nvidia, profiling, deep learning, ' \
             'machine learning, supervised learning, ' \
             'unsupervised learning, reinforcement learning, ',
    platforms=["Linux"],
)
