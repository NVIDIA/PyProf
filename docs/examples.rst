..
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

.. _section-examples:

Examples
========

This section provides several real examples on how to profile with PyPRrof.

Profile Lenet
-------------

Navigate to the lenet example. ::

  $ cd pyprof/examples

Run nsight systems to profile the network. ::

  $ nsys profile -f true -o lenet --export sqlite python lenet.py

Parse the resulting lenet.sqlite database. ::

  $ python -m pyprof.parse lenet.sqlite > lenet.dict

Run the prof script on the resulting dictionary. ::

  $ python -m pyprof.prof --csv lenet.dict > lenet.csv
