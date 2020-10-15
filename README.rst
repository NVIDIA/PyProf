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

|License|

PyProf - PyTorch Profiling tool
===============================

    **NOTE: You are currently on the r20.11 branch which tracks stabilization
    towards the release. This branch is not usable during stabilization.**

.. overview-begin-marker-do-not-remove

.. overview-end-marker-do-not-remove

Quick Installation Instructions
-------------------------------

.. quick-install-start-marker-do-not-remove

* Clone the git repository ::
    
    $ git clone https://github.com/NVIDIA/PyProf.git

* Navigate to the top level PyProf directory

* Install PyProf ::

   $ pip install .

* Verify installation is complete with pip list ::

   $ pip list | grep pyprof 

* Should display ::

   pyprof            3.6.0.dev0

.. quick-install-end-marker-do-not-remove

Quick Start Instructions
------------------------

.. quick-start-start-marker-do-not-remove

* Add the following lines to the PyTorch network you want to profile: ::

    import torch.cuda.profiler as profiler
    import pyprof
    pyprof.init()

* Profile with NVProf or Nsight Systems to generate a SQL file. ::

    $ nsys profile -f true -o net --export sqlite python net.py

* Run the parse.py script to generate the dictionary. ::
  
    $ python -m pyprof.parse net.sqlite > net.dict

* Run the prof.py script to generate the reports. ::

    $ python -m pyprof.prof --csv net.dict

.. quick-start-end-marker-do-not-remove

.. |License| image:: https://img.shields.io/badge/License-Apache2-green.svg
   :target: http://www.apache.org/licenses/LICENSE-2.0
