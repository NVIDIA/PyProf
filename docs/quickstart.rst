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

.. _section-quickstart:

Quickstart
==========

PyProf is available in the following ways:

* As installable python code located in GitHub.

.. _section-quickstart-prerequisites:

Prerequisites
-------------

    TODO: List any prerequisites, including point to instructions on how to
    install either 

.. _section-quickstart-installing-from-github:

Installing from GitHub
----------------------

Make sure you complete the steps in :ref:`section-quickstart-prerequisites`
before attempting to install PyProf. See :ref:`section-installing-from-github`
for details on how to install from GitHub

.. _section-quickstart-profile-with-pyprof:

Profile with PyProf
-------------------

Add the following lines to the PyTorch network you want to profile: ::

  import torch.cuda.profiler as profiler
  import pyprof
  pyprof.init()

Profile with NVProf to generate a SQL (NVVP) file. This file can be opened 
with NVVP, as usual.

  *TODO:* Continue with quickstart. This should be brief, just the CLI with
  little explanation. It is a quickstart guide for reason.
