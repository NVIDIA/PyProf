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

.. _section-install:

Installing PyProf
=================

PyProf is available from GitHub.

.. _section-installing-from-github:

Installing from GitHub
----------------------

.. include:: ../README.rst
   :start-after: quick-install-start-marker-do-not-remove
   :end-before: quick-install-end-marker-do-not-remove

.. _section-installing-from-ngc:

Install from NGC Container
--------------------------

PyProf is available in the PyTorch container on the `NVIDIA GPU Cloud (NGC) 
<https://ngc.nvidia.com>`_.

Before you can pull a container from the NGC container registry, you
must have Docker and nvidia-docker installed. For DGX users, this is
explained in `Preparing to use NVIDIA Containers Getting Started Guide
<http://docs.nvidia.com/deeplearning/dgx/preparing-containers/index.html>`_.
For users other than DGX, follow the `nvidia-docker installation
documentation <https://github.com/NVIDIA/nvidia-docker>`_ to install
the most recent version of CUDA, Docker, and nvidia-docker.

After performing the above setup, you can pull the PyProf container
using the following command::

  docker pull nvcr.io/nvidia/pytorch:20.11-py3

Replace *20.11* with the version of PyTorch container that you want to pull.
