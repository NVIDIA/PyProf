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

* As :ref:`installable python code located in GitHub <section-installing-from-github>`.

* As a pre-built Docker container available from the `NVIDIA GPU Cloud (NGC) 
  <https://ngc.nvidia.com>`_. For more information, see :ref:`section-installing-from-ngc`.

* As a buildable docker container. You can :ref:`build your
  own container using Docker <section-quickstart-building-with-docker>`

.. _section-quickstart-prerequisites:

Prerequisites
-------------

* If you are installing directly from GitHub or building your own docker 
  container, you will need to clone the PyProf GitHub repo. Go to 
  https://github.com/NVIDIA/PyProf and then select the *clone* or *download* 
  drop down button. After cloning the repo be sure to select the r<xx.yy> 
  release branch that corresponds to the version of PyProf want to use::

  $ git checkout r21.04

* If you are starting with a pre-built NGC container, you will need to install 
  Docker and nvidia-docker. For DGX users, see `Preparing to use NVIDIA Containers
  <http://docs.nvidia.com/deeplearning/dgx/preparing-containers/index.html>`_.
  For users other than DGX, see the `nvidia-docker installation documentation 
  <https://github.com/NVIDIA/nvidia-docker>`_.

.. _section-quickstart-using-a-prebuilt-docker-container:

Using a Prebuilt Docker Containers
----------------------------------

Use docker pull to get the PyTorch container from NGC::

  $ docker pull nvcr.io/nvidia/pytorch:<xx.yy>-py3

Where <xx.yy> is the version of PyProf that you want to pull. Once you have the 
container, you can run the container with the following command::

  $ docker run --gpus=1 --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v/full/path/to/example/model/repository:/models <docker image>

Where <docker image> is *nvcr.io/nvidia/pytorch:<xx.yy>-py3*.

.. _section-quickstart-building-with-docker:

Building With Docker
--------------------

Make sure you complete the step in 
:ref:`section-quickstart-prerequisites` before attempting to build the PyProf 
container. To build PyProf from source, change to the root directory of
the GitHub repo and checkout the release version of the branch that
you want to build (or the `main` branch if you want to build the
under-development version)::

  $ git checkout r21.04

Then use docker to build::

  $ docker build --pull -t pyprof .

After the build completes you can run the container with the following command::

  $ docker run --gpus=1 --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v/full/path/to/example/model/repository:/models <docker image>

Where <docker image> is *pyprof*.

.. _section-quickstart-profile-with-pyprof:

Profile with PyProf
-------------------

.. include:: ../README.rst
   :start-after: quick-start-start-marker-do-not-remove
   :end-before: quick-start-end-marker-do-not-remove
