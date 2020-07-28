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

* Clone the git repository ::
    
    $ git clone https://github.com/NVIDIA/PyProf.git

* Navigate to the top level PyProf directory

* Install PyProf ::

   $ pip install .

* Verify installation is complete with pip list ::

   $ pip list | grep pyprof 

* Should display ``pyprof 3.2.0``.
    