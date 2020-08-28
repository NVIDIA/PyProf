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

.. overview-begin-marker-do-not-remove

PyProf is a tool that profiles and analyzes the GPU performance of PyTorch
models. PyProf aggregates kernel performance from `Nsight Systems
<https://developer.nvidia.com/nsight-systems>`_ or `NvProf
<https://developer.nvidia.com/nvidia-visual-profiler>`_.

What's New in 3.3.0
-------------------

* Capture PyTorch API information and data loading configuration

* Added CUTLASS to the list of GEMM kernels

* Added optional function stack tracing to NVTX markers. Enable with ::

   pyprof.init(enable_function_stack=True)

Known Issues
------------

* Forward-Backward kernel correlation heuristics do not work correctly with 
  PyTorch 1.6. Recommended work arounds include:

  * Use with PyTorch 1.5
  * Use DLProf in the `20.08 NGC Pytorch container <https://ngc.nvidia.com/catalog/containers/nvidia:pytorch>`_

Features
--------

* Identifies the layer that launched a kernel: e.g. the association of 
  `ComputeOffsetsKernel` with a concrete PyTorch layer or API is not obvious.

* Identifies the tensor dimensions and precision: without knowing the tensor 
  dimensions and precision, it's impossible to reason about whether the actual 
  (silicon) kernel time is close to maximum performance of such a kernel on 
  the GPU. Knowing the tensor dimensions and precision, we can figure out the 
  FLOPs and bandwidth required by a layer, and then determine how close to 
  maximum performance the kernel is for that operation.

* Forward-backward correlation: PyProf determines what the forward pass step 
  is that resulted in the particular weight and data gradients (wgrad, dgrad), 
  which makes it possible to determine the tensor dimensions required by these
  backprop steps to assess their performance.
 
* Determines Tensor Core usage: PyProf can highlight the kernels that use 
  `Tensor Cores <https://developer.nvidia.com/tensor-cores>`_.
 
* Correlate the line in the user's code that launched a particular kernel (program trace).

.. overview-end-marker-do-not-remove

The current release of PyProf is 3.3.0 and is available in the 20.08 release of
the PyTorch container on `NVIDIA GPU Cloud (NGC) <https://ngc.nvidia.com>`_. The 
branch for this release is `r20.08
<https://github.com/NVIDIA/PyProf/tree/r20.08>`_.

Documentation
-------------

The User Guide can be found in the 
`documentation for current release 
<https://docs.nvidia.com/deeplearning/frameworks/pyprof-user-guide/index.html>`_, and 
provides instructions on how to install and profile with PyProf.

An `FAQ <https://docs.nvidia.com/deeplearning/frameworks/pyprof-user-guide/index.html/faqs.rst>`_ provides
answers for frequently asked questions.

The `Release Notes 
<https://docs.nvidia.com/deeplearning/frameworks/pyprof-release-notes/index.html>`_
indicate the required versions of the NVIDIA Driver and CUDA, and also describe 
which GPUs are supported by PyProf.

Contributing
------------

Contributions to PyProf are more than welcome. To
contribute make a pull request and follow the guidelines outlined in
the `Contributing <CONTRIBUTING.md>`_ document.

Reporting problems, asking questions
------------------------------------

We appreciate any feedback, questions or bug reporting regarding this
project. When help with code is needed, follow the process outlined in
the Stack Overflow (https://stackoverflow.com/help/mcve)
document. Ensure posted examples are:

* minimal – use as little code as possible that still produces the
  same problem

* complete – provide all parts needed to reproduce the problem. Check
  if you can strip external dependency and still show the problem. The
  less time we spend on reproducing problems the more time we have to
  fix it

* verifiable – test the code you're about to provide to make sure it
  reproduces the problem. Remove all other problems that are not
  related to your request/question.

.. |License| image:: https://img.shields.io/badge/License-Apache2-green.svg
   :target: http://www.apache.org/licenses/LICENSE-2.0
