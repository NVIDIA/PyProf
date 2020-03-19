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

Profiling PyTorch with PyProf
=============================

    TODO: this chapter should go into the details of profiling, 
    including any options.

.. _section-profile-enable-profiler:

Enable Profiler in PyTorch Network
----------------------------------

  *TODO:* provide more detail about `torch.cuda.profiler`, why it is needed
  and how to access it. The follow is cut and pasted from old README and needs
  to be expanded.

Add the following lines to your PyTorch network: ::

  import torch.cuda.profiler as profiler
  import pyprof
  pyprof.init()


Run the training/inference loop with the `PyTorch's NVTX context manager
<https://pytorch.org/docs/stable/_modules/torch/autograd/profiler.html#emit_nvtx>`_
with ``torch.autograd.profiler.emit_nvtx()``. Optionally, you can
use ``profiler.start()`` and ``profiler.stop()`` to pick an iteration
(say after warm-up) for which you would like to capture data.
Here's an example: ::

    iters = 500
    iter_to_capture = 100

    # Define network, loss function, optimizer etc.

    # PyTorch NVTX context manager
    with torch.autograd.profiler.emit_nvtx():

        for iter in range(iters):

            if iter == iter_to_capture:
                profiler.start()

            output = net(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            if iter == iter_to_capture:
                profiler.stop()

.. _section-profile-with-nvprof:

Profile with NVprof
-------------------

Run NVprof to generate a SQL (NVVP) file. This file can be opened with NVVP.

If using ``profiler.start()`` and ``profiler.stop()`` in ``net.py`` ::

  $ nvprof -f -o net.sql --profile-from-start off -- python net.py

For all other profiling ::

  $ nvprof -f -o net.sql -- python net.py

**Note:** if you're experiencing issues with hardware counters and you get 
a message such as ::

  **_ERR_NVGPUCTRPERM The user running <tool_name/application_name> does not 
  have permission to access NVIDIA GPU Performance Counters on the target device_**
  
Please follow the steps described in :ref:`section-profile-hardware-counters`.

.. _section-profile-with-nsys:

Profile with Nsight Systems
---------------------------

Run Nsight Systems to generate a sqlite file.

If using ``profiler.start()`` and ``profiler.stop()`` in ``net.py`` ::

  $ nsys profile -f true -o net -c cudaProfilerApi --stop-on-range-end true --export sqlite python net.py

For all other profiling ::

  $ nsys profile -f true -o net --export sqlite python net.py

.. _section-profile-hardware-counters:

Hardware Counters
-----------------

  *TODO:* Continue filling section in