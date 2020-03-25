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

Overview
--------
For FLOP and bandwidth calculations, we use a relatively straightforward approach. 
For example, for matrices AMxK and BKxN, the FLOP count for a matrix multiplication is 
2 * M * N * K, and bandwidth is M * K + N * K + M * N. Note that the numbers PyProf 
generates are based on the algorithm, not the actual performance of the specific kernel. 
For more details, see `NVIDIA's Deep Learning Performance Guide 
<https://docs.nvidia.com/deeplearning/performance/index.html>`_.

Using the information provided by PyProf, the user can identify various issues to 
help tune the network. For instance, according to the `Tensor Core Performance Guide 
<https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#tensor-core-shape>`_, 
the M, N and K dimensions that result in Tensor Core usage need to be divisible by 8. 
In fact, PyProf comes with a flag that lets the user obtain information regarding 
whether Tensor Cores were used by the kernel. Other useful information might include 
knowing that a particular kernel did not exploit much thread parallelism, as 
determined by the grid/block dimensions. Since many PyTorch kernels are open-source 
(or even custom written by the user, as in CUDA Extensions), this provides the user 
with information that helps root cause performance issues and prioritize optimization work.


.. _section-profile-enable-profiler:

Enable Profiler in PyTorch Network
----------------------------------

  *TODO:* provide more detail about `torch.cuda.profiler`, why it is needed
  and how to access it. The follow is cut and pasted from old README and needs
  to be expanded.


Pyprof makes use of the profiler functionality available in `Pytorch
<https://pytorch.org/docs/stable/autograd.html#profiler>`_.
The profiler allows you to inspect the cost of different operators 
inside your model, both CPU and GPU, via the "emit_nvtx()" function.

To enable the profiler, you must add the following
lines to your PyTorch network: ::

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

If you are not using Nvprof, skip ahead to :ref:`section-profile-with-nsys`.

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

Run Nsight Systems to generate a SQLite file.

If using ``profiler.start()`` and ``profiler.stop()`` in ``net.py`` ::

  $ nsys profile -f true -o net -c cudaProfilerApi --stop-on-range-end true --export sqlite python net.py

For all other profiling ::

  $ nsys profile -f true -o net --export sqlite python net.py

.. _section-parse-sql-file:

Parse the SQL file
------------------
Run parser on the SQL file. The output is an ASCII file. Each line
is a python dictionary which contains information about the kernel name,
duration, parameters etc. This file can be used as input to other custom
scripts as well. **Note:** Nsys will create a file called net.sqlite. ::

    python -m pyprof.parse net.sqlite > net.dict
   
Run the prof script
-------------------
Using the python dictionary created in step 3 as the input, Pyprof can produce 
a CSV output, a columnated output (similar to `column -t` for terminal 
readability) and a space separated output (for post processing by AWK 
for instance). It produces 20 columns of information for every GPU kernel 
but you can select a subset of columns using the `-c` flag. 
Note that a few columns might have the value "na" implying either its a work 
in progress or the tool was unable to extract that information. Assuming 
the directory is `prof`, here are a few examples of how to use `prof.py`. ::

  # Print usage and help. Lists all available output columns.
    python -m pyprof.prof -h

  # Columnated output of width 150 with some default columns.
    python -m pyprof.prof -w 150 net.dict

  # CSV output.
    python -m pyprof.prof --csv net.dict

  # Space seperated output.
    python -m pyprof.prof net.dict

  # Columnated output of width 130 with columns index,direction,kernel name,parameters,silicon time.
    python -m pyprof.prof -w 130 -c idx,dir,kernel,params,sil net.dict

  # CSV output with columns index,direction,kernel name,parameters,silicon time.
    python -m pyprof.prof --csv -c idx,dir,kernel,params,sil net.dict

  # Space separated output with columns index,direction,kernel name,parameters,silicon time.
    python -m pyprof.prof -c idx,dir,kernel,params,sil net.dict

  # Input redirection.
    python -m pyprof.prof < net.dict

.. csv-table:: Options for prof.py
  :header: "Command", "Description"
  :widths: 25, 120

  "file", "Input file for prof.py. Generated by parse.py"
  "c", "See column option table below"
  "csv", "Print a csv output. Exclusively use --csv or -w"
  "w", "Width of columnated output. Exclusively use --csv or -w"
  
|

.. csv-table:: Column Options 
  :header: "Option", "Description"
  :widths: 25, 120
    
  "idx", "Index"
  "seq", "PyTorch Sequence Id"
  "altseq", "PyTorch Alternate Sequence Id"
  "tid", "Thread Id"
  "layer", "User annotated NVTX string (can be nested)"
  "trace", "Function Call Trace"
  "dir", "Direction"
  "sub", "Sub Sequence Id"
  "mod", "Module"
  "op", "Operattion"
  "kernel",   "Kernel Name"
  "params",   "Parameters"
  "sil", "Silicon Time (in ns)"
  "tc", "Tensor Core Usage"
  "device", "GPU Device Id"
  "stream", "Stream Id"
  "grid", "Grid Dimensions"
  "block", "Block Dimensions"
  "flops", "Floating point ops (FMA = 2 FLOPs)"
  "bytes", "Number of bytes in and out of DRAM"

The **default** options are "idx,dir,sub,mod,op,kernel,params,sil".

.. _section-profile-hardware-counters:

Hardware Counters
-----------------

Profiling GPU workloads may require access to hardware performance 
counters. Due to a fix in recent NVIDIA drivers addressing CVE‑2018‑6260, 
the hardware counters are disabled by default, and require elevated 
privileges to be enabled again. If you're using a recent driver, 
you may see the following message when trying to run nvprof:

**_ERR_NVGPUCTRPERM The user running <tool_name/application_name> does not have permission to access NVIDIA GPU Performance Counters on the target device._**

For details, see `here <https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters>`_.

*Permanent solution*

Follow the steps here. The current steps for Linux are: ::

  sudo systemctl isolate multi-user
  sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia-vgpu-vfio nvidia
  sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
  sudo systemctl isolate graphical

The above steps should result in a permanent change.

*Temporary solution*

When running on bare metal, you can run nvprof with sudo.

If you're running in a Docker image, you can temporarily elevate your 
privileges with one of the following (oldest to newest syntax): ::

  nvidia-docker run --privileged
  docker run --runtime nvidia --privileged
  docker run --gpus all --privileged
