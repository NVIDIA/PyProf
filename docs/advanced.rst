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

Advanced PyProf Usage 
=====================

This section demonstrates some advanced techniques to get even more from your
PyProf profiles.

.. _section-layer-annotation:

Layer Annotation
----------------

Adding custom NVTX ranges to the model layers will allow PyProf to aggregate
profile results based on the ranges. ::

  # examples/user_annotation/resnet.py
  # Use the “layer:” prefix
  
  class Bottleneck(nn.Module):
    def forward(self, x):
      nvtx.range_push("layer:Bottleneck_{}".format(self.id))  # NVTX push marker
      
      nvtx.range_push("layer:Conv1")                          # Nested NVTX push/pop markers
      out = self.conv1(x)
      nvtx.range_pop()
      
      nvtx.range_push("layer:BN1")                            # Use the “layer:” prefix
      out = self.bn1(out)
      nvtx.range_pop()
      
      nvtx.range_push("layer:ReLU")
      out = self.relu(out)
      nvtx.range_pop()
      
      ...
      
      nvtx.range_pop()                                        # NVTX pop marker.return out

.. _section-custom-function:

Custom Function
---------------

The following is example of how to enable Torch Autograd to profile a custom
function. ::

  # examples/custom_func_module/custom_function.py
  
  import torch
  import pyprof
  pyprof.init()
  
  class Foo(torch.autograd.Function):
    @staticmethoddef forward(ctx, in1, in2):
      out = in1 + in2                    # This could be a custom C++ function
      return out
    @staticmethod
    def backward(ctx, grad):
      in1_grad, in2_grad = grad, grad    # This could be a custom C++ function
      return in1_grad, in2_grad
  
  # Hook the forward and backward functions to pyprof
  pyprof.wrap(Foo, 'forward')
  pyprof.wrap(Foo, 'backward')

.. _section-custom-module:

Custom Module
---------------

The following is example of how to enable Torch Autograd to profile a custom
module. ::

  # examples/custom_func_module/custom_module.py
  
  import torch
  import pyprof
  pyprof.init()
  
  class Foo(torch.nn.Module):
    def __init__(self, size):
      super(Foo, self).__init__()
      self.n = torch.nn.Parameter(torch.ones(size))
      self.m = torch.nn.Parameter(torch.ones(size))
      
    def forward(self, input):
      return self.n*input + self.m         # This could be a custom C++ function.
  
  # Hook the forward function to pyprof
  pyprof.wrap(Foo, 'forward')

Extensibility
-------------

* For custom functions and modules, users can add flops and bytes calculation

* Python code is easy to extend - no need to recompile, no need to change the 
  PyTorch backend and resolve merge conflicts on every version upgrade

Actionable Items
----------------

The following list provides some common actionable items to consider when 
analyzing profile results and deciding on how best to improve the performance. 
For more customized and directed actionable items, consider using the `NVIDIA 
Deep Learning Profiler <https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/index.html>`_ 
that provide direct *Expert Systems* feedback based on the profile.

* NvProf/ NsightSystems tell us what the hotspots are, but not if we can act on 
  them.

* If a kernel runs close to max perf based on FLOPs and bytes (and maximum FLOPs
  and bandwidth of the GPU), then there’s no point in optimizing it even if it’s
  a hotspot.
  
* If the ideal timing based on FLOPs and bytes (max(compute_time, 
  bandwidth_time)) is much shorter than the silicon time, there’s scope for 
  improvement.
  
* Tensor Core usage (conv): for Volta, convolutions should have the input 
  channel count (C) and the output channel count (K) divisible by 8, in order to
  use tensor cores. For Turing, it’s optimal for C and K to be divisible by 16. 
  
* Tensor core usage (GEMM): M, N and K divisible by 8 (Volta) or 16 (Turing) (https://docs.nvidia.com/deeplearning/sdk/dl-performance-guide/index.html)  
