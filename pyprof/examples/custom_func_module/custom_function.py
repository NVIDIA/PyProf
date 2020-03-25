#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import torch
import torch.cuda.profiler as profiler
import pyprof
#Initialize pyprof
pyprof.init()


class Foo(torch.autograd.Function):

    @staticmethod
    def forward(ctx, in1, in2):
        out = in1 + in2  #This could be a custom C/C++ function.
        return out

    @staticmethod
    def backward(ctx, grad):
        in1_grad = grad  #This could be a custom C/C++ function.
        in2_grad = grad  #This could be a custom C/C++ function.
        return in1_grad, in2_grad


#Hook the forward and backward functions to pyprof
pyprof.nvtx.wrap(Foo, 'forward')
pyprof.nvtx.wrap(Foo, 'backward')

foo = Foo.apply

x = torch.ones(4, 4).cuda()
y = torch.ones(4, 4).cuda()

with torch.autograd.profiler.emit_nvtx():
    profiler.start()
    z = foo(x, y)
    profiler.stop()
