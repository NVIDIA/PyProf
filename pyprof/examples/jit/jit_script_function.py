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

#The following creates an object "foo" of type ScriptModule
#The new object has a function called "forward"


@torch.jit.script
def foo(x, y):
    return torch.sigmoid(x) + y


#Initialize pyprof after the JIT step
pyprof.init()

#Assign a name to the object "foo"
foo.__name__ = "foo"

#Hook up the forward function to pyprof
pyprof.nvtx.wrap(foo, 'forward')

x = torch.zeros(4, 4).cuda()
y = torch.ones(4, 4).cuda()

with torch.autograd.profiler.emit_nvtx():
    profiler.start()
    z = foo(x, y)
    profiler.stop()
    print(z)
