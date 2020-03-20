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


def foo(x, y):
    return torch.sigmoid(x) + y


x = torch.zeros(4, 4).cuda()
y = torch.ones(4, 4).cuda()

#JIT the function using tracing
#This returns an object of type ScriptModule with a forward method.
traced_foo = torch.jit.trace(foo, (x, y))

#Initialize pyprof after the JIT step
pyprof.init()

#Assign a name to the object "traced_foo"
traced_foo.__dict__['__name__'] = "foo"

#Hook up the forward function to pyprof
pyprof.nvtx.wrap(traced_foo, 'forward')

with torch.autograd.profiler.emit_nvtx():
    profiler.start()
    z = traced_foo(x, y)
    profiler.stop()
    print(z)
