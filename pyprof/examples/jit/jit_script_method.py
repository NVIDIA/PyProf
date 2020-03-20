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


class Foo(torch.jit.ScriptModule):

    def __init__(self, size):
        super(Foo, self).__init__()
        self.n = torch.nn.Parameter(torch.ones(size))
        self.m = torch.nn.Parameter(torch.ones(size))

    @torch.jit.script_method
    def forward(self, input):
        return self.n * input + self.m


#Initialize pyprof after the JIT step
pyprof.init()

#Hook up the forward function to pyprof
pyprof.nvtx.wrap(Foo, 'forward')

foo = Foo(4)
foo.cuda()
x = torch.ones(4).cuda()

with torch.autograd.profiler.emit_nvtx():
    profiler.start()
    z = foo(x)
    profiler.stop()
    print(z)
