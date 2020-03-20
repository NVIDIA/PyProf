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
"""
This simple file provides an example of how to
 - import the pyprof library and initialize it
 - use the emit_nvtx context manager
 - start and stop the profiler

Only kernels within profiler.start and profiler.stop calls are profiled.
To profile
$ nvprof -f -o simple.sql --profile-from-start off ./simple.py
"""

import sys
import torch
import torch.cuda.profiler as profiler

#Import and initialize pyprof
import pyprof
pyprof.init()

a = torch.randn(5, 5).cuda()
b = torch.randn(5, 5).cuda()

#Context manager
with torch.autograd.profiler.emit_nvtx():

    #Start profiler
    profiler.start()

    c = a + b
    c = torch.mul(a, b)
    c = torch.matmul(a, b)
    c = torch.argmax(a, dim=1)
    c = torch.nn.functional.pad(a, (1, 1))

    #Stop profiler
    profiler.stop()
