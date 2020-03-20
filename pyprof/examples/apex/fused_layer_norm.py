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
import fused_layer_norm_cuda
from apex.normalization import FusedLayerNorm
import pyprof

pyprof.init()
pyprof.wrap(fused_layer_norm_cuda, 'forward')
pyprof.wrap(fused_layer_norm_cuda, 'backward')
pyprof.wrap(fused_layer_norm_cuda, 'forward_affine')
pyprof.wrap(fused_layer_norm_cuda, 'backward_affine')

input = torch.randn(20, 5, 10, 10).cuda()

# With Learnable Parameters
m = FusedLayerNorm(input.size()[1:]).cuda()
output = m(input)

# Without Learnable Parameters
m = FusedLayerNorm(input.size()[1:], elementwise_affine=False).cuda()
output = m(input)

# Normalize over last two dimensions
m = FusedLayerNorm([10, 10]).cuda()
output = m(input)

# Normalize over last dimension of size 10
m = FusedLayerNorm(10).cuda()
output = m(input)
