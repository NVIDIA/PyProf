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
import fused_adam_cuda
from apex.optimizers import FusedAdam, FP16_Optimizer
import pyprof

pyprof.init()
pyprof.wrap(fused_adam_cuda, 'adam')

model = torch.nn.Linear(10, 20).cuda().half()
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = FusedAdam(model.parameters())
optimizer = FP16_Optimizer(optimizer)

x = torch.ones(32, 10).cuda().half()
target = torch.empty(32, dtype=torch.long).random_(20).cuda()
y = model(x)
loss = criterion(y, target)
optimizer.zero_grad()
loss.backward()
optimizer.step()
