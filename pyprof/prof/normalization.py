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

from .base import OperatorLayerBase
from .tensor import Tensor

class BatchNorm(OperatorLayerBase):

    def __init__(self, d):
        marker = eval(d.argMarker[0])
        mod = marker['mod']
        op = marker['op']
        args = marker['args']

        self._mod = mod
        self._op = op

        assert (op == "batch_norm")
        assert (len(args) >= 1)
        i = args[0]
        assert (i['type'] == "tensor")

        self.inp = Tensor(i['shape'], i['dtype'])
        self.dir = d.dir
        self.sub = d.sub

    def params(self):
        return str(self.inp)

    def tc(self):
        return "-"

    def op(self):
        return self._op

    def mod(self):
        return self._mod

    def flops(self):
        # Variance algo-dependent, but this is a reasonable value.
        return self.inp.size * 8

    def bytes(self):
        b = self.inp.bytes
        if self.dir == "fprop":
            b *= 4
        else:
            b *= 5

        if self.sub > 0:
            return 0
        else:
            return b
