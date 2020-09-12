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

class Convert(OperatorLayerBase):
    """
	Class to handle convert operations.
	"""
    ops = ["byte", "char", "double", "float", "half", "int", "long",
           "short", "to"]

    def __init__(self, d):
        marker = eval(d.argMarker[0])
        mod = marker['mod']
        op = marker['op']
        args = marker['args']

        self._mod = mod
        self._op = op

        assert (mod == "Tensor")
        assert (op in Convert.ops)
        assert (len(args) == 1)

        t = args[0]
        if t['type'] == "tensor":
            self.inp = Tensor(t['shape'], t['dtype'])
        else: # scalar
            self.inp = Tensor([], t['type'])

        if op == "to":
            # make the output the same as input
            self.out = self.inp
        else:
            self.out = Tensor(self.inp.shape, op)

    def params(self):
        return str(self.inp)

    def op(self):
        return self._op

    def mod(self):
        return self._mod

    def tc(self):
        return "-"

    def flops(self):
        return 0

    def bytes(self):
        return self.inp.bytes + self.out.bytes
