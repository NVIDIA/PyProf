# -*- coding: utf-8 -*-

# Copyright (c) 2020, Aditya Agrawal.
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

def readMarker(d):
    marker = eval(d.argMarker[0])
    return marker['mod'], marker['op'], marker['args']

class OneZero(OperatorLayerBase):

    ops = ["ones", "ones_like", "zero_", "zeros", "zeros_like"]

    def __init__(self, d):
        mod, op, args = readMarker(d)
        assert mod in ["torch", "Tensor"]
        assert op in OneZero.ops

        self.mod_ = mod
        self.op_ = op

        assert(len(args) == 1)
        arg = args[0]
        self.input = Tensor(arg['shape'], arg['dtype'])

    def params(self):
        return str(self.input)

    def tc(self):
        return "-"

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

    def bytes(self):
        return self.input.bytes

    def flops(self):
        return 0

class Fill(OperatorLayerBase):

    def __init__(self, d):
        mod, op, args = readMarker(d)
        assert mod == "Tensor"
        assert op == "fill_"

        self.mod_ = mod
        self.op_ = op

        assert(len(args) == 2)
        arg = args[0]
        self.input = Tensor(arg['shape'], arg['dtype'])

    def params(self):
        return str(self.input)

    def tc(self):
        return "-"

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

    def bytes(self):
        return self.input.bytes

    def flops(self):
        return 0

class Full(OperatorLayerBase):

    def __init__(self, d):
        mod, op, args = readMarker(d)
        assert mod == "torch"
        assert op == "full"

        self.mod_ = mod
        self.op_ = op

        assert(len(args) == 2)
        arg1, arg2 = args
        assert arg1['type'] in ['list', 'tuple']
        # TODO: Add more types for arg2
        assert arg2['type'] in ['float', 'int']
        self.output = Tensor(arg1['value'], arg2['type'])

    def params(self):
        return str(self.output)

    def tc(self):
        return "-"

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

    def bytes(self):
        return self.output.bytes

    def flops(self):
        return 0
