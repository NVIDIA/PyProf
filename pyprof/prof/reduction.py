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

from collections import OrderedDict
from .utility import Utility
from .base import OperatorLayerBase
from .tensor import Tensor


class Mean(OperatorLayerBase):

    def __init__(self, d):
        marker = eval(d.argMarker[0])
        mod = marker['mod']
        op = marker['op']
        args = marker['args']

        self.mod_ = mod
        self.op_ = op

        assert (mod in ["torch", "Tensor"])
        assert (op == "mean")

        #Filter out named parameters
        args = list(filter(lambda x: x['name'] == '', args))

        assert (len(args) <= 2)
        i = args[0]

        # The input can be a scalar or a tensor
        if 'shape' in i:  # tensor
            self.input = Tensor(i['shape'], i['dtype'])
        else:  # scalar
            assert ('value' in i)
            self.input = Tensor([], i['type'])

        self.dir = d.dir
        self.sub = d.sub

    def params(self):
        return str(self.input)

    def tc(self):
        return "-"

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

    def bytes(self):
        if self.sub == 0:
            return self.input.bytes + self.input.itemsize
        else:
            return 0

    def flops(self):
        if self.sub == 0:
            return self.input.size + 1
        else:
            return 0


class Sum(OperatorLayerBase):

    def __init__(self, d):
        marker = eval(d.argMarker[0])
        mod = marker['mod']
        op = marker['op']
        args = marker['args']

        self.marker = marker
        self.mod_ = mod
        self.op_ = op
        self.args = args

        assert (mod in ["torch", "Tensor"])
        assert (op == "sum")
        assert (len(args) >= 1)

        #Get input
        if (args[0]['name'] == ""):
            i = args[0]
        else:
            i = list(filter(lambda x: x['name'] == "input", args))[0]

        self.shape = i['shape']
        self.type = i['dtype']
        self.sub = d.sub

    def params(self):
        p = OrderedDict([('T', self.shape), ('type', self.type)])
        return p

    def tc(self):
        return "-"

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

    def elems(self):
        return Utility.numElems(self.shape)

    def flops(self):
        # Note: This is incorrect, need to calculate actual flops (say via nvprof)
        return self.elems()

    def bytes(self):
        b = self.elems() * Utility.typeToBytes(self.type)
        if self.sub == 0:
            return b
        else:
            return 0


class Norm(OperatorLayerBase):

    def __init__(self, d):
        marker = eval(d.argMarker[0])
        mod = marker['mod']
        op = marker['op']
        args = marker['args']

        self.marker = marker
        self.mod_ = mod
        self.op_ = op
        self.args = args

        assert (mod in ["torch", "Tensor"])
        assert (op == "norm")
        #assert (len(args) == 1)
        i = args[0]
        self.shape = i['shape']
        self.type = i['dtype']
        self.sub = d.sub

    def params(self):
        p = OrderedDict([('T', self.shape), ('type', self.type)])
        return p

    def elems(self):
        return Utility.numElems(self.shape)

    def bytes(self):
        b = self.elems() * Utility.typeToBytes(self.type)
        if self.sub == 0:
            return b
        else:
            return 0

    def flops(self):
        # square and add plus sqrt
        f = 2 * self.elems() + 1
        if self.sub == 0:
            return f
        else:
            return 0

    def tc(self):
        return "-"

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_
