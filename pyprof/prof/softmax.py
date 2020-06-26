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


class Softmax(OperatorLayerBase):

    def __init__(self, d):
        marker = eval(d.argMarker[0])
        mod = marker['mod']
        op = marker['op']
        args = marker['args']

        self.marker = marker
        self.mod_ = mod
        self.op_ = op
        self.args = args

        assert (mod == "torch.nn.functional")
        assert (op == "softmax")

        #Filter out named parameters
        args = list(filter(lambda x: x['name'] == '', args))

        assert (len(args) <= 2)
        self.shape = args[0]['shape']
        self.type = args[0]['dtype']
        self.dir = d.dir

        return

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

    def tc(self):
        return "-"

    def params(self):
        p = OrderedDict([('T', self.shape), ('type', self.type)])
        return p

    def elems(self):
        return Utility.numElems(self.shape)

    def flops(self):
        # Note: exp, sum-reduce, divide
        #flops = elems * 3
        return 0

    def bytes(self):
        b = self.elems() * Utility.typeToBytes(self.type)
        b *= 3 if self.dir == "fprop" else 5  #verify
        return b


class LogSoftmax(OperatorLayerBase):

    def __init__(self, d):
        marker = eval(d.argMarker[0])
        mod = marker['mod']
        op = marker['op']
        args = marker['args']

        self.marker = marker
        self.mod_ = mod
        self.op_ = op
        self.args = args

        assert (mod in ["torch", "torch.nn.functional"])
        assert (op == "log_softmax")

        #Filter out named parameters
        args = list(filter(lambda x: x['name'] == '', args))

        assert (len(args) <= 2)

        #Get input
        if (args[0]['name'] == ""):
            i = args[0]
        else:
            i = list(filter(lambda x: x['name'] == "input", args))[0]

        t = i['dtype']

        self.shape = i['shape']
        self.type = i['dtype']
        self.dir = d.dir
        return

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

    def tc(self):
        return "-"

    def params(self):
        p = OrderedDict([('T', self.shape), ('type', self.type)])
        return p

    def elems(self):
        return Utility.numElems(self.shape)

    def flops(self):
        # Note: exp, sum-reduce, divide, log
        #flops = elems * 4
        return 0

    def bytes(self):
        b = self.elems() * Utility.typeToBytes(self.type)
        b *= 3 if self.dir == "fprop" else 5  #verify
        return b
