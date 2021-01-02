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


class Softmax(OperatorLayerBase):

    def __init__(self, d):
        marker = eval(d.argMarker[0])
        mod = marker['mod']
        op = marker['op']
        args = marker['args']

        self.mod_ = mod
        self.op_ = op

        assert (mod == "torch.nn.functional")
        assert (op == "softmax")

        #Filter out named parameters
        args = list(filter(lambda x: x['name'] == '', args))

        assert (len(args) <= 2)
        arg = args[0]
        self.input = Tensor(arg['shape'], arg['dtype'])
        self.dir = d.dir
        return

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

    def tc(self):
        return "-"

    def params(self):
        return str(self.input)

    def flops(self):
        # An approximation
        # http://ai.stanford.edu/~paskin/slam/javadoc/javaslam/util/Flops.html#exp()
        # TODO: consider direction
        e = self.input.size
        f = e * 20  # denominator, exp all elements and reduce
        f += e * 20  # numerator, exp all elements and divide
        return f

    def bytes(self):
        # TODO: verify
        b = self.input.bytes
        # fprop is 2 reads, 1 write
        # bprop is 4 reads, 1 write
        b *= 3 if self.dir == "fprop" else 5
        return b


class LogSoftmax(OperatorLayerBase):

    def __init__(self, d):
        marker = eval(d.argMarker[0])
        mod = marker['mod']
        op = marker['op']
        args = marker['args']

        self.mod_ = mod
        self.op_ = op

        assert (mod in ["torch", "Tensor", "torch.nn.functional"])
        assert (op == "log_softmax")

        #Filter out named parameters
        args = list(filter(lambda x: x['name'] == '', args))

        assert (len(args) <= 2)

        #Get input
        if (args[0]['name'] == ""):
            i = args[0]
        else:
            i = list(filter(lambda x: x['name'] == "input", args))[0]

        self.input = Tensor(i['shape'], i['dtype'])
        self.dir = d.dir
        return

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

    def tc(self):
        return "-"

    def params(self):
        return str(self.input)

    def flops(self):
        # An approximation
        # http://ai.stanford.edu/~paskin/slam/javadoc/javaslam/util/Flops.html#exp()
        # TODO: consider direction
        e = self.input.size
        f = e * 20  # denominator, exp all elements and reduce
        f += e  # numerator, just a subtraction
        return f

    def bytes(self):
        # TODO: verify
        b = self.input.bytes
        # fprop is 2 reads, 1 write
        # bprop is 4 reads, 1 write
        b *= 3 if self.dir == "fprop" else 5
        return b
