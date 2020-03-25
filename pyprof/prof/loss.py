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

#TODO: Add support for additional loss functions.


class MSELoss(OperatorLayerBase):

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
        assert (op == "mse_loss")
        assert (len(args) == 3)

        #Get input, target and reduction
        if (args[0]['name'] == ""):
            x = args[0]
        else:
            x = list(filter(lambda x: x['name'] == "input", args))[0]

        if (args[1]['name'] == ""):
            y = args[1]
        else:
            y = list(filter(lambda x: x['name'] == "target", args))[0]

        if (args[2]['name'] == ""):
            r = args[2]
        else:
            r = list(filter(lambda x: x['name'] == "reduction", args))[0]

        assert (x['type'] == y['type'] == "tensor")
        assert (x['shape'] == y['shape'])
        assert (x['dtype'] == y['dtype'])
        assert (r['type'] == "str")
        assert (r['value'] in ["none", "mean", "sum"])

        self.shape = x['shape']
        self.type = x['dtype']
        self.red = r['value']
        self.dir = d.dir

    def params(self):
        p = OrderedDict([('T', self.shape), ('type', self.type), ('red', self.red)])
        return p

    def elems(self):
        red = self.red
        e = Utility.numElems(self.shape)

        if self.dir == "fprop":
            if red == "none":
                e *= 3
            else:
                e *= 2
        else:
            if red == "none":
                e *= 4
            else:
                e *= 3
        return e

    def bytes(self):
        return self.elems() * Utility.typeToBytes(self.type)

    def flops(self):
        return self.elems() * 2 + 1

    def tc(self):
        return "-"

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_
