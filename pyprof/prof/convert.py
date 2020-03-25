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


class Convert(OperatorLayerBase):
    """
	Class to handle convert operations.
	"""
    ops = ["byte", "char", "double", "float", "half", "int", "long", "short", "to"]

    def __init__(self, d):
        marker = eval(d.argMarker[0])
        mod = marker['mod']
        op = marker['op']
        args = marker['args']

        self.marker = marker
        self.mod_ = mod
        self.op_ = op
        self.args = args

        assert (mod == "Tensor")
        assert (op in Convert.ops)
        assert (len(args) == 1)

        #The argument could be a tensor or scalar
        t = args[0]
        if t['type'] == "tensor":
            shape = t['shape']
            stype = t['dtype']
        else:
            shape = (1, )
            stype = t['type']
        if self.op_ == "to":
            op = stype

        self.shape = shape
        self.stype = stype
        self.dtype = op

    def params(self):
        p = OrderedDict([('T', self.shape), ('stype', self.stype), ('dtype', self.dtype)])
        return p

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

    def tc(self):
        return "-"

    def elems(self):
        return Utility.numElems(self.shape)

    def flops(self):
        return 0

    def bytes(self):
        b = self.elems() * (Utility.typeToBytes(self.stype) + Utility.typeToBytes(self.dtype))
        return b
