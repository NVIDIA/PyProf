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

class Dropout(OperatorLayerBase):

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
        assert (op == "dropout")

        self.inp = Tensor(args[0]['shape'], args[0]['dtype'])
        self.dir = d.dir

        return

    def params(self):
        return str(self.inp)

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

    def tc(self):
        return "-"

    def bytes(self):
        #Ignoring the cost of writing and reading the mask
        return self.inp.bytes * 2

    def flops(self):
        # Note: This is approximate and depends on the RNG
        return 5 * self.inp.size
