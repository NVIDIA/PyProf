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


class Embedding(OperatorLayerBase):

    def __init__(self, d):
        marker = eval(d.argMarker[0])
        mod = marker['mod']
        op = marker['op']
        args = marker['args']

        self.mod_ = mod
        self.op_ = op

        assert (mod == "torch.nn.functional")
        assert (op == "embedding")

        input = args[0]
        embedding = args[1]

        self.input = Tensor(input['shape'], input['dtype'])
        self.embedding = Tensor(embedding['shape'], embedding['dtype'])

        assert (len(self.embedding.shape) == 2)

        self.dir = d.dir
        self.sub = d.sub
        return

    def params(self):
        return str(self.input) + ";" + str(self.embedding)

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

    def tc(self):
        return "-"

    def bytes(self):
        b = 0
        if self.dir == "fprop":
            # read indices
            b += self.input.bytes
            # read and write the embedding values
            b += 2 * self.input.size * self.embedding.shape[1] * self.embedding.itemsize
        else:
            # 3 times the size of the incoming gradient
            b = 3 * self.input.size * self.embedding.shape[1] * self.embedding.itemsize

            if self.sub > 0:
                b = 0

        return b

    def flops(self):
        # Note: not implemented yet
        return 0
