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


class Activation(OperatorLayerBase):
    """
	This class handles the various activation functions.
	"""

    ops = [
        "celu", "elu", "elu_", "hardshrink", "hardtanh", "hardtanh_", "leaky_relu", "leaky_relu_", "logsigmoid",
        "prelu", "relu", "relu_", "relu6", "rrelu", "rrelu_", "selu", "sigmoid", "softplus", "softshrink", "softsign",
        "tanh", "tanhshrink", "threshold", "threshold_"
    ]

    def __init__(self, d):
        marker = eval(d.argMarker[0])
        mod = marker['mod']
        op = marker['op']
        args = marker['args']

        self.mod_ = mod
        self.op_ = op

        assert (mod in ["torch.nn.functional", "torch", "Tensor"])

        #Filter out named parameters
        args = list(filter(lambda x: x['name'] == '', args))

        assert (len(args) >= 1)
        arg = args[0]
        assert (arg['type'] == "tensor")

        self.input = Tensor(arg['shape'], arg['dtype'])
        self.dir = d.dir

    def params(self):
        return str(self.input)

    def flops(self):
        # TODO: revise based on op
        return self.input.size

    def bytes(self):
        # TODO: revise based on op
        direction = self.dir
        b = self.input.bytes
        # fprop is 1 read, 1 write
        # bprop is 2 reads, 1 write
        b *= 2 if direction == "fprop" else 3
        return b

    def tc(self):
        return "-"

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_
