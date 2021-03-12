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

import numpy as np
from .base import OperatorLayerBase
from .tensor import Tensor
from functools import reduce
import operator

class Pointwise(OperatorLayerBase):

    # TODO: Add more operators.
    # TODO: Determining the output dtype is tricky.
    # TODO: Refine calculations based on direction.
    # TODO: Refine calculations for non-arithmetic ops.

    # Unary
    unary = ["abs", "abs_", "neg", "neg_", "reciprocal", "reciprocal_"]
    unary += ["__abs__", "__neg__"]

    # Unary bitwise
    unary += ["__invert__"]

    # Exponential and log (unary)
    exp_log = ["exp", "exp_", "exp1m", "exp1m_", "log", "log_",
               "log10", "log10_", "log1p", "log1p_", "log2", "log2_"]

    # Sqrt (unary)
    sqrt = ["rsqrt", "rsqrt_", "sqrt", "sqrt_"]

    # Representation (unary)
    representation = ["ceil", "ceil_", "clamp", "clamp_", "floor", "floor_",
                      "frac", "frac_", "round", "round_", "sign", "sign_",
                      "trunc", "trunc_"]

    # Trigonometric and transcendental (unary)
    trig_trans = ["acos", "acos_", "asin", "asin_", "atan", "atan_",
                  "atan2", "atan2_", "cos", "cos_", "cosh", "cosh_",
                  "sin", "sin_", "sinh", "sinh_", "tan", "tan_",
                  "sigmoid", "sigmoid_", "tanh", "tanh_"]

    # Error (unary)
    error = ["erf", "erf_", "erfc", "erfc_", "erfinv", "erfinv_"]

    # Binary
    binary = ["add", "add_", "div", "div_", "mul", "mul_",
              "remainder", "remainder_", "sub", "sub_"]
    binary += ["__add__", "__sub__", "__mul__", "__floordiv__",
               "__truediv__", "__mod__"]
    binary += ["__radd__", "__rsub__", "__rmul__", "__rdiv__",
               "__rtruediv__", "__rfloordiv__"]
    binary += ["fmod", "fmod_"]

    # Binary inplace
    ibinary = ["__iadd__", "__isub__", "__imul__", "__itruediv__"]

    # Power (binary)
    power = ["pow", "pow_", "__pow__", "__rpow__"]

    # Comparison (binary)
    comp = ["lt", "lt_", "gt", "gt_", "ge", "ge_", "le", "le_",
            "eq", "eq_", "ne", "ne_"]
    comp += ["__lt__", "__gt__", "__ge__", "__le__", "__eq__", "__ne__"]

    # Logical (binary)
    logical = ["__and__", "__or__", "__xor__", "__lshift__", "__rshift__"]

    # Logical inplace (binary)
    ilogical = ["__iand__", "__ior__", "__ixor__", "__ilshift__", "__irshift__"]

    # Ternary
    ternary = ["addcdiv", "addcdiv_", "addcmul", "addcmul_"]

    # Misc
    misc = ["digamma", "lerp", "lerp_", "mvlgamma"]

    ops = unary + binary + ibinary + comp + logical + ilogical + \
          ternary + exp_log + power + sqrt + representation + trig_trans + \
          error + misc

    def __init__(self, d):
        marker = eval(d.argMarker[0])
        mod = marker['mod']
        op = marker['op']
        args = marker['args']

        self.marker = marker
        self.mod_ = mod
        self.op_ = op
        self.args = args

        self.dir = d.dir
        assert (d.dir in ["fprop", "bprop"])
        assert (op in Pointwise.ops)

        # Filter out all named parameters (kwargs).
        # This might require revisiting in future.
        args = list(filter(lambda x: x['name'] == "", args))

        # Filter out non tensors
        #args = list(filter(lambda x: x['type'] == "tensor", args))

        assert (len(args) <= 4)
        self.input = []

        for arg in args:
            t = arg['type']
            if (t == "tensor"):
                tensor = Tensor(arg['shape'], arg['dtype'])
            elif t in ['float', 'int']:
                tensor = Tensor([], t)
            else:
                assert False

            self.input.append(tensor)

    def params(self):
        return ";".join([str(t) for t in self.input])

    def tc(self):
        return "-"

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

    def bytes_flops(self):
        b = f = 0

        # Unary
        if self.op() in Pointwise.unary + Pointwise.representation:
            # Relaxing assert. clamp has > 1 input arguments.
            assert (len(self.input) >= 1)
            b = 2 * self.input[0].bytes
            f = self.input[0].size

        elif self.op() in Pointwise.exp_log + Pointwise.trig_trans + \
                Pointwise.sqrt + Pointwise.error:
            assert (len(self.input) == 1)
            b = 2 * self.input[0].bytes
            f = self.input[0].size * 20 # estimate

        # Binary
        elif self.op() in Pointwise.comp + \
                Pointwise.binary + Pointwise.ibinary + \
                Pointwise.logical + Pointwise.ilogical:

            assert (len(self.input) == 2)
            out = Tensor.broadcast(self.input)

            if self.dir == "fprop":
                b = reduce(operator.add, [t.bytes for t in self.input])
                # The output of comparison is bool
                if self.op() in Pointwise.comp:
                    out = Tensor(out.shape, "bool")
                b += out.bytes
                f = out.size
            else:
                if (self.op() in ["add", "__add__", "sub", "__sub__", "__isub__"]):
                    b = 2 * out.bytes
                    f = 0
                elif (self.op() in ["__mul__", "__imul__", "__rmul__", "div", "__truediv__"]):
                    b = 3 * out.bytes
                    f = out.size
                else:
                    e = f'{self.op()} bprop not supported yet. Please file a bug.'
                    assert False, e

        elif self.op() in Pointwise.power:
            assert (len(self.input) == 2)
            out = Tensor.broadcast(self.input)
            b = reduce(operator.add, [t.bytes for t in self.input])
            b += out.bytes
            f = out.size * 20 # estimate

        # Ternary
        elif self.op() in Pointwise.ternary:
            # Remove scalars
            tensors = list(filter(lambda x: x.shape != [], self.input))
            assert len(tensors) == 3
            out = Tensor.broadcast(tensors)
            b = reduce(operator.add, [t.bytes for t in tensors])
            b += out.bytes
            f = 3 * out.size

        else:
            e = f'{self.op()} not supported yet. Please file a bug.'
            assert False, e

        return b, f

    def bytes(self):
        b, f = self.bytes_flops()
        return b

    def flops(self):
        b, f = self.bytes_flops()
        return f
