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
import numpy as np
from .base import OperatorLayerBase
from .tensor import Tensor
from functools import reduce
import operator


class Cat(OperatorLayerBase):

    def __init__(self, d):
        marker = eval(d.argMarker[0])
        mod = marker['mod']
        op = marker['op']
        args = marker['args']
        self.mod_ = mod
        self.op_ = op

        assert (mod == "torch")
        assert (op == "cat")
        assert (len(args) >= 2)

        dtype = args[0]['dtype']
        tensors = []

        # Get all tensor arguments
        args = filter(lambda x: x['type'] == "tensor", args)

        for arg in args:
            assert (arg['dtype'] == dtype)
            t = Tensor(arg['shape'], dtype)
            tensors.append(t)

        self.input = tensors
        self.sub = d.sub

    def params(self):
        return ";".join([str(t) for t in self.input])

    def flops(self):
        return 0

    def tc(self):
        return "-"

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

    def bytes(self):
        # 1 read, 1 write
        b = 2 * reduce(operator.add, [t.bytes for t in self.input])
        return b if (self.sub == 0) else 0


class Reshape(OperatorLayerBase):

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
        assert (op == "reshape")

        #Temporarily commenting three lines
        #assert (len(args) == 2)
        #t,s = args
        #assert s['type'] == "tuple"

        t = args[0]
        assert t['type'] == "tensor"
        self.type = t['dtype']
        self.shape = t['shape']

    def params(self):
        p = OrderedDict([('T', self.shape), ('type', self.type)])
        return p

    def flops(self):
        return 0

    def tc(self):
        return "-"

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

    def bytes(self):
        return 0


class Gather(OperatorLayerBase):

    def __init__(self, d):
        marker = eval(d.argMarker[0])
        mod = marker['mod']
        op = marker['op']
        args = marker['args']

        self.marker = marker
        self.mod_ = mod
        self.op_ = op
        self.args = args

        assert (mod == "Tensor") or (mod == "torch")
        assert (op == "gather")

        #Filter out the "out" parameter
        args = list(filter(lambda x: x['name'] != 'out', args))
        assert (len(args) == 3)

        #Get input
        if (args[0]['name'] == ""):
            arg = args[0]
        else:
            arg = list(filter(lambda x: x['name'] == "input", args))[0]

        assert (arg['type'] == "tensor")

        self.shape = arg['shape']
        self.type = arg['dtype']

    def params(self):
        p = OrderedDict([('T', self.shape), ('type', self.type)])
        return p

    def flops(self):
        return 0

    def tc(self):
        return "-"

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

    def bytes(self):
        return 2 * Utility.numElems(self.shape) * Utility.typeToBytes(self.type)


class MaskedScatter(OperatorLayerBase):

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
        assert (op == "masked_scatter_")
        assert (len(args) == 3)

        dst, mask, src = args
        assert (dst['type'] == mask['type'] == src['type'] == "tensor")
        assert (mask['dtype'] == "uint8")
        assert (dst['dtype'] == src['dtype'])
        assert (dst['shape'] == mask['shape'])

        self.shape = dst['shape']
        self.type = dst['dtype']
        self.seqId = d.seqId
        self.sub = d.sub

    def params(self):
        p = OrderedDict([('T', self.shape), ('type', self.type)])
        return p

    def flops(self):
        return 0

    def tc(self):
        return "-"

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

    def bytes(self):
        elems = Utility.numElems(self.shape)

        #src and dst
        b = 2 * elems * Utility.typeToBytes(self.type)

        #mask (uint8)
        b += elems

        if (self.sub > 0):
            b = 0
        return b


class Nonzero(OperatorLayerBase):

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
        assert (op == "nonzero")
        assert (len(args) == 1)

        arg = args[0]
        self.shape = arg['shape']
        self.type = arg['dtype']
        self.seqId = d.seqId
        self.sub = d.sub

    def params(self):
        p = OrderedDict([('T', self.shape), ('type', self.type)])
        return p

    def flops(self):
        return 0

    def tc(self):
        return "-"

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

    def bytes(self):
        elems = Utility.numElems(self.shape)
        dim = len(self.shape)

        #input tensor
        b = elems * Utility.typeToBytes(self.type)

        #in the worst case, the output is a (elems x dim) tensor of type "long"
        b += elems * dim * Utility.typeToBytes("int64")

        if self.sub > 0:
            return 0
        else:
            return b


class IndexSelect(OperatorLayerBase):

    def __init__(self, d):
        marker = eval(d.argMarker[0])
        mod = marker['mod']
        op = marker['op']
        args = marker['args']

        self.marker = marker
        self.mod_ = mod
        self.op_ = op
        self.args = args

        assert (mod == "Tensor") or (mod == "torch")
        assert (op == "index_select")

        #Filter out the "out" parameter
        args = list(filter(lambda x: x['name'] != 'out', args))
        assert (len(args) == 3)

        #Get input, dim and index
        if (args[0]['name'] == ""):
            t = args[0]
        else:
            t = list(filter(lambda x: x['name'] == "input", args))[0]

        if (args[1]['name'] == ""):
            d = args[1]
        else:
            d = list(filter(lambda x: x['name'] == "dim", args))[0]

        if (args[2]['name'] == ""):
            i = args[2]
        else:
            i = list(filter(lambda x: x['name'] == "index", args))[0]

        assert (t['type'] == i['type'] == "tensor")
        assert (d['type'] == "int")
        assert (i['dtype'] == "int64")
        assert (len(i['shape']) == 1)

        shape = t['shape']
        dim = d['value']
        indices = i['shape'][0]
        assert (dim < len(shape))

        self.shape = shape
        self.dim = dim
        self.indices = indices
        self.type = t['dtype']

    def params(self):
        p = OrderedDict([('T', self.shape), ('D', self.dim), ('I', self.indices), ('type', self.type)])
        return p

    def tc(self):
        return "-"

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

    def flops(self):
        return 0

    def bytes(self):
        #determine the shape of the output tensor
        shape = list(self.shape)
        shape[self.dim] = self.indices
        shape = tuple(shape)

        b = 0

        #time to read the input and write the output
        elems = Utility.numElems(shape)
        b += 2 * elems * Utility.typeToBytes(self.type)

        #time to read the indices
        b += self.indices * Utility.typeToBytes("int64")

        return b


class MaskedSelect(OperatorLayerBase):

    def __init__(self, d):
        marker = eval(d.argMarker[0])
        mod = marker['mod']
        op = marker['op']
        args = marker['args']

        self.marker = marker
        self.mod_ = mod
        self.op_ = op
        self.args = args
        self.sub = d.sub

        assert (mod == "Tensor") or (mod == "torch")
        assert (op == "masked_select")

        #Filter out the "out" parameter
        args = list(filter(lambda x: x['name'] != 'out', args))
        assert (len(args) == 2)

        #Get input and mask
        if (args[0]['name'] == ""):
            t = args[0]
        else:
            t = list(filter(lambda x: x['name'] == "input", args))[0]

        if (args[1]['name'] == ""):
            m = args[1]
        else:
            m = list(filter(lambda x: x['name'] == "mask", args))[0]

        assert (m['dtype'] == "uint8")

        tensor = t['shape']
        mask = m['shape']

        #check for broadcast condition
        if (tensor != mask):
            array1 = np.empty(list(tensor))
            array2 = np.empty(list(mask))
            try:
                out = np.broadcast(array1, array2).shape
            except:
                assert False

        self.tshape = tensor
        self.mshape = mask
        self.type = t['dtype']

    def params(self):
        p = OrderedDict([('T', self.tshape), ('M', self.mshape), ('type', self.type)])
        return p

    def tc(self):
        return "-"

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

    def bytes(self):
        tensor = self.tshape
        mask = self.mshape
        t = self.type

        #in the worst case, #output elements = #input elements
        b = 2 * Utility.numElems(tensor) * Utility.typeToBytes(t)

        #mask tensor (assuming uint8)
        b += Utility.numElems(mask)
        return b

    def flops(self):
        return 0
