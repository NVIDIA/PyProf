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

from functools import reduce


class Utility(object):

    @staticmethod
    def numElems(shape):
        assert (type(shape) == tuple)
        return reduce(lambda x, y: x * y, shape, 1)

    @staticmethod
    def typeToBytes(t):
        if (t in ["uint8", "int8", "byte", "char", "bool"]):
            return 1
        elif (t in ["float16", "half", "int16", "short"]):
            return 2
        elif (t in ["float32", "float", "int32", "int"]):
            return 4
        elif (t in ["int64", "long", "float64", "double"]):
            return 8
        assert False

    @staticmethod
    def typeToString(t):
        if (t in ["uint8", "byte", "char"]):
            return "uint8"
        elif (t in [
                "int8",
        ]):
            return "int8"
        elif (t in [
                "int16",
                "short",
        ]):
            return "int16"
        elif (t in ["float16", "half"]):
            return "fp16"
        elif (t in ["float32", "float"]):
            return "fp32"
        elif (t in [
                "int32",
                "int",
        ]):
            return "int32"
        elif (t in ["int64", "long"]):
            return "int64"
        elif (t in [
                "float64",
                "double",
        ]):
            return "fp64"
        elif (t in [
                "bool",
        ]):
            return "bool"
        assert False

    @staticmethod
    def hasNVTX(marker):
        if type(marker) is str:
            try:
                marker = eval(marker)
            except:
                return False

        if type(marker) is dict:
            keys = marker.keys()
            return ("mod" in keys) and ("op" in keys) and ("args" in keys)
        else:
            return False

    @staticmethod
    def isscalar(t):
        return (t in ["float", "int"])
