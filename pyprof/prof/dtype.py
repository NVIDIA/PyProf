# -*- coding: utf-8 -*-

# Copyright (c) 2020, Aditya Agrawal.
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

class Dtype(object):

    _types = {
        "uint8" : (1, "uint8"),
        "int8" : (1, "int8"),
        "byte" : (1, "byte"),
        "char" : (1, "char"),
        "bool" : (1, "bool"),

        "float16" : (2, "fp16"),
        "half" : (2, "fp16"),
        "int16" : (2, "int16"),
        "short" : (2, "int16"),

        "float32" : (4, "fp32"),
        "float" : (4, "fp32"),
        "int32" : (4, "int32"),
        "int" : (4, "int32"),

        "int64" : (8, "int64"),
        "long" : (8, "int64"),
        "float64" : (8, "fp64"),
        "double" : (8, "fp64"),
    }

    @staticmethod
    def types():
        t = Dtype._types.keys()
        return list(t)

    def __init__(self, dtype):
        assert dtype in Dtype.types()
        size, name = Dtype._types[dtype]
        self._itemsize = size
        self._name = name

    def __str__(self):
        return self._name

    @property
    def itemsize(self):
        return self._itemsize
        
def main():
    print(Dtype.types())
    for i in Dtype.types():
        dt = Dtype(i)
        print(i, dt, dt.itemsize)

if __name__ == '__main__':
    main()
