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

from functools import reduce
from .dtype import Dtype

class Tensor(object):
    def __init__(self, shape, dtype):
        assert type(shape) in [tuple, list]
        assert dtype in Dtype.types()
        self._shape = list(shape)
        self._dtype = dtype

    def __str__(self):
        t = Dtype(self.dtype)
        return str(self.shape).replace(" ", "") + str(t)

    @property
    def ndim(self):
        # can be 0 for scalars
        return len(self._shape)

    @property
    def shape(self):
        # can be () for scalars
        return self._shape

    @property
    def size(self):
        # number of elements
        return reduce(lambda x, y: x * y, self.shape, 1)

    @property
    def dtype(self):
        return self._dtype

    @property
    def itemsize(self):
        return Dtype(self.dtype).itemsize

    @property
    def bytes(self):
        return self.size * self.itemsize

def main():
    for shape in [(), (1,), (3,7), (3,7,11)]:
        for dt in Dtype.types():
            t = Tensor(shape, dt)
            print(t.ndim, str(t.shape).replace(" ", ""), \
                    t.size, t.dtype, t.itemsize, t.bytes, t)

if __name__ == '__main__':
    main()
