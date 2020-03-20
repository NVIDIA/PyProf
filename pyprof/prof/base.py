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

from abc import ABC, abstractmethod


class OperatorLayerBase(ABC):
    """
	Base class for all layers and operators.
	Every derived class should have the following functions.
	"""

    @abstractmethod
    def tc(self):
        """
		Tensor core usage by the kernel.
		Return "1" (yes), "0" (no, but possible), "-" (not applicable)
		"""
        pass

    @abstractmethod
    def params(self):
        """
		Kernel parameters to be printed.
		"""
        pass

    @abstractmethod
    def flops(self):
        """
		Note that 1 FMA = 2 flops.
		"""
        pass

    @abstractmethod
    def bytes(self):
        pass

    @abstractmethod
    def mod(self):
        """
		Name of the module/class e.g. torch.nn.functional.
		"""
        pass

    @abstractmethod
    def op(self):
        """
		Name of the operator e.g. sigmoid.
		"""
        pass
