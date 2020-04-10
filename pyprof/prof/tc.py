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


class TC_Whitelist:
    whitelist = ['h884', 's884', 'h1688', 's1688', 'hmma', 'i8816', '16816',
                 'dgrad_1x1_stride_2x2', 'first_layer_wgrad_kernel', 'conv1x1',
                 'conv2d_c1_k1', 'direct_group', 'xmma_implicit_gemm',
                 'xmma_sparse_conv', 'xmma_warp_specialized_implicit_gemm',
                 'xmma_gemm', 'xmma_sparse_gemm', 'c1688']
    def __contains__(self, item):
        for pattern in self.whitelist:
            if pattern in item:
                return True
        return False
