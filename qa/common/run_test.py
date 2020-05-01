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
'''
This is a test script to run the L0 tests.
'''
import unittest
import sys

test_dirs = ["run_pyprof_nvtx", "run_pyprof_data"]

runner = unittest.TextTestRunner(verbosity=2)

errcode = 0

for test_dir in test_dirs:
    suite = unittest.TestLoader().discover(test_dir)

    print("\nExecuting tests from " + test_dir)

    result = runner.run(suite)

    if not result.wasSuccessful():
        errcode = 1

sys.exit(errcode)
