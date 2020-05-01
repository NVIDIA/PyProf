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
This test exercises the tracemarker get_func_stack() functionality
'''
import inspect
import unittest

import pyprof


class TestPyProfFuncStack(unittest.TestCase):

    def __init__(self, testName):
        super().__init__(testName)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def compare_funcstack(self, actual_tracemarker, expected_str):
        # Given a funcstack string, remove TestPyProfFuncStack::run and everything above it
        #
        def remove_test_class_hierarchy(x):
            separator = "/"
            fn_split = x.split(separator)
            for i, n in enumerate(fn_split):
                if (n == "TestPyProfFuncStack::run"):
                    fn_split = fn_split[i+1:]
                    break

            joined = separator.join(fn_split)
            return joined

        tracemarker_dict = eval(actual_tracemarker)
        actual_func_stack = remove_test_class_hierarchy(tracemarker_dict["funcStack"])
        self.assertEqual(expected_str,actual_func_stack)

    # Basic function hierarchy test
    # Function stack is func1->func2->func3->ignored.
    # Test that the leaf function's funcStack is TestPyProfFuncStack::test_basic/func1/func2/func3
    #
    def test_basic(self):
        def ignored():
            tracemarker = pyprof.nvtx.nvmarker.traceMarker()
            self.compare_funcstack(tracemarker,"TestPyProfFuncStack::test_basic/func1/func2/func3")
            
        def func3():
            ignored()

        def func2():
            func3()

        def func1():
            func2()

        func1()

    # Test that 'wrapper_func' is ignored in hierarchy
    # Function stack is func1->func2->ExecutableClass::__call__->outer_func1->outer_func2->outer_ignored
    # Test that the leaf function's funcStack is TestPyProfFuncStack::test_ignore_wrapper_func/func1/func2/func3
    #
    def test_ignore_wrapper_func(self):
           
        def ignored():
            tracemarker = pyprof.nvtx.nvmarker.traceMarker()
            self.compare_funcstack(tracemarker,"TestPyProfFuncStack::test_ignore_wrapper_func/func1/func2/func3")

        def func3():
            ignored()

        def wrapper_func():
            func3()

        def func2():
            wrapper_func()

        def func1():
            func2()

        func1()

    # Test that '__call__' is ignored in hierarchy
    # Function stack is func1->func2->abc::__call__->func3->ignored.
    # Test that the leaf function's funcStack is TestPyProfFuncStack::test_ignore_class_call/func1/func2/func3
    #
    def test_ignore_class_call(self):

        def ignored():
            tracemarker = pyprof.nvtx.nvmarker.traceMarker()
            self.compare_funcstack(tracemarker,"TestPyProfFuncStack::test_ignore_class_call/func1/func2/func3")
            
        def func3():
            ignored()

        class ExecutableClass:
            def __call__(self):
                func3()

        def func2():
            x = ExecutableClass()
            x()

        def func1():
            func2()

        func1()

def run_tests(test_name):
    dummy = TestPyProfFuncStack(test_name)
    test_cases = list(
        filter(lambda x: 'test_' in x, map(lambda x: x[0], inspect.getmembers(dummy, predicate=inspect.ismethod)))
    )
    print(f'Running tests for {test_name}')
    suite = unittest.TestSuite()
    for test_case in test_cases:
        suite.addTest(TestPyProfFuncStack(test_case))
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    run_tests("test_basic")
