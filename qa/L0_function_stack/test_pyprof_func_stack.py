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
from pyprof.nvtx.config import Config
from pyprof.nvtx.dlprof import DLProf

config = Config(enable_function_stack=True)
dlprof = DLProf()


class TestPyProfFuncStack(unittest.TestCase):

    def __init__(self, testName):
        super().__init__(testName)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def compare_funcstack(self, actual_tracemarker, expected_str):
        # Given a funcstack string, remove TestPyProfFuncStack::__call__/run and everything above it
        #
        def remove_test_class_hierarchy(x):
            separator = "/"
            fn_split = x.split(separator)
            split = 0
            # Find the LAST instance of run in the split
            #
            for i, n in enumerate(fn_split):
                if (n == "TestPyProfFuncStack::run"):
                    split = i + 1

            fn_split = fn_split[split:]
            joined = separator.join(fn_split)
            return joined

        tracemarker_dict = eval(actual_tracemarker)
        actual_func_stack = remove_test_class_hierarchy(tracemarker_dict["funcStack"])
        self.assertEqual(expected_str, actual_func_stack, f"Expected: {expected_str}\nActual: {actual_func_stack}")

    # Basic function hierarchy test
    # Function stack is func1->func2->func3->verify
    # Local function 'verify' gets recognized as a member of TestPyProfFuncStack because it uses 'self'
    #
    def test_basic(self):

        def verify():
            tracemarker = pyprof.nvtx.nvmarker.traceMarker("opname")
            self.compare_funcstack(
                tracemarker, "TestPyProfFuncStack::test_basic/func1/func2/func3/TestPyProfFuncStack::verify/opname"
            )

        def func3():
            verify()

        def func2():
            func3()

        def func1():
            func2()

        func1()

    # Test that 'always_benchmark_wrapper' is ignored in hierarchy
    # Test that 'wrapper_func' is ignored in hierarchy
    # Function stack is func1->func2->always_benchmark_wrapper->func3->wrapper_func->verify
    # Local function 'verify' gets recognized as a member of TestPyProfFuncStack because it uses 'self'
    #
    def test_ignore_wrapper_func(self):

        def verify():
            tracemarker = pyprof.nvtx.nvmarker.traceMarker("opname")
            self.compare_funcstack(
                tracemarker,
                "TestPyProfFuncStack::test_ignore_wrapper_func/func1/func2/func3/TestPyProfFuncStack::verify/opname"
            )

        def wrapper_func():
            verify()

        def func3():
            wrapper_func()

        def always_benchmark_wrapper():
            func3()

        def func2():
            always_benchmark_wrapper()

        def func1():
            func2()

        func1()

    # Test that lambdas are NOT ignored in hierarchy
    # Function stack is func1->func2->lambda->func3->verify
    # Local function 'verify' gets recognized as a member of TestPyProfFuncStack because it uses 'self'
    #
    def test_ignore_lambda(self):

        def verify():
            tracemarker = pyprof.nvtx.nvmarker.traceMarker("opname")
            self.compare_funcstack(
                tracemarker,
                "TestPyProfFuncStack::test_ignore_lambda/func1/func2/<lambda>/func3/TestPyProfFuncStack::verify/opname"
            )

        def func3():
            verify()

        def func2():
            x = lambda: func3()
            x()

        def func1():
            func2()

        func1()

    # Test that duplicates are ignored in hierarchy
    #
    # Function stack is func1->func1->func1->func1->func2->verify
    # Local function 'verify' gets recognized as a member of TestPyProfFuncStack because it uses 'self'
    #
    def test_ignore_duplicates(self):

        def verify():
            tracemarker = pyprof.nvtx.nvmarker.traceMarker("opname")
            self.compare_funcstack(
                tracemarker,
                "TestPyProfFuncStack::test_ignore_duplicates/func1/func2/TestPyProfFuncStack::verify/opname"
            )

        def func2():
            verify()

        def func1(count):
            if (count > 0):
                func1(count - 1)
            else:
                func2()

        func1(3)

    # Function stack is func1->func2->wrapper_func. It is called 4 times.
    #
    # Only the 4th time is any checking done
    #
    # On that 4th call, it will be the 2nd time executing func2, from func1, and
    # it will be the 2nd time executing wrapper_func from that 2nd call of func2.
    #
    # Even though wrapper_func is omitted from the func stack, its call count should
    # be passed on to the opname.
    #
    def test_uniquified_nodes(self):

        def verify(check):
            tracemarker = pyprof.nvtx.nvmarker.traceMarker("opname")
            if (check):
                self.compare_funcstack(
                    tracemarker,
                    "TestPyProfFuncStack::test_uniquified_nodes/func1/func2(2)/TestPyProfFuncStack::verify/opname(2)"
                )

        def wrapper_func(check):
            verify(check)

        def func2(check):
            wrapper_func(False)
            wrapper_func(check)

        def func1():
            func2(False)
            func2(True)

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
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)


if __name__ == '__main__':
    run_tests("test_basic")
