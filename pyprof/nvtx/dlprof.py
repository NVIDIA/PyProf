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

import torch
import inspect as ins


class DLProf(object):
    _instance = None

    # Overloading the __new__ method enables singleton behavior
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DLProf, cls).__new__(cls)
            cls.call_id = 0  # input op tracking identifier
            cls.op_to_out_tensor_map = {}  # map from tensor ptr to to call_id
            cls.call_id_to_op_map = {}  # map from call_id to op name
            # Nested dicts of this run's frame names to help uniquify them
            # func_map[(partial_func_stack,frame_name)][filename+lineno] = frame_name_to_use
            #
            cls.func_map = {}
        return cls._instance

    # Return True if the name in the hierarchy should be skipped
    @classmethod
    def should_skip_frame_name(cls, name, prev_name):
        # __call__:
        #    Much of Torch library is implemented in this way. Ignore these extra layers
        # wrapper_func and always_benchmark_warpper:
        #    Are functions in this file. If there are nested monkeypatched functions
        #    we don't want it to show up
        # <*>:
        #    Things like <module>, <genexpr>, <lamba> which don't add any information
        #    and break html
        # name==prev_name:
        #    Remove back-to-back duplicates of the same function name.
        #    This is common when python calls the inheritence stack
        #    For example:
        #      This: ModelAndLoss::forward/ResNet::forward/Sequential::forward/Bottleneck::forward/BatchNorm2d::forward
        #      Comes to this function as: forward/forward/forward/forward/forward
        #      Leaves this function as: forward
        #
        for prefix in ["__call__", "wrapper_func", "always_benchmark_wrapper"]:
            if name.startswith(prefix):
                return True
        if name.startswith("<") and name.endswith(">"):
            return True
        if name == prev_name:
            return True
        return False

    # Given a function stack, clean it up to remove unwanted fields as
    # well as removing any back-to-back duplicates
    @classmethod
    def cleanup_func_stack(cls, func_stack, op_name):

        ret = ""
        prev_fn_name = ""
        suffix = ""

        x = func_stack.split("/")
        for fn_name in x:

            # This is used to detect when the same torch op was called
            # multiple times from the same parent function. Capture the
            # count as a 'suffix' and put it on the end of the op name
            #
            # For example, if we end up with these:
            #   a/b/c/wrapper_func
            #   a/b/c/wrapper_func(2)
            # Both would end up as a/b/c after the wrapper function is ignored
            # However, we want to keep the information that the resulting torch op
            # called by wrapper_func was called 2 different times from the same function 'c'
            #
            # This code changes "wrapper_func(2)" to "(2)" so that it doesn't get filtered
            # out by should_skip_frame_name()
            #
            if fn_name.startswith("wrapper_func("):
                suffix = fn_name.replace("wrapper_func", "")
            if fn_name.startswith("always_benchmark_wrapper("):
                suffix = fn_name.replace("always_benchmark_wrapper", "")

            if not DLProf.should_skip_frame_name(fn_name, prev_fn_name):
                ret += "/" + fn_name
                prev_fn_name = fn_name
        ret += "/" + op_name + suffix
        return ret

    @classmethod
    def build_function_stack(cls, index, func_stack, frame_name, prev_fn, op_name, stack, ins_frame):

        # Build funcStack
        fn_name = frame_name
        # Capture class name
        #
        # Iterate through the stack frames (like a linked list) until we get
        # to the detailed frame we want. This is much faster and less
        # expensive than extracting the entire frame stack every time
        #
        # ins stack is backwards from traceback, so depth is inverse
        # of current traceback depth
        #
        depth = len(stack) - index
        for _ in range(1, depth):
            ins_frame = ins_frame.f_back

        # Grab the class name if it exists
        #
        if 'self' in ins_frame.f_locals:
            fn_name = ins_frame.f_locals['self'].__class__.__name__ + "::" + fn_name
        key = (func_stack, frame_name, "")
        if (fn_name in ["wrapper_func", "always_benchmark_wrapper"]):
            key = (func_stack, frame_name, op_name)

        if key not in cls.func_map.keys():
            cls.func_map[key] = {}

        # If we have been to this stack depth with all the same
        # information, use the stored name
        #
        if prev_fn in cls.func_map[key].keys():
            fn_name = cls.func_map[key][prev_fn]
        else:
            # If we have been do this stack depth and have called
            # this function at least once but didn't hit in the dict
            # above, then this is a repeat call. Postpend a count
            # to the fn_name to uniquify it
            #
            if len(cls.func_map[key]) > 0:
                fn_name = fn_name + "(" + str(1 + len(cls.func_map[key])) + ")"

            # Store this new unique stack information with the
            # determined fn_name
            #
            cls.func_map[key][prev_fn] = fn_name

        return fn_name

    @classmethod
    def capture_inputs(cls, input_callid_list, *args):
        input_tensors = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                input_tensors.append({
                    'ptr': arg.data_ptr(),
                })
            elif isinstance(arg, list) or isinstance(arg, tuple):
                for item in arg:
                    if isinstance(item, torch.Tensor):
                        input_tensors.append({
                            'ptr': item.data_ptr(),
                        })
                        if isinstance(item, list) or isinstance(item, tuple):
                            for item2 in item:
                                if isinstance(item2, torch.Tensor):
                                    input_tensors.append({
                                        'ptr': item2.data_ptr(),
                                    })
        for input_id, _ in enumerate(input_tensors):
            input_ptr = input_tensors[input_id]['ptr']
            if input_ptr in cls.op_to_out_tensor_map:
                input_callid_info = cls.op_to_out_tensor_map[input_ptr]
                if input_callid_info not in input_callid_list:
                    input_callid_list.append(input_callid_info)

    @classmethod
    def capture_outputs(cls, call_id, result):
        output_tensors = []
        if isinstance(result, torch.Tensor):
            output_tensors.append({
                'ptr': result.data_ptr(),
            })
        elif isinstance(result, list) or isinstance(result, tuple):
            for item in result:
                if isinstance(item, torch.Tensor):
                    output_tensors.append({
                        'ptr': item.data_ptr(),
                    })
        for out_port, _ in enumerate(output_tensors):
            output_ptr = output_tensors[out_port]['ptr']
            cls.op_to_out_tensor_map[output_ptr] = f"{call_id}"