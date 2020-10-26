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
"""
This file intercepts (monkey patches) the following functions and adds NVTX markers.
	torch.*
	torch.Tensor.*
	torch.nn.functional.*
	torch.nn.*.forward

The NVTX markers (one or more) contain the following information
	call trace (a list of file_name:line_number)
	extra_repr() from torch.nn modules
	module/class name
	function name
	inputs (args and kwargs)
		scalar: name, type and value
		tensor: name, shape and datatype
		numpy: name, shape and datatype
		list/tuple: a sequence of scalars or tensors or numpy arrays
"""

import torch
import torch.cuda.nvtx as nvtx
import numpy
import inspect as ins
import traceback
import math
import json
import importlib
from .config import Config
from .dlprof import DLProf

# Singleton object tracking dlprof specific information
dlprof = DLProf()
# flag to control wrapping ops in nvtx markers
wrappers_enabled = True


def isfunc(mod, f):
    assert hasattr(mod, f)
    attr = getattr(mod, f)

    #Ignore functions like _add
    if (len(f) >= 2):
        if f[0] == "_" and f[1] != "_":
            return False

    #Ignore functions from this list
    ignore = [
        '__all__', '__array__', '__array_priority__', '__array_wrap__', '__bool__', '__builtins__', '__cached__',
        '__class__', '__deepcopy__', '__delattr__', '__delitem__', '__dict__', '__dir__', '__doc__', '__file__',
        '__format__', '__getattribute__', '__getitem__', '__hash__', '__index__', '__init__', '__init_subclass__',
        '__iter__', '__len__', '__loader__', '__module__', '__name__', '__new__', '__nonzero__', '__package__',
        '__path__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__setattr__', '__setitem__',
        '__setstate__', '__sizeof__', '__spec__', '__str__', '__subclasshook__', '__version__', '__weakref__'
    ]

    #Add functions to this list if they cause recursion
    ignore += ['size', 'tolist', 'dim', 'is_storage', 'item', 'data_ptr']
    if f in ignore:
        return False
    return ins.ismethod(attr) or ins.isfunction(attr) or ins.ismethoddescriptor(attr) or ins.isbuiltin(attr)


# Returns a dict string with a tracemarker and function stack in it
#
def traceMarker(op_name):

    config = Config()

    # Return a trace marker string and func_stack string
    #
    def get_trace_info(op_name):
        cadena = []
        stack = traceback.extract_stack()
        func_stack = ""

        # Previous frame name and line. This is the file and line
        # that CALLED the frame we are in
        #
        prev_fn = ""

        # Starting at index of 2 to ignore this function and its parent (traceMarker).
        # Intentionally leaving in wrapper_func and other functions in this file as they
        # may be needed to uniquify the node name
        #
        for idx in range(len(stack) - 2):
            frame = stack[idx]

            # Build traceMarker
            #

            # Don't include any functions from this file (nvmarker.py)
            # Also skip repeated back to back cases of the same file/line (recursive calls)
            #
            fnl = "{}:{}".format(frame.filename, frame.lineno)
            if (not frame.filename.endswith("nvmarker.py") and fnl != prev_fn):
                cadena.append(fnl)

            # Early exit if we aren't doing any funcStack code
            #
            if not config.func_stack_enabled:
                continue
            else:
                ins_frame = ins.currentframe()
                fn_name = dlprof.build_function_stack(idx, func_stack, frame.name, prev_fn, op_name, stack, ins_frame)
                del ins_frame
            prev_fn = fnl

            # Append this frame's info into the function stack
            #
            func_stack = func_stack + "/" + fn_name

        if config.func_stack_enabled:
            func_stack = dlprof.cleanup_func_stack(func_stack, op_name)

        return cadena, func_stack

    d = {}
    tm, fs = get_trace_info(op_name)
    d['traceMarker'] = tm
    if config.func_stack_enabled:
        d['funcStack'] = fs
    return str(d)


def modMarker(mod, fn_name, args):
    """
	Returns the stringified extra_repr() of a module.
	"""
    assert (fn_name == 'forward')
    assert (len(args) > 0)
    d = {}
    d['mod'] = mod.__name__
    d['strRepr'] = args[0].extra_repr()
    return str(d)


def add_wrapper(mod, fn_name):

    config = Config()

    # Get a pointer to the original function
    func = getattr(mod, fn_name)

    # Check if the mod has a string representation
    # and is not a Script or Traced module (used by JIT)
    # yapf: disable
    s = hasattr(mod, "extra_repr") and (type(mod) is not torch.jit.ScriptModule
                                       ) and (type(mod) is not torch.jit.TopLevelTracedModule)
    # yapf: enable

    def wrapper_func(*args, **kwargs):

        global wrappers_enabled
        traceMarker_str = ""
        input_callid_list = []

        if config.capture_input_ops:
            dlprof.capture_inputs(input_callid_list, *args)

        if wrappers_enabled:
            # Push trace marker
            traceMarker_str = traceMarker(fn_name)
            nvtx.range_push(traceMarker_str)

            # Push module marker
            if s:
                m = modMarker(mod, fn_name, args)
                nvtx.range_push(m)

            # Create and push argument marker
            #
            # Disable wrappers while getting the argMarker in case it
            # ends up executing another wrapped function
            wrappers_enabled = False
            if config.capture_input_ops:
                cadena = argMarker(mod, fn_name, args, kwargs, dlprof.call_id, input_callid_list)
            else:
                cadena = argMarker(mod, fn_name, args, kwargs)
            nvtx.range_push(cadena)
            wrappers_enabled = True

        # Call the original function
        result = func(*args, **kwargs)

        if wrappers_enabled:
            # Pop argumet marker
            nvtx.range_pop()

            # Pop module marker
            if s:
                nvtx.range_pop()

            # Pop trace marker
            nvtx.range_pop()

        if config.capture_input_ops:
            dlprof.capture_outputs(dlprof.call_id, result)
            # Store the callid -> op_name mapping
            if traceMarker_str is not "":
                traceMarker_str = traceMarker_str.replace("\'", "\"")
                traceMarker_dict = json.loads(traceMarker_str)
                dlprof.call_id_to_op_map[dlprof.call_id] = traceMarker_dict['funcStack']
            dlprof.call_id = dlprof.call_id + 1

        return result

    setattr(mod, fn_name, wrapper_func)


def argMarker(mod, op, args, kwargs, idx=-1, inputid_list=[]):
    #For this function args is a tuple and kwargs is a dict
    config = Config()

    def tensor(arg, name=""):
        if config.capture_input_ops:
            cid = dlprof.op_to_out_tensor_map.get(arg.data_ptr(), -1)
            name = dlprof.call_id_to_op_map.get(int(cid), "")
        a = {}
        a['name'] = name
        a['type'] = "tensor"
        a['shape'] = tuple(arg.size())
        a['dtype'] = str(arg.dtype).split(".")[-1]
        cadena['args'].append(a)

    def ndarray(arg, name=""):
        a = {}
        a['name'] = name
        a['type'] = "ndarray"
        a['shape'] = arg.shape
        a['dtype'] = str(arg.dtype).split(".")[-1]
        cadena['args'].append(a)

    def seq(arg, name=""):
        assert issequence(arg)
        a = {}
        a['name'] = name
        if isinstance(arg, list):
            a['type'] = "list"
            a['value'] = arg
        else:
            a['type'] = "tuple"
            # The arg could be torch.Size, which is a subclass of tuple
            # Therefore, explicitly convert to tuple
            a['value'] = tuple(arg)

        cadena['args'].append(a)

    def scalar(arg, name=""):
        a = {}
        a['name'] = name
        a['type'] = type(arg).__name__
        #handle the case when the argument is +/- inf or nan
        if arg == float('inf'):
            a['value'] = "inf"
        elif arg == float('-inf'):
            a['value'] = "-inf"
        elif isinstance(arg, float) and math.isnan(arg):
            a['value'] = "nan"
        else:
            a['value'] = arg
        cadena['args'].append(a)

    def isscalar(arg):
        return (type(arg) is int) or (type(arg) is float) or (type(arg) is bool) or (arg is None) or (type(arg) is str)

    def issequence(arg):
        return isinstance(arg, list) or isinstance(arg, tuple)

    def foo(args, name):
        #args should be an iterable sequence e.g. list or tuple
        for arg in args:
            if isinstance(arg, torch.Tensor):
                if arg.dim() == 0:
                    scalar(arg.item(), name)
                else:
                    tensor(arg, name)
            elif isinstance(arg, numpy.ndarray):
                ndarray(arg, name)
            elif (isscalar(arg)):
                scalar(arg, name)
            elif issequence(arg):
                if (len(arg) == 0) or isscalar(arg[0]):  #An empty sequence or a sequence of scalars
                    seq(arg, name)
                else:  # A sequence of tensors or numpy arrays
                    foo(arg, name)
            '''
			else:
				print("The following arg is none of Tensor, numpy array, scalar but a %s" % (str(type(arg))))
				print("Mod: %s" % str(mod.__name__))
				print("Op: %s" % str(op))
				print(dir(arg))
			'''

    cadena = {}
    cadena['mod'] = mod.__name__
    cadena['op'] = op
    if config.capture_input_ops:
        cadena['callid'] = idx
        cadena['input_callids'] = inputid_list
    cadena['args'] = []

    foo(args, "")
    for k, v in kwargs.items():
        foo((v, ), k)

    return str(cadena)


def patchClass(cls):
    for f in dir(cls):
        if isfunc(cls, f):
            add_wrapper(cls, f)


def patch_torch_classes():
    """Monkey-patch all classes in torch"""
    for cls in [
            torch,
            torch.Tensor,
            torch.nn.functional,
    ]:
        patchClass(cls)


def patch_torch_nn_forward_functions():
    """Monkey-patch all forward functions in torch.nn libraries"""
    for cls in [torch.nn.RNN, torch.nn.RNNCell, torch.nn.LSTM, torch.nn.LSTMCell, torch.nn.GRU, torch.nn.GRUCell]:
        if isfunc(cls, 'forward'):
            add_wrapper(cls, 'forward')


def patch_dataloader():
    """Monkey-patch the dataloader in torch.utils.data"""
    mod = torch.utils.data.dataloader
    old_iter = mod.DataLoader.__iter__

    def new_iter(self, *args, **kwargs):

        # Push trace marker
        nvtx.range_push(traceMarker("Dataloader"))

        # First pass is for creating the dataloader + returning the first data
        cadena = argMarker(mod, "DataLoader", args, kwargs)
        nvtx.range_push(cadena)

        for x in old_iter(self, *args, **kwargs):

            # Pop tracemarker
            nvtx.range_pop()

            # Dataloader stop, Model start
            nvtx.range_pop()

            yield x

            # Push trace marker
            nvtx.range_push(traceMarker("DataLoader"))

            # Model stop, dataloader start
            cadena = argMarker(mod, "DataLoader", args, kwargs)
            nvtx.range_push(cadena)

        # Pop the last iteration before returning
        nvtx.range_pop()
        nvtx.range_pop()

    mod.DataLoader.__iter__ = new_iter


def patch_apex():
    """Monkey-patch functions in APEX"""

    if importlib.util.find_spec("amp_C") is not None:
        import amp_C
        patchClass(amp_C)

    if importlib.util.find_spec("fused_adam_cuda") is not None:
        import fused_adam_cuda
        patchClass(fused_adam_cuda)

    if importlib.util.find_spec("fused_lamb_cuda") is not None:
        import fused_lamb_cuda
        patchClass(fused_lamb_cuda)

    if importlib.util.find_spec("fused_layer_norm_cuda") is not None:
        import fused_layer_norm_cuda
        patchClass(fused_layer_norm_cuda)

    if importlib.util.find_spec("distributed_lamb_cuda") is not None:
        import distributed_lamb_cuda
        patchClass(distributed_lamb_cuda)

    if importlib.util.find_spec("xentropy_cuda") is not None:
        import xentropy_cuda
        patchClass(xentropy_cuda)

    if importlib.util.find_spec("mlp_cuda") is not None:
        import mlp_cuda
        patchClass(mlp_cuda)

    patch_apex_module("apex.amp")
    patch_apex_module("apex.contrib.groupbn")
    patch_apex_module("apex.contrib.multihead_attn")
    patch_apex_module("apex.contrib.optimizers")
    patch_apex_module("apex.contrib.sparsity")
    patch_apex_module("apex.contrib.xentropy")
    patch_apex_module("apex.fp16_utils")
    patch_apex_module("apex.mlp")
    patch_apex_module("apex.multi_tensor_apply")
    patch_apex_module("apex.optimizers")
    patch_apex_module("apex.parallel")
    #patch_apex_module("apex.reparameterization") # FIXME
    #patch_apex_module("apex.RNN") # FIXME


def is_same_module_or_submodule(orig, incoming):
    if incoming is None:
        return False
    if orig == incoming:
        return True
    if incoming.__name__.startswith(orig.__name__):
        return True
    return False


def patch_apex_module(modstr):
    if importlib.util.find_spec(modstr) is not None:
        mod = importlib.import_module(modstr)

        for n, v in ins.getmembers(mod):
            if is_same_module_or_submodule(mod, ins.getmodule(v)):
                if (ins.isclass(v)):
                    for key in v.__dict__:
                        if (ins.isfunction(v.__dict__[key])):
                            # FIXME SHOULD I ONLY BE DOING FORWARD??
                            add_wrapper(v, key)
                if (ins.isfunction(v)):
                    add_wrapper(mod, n)


def push_nvtx_model_config(config):
    """
    Helper function to dump the passed in dict config as an nvtx
    marker with "model_config" key
    """
    nvtx_msg = json.dumps({"model_config": config})
    nvtx.range_push(nvtx_msg)


def patch_dataloader_init():
    """
    Capture dataloader config (num_workers and pin_memory) and
    emit a model_config nvtx range with the information
    """
    mod = torch.utils.data.dataloader
    old_init = mod.DataLoader.__init__

    def new_init(self, *args, **kwargs):

        num_workers = kwargs.get("num_workers", 0)
        pin_memory = kwargs.get("pin_memory", False)

        push_nvtx_model_config({"num_workers": num_workers, "pin_memory": pin_memory})
        old_init(self, *args, **kwargs)
        nvtx.range_pop()

    mod.DataLoader.__init__ = new_init


# Flag to indicate that cudnn_benchmark_disabled has already been reported
#
cudnn_benchmark_disabled_reported = False


def patch_with_always_benchmark(mod, fn_name):
    """
    Patch the given mod/function so that if it is ever executed and 
    torch.backends.cudnn.benchmark is not true, it will emit an nvtx
    range to report that fact
    """
    assert isfunc(mod, fn_name)
    old_fn = getattr(mod, fn_name)

    def always_benchmark_wrapper(*args, **kwargs):
        global cudnn_benchmark_disabled_reported

        add_nvtx = not torch.backends.cudnn.benchmark and not cudnn_benchmark_disabled_reported
        if add_nvtx:
            cudnn_benchmark_disabled_reported = True
            push_nvtx_model_config({"cudnn_benchmark_disabled": True})

        result = old_fn(*args, **kwargs)

        if add_nvtx:
            nvtx.range_pop()

        return result

    setattr(mod, fn_name, always_benchmark_wrapper)


def patch_never_call(mod, fn_name, key):
    """
    Patch the given mod/function. If the function is executed, emit 
    an nvtx_range with data indicating that 'key' was true
    """
    old_fn = getattr(mod, fn_name)

    def wrapper_func(*args, **kwargs):
        push_nvtx_model_config({key: True})
        result = old_fn(*args, **kwargs)
        nvtx.range_pop()
        return result

    setattr(mod, fn_name, wrapper_func)


def patch_never_call_with_args(mod, fn_name, key, bad_args):
    """
    Patch the given mod/function. If the function is executed 
    and any of the bad args have any of the listed bad values, 
    emit an nvtx_range with data indicating that 'key' was true
    """
    old_fn = getattr(mod, fn_name)

    def wrapper_func(*args, **kwargs):

        signature = ins.signature(old_fn)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()

        problem = False
        for k, v in bound.arguments.items():
            if k in bad_args:
                if v in bad_args[k]:
                    problem = True

        if problem:
            push_nvtx_model_config({key: True})

        result = old_fn(*args, **kwargs)

        if problem:
            nvtx.range_pop()

        return result

    setattr(mod, fn_name, wrapper_func)


def patch_model_configs():
    """
    Patch functions that help gather high-level configuration options for the model.
    All resulting nvtx ranges will have 'model_config' as the primary key
    """

    patch_dataloader_init()

    patch_with_always_benchmark(torch.nn.functional, "conv1d")
    patch_with_always_benchmark(torch.nn.functional, "conv2d")
    patch_with_always_benchmark(torch.nn.functional, "conv3d")
    patch_with_always_benchmark(torch.nn.functional, "conv_transpose1d")
    patch_with_always_benchmark(torch.nn.functional, "conv_transpose2d")
    patch_with_always_benchmark(torch.nn.functional, "conv_transpose3d")

    patch_never_call(torch.autograd.detect_anomaly, "__init__", "detect_anomaly")
    patch_never_call(torch.autograd, "gradcheck", "gradcheck")
    patch_never_call(torch.autograd, "gradgradcheck", "gradgradcheck")
    patch_never_call(torch.autograd.profiler.record_function, "__init__", "record_function")

    # Patch both AMP libraries
    #
    import importlib
    if importlib.util.find_spec("apex") is not None and importlib.util.find_spec("apex.amp") is not None:
        import apex.amp
        patch_never_call_with_args(apex.amp, "initialize", "amp_enabled", {"enabled": {True}})
    patch_never_call_with_args(torch.cuda.amp, "autocast", "amp_enabled", {"enabled": {True}})

    patch_never_call_with_args(torch.autograd.profiler.profile, "__init__", "profile", {"enabled": {True}})
    patch_never_call_with_args(torch.autograd.set_detect_anomaly, "__init__", "detect_anomaly", {"mode": {True}})
    patch_never_call_with_args(torch.autograd.profiler.emit_nvtx, "__init__", "emit_nvtx", {"enabled": {True}})


def init(*args, **kwargs):
    """
    Initialize pyprof and monkey-patch Torch functions

    Kwargs:
        enable_function_stack (bool): When true, function stack information 
            will be added to NVTX markers
        capture_input_ops (bool): When true, input tensor names will be added 
            to NVTX markers and enable_function_stack is set to True.
    """

    Config(*args, **kwargs)

    print("Initializing NVTX monkey patches")

    patch_apex()
    patch_dataloader()
    patch_torch_classes()
    patch_torch_nn_forward_functions()
    patch_model_configs()

    print("Done with NVTX monkey patching")
