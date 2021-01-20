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

import cxxfilt, struct, binascii

#Helper functions


def demangle(name):
    """
	Demangle a C++ string
	"""
    result = name
    try:
        result = cxxfilt.demangle(name)
    except:
        pass
    return result


def getShortName(name):
    """
	Returns a shorter kernel name
	"""
    sname = name.split("<")[0] \
       .replace("void ", "") \
       .replace("at::","") \
       .replace("cuda::", "") \
       .replace("native::","") \
       .replace("(anonymous namespace)::", "")
    sname = sname.split("(")[0]
    return sname


class Kernel(object):
    """
	This class stores information about a kernel.
	"""

    kernels = []
    profStart = 0

    def __init__(self):
        self.kNameId = None
        self.kShortName = None
        self.kLongName = None
        self.kStartTime = None  #GPU start time
        self.kEndTime = None  #GPU end time
        self.kDuration = None
        self.device = None
        self.stream = None
        self.grid = ()
        self.block = ()
        self.corrId = None
        self.rStartTime = None  #CPU start time
        self.rEndTime = None  #CPU end time
        self.rDuration = None
        self.tid = None
        self.pid = None
        self.objId = None
        self.timeOffset = None

        self.layerMarkers = []
        self.traceMarkers = []
        self.fnNameMarkers = []
        self.reprMarkers = []
        self.pyprofMarkers = []
        self.seqMarkers = []
        self.otherMarkers = []
        self.altMarkers = []
        self.seqId = []
        self.altSeqId = []
        self.layer = []

        self.callid = []         ## Input node tracking
        self.input_callids = []  ## Input node tracking

        self.subSeqId = None
        self.dir = None
        self.mod = []
        self.op = []
        self.unique_name = None

    def setKernelInfo(self, info):
        self.kNameId = info['kNameId']
        self.corrId = int(info['correlationId'])
        start = int(info['start'])
        end = int(info['end'])
        assert end > start, "This assertion can fail for very large profiles. It usually fails when start = end = 0."
        self.kStartTime = start
        self.kEndTime = end
        self.kDuration = end - start
        assert (start > Kernel.profStart)
        self.device = int(info['deviceId'])
        self.stream = int(info['streamId'])
        self.grid = (info['gridX'], info['gridY'], info['gridZ'])
        self.block = (info['blockX'], info['blockY'], info['blockZ'])
        self.timeOffset = Kernel.profStart
        self.setKernelName(info['name'])
        self.setRunTimeInfo(info)

    def setKernelName(self, name):
        cadena = demangle(name)
        self.kLongName = cadena
        self.kShortName = getShortName(cadena)

    def setRunTimeInfo(self, info):
        self.rStartTime = info['rStart']
        self.rEndTime = info['rEnd']
        self.rDuration = info['rEnd'] - info['rStart']
        self.pid = info['pid']
        self.tid = info['tid']
        self.objId = info['objId']
        assert (self.rStartTime < self.rEndTime)
        if self.kStartTime:
            assert (self.rStartTime < self.kStartTime)

    def setMarkerInfo(self, info):
        self.layerMarkers, self.traceMarkers, self.fnNameMarkers, self.reprMarkers, self.pyprofMarkers, self.seqMarkers, self.otherMarkers, self.altMarkers, self.seqId, self.altSeqId, self.layer = info
        self.subSeqId = 0

    def setDirection(self):
        """
		Set direction (fprop, bprop) based on PyTorch sequence markers.
		It is a heuristic and not a foolproof method.
		"""
        if any("Backward, seq = " in x for x in self.seqMarkers) or \
         any("backward, seq = " in x for x in self.seqMarkers) or \
         any("Backward0, seq = " in x for x in self.seqMarkers):
            self.dir = "bprop"
        else:
            self.dir = "fprop"

    def setUniqueName(self):
        '''
        Maps unique name set by funcStack to the callid
        Must be called after setOp
        '''
        if self.callid:
            assert len(self.callid) == 1, "Only 1 callid expected per op"
            if  self.callid[0] != 'na':
                assert self.fnNameMarkers, "Call id {} doesn't have fnName marker {}".format(self.callid, self.traceMarkers)
                self.unique_name = self.fnNameMarkers
        return

    def setOp(self):
        """
		Detect and set the class/module (mod) and operation (op)
		of the kernel e.g. torch.nn.functional / linear, torch / sigmoid.
		The lookup sequence we use is
			NVTX markers inserted by pyprof
			NVTX markers inserted by PyTorch in bprop
			NVTX markers inserted by PyTorch in fprop
		It is a heuristic and not a foolproof method.
		"""

        def sanitize(name):
            name = name.replace("torch","") \
               .replace("autograd","") \
               .replace("_backward","") \
               .replace("::","") \
               .replace("jit","") \
               .replace("(anonymous namespace)","")
            head, sep, tail = name.partition("Backward")
            return head

        check_func_stack = False
        #Check pyprof markers
        for m in self.pyprofMarkers:
            assert ("mod" in m) and ("op" in m) and ("args" in m)
            t = eval(m)
            self.op.append(t['op'])
            self.mod.append(t['mod'])
            ## Begin Input node tracking
            if 'callid' in t:
                if t['callid'] not in self.callid:
                    self.callid.append(t['callid'])
                in_callids = t['input_callids']
                for item in in_callids:
                    if item not in self.input_callids:
                        self.input_callids.append(item)
            ## End Input node tracking


        if len(self.op):
            return

        #Check bprop kernel markers
        for m in self.seqMarkers:
            if ", seq = " in m:
                op = m.split(",")[0]
                if ("backward, seq = " in m) or ("Backward, seq = " in m):
                    op = sanitize(op)
                self.op.append(op)
                self.mod.append('na')
                ## Begin Input node tracking
                if 'na' not in self.callid:
                    self.callid.append('na')
                self.input_callids.append('na')
                ## End Input node tracking


        if len(self.op):
            return

        #If nothing else
        if len(self.otherMarkers):
            self.op.append(self.otherMarkers[0])
        self.mod.append('na')
        ## Begin Input node tracking
        if 'na' not in self.callid:
            self.callid.append('na')
        self.input_callids.append('na')
        ## End Input node tracking

    def print(self):
        """
		Print kernel information. This is used by prof.py.
		"""

        a = lambda: None
        a.kShortName = self.kShortName
        a.kDuration = self.kDuration
        #a.layerMarkers = self.layerMarkers
        a.layer = self.layer
        a.trace = self.traceMarkers
        a.reprMarkers = self.reprMarkers
        a.marker = self.pyprofMarkers
        a.seqMarker = self.seqMarkers

        a.callid = self.callid                 # Input node tracking
        a.input_callids = self.input_callids   # Input node tracking

        a.seqId = self.seqId
        a.subSeqId = self.subSeqId
        a.altSeqId = self.altSeqId

        a.dir = self.dir
        a.mod = self.mod
        a.op = self.op

        a.tid = self.tid
        a.device = self.device
        a.stream = self.stream
        a.grid = self.grid
        a.block = self.block
        a.kLongName = self.kLongName
        a.unique_name = self.unique_name

        print(a.__dict__)
