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

from .utility import Utility


class Data(object):
    """
	Class to store all the data for every kernel e.g. name, bytes, flops, device, stream etc.
	"""

    def __init__(self, kernel):
        #Available from NVprof
        self.tid = kernel['tid']
        self.device = kernel['device']
        self.stream = kernel['stream']
        self.grid = str(kernel['grid']).replace(" ", "").replace("(", "").replace(")", "")
        self.block = str(kernel['block']).replace(" ", "").replace("(", "").replace(")", "")
        self.name = kernel['kShortName'].replace(" ", "_")
        self.lName = kernel['kLongName']
        self.sil = kernel['kDuration']  #units ns

        self.index = None

        #Markers
        self.argMarker = kernel['marker']
        self.modMarker = kernel['reprMarkers']
        self.seqMarker = kernel['seqMarker']

        self.layer = kernel['layer']
        self.trace = kernel['trace']

        self.seqId = kernel['seqId']
        self.altSeqId = kernel['altSeqId']

        self.dir = kernel['dir']
        self.sub = kernel['subSeqId']

        self.mod = "na"
        self.op = "na"
        self.params = {"na": "na"}
        self.tc = "na"
        self.flops = 0
        self.bytes = 0

    def setParams(self, params):
        # TODO: Remove the else block after refactoring.
        if type(params) == str:
          self.params = params
        else:
          #Remove space from params
          qaz = ""
          for key, value in params.items():
              if "type" not in key:
                  qaz += "{}={},".format(key, value)
              else:
                  if type(value) is str:
                      qaz += "{},".format(Utility.typeToString(value))
                  else:
                      qaz += "{}".format(value)
          
          self.params = qaz.replace(" ", "")
