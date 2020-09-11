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
Parse the SQLite3 database from NVprof or Nsight and print a dictionary for every kernel.
"""

import sys
import os
import argparse
from tqdm import tqdm

from .db import DB
from .kernel import Kernel
from .nvvp import NVVP
from .nsight import Nsight


def parseArgs():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Parse SQLite3 DB from NVprof or Nsight.")
    parser.add_argument("file", type=str, default=None, help="SQLite3 database.")

    args = parser.parse_args()

    if not os.path.isfile(args.file):
        raise parser.error("No such file '{}'.".format(args.file))

    return args


def dbIsNvvp(db):
    cmd = "SELECT * FROM sqlite_master where type='table' AND name='StringTable'"
    result = db.select(cmd)
    return True if len(result) == 1 else False


def main():
    args = parseArgs()

    db = DB(args.file)
    nvvp = None
    if dbIsNvvp(db):
        nvvp = NVVP(db)
    else:
        nvvp = Nsight(db)

    kInfo = nvvp.getKernelInfo()
    if len(kInfo) == 0:
        print("Found 0 kernels. Exiting.", file=sys.stderr)
        db.close()
        sys.exit(0)
    else:
        print("Found {} kernels. Getting info for each kernel.".format(len(kInfo)), file=sys.stderr)

    nvvp.createMarkerTable()

    prevSeqId = -1
    prevSubSeqId = -1
    prevOp = "na"

    Kernel.profStart = nvvp.getProfileStart()

    for i in tqdm(range(len(kInfo)), ascii=True):
        info = kInfo[i]
        k = Kernel()

        #Set kernel info
        k.setKernelInfo(info)

        #Get runtime info
        info = nvvp.getCPUInfo(k.corrId)
        k.setRunTimeInfo(info)

        #Get and set marker and seqid info
        info = nvvp.getMarkerInfo(k.objId, k.rStartTime, k.rEndTime)
        k.setMarkerInfo(info)

        #If the seqId contains both 0 and non zero integers, remove 0.
        if any(seq != 0 for seq in k.seqId) and (0 in k.seqId):
            k.seqId.remove(0)

        #Set direction (it uses seq id)
        k.setDirection()

        #Set op
        k.setOp()

        #The following code is based on heuristics.
        #TODO: Refactor.
        #Assign subSeqId, adjust seqId and altSeqId
        #seqId can be 0.
        #A kernel can have multiple seqIds both in fprop and bprop.
        #In bprop, seqIds might not decrease monotonically. I have observed a few blips.
        if len(k.seqId):
            assert (k.dir in ["fprop", "bprop"])
            if (k.dir == "fprop"):
                #Check if there is a sequence id larger than the previous
                inc = (k.seqId[-1] > prevSeqId)
                if inc:
                    currSeqId = [x for x in k.seqId if x > prevSeqId][0]
                else:
                    currSeqId = prevSeqId
            else:
                currSeqId = k.seqId[0]

            #if ((currSeqId == prevSeqId) and (k.op == prevOp)):
            if ((currSeqId == prevSeqId) and (k.op == prevOp)) or ((k.op[0] == "forward") and (k.op == prevOp) and
                                                                   (k.mod[0] in ["LSTMCell", "GRUCell", "RNNCell"])):
                #The second condition is to trap cases when pytorch does not use cudnn for a LSTMCell.
                k.subSeqId = prevSubSeqId + 1

            prevSeqId = currSeqId
            prevSubSeqId = k.subSeqId
            prevOp = k.op

            #Keep currSeqId in k.seqId, move everything else to k.altSeqId
            for s in k.seqId:
                if s != currSeqId:
                    k.seqId.remove(s)
                    k.altSeqId.append(s)

            for s in k.altSeqId:
                if s == currSeqId:
                    k.altSeqId.remove(s)

            k.altSeqId = list(set(k.altSeqId))
            if (len(k.altSeqId)):
                (k.altSeqId).sort()

        k.print()

    db.close()


if __name__ == '__main__':
    main()
