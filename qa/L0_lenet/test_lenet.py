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
This test runs lenet through the 3 steps on pyprof. 
It ensures:
- A database is created from nsys
- A dict is created from pyprof.parse
- A csv with valid data is created from pyprof.prof
'''

import subprocess
from pathlib import Path
import unittest
import csv

unittest.TestLoader.sortTestMethodsUsing = None


class TestPyprofWithLenet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pyprof_path = Path("/opt/pytorch/pyprof/pyprof/examples")

    def test_run_nsys(self):
        # Print a blank line to make the test output more readable
        print()
        command = "nsys profile -f true -o lenet --export sqlite python " + self.pyprof_path.as_posix() + "/lenet.py"
        command_tokens = command.split()

        ret_val = subprocess.run(command_tokens)

        self.assertEqual(ret_val.returncode, 0)
        db_path = Path('./lenet.sqlite')
        self.assertTrue(db_path.exists())

    def test_run_parse(self):
        command = "python -m pyprof.parse lenet.sqlite"
        command_tokens = command.split()

        with open("lenet.dict", "w") as f:
            ret_val = subprocess.run(command_tokens, stdout=f)

        self.assertEqual(ret_val.returncode, 0)
        dict_path = Path('./lenet.dict')
        self.assertTrue(dict_path.exists())

    def test_run_profile(self):
        lenet_csv = "./lenet.csv"
        command = "python -m pyprof.prof --csv lenet.dict"
        command_tokens = command.split()
        with open(lenet_csv, "w") as f:
            ret_val = subprocess.run(command_tokens, stdout=f)

        self.assertEqual(ret_val.returncode, 0)
        csv_path = Path(lenet_csv)
        self.assertTrue(csv_path.exists())

        directions = ["bprop", "fprop"]
        ops = [
            "",  # covers the "reduce_kernel" kernel, op will be an empty string in the report
            "add_",
            "backward",
            "bias",
            "conv2d",
            "linear",
            "max_pool2d",
            "mse_loss",
            "relu",
            "sum",
        ]

        with open("lenet.csv", "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # verify direction
                self.assertTrue(row['Direction'] in directions, f"Row direction: {row['Direction']}")
                # verify op
                self.assertTrue(row['Op'] in ops, f"Row op: {row['Op']}")
            # verify final id is in the range
            # Which kernel cuDNN uses is nondeterministic.
            # While the exact number of kernels is not clear, for this network, it should be [60, 70]
            self.assertTrue(int(row['Idx']) in range(65, 75), f"Final Idx: {row['Idx']}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
