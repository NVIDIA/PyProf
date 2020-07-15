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

class Config:
    __instance = None
    enable_func_stack = False

    @staticmethod
    def getInstance():
        if Config.__instance == None:
            Config()
        return Config.__instance
    
    def __init__(self):
        if Config.__instance != None:
            raise Exception("This is a singleton")
        else:
            Config.__instance = self
    
    def setConfig(self, **kwargs):
        print("tkg in set_config")
        for k,v in kwargs.items():
            print(k,v)
        self.enable_func_stack = kwargs.get("enable_function_stack", False)

    def isFuncStackEnabled(self):
        return self.enable_func_stack