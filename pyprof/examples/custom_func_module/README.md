<!-- 
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. 
-->

This directory has examples which show how to intercept (monkey patch) custom
functions and modules with `pyprof`. No changes are required in `pyprof/parse`,
however, users can add support for bytes and flops calculation for custom
functions and modules in `pyprof/prof` by extending the `OperatorLayerBase` 
class.
