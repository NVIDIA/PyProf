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

*As of this writing, these examples do not work
because of changes being proposed in PyTorch.*

There are two ways to use PyTorch JIT
 - Scripting
 - Tracing

In addition, we can JIT a
 - Stand alone function
 - Class / class method

This directory has an example for each of the 4 cases.
Intercepting (monkey patching) JITted code has a few extra steps,
which are explained through comments.
