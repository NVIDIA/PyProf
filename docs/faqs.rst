..
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

.. _section-faqs:

PyProf FAQs
===========

**How do I intercept the Adam optimizer in APEX?** ::

	import pyprof
	import fused_adam_cuda
	pyprof.nvtx.wrap(fused_adam_cuda, 'adam')

**What is the correct initialization if you are using JIT and/or AMP?**

#. Let any JIT to finish.
#. Initlialize pyprof ``pyprof.init()``.
#. Initialize AMP.

**How do I profile with ``torch.distributed.launch``?** ::

	nvprof -f -o net%p.sql --profile-from-start off --profile-child-processes \
		python -m torch.distributed.launch net.py
    