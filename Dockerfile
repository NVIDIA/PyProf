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

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:21.04-py3

############################################################################
## Install PyProf
############################################################################
FROM $BASE_IMAGE

ARG PYPROF_VERSION=3.11.0dev
ARG PYPROF_CONTAINER_VERSION=21.06dev

# Copy entire repo into container even though some is not needed for the 
# build itself... because we want to be able to copyright check on 
# files that aren't directly needed for build.    
WORKDIR /opt/pytorch/pyprof
RUN rm -fr *
COPY . .

RUN pip uninstall -y pyprof
RUN pip install --no-cache-dir .

# Generating the docs requires the docs source so copy that into the L0_docs so 
# that it is available when the test runs.
RUN cp VERSION qa/L0_docs/. && \
    cp README.rst qa/L0_docs/. && \
    cp -r docs qa/L0_docs/.
    
