#!/bin/bash
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

TEST_LOG="./docs.log"

rm -f $TEST_LOG
RET=0

apt-get update && \
    apt-get install -y --no-install-recommends python3-pip zip doxygen && \
    rm -rf /root/.cache/pip && \
    pip uninstall -y Sphinx && \
    pip3 install --upgrade setuptools wheel && \
    pip3 install --upgrade sphinx==2.4.4 sphinx-rtd-theme==0.4.3 \
         nbsphinx==0.6.0 breathe==4.14.1

set +e

# Set visitor script to be included on every HTML page
export VISITS_COUNTING_SCRIPT=//assets.adobedtm.com/b92787824f2e0e9b68dc2e993f9bd995339fe417/satelliteLib-7ba51e58dc61bcb0e9311aadd02a0108ab24cc6c.js

(cd docs && rm -f pyprof_docs.zip && \
        make BUILDDIR=/opt/pytorch/pyprof/qa/L0_docs/build clean html) > $TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

(cd build && zip -r ../pyprof_docs.zip html)
if [ $? -ne 0 ]; then
    RET=1
fi

set -e

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $TEST_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
