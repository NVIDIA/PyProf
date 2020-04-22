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

set -e

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
PYPROF="$SCRIPTPATH/../.."

parse="python $PYPROF/parse/parse.py"
prof="python $PYPROF/prof/prof.py"

for f in *.py
do
	base=`basename $f .py`
	sql=$base.sql
	dict=$base.dict

	#NVprof
	echo "nvprof -fo $sql python $f"
	nvprof -fo $sql python $f

	#Parse
	echo $parse $sql
	$parse $sql > $dict

	#Prof
	echo $prof $dict
	$prof -w 130 $dict
	\rm $sql $dict
done
