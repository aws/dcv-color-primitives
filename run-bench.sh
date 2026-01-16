#!/bin/bash

# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

while [[ "$#" -gt 0 ]]
do case $1 in
    -c|--use-cycles) use_cycles="$2"
    shift;;
    -f|--filter) filter="$2"
esac
shift
done

EXEC=`cargo bench --features std --no-run 2>&1 | grep -o 'benches-[0-9a-f]*'`
DIR=`pwd`

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Drop file system cache
    echo 3 | sudo tee /proc/sys/vm/drop_caches
    sudo sync

    # Disable address space randomization
    echo 0 | sudo tee /proc/sys/kernel/randomize_va_space
    TASKSET="taskset -c 0"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS-specific optimizations
    sudo purge
    TASKSET=""
else
    # Other systems
    TASKSET=""
fi

sudo rm -Rf ${DIR}/target/criterion

sudo DCP_USE_CYCLES=${use_cycles} nice -n -5 ${TASKSET} ${DIR}/target/release/deps/${EXEC} --bench -n ${filter} | grep -i "time:\|thrpt:"
