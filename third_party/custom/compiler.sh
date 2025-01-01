#!/usr/bin/env bash

# BSD 2-Clause License
#
# Copyright (c) 2024, Christoph Neuhauser
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Wrapper over the C/C++ compiler automatically determining whether C or C++ is used.

if [[ -z "$CXX_REAL" ]]; then
    CXX="c++"
else
    CXX="$CXX_REAL"
fi
if [[ -z "$CC_REAL" ]]; then
    CC="cc"
else
    CC="$CC_REAL"
fi

use_cc=false
last_arg="-c"
counter=0
for arg in "$@"
do
    if [ $last_arg = "-c" ] && [[ $arg == *.c ]]; then
        use_cc=true
    fi
    if [[ $arg == *huf_decompress_amd64.c ]]; then
        set -- "${@:1:${counter}}" "${arg%?}S" "${@:$((counter+2))}"
    fi
    last_arg="$arg"
    counter=$((counter+1))
done

if [ $use_cc = true ]; then
    "$CC" "$@"
else
    "$CXX" "$@"
fi
