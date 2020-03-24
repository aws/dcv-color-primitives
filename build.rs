// Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify,
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
use std::env;
use std::fs;
use std::path::Path;

#[cfg(target_arch = "x86")]
use core::arch::x86::{__cpuid, _xgetbv};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{__cpuid, _xgetbv};

fn cpuid(functionnumber: u32, output: &mut [u32; 4]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        unsafe {
            let result = __cpuid(functionnumber);
            output[0] = result.eax;
            output[1] = result.ebx;
            output[2] = result.ecx;
            output[3] = result.edx;
        }
    }
}

fn main() {
    // This build script generates the cargo build config file (.cargo/config)
    // The rust flags are set in order to avoid generating illegal instructions
    // on the machine on which the build is triggered.
    let features = &mut [0u32; 4];
    cpuid(0, features);

    if features[0] != 0 {
        cpuid(1, features);
        if (features[3] & (1 << 26)) != 0 {
            println!("cargo:rustc-cfg=target_feature=\"sse2\"");

            // AVX is supported if all the following conditions hold:
            // - OS uses XSAVE/XRSTOR
            // - AVX supported by CPU
            // - AVX registers are restored at context switch
            // See https://software.intel.com/en-us/blogs/2011/04/14/is-avx-enabled/
            let xcr_feature_mask =
                if (features[2] & (1 << 27)) != 0 && (features[2] & (1 << 28)) != 0 {
                    unsafe { _xgetbv(0) }
                } else {
                    0
                };

            if (xcr_feature_mask & 0x6) == 0x6 {
                println!("cargo:rustc-cfg=target_feature=\"avx\"");

                cpuid(7, features);
                if (features[1] & (1 << 5)) != 0 {
                    println!("cargo:rustc-cfg=target_feature=\"avx2\"");
                }
            }
        }
    }
}
