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
#[cfg(target_arch = "x86")]
use core::arch::x86::__cpuid;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::__cpuid;

const CPU_MANUFACTURER_LABEL: &[u8] = "{cpu-manufacturer:".as_bytes();
const INST_SET_LABEL: &[u8] = ",instruction-set:".as_bytes();
const TRAILING_LABEL: &[u8] = "}".as_bytes();

macro_rules! define_enum_with_max_len {
    (
        $(#[$enum_attr:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$variant_attr:meta])*
                $variant:ident,
            )*
        }
    ) => {
        $(#[$enum_attr])*
        $vis enum $name {
            $(
                $(#[$variant_attr])*
                $variant,
            )*
        }

        impl $name {
            const fn max_len() -> usize {
                const fn max(a: usize, b: usize) -> usize {
                    if a > b { a } else { b }
                }
                let mut max_len = 0;
                $(
                    $(#[$variant_attr])*
                    {
                        max_len = max(max_len, stringify!($variant).as_bytes().len());
                    }
                )*
                max_len
            }
        }
    };
}

macro_rules! manufacturer {
    ($manufacturer:ident) => {
        paste::paste! {
            (CpuManufacturer::$manufacturer, stringify!($manufacturer).as_bytes() )
        }
    };
}

macro_rules! instruction_set {
    ($instruction_set:ident) => {
        paste::paste! {
            (InstructionSet::$instruction_set, stringify!($instruction_set).as_bytes() )
        }
    };
}

define_enum_with_max_len! {
    enum CpuManufacturer {
        Unknown,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Intel,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Amd,
    }
}

define_enum_with_max_len! {
    #[allow(unused)]
    pub enum InstructionSet {
        X86,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Sse2,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Avx2,
        #[cfg(target_arch = "aarch64")]
        Neon,
    }
}

const fn max_desc_len() -> usize {
    CPU_MANUFACTURER_LABEL.len()
        + CpuManufacturer::max_len()
        + INST_SET_LABEL.len()
        + InstructionSet::max_len()
        + TRAILING_LABEL.len()
        + 1
}

pub type AccelerationDescriptor = [u8; max_desc_len()];

fn write_desc(desc: &mut [u8], offset: usize, value: &[u8]) -> usize {
    desc[offset..offset + value.len()].copy_from_slice(value);
    offset + value.len()
}

fn get_desc(manufacturer: &[u8], set: &[u8]) -> (AccelerationDescriptor, usize) {
    let mut desc = [0u8; max_desc_len()];
    let mut size = write_desc(&mut desc, 0, CPU_MANUFACTURER_LABEL);
    size = write_desc(&mut desc, size, manufacturer);
    size = write_desc(&mut desc, size, INST_SET_LABEL);
    size = write_desc(&mut desc, size, set);
    size = write_desc(&mut desc, size, TRAILING_LABEL);

    (desc, size)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const fn four_cc(a: u8, b: u8, c: u8, d: u8) -> u32 {
    ((d as u32) << 24) | ((c as u32) << 16) | ((b as u32) << 8) | (a as u32)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn cpuid(functionnumber: u32, output: &mut [u32; 4]) {
    unsafe {
        let result = __cpuid(functionnumber);
        output[0] = result.eax;
        output[1] = result.ebx;
        output[2] = result.ecx;
        output[3] = result.edx;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn compare_cpu_manufacturer(features: &[u32; 4], name: &[u8; 12]) -> u32 {
    features[1].wrapping_sub(four_cc(name[0], name[1], name[2], name[3]))
        | features[3].wrapping_sub(four_cc(name[4], name[5], name[6], name[7]))
        | features[2].wrapping_sub(four_cc(name[8], name[9], name[10], name[11]))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn get() -> (InstructionSet, AccelerationDescriptor, usize) {
    let mut manufacturer = manufacturer!(Unknown);
    let mut set = instruction_set!(X86);

    let features = &mut [0; 4];
    cpuid(0, features);

    if features[0] != 0 {
        if compare_cpu_manufacturer(features, b"GenuineIntel") == 0 {
            manufacturer = manufacturer!(Intel);
        } else if (compare_cpu_manufacturer(features, b"AuthenticAMD") == 0)
            | (compare_cpu_manufacturer(features, b"AMDisbetter!") == 0)
        {
            manufacturer = manufacturer!(Amd);
        }

        // This ensures we always use hardware intrinsics and we do not use software emulation
        cpuid(1, features);
        if (features[3] & (1 << 26)) != 0 {
            // On AMD cpus, all encode/decode using avx2 have worse performance than sse2 ones
            // For now, disable the avx2 path even if supported.
            // See https://en.wikipedia.org/wiki/CPUID for additional details
            match manufacturer.0 {
                CpuManufacturer::Amd => features[1] = 0,
                _ => cpuid(7, features),
            }

            set = if (features[1] & (1 << 5)) == 0 {
                instruction_set!(Sse2)
            } else {
                instruction_set!(Avx2)
            };
        }
    }

    let (desc, size) = get_desc(manufacturer.1, set.1);
    (set.0, desc, size)
}

#[cfg(target_arch = "aarch64")]
pub fn get() -> (InstructionSet, AccelerationDescriptor, usize) {
    const MANUFACTURER: (CpuManufacturer, &[u8]) = manufacturer!(Unknown);
    const SET: (InstructionSet, &[u8]) = instruction_set!(Neon);
    let (desc, size) = get_desc(MANUFACTURER.1, SET.1);
    (SET.0, desc, size)
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
pub fn get() -> (InstructionSet, AccelerationDescriptor, usize) {
    const MANUFACTURER: (CpuManufacturer, &[u8]) = manufacturer!(Unknown);
    const SET: (InstructionSet, &[u8]) = instruction_set!(X86);
    let (desc, size) = get_desc(MANUFACTURER.1, SET.1);
    (SET.0, desc, size)
}
