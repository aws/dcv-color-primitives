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

#[cfg_attr(coverage_nightly, coverage(off))]
const fn u8_to_fix(x: i32, frac_bits: i32) -> i32 {
    x << frac_bits
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
pub const fn i32x2_to_i32(x: i32, y: i32) -> i32 {
    let val = (((x & 0xFFFF) as u32) << 16) | ((y & 0xFFFF) as u32);

    // Checked: we want to reinterpret the bits
    val as i32
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub const fn i32_to_i16(x: i32) -> i16 {
    // Checked: we want to reinterpret the bits
    let val = (x & 0xFFFF) as u32;

    // Checked: we are extracting the lower part of a 32-bit integer
    val as i16
}

#[cfg(target_arch = "aarch64")]
struct AssertI16<const N: i32>;

#[cfg(target_arch = "aarch64")]
struct AssertU16<const N: i32>;

#[cfg(target_arch = "aarch64")]
impl<const N: i32> AssertI16<N> {
    const OK: () = assert!(
        N >= i16::MIN as i32 && N <= i16::MAX as i32,
        "must be in i16 range"
    );
}

#[cfg(target_arch = "aarch64")]
impl<const N: i32> AssertU16<N> {
    const OK: () = assert!(N >= 0 && N <= u16::MAX as i32, "must be in u16 range");
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub const fn as_i16<const X: i32>() -> i16 {
    let () = AssertI16::<X>::OK;
    let val = (X & 0xFFFF) as u32;
    val as i16
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub const fn as_u16<const X: i32>() -> u16 {
    let () = AssertU16::<X>::OK;
    (X & 0xFFFF) as u16
}

pub fn wg_index(x: usize, y: usize, w: usize, h: usize) -> usize {
    (h * y) + (x * w)
}

pub fn lower_multiple_of_pot(x: usize, p: usize) -> usize {
    x & !(p - 1)
}

pub fn out_of_bounds(size: usize, stride: usize, height_minus_one: usize, width: usize) -> bool {
    size < width
        || (height_minus_one != 0
            && ((stride > usize::MAX / height_minus_one)
                || (stride * height_minus_one > size - width)))
}

pub fn compute_stride(stride: usize, def: usize) -> usize {
    if stride == 0 { def } else { stride }
}

pub const FIX16: i32 = 16;
pub const FIX18: i32 = 18;
pub const FIX16_HALF: i32 = 1 << (FIX16 - 1);
pub const FIX18_HALF: i32 = 1 << (FIX16 + 1);
pub const FIX6: i32 = 6;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub const SHORT_HALF: i32 = 16384;

// Auto-generated coefficient tables
include!(concat!(env!("OUT_DIR"), "/weights.rs"));

// Other defines
pub const Y_MIN: i32 = 16;
pub const C_HALF: i32 = 128;
const FIX16_Y_MIN: i32 = u8_to_fix(Y_MIN, FIX16);
pub const FIX16_C_HALF: i32 = u8_to_fix(C_HALF, FIX16);
pub const FIX18_C_HALF: i32 = u8_to_fix(C_HALF, FIX18);
pub const Y_OFFSET: i32 = FIX16_Y_MIN + FIX16_HALF;
pub const DEFAULT_ALPHA: u8 = 255;

#[derive(Copy, Clone, Debug)]
pub enum Sampler {
    Argb,
    Bgra,
    Bgr,
    Length,
}

#[derive(Debug)]
pub enum Colorimetry {
    Bt601,
    Bt709,
    Bt2020,
    Bt601FR,
    Bt709FR,
    Bt2020FR,
    Length,
}
