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

// Coefficient table for 601
pub const XR_601: i32 = 16829;
pub const XG_601: i32 = 33039;
pub const XB_601: i32 = 6416;
pub const YR_601: i32 = -9714;
pub const YG_601: i32 = -19071;
pub const ZG_601: i32 = -24103;

pub const XXYM_601: i32 = 19077;
pub const RCRM_601: i32 = 26149;
pub const GCRM_601: i32 = 13320;
pub const GCBM_601: i32 = 6419;
pub const BCBM_601: i32 = 33050;
pub const RN_601: i32 = 14234;
pub const GP_601: i32 = 8709;
pub const BN_601: i32 = 17685;

// Coefficient table for 709
pub const XR_709: i32 = 11966;
pub const XG_709: i32 = 40254;
pub const XB_709: i32 = 4064;
pub const YR_709: i32 = -6596;
pub const YG_709: i32 = -22189;
pub const ZG_709: i32 = -26145;

pub const XXYM_709: i32 = 19077;
pub const RCRM_709: i32 = 29372;
pub const GCRM_709: i32 = 8731;
pub const GCBM_709: i32 = 3494;
pub const BCBM_709: i32 = 34610;
pub const RN_709: i32 = 15846;
pub const GP_709: i32 = 4952;
pub const BN_709: i32 = 18465;

// Coefficient table for 601 (full range)
pub const XR_601FR: i32 = 19595;
pub const XG_601FR: i32 = 38470;
pub const XB_601FR: i32 = 7471;
pub const YR_601FR: i32 = -11058;
pub const YG_601FR: i32 = -21709;
pub const ZG_601FR: i32 = -27439;

pub const XXYM_601FR: i32 = 16384;
pub const RCRM_601FR: i32 = 22970;
pub const GCRM_601FR: i32 = 11700;
pub const GCBM_601FR: i32 = 5638;
pub const BCBM_601FR: i32 = 29032;
pub const RN_601FR: i32 = 11363;
pub const GP_601FR: i32 = 8633;
pub const BN_601FR: i32 = 14370;

// Coefficient table for 709 (full range)
pub const XR_709FR: i32 = 13933;
pub const XG_709FR: i32 = 46871;
pub const XB_709FR: i32 = 4732;
pub const YR_709FR: i32 = -7508;
pub const YG_709FR: i32 = -25259;
pub const ZG_709FR: i32 = -29763;

pub const XXYM_709FR: i32 = 16384;
pub const RCRM_709FR: i32 = 25802;
pub const GCRM_709FR: i32 = 7670;
pub const GCBM_709FR: i32 = 3069;
pub const BCBM_709FR: i32 = 30402;
pub const RN_709FR: i32 = 12768;
pub const GP_709FR: i32 = 5359;
pub const BN_709FR: i32 = 15050;

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
    Bt601FR,
    Bt709FR,
    Length,
}
