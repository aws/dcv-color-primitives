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

const fn u8_to_fix(x: i32, frac_bits: i32) -> i32 {
    x << frac_bits
}

pub const fn i32x2_to_i32(x: i32, y: i32) -> i32 {
    let val = (((x & 0xFFFF) as u32) << 16) | ((y & 0xFFFF) as u32);
    val as i32
}

pub const fn i32_to_i16(x: i32) -> i16 {
    let val = (x & 0xFFFF) as u32;
    val as i16
}

pub fn wg_index(x: usize, y: usize, w: usize, h: usize) -> usize {
    (h * y) + (x * w)
}

pub fn is_wg_multiple(x: u32, w: usize) -> bool {
    ((x as usize) & (w - 1)) == 0
}

pub const FIX16: i32 = 16;
pub const FIX18: i32 = 18;
pub const FIX16_HALF: i32 = 1 << (FIX16 - 1);
pub const FIX18_HALF: i32 = 1 << (FIX16 + 1);
pub const FIX6: i32 = 6;
pub const SHORT_HALF: i32 = 16384;

// Cooefficient table for 601
pub const XR_601: i32 = 16829;
pub const XG_601: i32 = 33039;
pub const XB_601: i32 = 6416;
pub const YR_601: i32 = -9713;
pub const YG_601: i32 = -19070;
pub const YB_601: i32 = 28784;
pub const ZR_601: i32 = 28784;
pub const ZG_601: i32 = -24102;
pub const ZB_601: i32 = -4680;

pub const XXYM_601: i32 = 19077;
pub const RCRM_601: i32 = 26149;
pub const GCRM_601: i32 = 13320;
pub const GCBM_601: i32 = 6419;
pub const BCBM_601: i32 = 33050;
pub const RN_601: i32 = 14234;
pub const GP_601: i32 = 8709;
pub const BN_601: i32 = 17685;

// Cooefficient table for 709
pub const XR_709: i32 = 11966;
pub const XG_709: i32 = 40254;
pub const XB_709: i32 = 4064;
pub const YR_709: i32 = -6595;
pub const YG_709: i32 = -22188;
pub const YB_709: i32 = 28784;
pub const ZR_709: i32 = 28784;
pub const ZG_709: i32 = -26144;
pub const ZB_709: i32 = -2638;

pub const XXYM_709: i32 = 19077;
pub const RCRM_709: i32 = 29372;
pub const GCRM_709: i32 = 8731;
pub const GCBM_709: i32 = 3494;
pub const BCBM_709: i32 = 34610;
pub const RN_709: i32 = 15846;
pub const GP_709: i32 = 4952;
pub const BN_709: i32 = 18465;

// Other defines
pub const Y_MIN: i32 = 16;
pub const C_HALF: i32 = 128;
pub const FIX16_Y_MIN: i32 = u8_to_fix(Y_MIN, FIX16);
pub const FIX16_C_HALF: i32 = u8_to_fix(C_HALF, FIX16);
pub const FIX18_C_HALF: i32 = u8_to_fix(C_HALF, FIX18);
pub const Y_OFFSET: i32 = FIX16_Y_MIN + FIX16_HALF;
pub const C_OFFSET: i32 = FIX18_C_HALF + FIX18_HALF;
pub const C_OFFSET16: i32 = FIX16_C_HALF + FIX16_HALF;
pub const DEFAULT_ALPHA: u8 = 255;

#[derive(Copy, Clone)]
pub enum Sampler {
    Argb,
    Bgra,
    Bgr,
    BgrOverflow,
    Length,
}

pub enum Colorimetry {
    Bt601,
    Bt709,
    Length,
}

pub enum PixelFormatChannels {
    Three = 3,
    Four = 4,
}
