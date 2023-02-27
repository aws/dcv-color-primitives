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

#![allow(clippy::wildcard_imports)] // We are importing everything
use crate::convert_image::common::*;
use crate::{rgb_to_yuv_converter, yuv_to_rgb_converter};

use core::ptr::{read_unaligned as loadu, write_unaligned as storeu};

#[cfg(target_arch = "x86")]
use core::arch::x86::_bswap;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_bswap;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_bswap64;

#[cfg(all(not(target_arch = "x86"), not(target_arch = "x86_64")))]
#[inline(always)]
fn _bswap(x: i32) -> i32 {
    let mut y2: [i8; 4] = [0; 4];
    let y: i32;
    let x2: [i8; 4];

    unsafe {
        x2 = std::mem::transmute::<i32, [i8; 4]>(x);

        for i in 0..4 {
            y2[i] = x2[3 - i];
        }

        y = std::mem::transmute::<[i8; 4], i32>(y2);
    }

    return y;
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn _bswap64(x: i64) -> i64 {
    (((_bswap(x as i32) as u64) << 32) | ((_bswap((x >> 32) as i32) as u64) & 0xFFFFFFFF)) as i64
}

pub const FORWARD_WEIGHTS: [[i32; 10]; Colorimetry::Length as usize] = [
    [
        XR_601, XG_601, XB_601, YR_601, YG_601, YB_601, ZR_601, ZG_601, ZB_601, Y_OFFSET,
    ],
    [
        XR_709, XG_709, XB_709, YR_709, YG_709, YB_709, ZR_709, ZG_709, ZB_709, Y_OFFSET,
    ],
    [
        XR_601FR, XG_601FR, XB_601FR, YR_601FR, YG_601FR, YB_601FR, ZR_601FR, ZG_601FR, ZB_601FR,
        FIX16_HALF,
    ],
    [
        XR_709FR, XG_709FR, XB_709FR, YR_709FR, YG_709FR, YB_709FR, ZR_709FR, ZG_709FR, ZB_709FR,
        FIX16_HALF,
    ],
];

pub const BACKWARD_WEIGHTS: [[i32; 8]; Colorimetry::Length as usize] = [
    [
        XXYM_601, RCRM_601, GCRM_601, GCBM_601, BCBM_601, RN_601, GP_601, BN_601,
    ],
    [
        XXYM_709, RCRM_709, GCRM_709, GCBM_709, BCBM_709, RN_709, GP_709, BN_709,
    ],
    [
        XXYM_601FR, RCRM_601FR, GCRM_601FR, GCBM_601FR, BCBM_601FR, RN_601FR, GP_601FR, BN_601FR,
    ],
    [
        XXYM_709FR, RCRM_709FR, GCRM_709FR, GCBM_709FR, BCBM_709FR, RN_709FR, GP_709FR, BN_709FR,
    ],
];

const SAMPLER_OFFSETS: [[usize; 3]; Sampler::Length as usize] =
    [[1, 2, 3], [2, 1, 0], [2, 1, 0], [2, 1, 0]];

/// Convert fixed point number approximation to uchar, using saturation
///
/// This is equivalent to the following code:
/// ```c
/// if (fix[8 + frac_bits:31] == 0) {
///      return fix >> frac_bits;  // extracts the integer part, no integer underflow
/// } else if (fix < 0) {
///      return 0;       // integer underflow occurred (we got a negative number)
/// } else {
///      return 255;     // no integer underflow occurred, fix is just bigger than 255
/// }
/// ```
///
/// We can get rid of the last branch (else if / else) by observing that:
/// - if fix is negative, fix[31] is 1, fix[31] + 255 = 256, when clamped to uint8 is 0 (just what we want)
/// -    <<  is positive, fix[31] is 0, fix[31] + 255 = 255, when clamped to uint8 is 255 (just what we want)
fn fix_to_u8_sat(fix: i32, frac_bits: i32) -> u8 {
    // Checked: we want the lower 8 bits
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    if (fix & !((256 << frac_bits) - 1)) == 0 {
        ((fix as u32) >> frac_bits) as u8
    } else {
        ((((fix as u32) >> 31) + 255) & 255) as u8
    }
}

/// Extract upper 16 bits of the 32-bit product: (a << 8) * b
///
/// Does the following:
/// ((a << 8) * b) >> 16
///
/// It is equivalent to:
/// (a * b) >> 8
///
/// Works for as long as a * b <= 0x7FFFFFFF (no integer overflow occurs)
/// This is fine for ycbcr to rgb using fixed point 8.14, because:
/// a is in range [0,255] << 8 = [0,65280]
/// b is in range [0,32767]
/// a * b is in range [0,2139029760] = [0,0x7F7F0100]
fn mulhi_i32(a: i32, b: i32) -> i32 {
    (a * b) >> 8
}

/// Deinterleave 2 uchar samples into 2 deinterleaved int
unsafe fn unpack_ui8x2_i32(image: *const u8) -> (i32, i32) {
    (i32::from(*image), i32::from(*image.add(1)))
}

/// Deinterleave 3 uchar samples into 3 deinterleaved int
///
/// sampler=0: little endian
/// sampler=1: big endian, base offset is one.
unsafe fn unpack_ui8x3_i32<const SAMPLER: usize>(image: *const u8) -> (i32, i32, i32) {
    let offsets: &[usize] = &SAMPLER_OFFSETS[SAMPLER];
    (
        i32::from(*image.add(offsets[0])),
        i32::from(*image.add(offsets[1])),
        i32::from(*image.add(offsets[2])),
    )
}

/// Perform affine transformation y = Ax + b, where:
/// - A = (ax, ay, az, 0)
/// - x = (x, y, z, 0)
/// - b = (0, 0, 0, bw)
fn affine_transform(x: i32, y: i32, z: i32, ax: i32, ay: i32, az: i32, bw: i32) -> i32 {
    (ax * x) + (ay * y) + (az * z) + bw
}

/// Converts fixed point number to int
fn fix_to_i32(fix: i32, frac_bits: i32) -> i32 {
    fix >> frac_bits
}

/// Truncate and interleave 2 int to 2 uchar
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
unsafe fn pack_i32x2(image: *mut u8, x: i32, y: i32) {
    // Checked: truncation is explicitly wanted
    *image = x as u8;
    *image.add(1) = y as u8;
}

/// Truncate and interleave 3 int into a dword
/// Last component is set to `DEFAULT_ALPHA`
unsafe fn pack_ui8x3(image: *mut u8, x: u8, y: u8, z: u8) {
    *image = x;
    *image.add(1) = y;
    *image.add(2) = z;
    *image.add(3) = DEFAULT_ALPHA;
}

/// Truncate and interleave 3 int
unsafe fn pack_rgb(image: *mut u8, x: u8, y: u8, z: u8) {
    *image = x;
    *image.add(1) = y;
    *image.add(2) = z;
}

// Called by sse2 and avx2 (process remainder)
#[inline(never)]
pub fn rgb_to_nv12<const SAMPLER: usize, const DEPTH: usize, const COLORIMETRY: usize>(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_strides: (usize, usize),
    dst_buffers: &mut (&mut [u8], &mut [u8]),
) {
    let (y_stride, uv_stride) = dst_strides;

    let weights = &FORWARD_WEIGHTS[COLORIMETRY];
    let xr = weights[0];
    let xg = weights[1];
    let xb = weights[2];
    let yr = weights[3];
    let yg = weights[4];
    let yb = weights[5];
    let zr = weights[6];
    let zg = weights[7];
    let zb = weights[8];
    let yo = weights[9];
    let co = if is_full_range::<COLORIMETRY>() {
        C_OFFSET - FIX18_HALF
    } else {
        C_OFFSET
    };

    let wg_width = width / 2;
    let wg_height = height / 2;

    unsafe {
        let src_group = src_buffer.as_ptr();
        let y_group = dst_buffers.0.as_mut_ptr();
        let uv_group = dst_buffers.1.as_mut_ptr();

        for y in 0..wg_height {
            for x in 0..wg_width {
                let (r00, g00, b00) = unpack_ui8x3_i32::<SAMPLER>(src_group.add(wg_index(
                    2 * x,
                    2 * y,
                    DEPTH,
                    src_stride,
                )));

                let (r10, g10, b10) = unpack_ui8x3_i32::<SAMPLER>(src_group.add(wg_index(
                    2 * x + 1,
                    2 * y,
                    DEPTH,
                    src_stride,
                )));

                pack_i32x2(
                    y_group.add(wg_index(2 * x, 2 * y, 1, y_stride)),
                    fix_to_i32(affine_transform(r00, g00, b00, xr, xg, xb, yo), FIX16),
                    fix_to_i32(affine_transform(r10, g10, b10, xr, xg, xb, yo), FIX16),
                );

                let (r01, g01, b01) = unpack_ui8x3_i32::<SAMPLER>(src_group.add(wg_index(
                    2 * x,
                    2 * y + 1,
                    DEPTH,
                    src_stride,
                )));

                let (r11, g11, b11) = unpack_ui8x3_i32::<SAMPLER>(src_group.add(wg_index(
                    2 * x + 1,
                    2 * y + 1,
                    DEPTH,
                    src_stride,
                )));

                pack_i32x2(
                    y_group.add(wg_index(2 * x, 2 * y + 1, 1, y_stride)),
                    fix_to_i32(affine_transform(r01, g01, b01, xr, xg, xb, yo), FIX16),
                    fix_to_i32(affine_transform(r11, g11, b11, xr, xg, xb, yo), FIX16),
                );

                let sr = (r00 + r10) + (r01 + r11);
                let sg = (g00 + g10) + (g01 + g11);
                let sb = (b00 + b10) + (b01 + b11);
                pack_i32x2(
                    uv_group.add(wg_index(x, y, 2, uv_stride)),
                    fix_to_i32(affine_transform(sr, sg, sb, yr, yg, yb, co), FIX18),
                    fix_to_i32(affine_transform(sr, sg, sb, zr, zg, zb, co), FIX18),
                );
            }
        }
    }
}

#[inline(never)]
pub fn rgb_to_i420<const SAMPLER: usize, const DEPTH: usize, const COLORIMETRY: usize>(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_strides: (usize, usize, usize),
    dst_buffers: &mut (&mut [u8], &mut [u8], &mut [u8]),
) {
    let (y_stride, u_stride, v_stride) = dst_strides;

    let weights = &FORWARD_WEIGHTS[COLORIMETRY];
    let xr = weights[0];
    let xg = weights[1];
    let xb = weights[2];
    let yr = weights[3];
    let yg = weights[4];
    let yb = weights[5];
    let zr = weights[6];
    let zg = weights[7];
    let zb = weights[8];
    let yo = weights[9];
    let co = if is_full_range::<COLORIMETRY>() {
        C_OFFSET - FIX18_HALF
    } else {
        C_OFFSET
    };

    let wg_width = width / 2;
    let wg_height = height / 2;

    unsafe {
        let src_group = src_buffer.as_ptr();
        let y_group = dst_buffers.0.as_mut_ptr();
        let u_group = dst_buffers.1.as_mut_ptr();
        let v_group = dst_buffers.2.as_mut_ptr();

        for y in 0..wg_height {
            for x in 0..wg_width {
                let (r00, g00, b00) = unpack_ui8x3_i32::<SAMPLER>(src_group.add(wg_index(
                    2 * x,
                    2 * y,
                    DEPTH,
                    src_stride,
                )));

                let (r10, g10, b10) = unpack_ui8x3_i32::<SAMPLER>(src_group.add(wg_index(
                    2 * x + 1,
                    2 * y,
                    DEPTH,
                    src_stride,
                )));

                pack_i32x2(
                    y_group.add(wg_index(2 * x, 2 * y, 1, y_stride)),
                    fix_to_i32(affine_transform(r00, g00, b00, xr, xg, xb, yo), FIX16),
                    fix_to_i32(affine_transform(r10, g10, b10, xr, xg, xb, yo), FIX16),
                );

                let (r01, g01, b01) = unpack_ui8x3_i32::<SAMPLER>(src_group.add(wg_index(
                    2 * x,
                    2 * y + 1,
                    DEPTH,
                    src_stride,
                )));

                let (r11, g11, b11) = unpack_ui8x3_i32::<SAMPLER>(src_group.add(wg_index(
                    2 * x + 1,
                    2 * y + 1,
                    DEPTH,
                    src_stride,
                )));

                pack_i32x2(
                    y_group.add(wg_index(2 * x, 2 * y + 1, 1, y_stride)),
                    fix_to_i32(affine_transform(r01, g01, b01, xr, xg, xb, yo), FIX16),
                    fix_to_i32(affine_transform(r11, g11, b11, xr, xg, xb, yo), FIX16),
                );

                let sr = (r00 + r10) + (r01 + r11);
                let sg = (g00 + g10) + (g01 + g11);
                let sb = (b00 + b10) + (b01 + b11);

                let u = u_group.add(wg_index(x, y, 1, u_stride));
                let v = v_group.add(wg_index(x, y, 1, v_stride));

                // Checked: this is proved to not go outside the 8-bit boundary
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                {
                    *u = fix_to_i32(affine_transform(sr, sg, sb, yr, yg, yb, co), FIX18) as u8;
                    *v = fix_to_i32(affine_transform(sr, sg, sb, zr, zg, zb, co), FIX18) as u8;
                }
            }
        }
    }
}

#[inline(never)]
pub fn rgb_to_i444<const SAMPLER: usize, const DEPTH: usize, const COLORIMETRY: usize>(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_strides: (usize, usize, usize),
    dst_buffers: &mut (&mut [u8], &mut [u8], &mut [u8]),
) {
    let (y_stride, u_stride, v_stride) = dst_strides;

    let weights = &FORWARD_WEIGHTS[COLORIMETRY];
    let xr = weights[0];
    let xg = weights[1];
    let xb = weights[2];
    let yr = weights[3];
    let yg = weights[4];
    let yb = weights[5];
    let zr = weights[6];
    let zg = weights[7];
    let zb = weights[8];
    let yo = weights[9];
    let co = if is_full_range::<COLORIMETRY>() {
        C_OFFSET16 - FIX16_HALF
    } else {
        C_OFFSET16
    };

    unsafe {
        let src_group = src_buffer.as_ptr();
        let y_group = dst_buffers.0.as_mut_ptr();
        let u_group = dst_buffers.1.as_mut_ptr();
        let v_group = dst_buffers.2.as_mut_ptr();

        for y in 0..height {
            for x in 0..width {
                let (r, g, b) =
                    unpack_ui8x3_i32::<SAMPLER>(src_group.add(wg_index(x, y, DEPTH, src_stride)));

                let y_data = y_group.add(wg_index(x, y, 1, y_stride));
                let u_data = u_group.add(wg_index(x, y, 1, u_stride));
                let v_data = v_group.add(wg_index(x, y, 1, v_stride));

                // Checked: this is proved to not go outside the 8-bit boundary
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                {
                    *y_data = fix_to_i32(affine_transform(r, g, b, xr, xg, xb, yo), FIX16) as u8;
                    *u_data = fix_to_i32(affine_transform(r, g, b, yr, yg, yb, co), FIX16) as u8;
                    *v_data = fix_to_i32(affine_transform(r, g, b, zr, zg, zb, co), FIX16) as u8;
                }
            }
        }
    }
}

#[inline(never)]
pub fn nv12_to_bgra<const COLORIMETRY: usize>(
    width: usize,
    height: usize,
    src_strides: (usize, usize),
    src_buffers: (&[u8], &[u8]),
    dst_stride: usize,
    dst_buffer: &mut [u8],
) {
    const DEPTH: usize = 4;
    let (y_stride, uv_stride) = src_strides;

    let weights = &BACKWARD_WEIGHTS[COLORIMETRY];
    let xxym = weights[0];
    let rcrm = weights[1];
    let gcrm = weights[2];
    let gcbm = weights[3];
    let bcbm = weights[4];
    let rn = weights[5];
    let gp = weights[6];
    let bn = weights[7];

    let wg_width = width / 2;
    let wg_height = height / 2;

    // In this fixed point approximation, we should take care
    // all the intermediate operations do never overflow nor underflow
    // the 16-bit integer arithmetic.
    //
    // This is not strictly necessary for scalar computations, but we could write
    // an optimized vector implementation that relies on 16-bit integers.
    // And we want identical results between scalar and vector.
    //
    // The proof of this is commented using the numerical analysis on side.
    unsafe {
        let y_group = src_buffers.0.as_ptr();
        let uv_group = src_buffers.1.as_ptr();
        let dst_group = dst_buffer.as_mut_ptr();

        for y in 0..wg_height {
            for x in 0..wg_width {
                let (cb, cr) = unpack_ui8x2_i32(uv_group.add(wg_index(x, y, 2, uv_stride)));

                // [-12600,10280]   Cr(16)              Cr(240)
                let sr = mulhi_i32(cr, rcrm) - rn;
                // [ -9795, 7476]   Cb(240)Cr(240)      Cb(16)Cr(16)
                let sg = -mulhi_i32(cb, gcbm) - mulhi_i32(cr, gcrm) + gp;
                // [-15620,13299]   Cb(16)              Cb(240)
                let sb = mulhi_i32(cb, bcbm) - bn;

                let (y00, y10) = unpack_ui8x2_i32(y_group.add(wg_index(2 * x, 2 * y, 1, y_stride)));

                // [ 1192, 17512]   Y(16)               Y(235)
                let sy00 = mulhi_i32(y00, xxym);

                pack_ui8x3(
                    dst_group.add(wg_index(2 * x, 2 * y, DEPTH, dst_stride)),
                    // [-14428,30811]   Y(16)Cb(16)         Y(235)Cb(240)
                    fix_to_u8_sat(sy00 + sb, FIX6),
                    // [ -8603,24988]   Y(16)Cb(240)Cr(240) Y(235)Cb(16)Cr(16)
                    fix_to_u8_sat(sy00 + sg, FIX6),
                    // [-11408,27792]   Y(16)Cr(16)         Y(235)Cr(240)
                    fix_to_u8_sat(sy00 + sr, FIX6),
                );

                let sy10 = mulhi_i32(y10, xxym);
                pack_ui8x3(
                    dst_group.add(wg_index(2 * x + 1, 2 * y, DEPTH, dst_stride)),
                    fix_to_u8_sat(sy10 + sb, FIX6),
                    fix_to_u8_sat(sy10 + sg, FIX6),
                    fix_to_u8_sat(sy10 + sr, FIX6),
                );

                let (y01, y11) =
                    unpack_ui8x2_i32(y_group.add(wg_index(2 * x, 2 * y + 1, 1, y_stride)));

                let sy01 = mulhi_i32(y01, xxym);
                pack_ui8x3(
                    dst_group.add(wg_index(2 * x, 2 * y + 1, DEPTH, dst_stride)),
                    fix_to_u8_sat(sy01 + sb, FIX6),
                    fix_to_u8_sat(sy01 + sg, FIX6),
                    fix_to_u8_sat(sy01 + sr, FIX6),
                );

                let sy11 = mulhi_i32(y11, xxym);
                pack_ui8x3(
                    dst_group.add(wg_index(2 * x + 1, 2 * y + 1, DEPTH, dst_stride)),
                    fix_to_u8_sat(sy11 + sb, FIX6),
                    fix_to_u8_sat(sy11 + sg, FIX6),
                    fix_to_u8_sat(sy11 + sr, FIX6),
                );
            }
        }
    }
}

#[inline(never)]
pub fn nv12_to_rgb<const COLORIMETRY: usize>(
    width: usize,
    height: usize,
    src_strides: (usize, usize),
    src_buffers: (&[u8], &[u8]),
    dst_stride: usize,
    dst_buffer: &mut [u8],
) {
    const DEPTH: usize = 3;
    let (y_stride, uv_stride) = src_strides;

    let weights = &BACKWARD_WEIGHTS[COLORIMETRY];
    let xxym = weights[0];
    let rcrm = weights[1];
    let gcrm = weights[2];
    let gcbm = weights[3];
    let bcbm = weights[4];
    let rn = weights[5];
    let gp = weights[6];
    let bn = weights[7];

    let wg_width = width / 2;
    let wg_height = height / 2;

    unsafe {
        let y_group = src_buffers.0.as_ptr();
        let uv_group = src_buffers.1.as_ptr();
        let dst_group = dst_buffer.as_mut_ptr();

        for y in 0..wg_height {
            for x in 0..wg_width {
                let (cb, cr) = unpack_ui8x2_i32(uv_group.add(wg_index(x, y, 2, uv_stride)));

                let sr = mulhi_i32(cr, rcrm) - rn;
                let sg = -mulhi_i32(cb, gcbm) - mulhi_i32(cr, gcrm) + gp;
                let sb = mulhi_i32(cb, bcbm) - bn;

                let (y00, y10) = unpack_ui8x2_i32(y_group.add(wg_index(2 * x, 2 * y, 1, y_stride)));

                let sy00 = mulhi_i32(y00, xxym);

                pack_rgb(
                    dst_group.add(wg_index(2 * x, 2 * y, DEPTH, dst_stride)),
                    fix_to_u8_sat(sy00 + sr, FIX6),
                    fix_to_u8_sat(sy00 + sg, FIX6),
                    fix_to_u8_sat(sy00 + sb, FIX6),
                );

                let sy10 = mulhi_i32(y10, xxym);
                pack_rgb(
                    dst_group.add(wg_index(2 * x + 1, 2 * y, DEPTH, dst_stride)),
                    fix_to_u8_sat(sy10 + sr, FIX6),
                    fix_to_u8_sat(sy10 + sg, FIX6),
                    fix_to_u8_sat(sy10 + sb, FIX6),
                );

                let (y01, y11) =
                    unpack_ui8x2_i32(y_group.add(wg_index(2 * x, 2 * y + 1, 1, y_stride)));

                let sy01 = mulhi_i32(y01, xxym);
                pack_rgb(
                    dst_group.add(wg_index(2 * x, 2 * y + 1, DEPTH, dst_stride)),
                    fix_to_u8_sat(sy01 + sr, FIX6),
                    fix_to_u8_sat(sy01 + sg, FIX6),
                    fix_to_u8_sat(sy01 + sb, FIX6),
                );

                let sy11 = mulhi_i32(y11, xxym);
                pack_rgb(
                    dst_group.add(wg_index(2 * x + 1, 2 * y + 1, DEPTH, dst_stride)),
                    fix_to_u8_sat(sy11 + sr, FIX6),
                    fix_to_u8_sat(sy11 + sg, FIX6),
                    fix_to_u8_sat(sy11 + sb, FIX6),
                );
            }
        }
    }
}

#[inline(never)]
pub fn i420_to_bgra<const COLORIMETRY: usize>(
    width: usize,
    height: usize,
    src_strides: (usize, usize, usize),
    src_buffers: (&[u8], &[u8], &[u8]),
    dst_stride: usize,
    dst_buffer: &mut [u8],
) {
    const DEPTH: usize = 4;
    let (y_stride, u_stride, v_stride) = src_strides;

    let weights = &BACKWARD_WEIGHTS[COLORIMETRY];
    let xxym = weights[0];
    let rcrm = weights[1];
    let gcrm = weights[2];
    let gcbm = weights[3];
    let bcbm = weights[4];
    let rn = weights[5];
    let gp = weights[6];
    let bn = weights[7];

    let wg_width = width / 2;
    let wg_height = height / 2;

    unsafe {
        let y_group = src_buffers.0.as_ptr();
        let u_group = src_buffers.1.as_ptr();
        let v_group = src_buffers.2.as_ptr();
        let dst_group = dst_buffer.as_mut_ptr();

        for y in 0..wg_height {
            for x in 0..wg_width {
                let cb = i32::from(*u_group.add(wg_index(x, y, 1, u_stride)));
                let cr = i32::from(*v_group.add(wg_index(x, y, 1, v_stride)));

                let sr = mulhi_i32(cr, rcrm) - rn;
                let sg = -mulhi_i32(cb, gcbm) - mulhi_i32(cr, gcrm) + gp;
                let sb = mulhi_i32(cb, bcbm) - bn;

                let (y00, y10) = unpack_ui8x2_i32(y_group.add(wg_index(2 * x, 2 * y, 1, y_stride)));

                let sy00 = mulhi_i32(y00, xxym);

                pack_ui8x3(
                    dst_group.add(wg_index(2 * x, 2 * y, DEPTH, dst_stride)),
                    fix_to_u8_sat(sy00 + sb, FIX6),
                    fix_to_u8_sat(sy00 + sg, FIX6),
                    fix_to_u8_sat(sy00 + sr, FIX6),
                );

                let sy10 = mulhi_i32(y10, xxym);
                pack_ui8x3(
                    dst_group.add(wg_index(2 * x + 1, 2 * y, DEPTH, dst_stride)),
                    fix_to_u8_sat(sy10 + sb, FIX6),
                    fix_to_u8_sat(sy10 + sg, FIX6),
                    fix_to_u8_sat(sy10 + sr, FIX6),
                );

                let (y01, y11) =
                    unpack_ui8x2_i32(y_group.add(wg_index(2 * x, 2 * y + 1, 1, y_stride)));

                let sy01 = mulhi_i32(y01, xxym);
                pack_ui8x3(
                    dst_group.add(wg_index(2 * x, 2 * y + 1, DEPTH, dst_stride)),
                    fix_to_u8_sat(sy01 + sb, FIX6),
                    fix_to_u8_sat(sy01 + sg, FIX6),
                    fix_to_u8_sat(sy01 + sr, FIX6),
                );

                let sy11 = mulhi_i32(y11, xxym);
                pack_ui8x3(
                    dst_group.add(wg_index(2 * x + 1, 2 * y + 1, DEPTH, dst_stride)),
                    fix_to_u8_sat(sy11 + sb, FIX6),
                    fix_to_u8_sat(sy11 + sg, FIX6),
                    fix_to_u8_sat(sy11 + sr, FIX6),
                );
            }
        }
    }
}

#[inline(never)]
pub fn i444_to_bgra<const COLORIMETRY: usize>(
    width: usize,
    height: usize,
    src_strides: (usize, usize, usize),
    src_buffers: (&[u8], &[u8], &[u8]),
    dst_stride: usize,
    dst_buffer: &mut [u8],
) {
    const DEPTH: usize = 4;
    let (y_stride, u_stride, v_stride) = src_strides;

    let weights = &BACKWARD_WEIGHTS[COLORIMETRY];
    let xxym = weights[0];
    let rcrm = weights[1];
    let gcrm = weights[2];
    let gcbm = weights[3];
    let bcbm = weights[4];
    let rn = weights[5];
    let gp = weights[6];
    let bn = weights[7];

    unsafe {
        let y_group = src_buffers.0.as_ptr();
        let u_group = src_buffers.1.as_ptr();
        let v_group = src_buffers.2.as_ptr();
        let dst_group = dst_buffer.as_mut_ptr();

        for y in 0..height {
            for x in 0..width {
                let l = i32::from(*y_group.add(wg_index(x, y, 1, y_stride)));
                let cb = i32::from(*u_group.add(wg_index(x, y, 1, u_stride)));
                let cr = i32::from(*v_group.add(wg_index(x, y, 1, v_stride)));

                let sr = mulhi_i32(cr, rcrm) - rn;
                let sg = -mulhi_i32(cb, gcbm) - mulhi_i32(cr, gcrm) + gp;
                let sb = mulhi_i32(cb, bcbm) - bn;
                let sl = mulhi_i32(l, xxym);

                pack_ui8x3(
                    dst_group.add(wg_index(x, y, DEPTH, dst_stride)),
                    fix_to_u8_sat(sl + sb, FIX6),
                    fix_to_u8_sat(sl + sg, FIX6),
                    fix_to_u8_sat(sl + sr, FIX6),
                );
            }
        }
    }
}

#[inline(never)]
pub fn rgb_to_bgra(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_stride: usize,
    dst_buffer: &mut [u8],
) {
    const HIGH_MASK: u64 = 0xFFFF_FFFF_0000_0000;
    const LOW_MASK: u64 = 0x0000_0000_FFFF_FFFF;
    const ALPHA_MASK: u64 = 0x0000_00FF_0000_00FF;
    const SRC_DEPTH: usize = 3;
    const DST_DEPTH: usize = 4;
    const ITEMS_PER_ITERATION: usize = 8;

    let src_stride_diff = src_stride - (SRC_DEPTH * width);
    let dst_stride_diff = dst_stride - (DST_DEPTH * width);

    unsafe {
        let src_group = src_buffer.as_ptr();
        let dst_group = dst_buffer.as_mut_ptr();
        let mut src_offset = 0;
        let mut dst_offset = 0;

        for y in 0..height {
            let is_last_line = y == height - 1;

            // For single swap iteration, since the swap is done on 32 bits while the input is only
            // 24 bits (RGB), to avoid reading an extra byte of memory that could be outside the
            // boundaries of the buffer it is necessary to check if the there is at least one byte
            // of stride in the input buffer, in that case we can read till the end of the buffer.
            let single_swap_iterations = if is_last_line || src_stride_diff < 1 {
                width - 1
            } else {
                width
            };

            // For multiple swap iteration, since each swap reads two extra bytes it is necessary
            // to check if there is enough space to read them without going out of boundaries.
            // Since each step retrieves items_per_iteration colors, it is checked if the width of
            // the image is a multiple of items_per_iteration,
            // in that case it is checked if there are 2 extra bytes of stride, if there is not
            // enough space then the number of multiple swap iterarions is reduced by items_per_iteration
            // since it is not safe to read till the end of the buffer, otherwise it is not.
            let multi_swap_iterations = if is_last_line
                || (ITEMS_PER_ITERATION * (width / ITEMS_PER_ITERATION) == width
                    && src_stride_diff < 2)
            {
                ITEMS_PER_ITERATION * ((width - 1) / ITEMS_PER_ITERATION)
            } else {
                ITEMS_PER_ITERATION * (width / ITEMS_PER_ITERATION)
            };

            let mut x = 0;

            // Retrieves items_per_iteration colors per cycle if possible
            for _ in (0..multi_swap_iterations).step_by(ITEMS_PER_ITERATION) {
                let rgb0: u64 = loadu(src_group.add(src_offset).cast());
                let rgb1: u64 = loadu(src_group.add(src_offset + 6).cast());
                let rgb2: u64 = loadu(src_group.add(src_offset + 12).cast());
                let rgb3: u64 = loadu(src_group.add(src_offset + 18).cast());

                let bgra: *mut i64 = dst_group.add(dst_offset).cast();

                // Checked: we want to reinterpret the bits
                #[allow(clippy::cast_possible_wrap)]
                {
                    storeu(
                        bgra,
                        _bswap64(
                            (((rgb0 >> 16) & LOW_MASK) | ((rgb0 << 40) & HIGH_MASK) | ALPHA_MASK)
                                as i64,
                        ),
                    );
                    storeu(
                        bgra.add(1),
                        _bswap64(
                            (((rgb1 >> 16) & LOW_MASK) | ((rgb1 << 40) & HIGH_MASK) | ALPHA_MASK)
                                as i64,
                        ),
                    );
                    storeu(
                        bgra.add(2),
                        _bswap64(
                            (((rgb2 >> 16) & LOW_MASK) | ((rgb2 << 40) & HIGH_MASK) | ALPHA_MASK)
                                as i64,
                        ),
                    );
                    storeu(
                        bgra.add(3),
                        _bswap64(
                            (((rgb3 >> 16) & LOW_MASK) | ((rgb3 << 40) & HIGH_MASK) | ALPHA_MASK)
                                as i64,
                        ),
                    );
                }

                x += ITEMS_PER_ITERATION;
                src_offset += SRC_DEPTH * ITEMS_PER_ITERATION;
                dst_offset += DST_DEPTH * ITEMS_PER_ITERATION;
            }

            // Retrieves the ramaining colors in the line
            while x < single_swap_iterations {
                let ptr: *const u32 = src_group.add(src_offset).cast();

                // Checked: we want to reinterpret the bits
                #[allow(clippy::cast_possible_wrap)]
                storeu(
                    dst_group.add(dst_offset).cast(),
                    _bswap(((loadu(ptr) << 8) | 0xFF) as i32),
                );

                x += 1;
                src_offset += SRC_DEPTH;
                dst_offset += DST_DEPTH;
            }

            // If the input stride is not at least 1 byte it directly copies byte per byte,
            // this could happen once per line
            if x < width {
                *dst_group.add(dst_offset) = *src_group.add(src_offset + 2);
                *dst_group.add(dst_offset + 1) = *src_group.add(src_offset + 1);
                *dst_group.add(dst_offset + 2) = *src_group.add(src_offset);
                *dst_group.add(dst_offset + 3) = 0xFF;

                src_offset += SRC_DEPTH;
                dst_offset += DST_DEPTH;
            }

            src_offset += src_stride_diff;
            dst_offset += dst_stride_diff;
        }
    }
}

#[inline(never)]
pub fn bgr_to_rgb(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_stride: usize,
    dst_buffer: &mut [u8],
) {
    const DEPTH: usize = 3;
    const ITEMS_PER_ITERATION: usize = 8;

    let limit_iterations = lower_multiple_of_pot(width, ITEMS_PER_ITERATION);

    unsafe {
        let src_group = src_buffer.as_ptr();
        let dst_group = dst_buffer.as_mut_ptr();

        for i in 0..height {
            let mut y = 0;
            let mut src_offset = src_stride * i;
            let mut dst_offset = dst_stride * i;

            while y < limit_iterations {
                let src_ptr: *const u64 = src_group.add(src_offset).cast();
                let bgr0 = loadu(src_ptr);
                let bgr1 = loadu(src_ptr.add(1));
                let bgr2 = loadu(src_ptr.add(2));

                // Checked: we want to reinterpret the bits
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_possible_wrap
                )]
                let (swap_rgb0, swap_rgb1, swap_rgb2) = (
                    // swap_rgb0: B0 G0 R0 B1 G1 R1 B2 G2
                    _bswap64(bgr0 as i64) as u64,
                    // swap_rgb1: R2 B3 G3 R3 B4 G4 R4 B5
                    _bswap64(bgr1 as i64) as u64,
                    // swap_rgb2: G5 R5 B6 G6 R6 B7 G7 R7
                    _bswap64(bgr2 as i64) as u64,
                );

                // rgb0: G2 R2 B1 G1 R1 B0 G0 R0
                let rgb0 = (swap_rgb0 >> 40)
                    | ((swap_rgb0 << 8) & 0x0000_FFFF_FF00_0000)
                    | (swap_rgb0 << 56)
                    | ((swap_rgb1 & 0xFF00_0000_0000_0000) >> 8);
                // rgb1: R5 B4 G4 R4 B3 G3 R3 B2
                let rgb1 = ((swap_rgb1 >> 24) & 0xFFFF_FF00)
                    | ((swap_rgb1 << 24) & 0x00FF_FFFF_0000_0000)
                    | ((swap_rgb2 & 0x00FF_0000_0000_0000) << 8)
                    | ((swap_rgb0 & 0xFF00) >> 8);
                // rgb2: B7 G7 R7 B6 G6 R6 B5 G5
                let rgb2 = ((swap_rgb2 & 0xFF_FFFF) << 40)
                    | ((swap_rgb2 >> 8) & 0xFF_FFFF_0000)
                    | (swap_rgb2 >> 56)
                    | ((swap_rgb1 & 0xFF) << 8);

                let dst_ptr: *mut u64 = dst_group.add(dst_offset).cast();
                storeu(dst_ptr, rgb0);
                storeu(dst_ptr.add(1), rgb1);
                storeu(dst_ptr.add(2), rgb2);

                src_offset += DEPTH * ITEMS_PER_ITERATION;
                dst_offset += DEPTH * ITEMS_PER_ITERATION;
                y += ITEMS_PER_ITERATION;
            }

            while y < width {
                *dst_group.add(dst_offset) = *src_group.add(src_offset + 2);
                *dst_group.add(dst_offset + 1) = *src_group.add(src_offset + 1);
                *dst_group.add(dst_offset + 2) = *src_group.add(src_offset);

                src_offset += DEPTH;
                dst_offset += DEPTH;
                y += 1;
            }
        }
    }
}

// Internal module functions
#[inline(never)]
fn bgra_to_rgb(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_stride: usize,
    dst_buffer: &mut [u8],
) {
    const SRC_DEPTH: usize = 4;
    const DST_DEPTH: usize = 3;
    const ITEMS_PER_ITERATION_4X: usize = 8;
    const HIGH_MASK: u64 = 0xFFFF_FF00_0000_0000;
    const LOW_MASK: u64 = 0x0000_00FF_FFFF_0000;

    let src_stride_diff = src_stride - (SRC_DEPTH * width);
    let dst_stride_diff = dst_stride - (DST_DEPTH * width);
    let limit_4x = lower_multiple_of_pot(width, ITEMS_PER_ITERATION_4X);

    unsafe {
        let src_group = src_buffer.as_ptr();
        let dst_group = dst_buffer.as_mut_ptr();

        for i in 0..height {
            let mut y = 0;
            let mut src_offset = ((SRC_DEPTH * width) + src_stride_diff) * i;
            let mut dst_offset = ((DST_DEPTH * width) + dst_stride_diff) * i;

            while y < limit_4x {
                let src_ptr: *const u64 = src_group.add(src_offset).cast();
                let bgra0 = loadu(src_ptr);
                let bgra1 = loadu(src_ptr.add(1));
                let bgra2 = loadu(src_ptr.add(2));
                let bgra3 = loadu(src_ptr.add(3));

                // Checked: we want to reinterpret the bits
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_possible_wrap
                )]
                let (rgb0, rgb1, rgb2, rgb3) = (
                    _bswap64((((bgra0 << 40) & HIGH_MASK) | ((bgra0 >> 16) & LOW_MASK)) as i64)
                        as u64,
                    _bswap64((((bgra1 << 40) & HIGH_MASK) | ((bgra1 >> 16) & LOW_MASK)) as i64)
                        as u64,
                    _bswap64((((bgra2 << 40) & HIGH_MASK) | ((bgra2 >> 16) & LOW_MASK)) as i64)
                        as u64,
                    _bswap64((((bgra3 << 40) & HIGH_MASK) | ((bgra3 >> 16) & LOW_MASK)) as i64)
                        as u64,
                );

                let dst_ptr: *mut u64 = dst_group.add(dst_offset).cast();
                storeu(dst_ptr, (rgb1 << 48) | rgb0);
                storeu(dst_ptr.add(1), (rgb1 >> 16) | (rgb2 << 32));
                storeu(dst_ptr.add(2), (rgb2 >> 32) | (rgb3 << 16));

                src_offset += SRC_DEPTH * ITEMS_PER_ITERATION_4X;
                dst_offset += DST_DEPTH * ITEMS_PER_ITERATION_4X;
                y += ITEMS_PER_ITERATION_4X;
            }

            while y < width {
                *dst_group.add(dst_offset) = *src_group.add(src_offset + 2);
                *dst_group.add(dst_offset + 1) = *src_group.add(src_offset + 1);
                *dst_group.add(dst_offset + 2) = *src_group.add(src_offset);

                src_offset += SRC_DEPTH;
                dst_offset += DST_DEPTH;
                y += 1;
            }
        }
    }
}

#[inline(never)]
fn nv12_rgb<const COLORIMETRY: usize, const DEPTH: usize>(
    width: u32,
    height: u32,
    last_src_plane: usize,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: usize,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    // Degenerate case, trivially accept
    if width == 0 || height == 0 {
        return true;
    }

    // Check there are sufficient strides and buffers
    if last_src_plane >= src_strides.len()
        || last_src_plane >= src_buffers.len()
        || dst_strides.is_empty()
        || dst_buffers.is_empty()
    {
        return false;
    }

    // Check subsampling limits
    let w = width as usize;
    let h = height as usize;
    let ch = h / 2;
    let rgb_stride = DEPTH * w;

    // Compute actual strides
    let src_strides = (
        compute_stride(src_strides[0], w),
        compute_stride(src_strides[last_src_plane], w),
    );
    let dst_stride = compute_stride(dst_strides[0], rgb_stride);

    // Ensure there is sufficient data in the buffers according
    // to the image dimensions and computed strides
    let mut src_buffers = (src_buffers[0], src_buffers[last_src_plane]);
    if last_src_plane == 0 {
        if src_buffers.0.len() < src_strides.0 * h {
            return false;
        }

        src_buffers = src_buffers.0.split_at(src_strides.0 * h);
    }

    let dst_buffer = &mut *dst_buffers[0];
    if out_of_bounds(src_buffers.0.len(), src_strides.0, h - 1, w)
        || out_of_bounds(src_buffers.1.len(), src_strides.1, ch - 1, w)
        || out_of_bounds(dst_buffer.len(), dst_stride, h - 1, rgb_stride)
    {
        return false;
    }

    if DEPTH == 4 {
        nv12_to_bgra::<COLORIMETRY>(w, h, src_strides, src_buffers, dst_stride, dst_buffer);
    } else {
        nv12_to_rgb::<COLORIMETRY>(w, h, src_strides, src_buffers, dst_stride, dst_buffer);
    }

    true
}

#[inline(never)]
fn i420_rgb<const COLORIMETRY: usize, const DEPTH: usize>(
    width: u32,
    height: u32,
    _last_src_plane: usize,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: usize,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    // Degenerate case, trivially accept
    if width == 0 || height == 0 {
        return true;
    }

    // Check there are sufficient strides and buffers
    if src_strides.len() < 3
        || src_buffers.len() < 3
        || dst_strides.is_empty()
        || dst_buffers.is_empty()
    {
        return false;
    }

    // Check subsampling limits
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;
    let rgb_stride = DEPTH * w;

    // Compute actual strides
    let src_strides = (
        compute_stride(src_strides[0], w),
        compute_stride(src_strides[1], cw),
        compute_stride(src_strides[2], cw),
    );
    let dst_stride = compute_stride(dst_strides[0], rgb_stride);

    // Ensure there is sufficient data in the buffers according
    // to the image dimensions and computed strides
    let src_buffers = (src_buffers[0], src_buffers[1], src_buffers[2]);
    let dst_buffer = &mut *dst_buffers[0];
    if out_of_bounds(src_buffers.0.len(), src_strides.0, h - 1, w)
        || out_of_bounds(src_buffers.1.len(), src_strides.1, ch - 1, cw)
        || out_of_bounds(src_buffers.2.len(), src_strides.2, ch - 1, cw)
        || out_of_bounds(dst_buffer.len(), dst_stride, h - 1, rgb_stride)
    {
        return false;
    }

    i420_to_bgra::<COLORIMETRY>(w, h, src_strides, src_buffers, dst_stride, dst_buffer);

    true
}

#[inline(never)]
fn i444_rgb<const COLORIMETRY: usize, const DEPTH: usize>(
    width: u32,
    height: u32,
    _last_src_plane: usize,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: usize,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    // Degenerate case, trivially accept
    if width == 0 || height == 0 {
        return true;
    }

    // Check there are sufficient strides and buffers
    if src_strides.len() < 3
        || src_buffers.len() < 3
        || dst_strides.is_empty()
        || dst_buffers.is_empty()
    {
        return false;
    }

    let w = width as usize;
    let h = height as usize;
    let rgb_stride = DEPTH * w;

    // Compute actual strides
    let src_strides = (
        compute_stride(src_strides[0], w),
        compute_stride(src_strides[1], w),
        compute_stride(src_strides[2], w),
    );
    let dst_stride = compute_stride(dst_strides[0], rgb_stride);

    // Ensure there is sufficient data in the buffers according
    // to the image dimensions and computed strides
    let src_buffers = (src_buffers[0], src_buffers[1], src_buffers[2]);
    let dst_buffer = &mut *dst_buffers[0];
    if out_of_bounds(src_buffers.0.len(), src_strides.0, h - 1, w)
        || out_of_bounds(src_buffers.1.len(), src_strides.1, h - 1, w)
        || out_of_bounds(src_buffers.2.len(), src_strides.2, h - 1, w)
        || out_of_bounds(dst_buffer.len(), dst_stride, h - 1, rgb_stride)
    {
        return false;
    }

    i444_to_bgra::<COLORIMETRY>(w, h, src_strides, src_buffers, dst_stride, dst_buffer);

    true
}

#[inline(never)]
fn rgb_i444<const SAMPLER: usize, const DEPTH: usize, const COLORIMETRY: usize>(
    width: u32,
    height: u32,
    _last_src_plane: usize,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: usize,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    // Degenerate case, trivially accept
    if width == 0 || height == 0 {
        return true;
    }

    // Check there are sufficient strides and buffers
    if src_strides.is_empty()
        || src_buffers.is_empty()
        || dst_strides.len() < 3
        || dst_buffers.len() < 3
    {
        return false;
    }

    let w = width as usize;
    let h = height as usize;
    let rgb_stride = DEPTH * w;

    // Compute actual strides
    let src_stride = compute_stride(src_strides[0], rgb_stride);
    let dst_strides = (
        compute_stride(dst_strides[0], w),
        compute_stride(dst_strides[1], w),
        compute_stride(dst_strides[2], w),
    );

    // Ensure there is sufficient data in the buffers according
    // to the image dimensions and computed strides
    let src_buffer = &src_buffers[0];
    let (y_plane, uv_plane) = dst_buffers.split_at_mut(1);
    let (u_plane, v_plane) = uv_plane.split_at_mut(1);
    let (y_plane, u_plane, v_plane) = (&mut *y_plane[0], &mut *u_plane[0], &mut *v_plane[0]);
    if out_of_bounds(src_buffer.len(), src_stride, h - 1, rgb_stride)
        || out_of_bounds(y_plane.len(), dst_strides.0, h - 1, w)
        || out_of_bounds(u_plane.len(), dst_strides.1, h - 1, w)
        || out_of_bounds(v_plane.len(), dst_strides.2, h - 1, w)
    {
        return false;
    }

    rgb_to_i444::<SAMPLER, DEPTH, COLORIMETRY>(
        w,
        h,
        src_stride,
        src_buffer,
        dst_strides,
        &mut (y_plane, u_plane, v_plane),
    );

    true
}

#[inline(never)]
fn rgb_i420<const SAMPLER: usize, const DEPTH: usize, const COLORIMETRY: usize>(
    width: u32,
    height: u32,
    _last_src_plane: usize,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: usize,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    // Degenerate case, trivially accept
    if width == 0 || height == 0 {
        return true;
    }

    // Check there are sufficient strides and buffers
    if src_strides.is_empty()
        || src_buffers.is_empty()
        || dst_strides.len() < 3
        || dst_buffers.len() < 3
    {
        return false;
    }

    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;
    let rgb_stride = DEPTH * w;

    // Compute actual strides
    let src_stride = compute_stride(src_strides[0], rgb_stride);
    let dst_strides = (
        compute_stride(dst_strides[0], w),
        compute_stride(dst_strides[1], cw),
        compute_stride(dst_strides[2], cw),
    );

    // Ensure there is sufficient data in the buffers according
    // to the image dimensions and computed strides
    let src_buffer = &src_buffers[0];
    let (y_plane, uv_plane) = dst_buffers.split_at_mut(1);
    let (u_plane, v_plane) = uv_plane.split_at_mut(1);
    let (y_plane, u_plane, v_plane) = (&mut *y_plane[0], &mut *u_plane[0], &mut *v_plane[0]);
    if out_of_bounds(src_buffer.len(), src_stride, h - 1, rgb_stride)
        || out_of_bounds(y_plane.len(), dst_strides.0, h - 1, w)
        || out_of_bounds(u_plane.len(), dst_strides.1, ch - 1, cw)
        || out_of_bounds(v_plane.len(), dst_strides.2, ch - 1, cw)
    {
        return false;
    }

    rgb_to_i420::<SAMPLER, DEPTH, COLORIMETRY>(
        w,
        h,
        src_stride,
        src_buffer,
        dst_strides,
        &mut (y_plane, u_plane, v_plane),
    );

    true
}

#[inline(never)]
fn rgb_nv12<const SAMPLER: usize, const DEPTH: usize, const COLORIMETRY: usize>(
    width: u32,
    height: u32,
    _last_src_plane: usize,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: usize,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    // Degenerate case, trivially accept
    if width == 0 || height == 0 {
        return true;
    }

    // Check there are sufficient strides and buffers
    if src_strides.is_empty()
        || src_buffers.is_empty()
        || last_dst_plane >= dst_strides.len()
        || last_dst_plane >= dst_buffers.len()
    {
        return false;
    }

    let w = width as usize;
    let h = height as usize;
    let ch = h / 2;
    let rgb_stride = DEPTH * w;

    // Compute actual strides
    let src_stride = compute_stride(src_strides[0], rgb_stride);
    let dst_strides = (
        compute_stride(dst_strides[0], w),
        compute_stride(dst_strides[last_dst_plane], w),
    );

    // Ensure there is sufficient data in the buffers according
    // to the image dimensions and computed strides
    let src_buffer = &src_buffers[0];
    if last_dst_plane == 0 && dst_buffers[last_dst_plane].len() < dst_strides.0 * h {
        return false;
    }

    let (y_plane, uv_plane) = if last_dst_plane == 0 {
        dst_buffers[last_dst_plane].split_at_mut(dst_strides.0 * h)
    } else {
        let (y_plane, uv_plane) = dst_buffers.split_at_mut(last_dst_plane);

        (&mut *y_plane[0], &mut *uv_plane[0])
    };

    if out_of_bounds(src_buffer.len(), src_stride, h - 1, rgb_stride)
        || out_of_bounds(y_plane.len(), dst_strides.0, h - 1, w)
        || out_of_bounds(uv_plane.len(), dst_strides.1, ch - 1, w)
    {
        return false;
    }

    rgb_to_nv12::<SAMPLER, DEPTH, COLORIMETRY>(
        w,
        h,
        src_stride,
        src_buffer,
        dst_strides,
        &mut (y_plane, uv_plane),
    );

    true
}

rgb_to_yuv_converter!(Argb, I420, Bt601);
rgb_to_yuv_converter!(Argb, I420, Bt601FR);
rgb_to_yuv_converter!(Argb, I420, Bt709);
rgb_to_yuv_converter!(Argb, I420, Bt709FR);
rgb_to_yuv_converter!(Argb, I444, Bt601);
rgb_to_yuv_converter!(Argb, I444, Bt601FR);
rgb_to_yuv_converter!(Argb, I444, Bt709);
rgb_to_yuv_converter!(Argb, I444, Bt709FR);
rgb_to_yuv_converter!(Argb, Nv12, Bt601);
rgb_to_yuv_converter!(Argb, Nv12, Bt601FR);
rgb_to_yuv_converter!(Argb, Nv12, Bt709);
rgb_to_yuv_converter!(Argb, Nv12, Bt709FR);
rgb_to_yuv_converter!(Bgr, I420, Bt601);
rgb_to_yuv_converter!(Bgr, I420, Bt601FR);
rgb_to_yuv_converter!(Bgr, I420, Bt709);
rgb_to_yuv_converter!(Bgr, I420, Bt709FR);
rgb_to_yuv_converter!(Bgr, I444, Bt601);
rgb_to_yuv_converter!(Bgr, I444, Bt601FR);
rgb_to_yuv_converter!(Bgr, I444, Bt709);
rgb_to_yuv_converter!(Bgr, I444, Bt709FR);
rgb_to_yuv_converter!(Bgr, Nv12, Bt601);
rgb_to_yuv_converter!(Bgr, Nv12, Bt601FR);
rgb_to_yuv_converter!(Bgr, Nv12, Bt709);
rgb_to_yuv_converter!(Bgr, Nv12, Bt709FR);
rgb_to_yuv_converter!(Bgra, I420, Bt601);
rgb_to_yuv_converter!(Bgra, I420, Bt601FR);
rgb_to_yuv_converter!(Bgra, I420, Bt709);
rgb_to_yuv_converter!(Bgra, I420, Bt709FR);
rgb_to_yuv_converter!(Bgra, I444, Bt601);
rgb_to_yuv_converter!(Bgra, I444, Bt601FR);
rgb_to_yuv_converter!(Bgra, I444, Bt709);
rgb_to_yuv_converter!(Bgra, I444, Bt709FR);
rgb_to_yuv_converter!(Bgra, Nv12, Bt601);
rgb_to_yuv_converter!(Bgra, Nv12, Bt601FR);
rgb_to_yuv_converter!(Bgra, Nv12, Bt709);
rgb_to_yuv_converter!(Bgra, Nv12, Bt709FR);
yuv_to_rgb_converter!(I420, Bt601, Bgra);
yuv_to_rgb_converter!(I420, Bt601FR, Bgra);
yuv_to_rgb_converter!(I420, Bt709, Bgra);
yuv_to_rgb_converter!(I420, Bt709FR, Bgra);
yuv_to_rgb_converter!(I444, Bt601, Bgra);
yuv_to_rgb_converter!(I444, Bt601FR, Bgra);
yuv_to_rgb_converter!(I444, Bt709, Bgra);
yuv_to_rgb_converter!(I444, Bt709FR, Bgra);
yuv_to_rgb_converter!(Nv12, Bt601, Bgra);
yuv_to_rgb_converter!(Nv12, Bt601FR, Bgra);
yuv_to_rgb_converter!(Nv12, Bt709, Bgra);
yuv_to_rgb_converter!(Nv12, Bt709FR, Bgra);
yuv_to_rgb_converter!(Nv12, Bt601, Rgb);
yuv_to_rgb_converter!(Nv12, Bt601FR, Rgb);
yuv_to_rgb_converter!(Nv12, Bt709, Rgb);
yuv_to_rgb_converter!(Nv12, Bt709FR, Rgb);

pub fn rgb_bgra(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    const SRC_DEPTH: usize = 3;
    const DST_DEPTH: usize = 4;

    // Degenerate case, trivially accept
    if width == 0 || height == 0 {
        return true;
    }

    // Check there are sufficient strides and buffers
    if src_strides.is_empty()
        || src_buffers.is_empty()
        || dst_strides.is_empty()
        || dst_buffers.is_empty()
    {
        return false;
    }

    let w = width as usize;
    let h = height as usize;

    // Compute actual strides
    let src_stride = compute_stride(src_strides[0], SRC_DEPTH * w);
    let dst_stride = compute_stride(dst_strides[0], DST_DEPTH * w);

    // Ensure there is sufficient data in the buffers according
    // to the image dimensions and computed strides
    let src_buffer = src_buffers[0];
    let dst_buffer = &mut *dst_buffers[0];
    if out_of_bounds(src_buffer.len(), src_stride, h - 1, w)
        || out_of_bounds(dst_buffer.len(), dst_stride, h - 1, w)
    {
        return false;
    }

    rgb_to_bgra(w, h, src_stride, src_buffer, dst_stride, dst_buffer);

    true
}

pub fn bgra_rgb(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    const SRC_DEPTH: usize = 4;
    const DST_DEPTH: usize = 3;

    // Degenerate case, trivially accept
    if width == 0 || height == 0 {
        return true;
    }

    // Check there are sufficient strides and buffers
    if src_strides.is_empty()
        || src_buffers.is_empty()
        || dst_strides.is_empty()
        || dst_buffers.is_empty()
    {
        return false;
    }

    let w = width as usize;
    let h = height as usize;

    // Compute actual strides
    let src_stride = compute_stride(src_strides[0], SRC_DEPTH * w);
    let dst_stride = compute_stride(dst_strides[0], DST_DEPTH * w);

    // Ensure there is sufficient data in the buffers according
    // to the image dimensions and computed strides
    let src_buffer = src_buffers[0];
    let dst_buffer = &mut *dst_buffers[0];
    if out_of_bounds(src_buffer.len(), src_stride, h - 1, w)
        || out_of_bounds(dst_buffer.len(), dst_stride, h - 1, w)
    {
        return false;
    }

    bgra_to_rgb(w, h, src_stride, src_buffer, dst_stride, dst_buffer);

    true
}

pub fn bgr_rgb(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    const DEPTH: usize = 3;

    // Degenerate case, trivially accept
    if width == 0 || height == 0 {
        return true;
    }

    // Check there are sufficient strides and buffers
    if src_strides.is_empty()
        || src_buffers.is_empty()
        || dst_strides.is_empty()
        || dst_buffers.is_empty()
    {
        return false;
    }

    let w = width as usize;
    let h = height as usize;

    // Compute actual strides
    let src_stride = compute_stride(src_strides[0], DEPTH * w);
    let dst_stride = compute_stride(dst_strides[0], DEPTH * w);

    // Ensure there is sufficient data in the buffers according
    // to the image dimensions and computed strides
    let src_buffer = src_buffers[0];
    let dst_buffer = &mut *dst_buffers[0];
    if out_of_bounds(src_buffer.len(), src_stride, h - 1, w)
        || out_of_bounds(dst_buffer.len(), dst_stride, h - 1, w)
    {
        return false;
    }

    bgr_to_rgb(w, h, src_stride, src_buffer, dst_stride, dst_buffer);

    true
}
