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

use crate::convert_image::common::*;

const FORWARD_WEIGHTS: [[i32; 9]; Colorimetry::Length as usize] = [
    [
        XR_601, XG_601, XB_601, YR_601, YG_601, YB_601, ZR_601, ZG_601, ZB_601,
    ],
    [
        XR_709, XG_709, XB_709, YR_709, YG_709, YB_709, ZR_709, ZG_709, ZB_709,
    ],
];

const BACKWARD_WEIGHTS: [[i32; 8]; Colorimetry::Length as usize] = [
    [
        XXYM_601, RCRM_601, GCRM_601, GCBM_601, BCBM_601, RN_601, GP_601, BN_601,
    ],
    [
        XXYM_709, RCRM_709, GCRM_709, GCBM_709, BCBM_709, RN_709, GP_709, BN_709,
    ],
];

const SAMPLER_OFFSETS: [[usize; 3]; Sampler::Length as usize] =
    [[1, 2, 3], [2, 1, 0], [2, 1, 0], [2, 1, 0]];

/// Convert fixed point number approximation to uchar, using saturation
///
/// This is equivalent to the following code:
/// if (fix[8 + frac_bits:31] == 0) {
///      return fix >> frac_bits;  // extracts the integer part, no integer underflow
/// } else if (fix < 0) {
///      return 0;       // integer underflow occurred (we got a negative number)
/// } else {
///      return 255;     // no integer underflow occurred, fix is just bigger than 255
/// }
///
/// We can get rid of the last branch (else if / else) by observing that:
/// - if fix is negative, fix[31] is 1, fix[31] + 255 = 256, when clamped to uint8 is 0 (just what we want)
/// -    <<  is positive, fix[31] is 0, fix[31] + 255 = 255, when clamped to uint8 is 255 (just what we want)
fn fix_to_u8_sat(fix: i32, frac_bits: i32) -> u8 {
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
unsafe fn unpack_ui8x3_i32(image: *const u8, sampler: Sampler) -> (i32, i32, i32) {
    let offsets: &[usize] = &SAMPLER_OFFSETS[sampler as usize];
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
unsafe fn pack_i32x2(image: *mut u8, x: i32, y: i32) {
    *image = x as u8;
    *image.add(1) = y as u8;
}

/// Truncate and interleave 3 int into a dword
/// Last component is set to DEFAULT_ALPHA
unsafe fn pack_ui8x3(image: *mut u8, x: u8, y: u8, z: u8) {
    *image = x;
    *image.add(1) = y;
    *image.add(2) = z;
    *image.add(3) = DEFAULT_ALPHA;
}

#[inline(always)]
fn lrgb_to_yuv(
    width: u32,
    height: u32,
    _last_src_plane: usize,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: usize,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
    channels: PixelFormatChannels,
    colorimetry: Colorimetry,
    sampler: Sampler,
) -> bool {
    if (last_dst_plane >= dst_strides.len())
        || (last_dst_plane >= dst_buffers.len())
        || src_strides.is_empty()
        || src_buffers.is_empty()
    {
        return false;
    }

    let depth = channels as usize;
    let col_count = width as usize;
    let line_count = height as usize;
    let packed_rgb_stride = depth * col_count;
    let packed_nv12_stride = col_count;

    let rgb_stride = if src_strides[0] == 0 {
        packed_rgb_stride
    } else {
        src_strides[0]
    };

    let y_stride = if dst_strides[0] == 0 {
        packed_nv12_stride
    } else {
        dst_strides[0]
    };

    let uv_stride = if dst_strides[last_dst_plane] == 0 {
        packed_nv12_stride
    } else {
        dst_strides[last_dst_plane]
    };

    let rgb_plane = &src_buffers[0];
    let (first, last) = dst_buffers.split_at_mut(last_dst_plane);
    let interplane_split = y_stride * line_count;
    if last_dst_plane == 0 && interplane_split > last[0].len() {
        return false;
    }

    let (y_plane, uv_plane) = if last_dst_plane == 0 {
        last[0].split_at_mut(interplane_split)
    } else {
        (&mut first[0][..], &mut last[0][..])
    };

    if line_count == 0 {
        return true;
    }

    let max_stride = usize::max_value() / line_count;
    if (y_stride > max_stride) || (uv_stride > max_stride) || (rgb_stride > max_stride) {
        return false;
    }

    let wg_height = line_count / 2;
    if y_stride * line_count > y_plane.len()
        || uv_stride * wg_height > uv_plane.len()
        || rgb_stride * line_count > rgb_plane.len()
    {
        return false;
    }

    // The following constants will be automatically propagated as immediate values
    // inside the operations by optimizing compilers.
    let col = colorimetry as usize;
    let xr = FORWARD_WEIGHTS[col][0];
    let xg = FORWARD_WEIGHTS[col][1];
    let xb = FORWARD_WEIGHTS[col][2];
    let yr = FORWARD_WEIGHTS[col][3];
    let yg = FORWARD_WEIGHTS[col][4];
    let yb = FORWARD_WEIGHTS[col][5];
    let zr = FORWARD_WEIGHTS[col][6];
    let zg = FORWARD_WEIGHTS[col][7];
    let zb = FORWARD_WEIGHTS[col][8];
    unsafe {
        let rgb_group = rgb_plane.as_ptr();
        let y_group = y_plane.as_mut_ptr();
        let uv_group = uv_plane.as_mut_ptr();
        let wg_width = col_count / 2;

        for y in 0..wg_height {
            for x in 0..wg_width {
                let (r00, g00, b00) = unpack_ui8x3_i32(
                    rgb_group.add(wg_index(2 * x, 2 * y, depth, rgb_stride)),
                    sampler,
                );

                let (r10, g10, b10) = unpack_ui8x3_i32(
                    rgb_group.add(wg_index(2 * x + 1, 2 * y, depth, rgb_stride)),
                    sampler,
                );

                pack_i32x2(
                    y_group.add(wg_index(2 * x, 2 * y, 1, y_stride)),
                    fix_to_i32(affine_transform(r00, g00, b00, xr, xg, xb, Y_OFFSET), FIX16),
                    fix_to_i32(affine_transform(r10, g10, b10, xr, xg, xb, Y_OFFSET), FIX16),
                );

                let (r01, g01, b01) = unpack_ui8x3_i32(
                    rgb_group.add(wg_index(2 * x, 2 * y + 1, depth, rgb_stride)),
                    sampler,
                );

                let (r11, g11, b11) = unpack_ui8x3_i32(
                    rgb_group.add(wg_index(2 * x + 1, 2 * y + 1, depth, rgb_stride)),
                    sampler,
                );

                pack_i32x2(
                    y_group.add(wg_index(2 * x, 2 * y + 1, 1, y_stride)),
                    fix_to_i32(affine_transform(r01, g01, b01, xr, xg, xb, Y_OFFSET), FIX16),
                    fix_to_i32(affine_transform(r11, g11, b11, xr, xg, xb, Y_OFFSET), FIX16),
                );

                let sr = (r00 + r10) + (r01 + r11);
                let sg = (g00 + g10) + (g01 + g11);
                let sb = (b00 + b10) + (b01 + b11);
                pack_i32x2(
                    uv_group.add(wg_index(x, y, 2, uv_stride)),
                    fix_to_i32(affine_transform(sr, sg, sb, yr, yg, yb, C_OFFSET), FIX18),
                    fix_to_i32(affine_transform(sr, sg, sb, zr, zg, zb, C_OFFSET), FIX18),
                );
            }
        }
    }

    true
}

#[inline(always)]
fn yuv_to_lrgb(
    width: u32,
    height: u32,
    last_src_plane: usize,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: usize,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
    channels: PixelFormatChannels,
    colorimetry: Colorimetry,
) -> bool {
    if (last_src_plane >= src_strides.len())
        || (last_src_plane >= src_buffers.len())
        || dst_strides.is_empty()
        || dst_buffers.is_empty()
    {
        return false;
    }

    let depth = channels as usize;
    let col_count = width as usize;
    let line_count = height as usize;
    let packed_nv12_stride = col_count;
    let packed_rgb_stride = depth * col_count;

    let y_stride = if src_strides[0] == 0 {
        packed_nv12_stride
    } else {
        src_strides[0]
    };

    let uv_stride = if src_strides[last_src_plane] == 0 {
        packed_nv12_stride
    } else {
        src_strides[last_src_plane]
    };

    let rgb_stride = if dst_strides[0] == 0 {
        packed_rgb_stride
    } else {
        dst_strides[0]
    };

    let rgb_plane = &mut dst_buffers[0];
    let (first, last) = src_buffers.split_at(last_src_plane);
    let interplane_split = y_stride * line_count;
    if last_src_plane == 0 && interplane_split > last[0].len() {
        return false;
    }

    let (y_plane, uv_plane) = if last_src_plane == 0 {
        last[0].split_at(interplane_split)
    } else {
        (first[0], last[0])
    };

    if line_count == 0 {
        return true;
    }

    let max_stride = usize::max_value() / line_count;
    if (y_stride > max_stride) || (uv_stride > max_stride) || (rgb_stride > max_stride) {
        return false;
    }

    let wg_height = line_count / 2;
    if y_stride * line_count > y_plane.len()
        || uv_stride * wg_height > uv_plane.len()
        || rgb_stride * line_count > rgb_plane.len()
    {
        return false;
    }

    let col = colorimetry as usize;
    let xxym = BACKWARD_WEIGHTS[col][0];
    let rcrm = BACKWARD_WEIGHTS[col][1];
    let gcrm = BACKWARD_WEIGHTS[col][2];
    let gcbm = BACKWARD_WEIGHTS[col][3];
    let bcbm = BACKWARD_WEIGHTS[col][4];
    let rn = BACKWARD_WEIGHTS[col][5];
    let gp = BACKWARD_WEIGHTS[col][6];
    let bn = BACKWARD_WEIGHTS[col][7];

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
        let y_group = y_plane.as_ptr();
        let uv_group = uv_plane.as_ptr();
        let rgb_group = rgb_plane.as_mut_ptr();
        let wg_width = col_count / 2;

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
                    rgb_group.add(wg_index(2 * x, 2 * y, depth, rgb_stride)),
                    // [-14428,30811]   Y(16)Cb(16)         Y(235)Cb(240)
                    fix_to_u8_sat(sy00 + sb, FIX6),
                    // [ -8603,24988]   Y(16)Cb(240)Cr(240) Y(235)Cb(16)Cr(16)
                    fix_to_u8_sat(sy00 + sg, FIX6),
                    // [-11408,27792]   Y(16)Cr(16)         Y(235)Cr(240)
                    fix_to_u8_sat(sy00 + sr, FIX6),
                );

                let sy10 = mulhi_i32(y10, xxym);
                pack_ui8x3(
                    rgb_group.add(wg_index(2 * x + 1, 2 * y, depth, rgb_stride)),
                    fix_to_u8_sat(sy10 + sb, FIX6),
                    fix_to_u8_sat(sy10 + sg, FIX6),
                    fix_to_u8_sat(sy10 + sr, FIX6),
                );

                let (y01, y11) =
                    unpack_ui8x2_i32(y_group.add(wg_index(2 * x, 2 * y + 1, 1, y_stride)));

                let sy01 = mulhi_i32(y01, xxym);
                pack_ui8x3(
                    rgb_group.add(wg_index(2 * x, 2 * y + 1, depth, rgb_stride)),
                    fix_to_u8_sat(sy01 + sb, FIX6),
                    fix_to_u8_sat(sy01 + sg, FIX6),
                    fix_to_u8_sat(sy01 + sr, FIX6),
                );

                let sy11 = mulhi_i32(y11, xxym);
                pack_ui8x3(
                    rgb_group.add(wg_index(2 * x + 1, 2 * y + 1, depth, rgb_stride)),
                    fix_to_u8_sat(sy11 + sb, FIX6),
                    fix_to_u8_sat(sy11 + sg, FIX6),
                    fix_to_u8_sat(sy11 + sr, FIX6),
                );
            }
        }
    }

    true
}

pub fn argb_lrgb_nv12_bt601(
    width: u32,
    height: u32,
    last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_to_yuv(
        width,
        height,
        last_src_plane as usize,
        src_strides,
        src_buffers,
        last_dst_plane as usize,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Four,
        Colorimetry::Bt601,
        Sampler::Argb,
    )
}

pub fn argb_lrgb_nv12_bt709(
    width: u32,
    height: u32,
    last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_to_yuv(
        width,
        height,
        last_src_plane as usize,
        src_strides,
        src_buffers,
        last_dst_plane as usize,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Four,
        Colorimetry::Bt709,
        Sampler::Argb,
    )
}

pub fn bgra_lrgb_nv12_bt601(
    width: u32,
    height: u32,
    last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_to_yuv(
        width,
        height,
        last_src_plane as usize,
        src_strides,
        src_buffers,
        last_dst_plane as usize,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Four,
        Colorimetry::Bt601,
        Sampler::Bgra,
    )
}

pub fn bgra_lrgb_nv12_bt709(
    width: u32,
    height: u32,
    last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_to_yuv(
        width,
        height,
        last_src_plane as usize,
        src_strides,
        src_buffers,
        last_dst_plane as usize,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Four,
        Colorimetry::Bt709,
        Sampler::Bgra,
    )
}

pub fn bgr_lrgb_nv12_bt601(
    width: u32,
    height: u32,
    last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_to_yuv(
        width,
        height,
        last_src_plane as usize,
        src_strides,
        src_buffers,
        last_dst_plane as usize,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Three,
        Colorimetry::Bt601,
        Sampler::Bgr,
    )
}

pub fn bgr_lrgb_nv12_bt709(
    width: u32,
    height: u32,
    last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_to_yuv(
        width,
        height,
        last_src_plane as usize,
        src_strides,
        src_buffers,
        last_dst_plane as usize,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Three,
        Colorimetry::Bt709,
        Sampler::Bgr,
    )
}

pub fn nv12_bt601_bgra_lrgb(
    width: u32,
    height: u32,
    last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    yuv_to_lrgb(
        width,
        height,
        last_src_plane as usize,
        src_strides,
        src_buffers,
        last_dst_plane as usize,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Four,
        Colorimetry::Bt601,
    )
}

pub fn nv12_bt709_bgra_lrgb(
    width: u32,
    height: u32,
    last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    yuv_to_lrgb(
        width,
        height,
        last_src_plane as usize,
        src_strides,
        src_buffers,
        last_dst_plane as usize,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Four,
        Colorimetry::Bt709,
    )
}
