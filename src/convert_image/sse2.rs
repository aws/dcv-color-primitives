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
use crate::convert_image::x86;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

const LANE_COUNT: usize = 16;
const LRGB_TO_YUV_WG_SIZE: usize = 4;
const YUV_TO_LRGB_WG_SIZE: usize = 1;
const LRGB_TO_YUV_WAVES: usize = LANE_COUNT / LRGB_TO_YUV_WG_SIZE;
const YUV_TO_LRGB_WAVES: usize = LANE_COUNT / YUV_TO_LRGB_WG_SIZE;

const fn mm_shuffle(z: i32, y: i32, x: i32, w: i32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w)
}

macro_rules! zero {
    () => {
        _mm_setzero_si128()
    };
}

macro_rules! xcgh_odd_even_words {
    () => {
        mm_shuffle(2, 3, 0, 1)
    };
}

const FORWARD_WEIGHTS: [[i32; 6]; Colorimetry::Length as usize] = [
    [
        i32x2_to_i32(XG_601 - SHORT_HALF, XR_601),
        i32x2_to_i32(SHORT_HALF, XB_601),
        i32x2_to_i32(ZG_601, ZR_601),
        i32x2_to_i32(YG_601, YR_601),
        i32x2_to_i32(0, ZB_601),
        i32x2_to_i32(0, YB_601),
    ],
    [
        i32x2_to_i32(XG_709 - SHORT_HALF, XR_709),
        i32x2_to_i32(SHORT_HALF, XB_709),
        i32x2_to_i32(ZG_709, ZR_709),
        i32x2_to_i32(YG_709, YR_709),
        i32x2_to_i32(0, ZB_709),
        i32x2_to_i32(0, YB_709),
    ],
];

const BACKWARD_WEIGHTS: [[i16; 8]; Colorimetry::Length as usize] = [
    [
        i32_to_i16(XXYM_601),
        i32_to_i16(RCRM_601),
        i32_to_i16(GCRM_601),
        i32_to_i16(GCBM_601),
        i32_to_i16(BCBM_601),
        i32_to_i16(RN_601),
        i32_to_i16(GP_601),
        i32_to_i16(BN_601),
    ],
    [
        i32_to_i16(XXYM_709),
        i32_to_i16(RCRM_709),
        i32_to_i16(GCRM_709),
        i32_to_i16(GCBM_709),
        i32_to_i16(BCBM_709),
        i32_to_i16(RN_709),
        i32_to_i16(GP_709),
        i32_to_i16(BN_709),
    ],
];

/// Convert fixed point to int (4-wide)
macro_rules! fix_to_i32_4x {
    ($fix:expr, $frac_bits:expr) => {
        _mm_srai_epi32($fix, $frac_bits);
    };
}

/// Convert fixed point to short (8-wide)
macro_rules! fix_to_i16_8x {
    ($fix:expr, $frac_bits:expr) => {
        _mm_srai_epi16($fix, $frac_bits)
    };
}

/// Convert short to 2D short vector (8-wide)
///
/// x:   --x7--x6 --x5--x4 --x3--x2 --x1--x0
/// y0:  --x3--x3 --x2--x2 --x1--x1 --x0--x0
/// y1:  --x7--x7 --x6--x6 --x5--x5 --x4--x4
unsafe fn i16_to_i16x2_8x(x: __m128i) -> (__m128i, __m128i) {
    (_mm_unpacklo_epi16(x, x), _mm_unpackhi_epi16(x, x))
}

/// Unpack 8 uchar samples into 8 short samples,
/// stored in big endian (8-wide)
///
/// image: g15g14g13g12 g11g10g9g18 g7g6g5g4 g3g2g1g0
/// res:   g7--g6-- g5--g4-- g3--g2-- g1--g0--
unsafe fn unpack_ui8_i16be_8x(image: *const u8) -> __m128i {
    let x = _mm_set1_epi64x(*(image as *const i64));
    _mm_unpacklo_epi8(zero!(), x)
}

/// Deinterleave 2 uchar samples into short samples,
/// stored in big endian (8-wide)
///
/// image: g7r7g6r6 g5r5g4r4 g3r3g2r2 g1r1g0r0
/// red:   r7--r6-- r5--r4-- r3--r2-- r1--r0--
/// green: g7--g6-- g5--g4-- g3--g2-- g1--g0--
unsafe fn unpack_ui8x2_i16be_8x(image: *const u8) -> (__m128i, __m128i) {
    let x = _mm_loadu_si128(image as *const __m128i);
    (
        _mm_slli_epi16(x, 8),
        _mm_slli_epi16(_mm_srli_epi16(x, 8), 8),
    )
}

/// Truncate and deinterleave 3 short samples into 4 uchar samples (8-wide)
/// Alpha set to DEFAULT_ALPHA
///
/// red:      --r7--r6 --r5--r4 --r3--r2 --r1--r0
/// green:    --r7--r6 --r5--r4 --r3--r2 --r1--r0
/// blue:     --r7--r6 --r5--r4 --r3--r2 --r1--r0
/// image[0]: ffr3g3b3 ffr2g2b2 ffr1g1b1 ffr0g0b0
/// image[1]: ffr7g7b7 ffr6g6b6 ffr5g5b5 ffr4g4b4
unsafe fn pack_i16x3_8x(image: *mut u8, red: __m128i, green: __m128i, blue: __m128i) {
    let x = _mm_packus_epi16(blue, red);
    let y = _mm_packus_epi16(green, _mm_srli_epi16(_mm_cmpeq_epi32(zero!(), zero!()), 8));
    let z = _mm_unpacklo_epi8(x, y);
    let w = _mm_unpackhi_epi8(x, y);

    _mm_storeu_si128(image as *mut __m128i, _mm_unpacklo_epi16(z, w));
    _mm_storeu_si128(
        image.add(LANE_COUNT) as *mut __m128i,
        _mm_unpackhi_epi16(z, w),
    );
}

/// Convert 3 deinterleaved uchar samples into 2 deinterleaved
/// short samples (4-wide)
///
/// image (sampler=0): a3r3g3b3 a2r2g2b2 a1r1g1b1 a0r0g0b0
/// image (sampler=1): b3g3r3a3 b2g2r2a2 b1g1r1a1 b0g0r0a0
/// image (sampler=2): ******** r3g3b3r2 g2b2r1g1 b1r0g0b0
/// image (sampler=3): -------- r3g3b3r2 g2b2r1g1 b1r0g0b0
///
/// green_red:         --g3--r3 --g2--r2 --g1--r1 --g0--r0
/// green_blue:        --g3--b3 --g2--b2 --g1--b1 --g0--b0
unsafe fn unpack_ui8x3_i16x2_4x(image: *const u8, sampler: Sampler) -> (__m128i, __m128i) {
    let line = match sampler {
        Sampler::BgrOverflow => _mm_set_epi32(
            0,
            *(image.offset(8) as *const i32),
            *(image.offset(4) as *const i32),
            *(image as *const i32),
        ),
        _ => _mm_loadu_si128(image as *const __m128i),
    };

    let aligned_line = match sampler {
        Sampler::Bgr | Sampler::BgrOverflow => _mm_unpacklo_epi64(
            _mm_unpacklo_epi32(line, _mm_srli_si128(line, 3)),
            _mm_unpacklo_epi32(_mm_srli_si128(line, 6), _mm_srli_si128(line, 9)),
        ),
        _ => line,
    };

    let (red, blue, green) = match sampler {
        Sampler::Argb => (
            _mm_srli_epi32(_mm_slli_epi32(aligned_line, 16), 24),
            _mm_srli_epi32(aligned_line, 24),
            _mm_srli_epi32(_mm_slli_epi32(_mm_srli_epi32(aligned_line, 16), 24), 8),
        ),
        _ => (
            _mm_srli_epi32(_mm_slli_epi32(aligned_line, 8), 24),
            _mm_srli_epi32(_mm_slli_epi32(aligned_line, 24), 24),
            _mm_srli_epi32(_mm_slli_epi32(_mm_srli_epi32(aligned_line, 8), 24), 8),
        ),
    };

    (_mm_or_si128(red, green), _mm_or_si128(blue, green))
}

/// Truncate int to uchar (4-wide)
///
/// red:      ******r3 ******r2 ******r1 ******r0
/// image[0]: r3r2r1r0
unsafe fn pack_i32_4x(image: *mut u8, red: __m128i) {
    let y = _mm_packs_epi32(red, red);
    let z = _mm_packus_epi16(y, y);
    *(image as *mut i32) = _mm_cvtsi128_si32(z);
}

unsafe fn affine_transform(xy: __m128i, zy: __m128i, weights: &[__m128i; 3]) -> __m128i {
    _mm_add_epi32(
        _mm_add_epi32(
            _mm_madd_epi16(xy, weights[0]),
            _mm_madd_epi16(zy, weights[1]),
        ),
        weights[2],
    )
}

/// Sum 2x2 neighborhood of 2D short vectors (2-wide)
///
/// xy0:    -y30-x30 -y20-x20 -y10-x10 -y00-x00
/// xy1:    -y31-x31 -y21-x21 -y11-x11 -y01-x01
/// return: -ys1-xs1 -ys1-xs1 -ys0-xs0 -ys0-xs0
///
/// xs0 = x00 + x10 + x01 + x11
/// xs1 = x20 + x30 + x21 + x31
/// ys0 = y00 + y10 + y01 + y11
/// ys1 = y20 + y30 + y21 + y31
unsafe fn sum_i16x2_neighborhood_2x(xy0: __m128i, xy1: __m128i) -> __m128i {
    _mm_add_epi16(
        _mm_add_epi16(xy0, _mm_shuffle_epi32(xy0, xcgh_odd_even_words!())),
        _mm_add_epi16(xy1, _mm_shuffle_epi32(xy1, xcgh_odd_even_words!())),
    )
}

/// Convert linear rgb to yuv colorspace (4-wide)
unsafe fn lrgb_to_yuv_4x(
    rgb0: *const u8,
    rgb1: *const u8,
    y0: *mut u8,
    y1: *mut u8,
    uv: *mut u8,
    sampler: Sampler,
    y_weigths: &[__m128i; 3],
    uv_weights: &[__m128i; 3],
) {
    let (rg0, bg0) = unpack_ui8x3_i16x2_4x(rgb0, sampler);
    pack_i32_4x(
        y0,
        fix_to_i32_4x!(affine_transform(rg0, bg0, y_weigths), FIX16),
    );

    let (rg1, bg1) = unpack_ui8x3_i16x2_4x(rgb1, sampler);
    pack_i32_4x(
        y1,
        fix_to_i32_4x!(affine_transform(rg1, bg1, y_weigths), FIX16),
    );

    let srg = sum_i16x2_neighborhood_2x(rg0, rg1);
    let sbg = sum_i16x2_neighborhood_2x(bg0, bg1);
    pack_i32_4x(
        uv,
        fix_to_i32_4x!(affine_transform(srg, sbg, uv_weights), FIX18),
    );
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

    let col = colorimetry as usize;
    unsafe {
        let y_weigths = [
            _mm_set1_epi32(FORWARD_WEIGHTS[col][0]),
            _mm_set1_epi32(FORWARD_WEIGHTS[col][1]),
            _mm_set1_epi32(Y_OFFSET),
        ];

        let uv_weights = [
            _mm_set_epi32(
                FORWARD_WEIGHTS[col][2],
                FORWARD_WEIGHTS[col][3],
                FORWARD_WEIGHTS[col][2],
                FORWARD_WEIGHTS[col][3],
            ),
            _mm_set_epi32(
                FORWARD_WEIGHTS[col][4],
                FORWARD_WEIGHTS[col][5],
                FORWARD_WEIGHTS[col][4],
                FORWARD_WEIGHTS[col][5],
            ),
            _mm_set1_epi32(C_OFFSET),
        ];

        let rgb_depth = depth * LRGB_TO_YUV_WAVES;
        let nv12_depth = LRGB_TO_YUV_WAVES;
        let read_bytes_per_line = ((col_count - 1) / LRGB_TO_YUV_WAVES) * rgb_depth + LANE_COUNT;

        let y_start = if (depth == 4) || (read_bytes_per_line <= rgb_stride) {
            line_count
        } else {
            line_count - 2
        };

        let rgb_group = rgb_plane.as_ptr();
        let y_group = y_plane.as_mut_ptr();
        let uv_group = uv_plane.as_mut_ptr();
        let wg_width = col_count / LRGB_TO_YUV_WAVES;
        let wg_height = y_start / 2;

        for y in 0..wg_height {
            for x in 0..wg_width {
                lrgb_to_yuv_4x(
                    rgb_group.add(wg_index(x, 2 * y, rgb_depth, rgb_stride)),
                    rgb_group.add(wg_index(x, 2 * y + 1, rgb_depth, rgb_stride)),
                    y_group.add(wg_index(x, 2 * y, nv12_depth, y_stride)),
                    y_group.add(wg_index(x, 2 * y + 1, nv12_depth, y_stride)),
                    uv_group.add(wg_index(x, y, nv12_depth, uv_stride)),
                    sampler,
                    &y_weigths,
                    &uv_weights,
                );
            }
        }

        // Handle leftover line
        if y_start != line_count {
            let wg_width = (col_count - LRGB_TO_YUV_WAVES) / LRGB_TO_YUV_WAVES;
            for x in 0..wg_width {
                lrgb_to_yuv_4x(
                    rgb_group.add(wg_index(x, y_start, rgb_depth, rgb_stride)),
                    rgb_group.add(wg_index(x, y_start + 1, rgb_depth, rgb_stride)),
                    y_group.add(wg_index(x, y_start, nv12_depth, y_stride)),
                    y_group.add(wg_index(x, y_start + 1, nv12_depth, y_stride)),
                    uv_group.add(wg_index(x, wg_height, nv12_depth, uv_stride)),
                    sampler,
                    &y_weigths,
                    &uv_weights,
                );
            }

            // Handle leftover pixels
            lrgb_to_yuv_4x(
                rgb_group.add(wg_index(wg_width, y_start, rgb_depth, rgb_stride)),
                rgb_group.add(wg_index(wg_width, y_start + 1, rgb_depth, rgb_stride)),
                y_group.add(wg_index(wg_width, y_start, nv12_depth, y_stride)),
                y_group.add(wg_index(wg_width, y_start + 1, nv12_depth, y_stride)),
                uv_group.add(wg_index(wg_width, wg_height, nv12_depth, uv_stride)),
                Sampler::BgrOverflow,
                &y_weigths,
                &uv_weights,
            );
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
    unsafe {
        let xxym = _mm_set1_epi16(BACKWARD_WEIGHTS[col][0]);
        let rcrm = _mm_set1_epi16(BACKWARD_WEIGHTS[col][1]);
        let gcrm = _mm_set1_epi16(BACKWARD_WEIGHTS[col][2]);
        let gcbm = _mm_set1_epi16(BACKWARD_WEIGHTS[col][3]);
        let bcbm = _mm_set1_epi16(BACKWARD_WEIGHTS[col][4]);
        let rn = _mm_set1_epi16(BACKWARD_WEIGHTS[col][5]);
        let gp = _mm_set1_epi16(BACKWARD_WEIGHTS[col][6]);
        let bn = _mm_set1_epi16(BACKWARD_WEIGHTS[col][7]);

        let y_group = y_plane.as_ptr();
        let uv_group = uv_plane.as_ptr();
        let rgb_group = rgb_plane.as_mut_ptr();
        let rgb_depth = 2 * YUV_TO_LRGB_WAVES;
        let nv12_depth = YUV_TO_LRGB_WAVES;
        let wg_width = col_count / YUV_TO_LRGB_WAVES;

        for y in 0..wg_height {
            for x in 0..wg_width {
                let (cb, cr) =
                    unpack_ui8x2_i16be_8x(uv_group.add(wg_index(x, y, nv12_depth, uv_stride)));

                let sb = _mm_sub_epi16(_mm_mulhi_epu16(cb, bcbm), bn);
                let sr = _mm_sub_epi16(_mm_mulhi_epu16(cr, rcrm), rn);
                let sg = _mm_sub_epi16(
                    gp,
                    _mm_add_epi16(_mm_mulhi_epu16(cb, gcbm), _mm_mulhi_epu16(cr, gcrm)),
                );

                let (sb_lo, sb_hi) = i16_to_i16x2_8x(sb);
                let (sr_lo, sr_hi) = i16_to_i16x2_8x(sr);
                let (sg_lo, sg_hi) = i16_to_i16x2_8x(sg);

                let y0 = _mm_loadu_si128(
                    y_group.add(wg_index(x, 2 * y, nv12_depth, y_stride)) as *const __m128i
                );

                let y00 = _mm_mulhi_epu16(_mm_unpacklo_epi8(zero!(), y0), xxym);
                pack_i16x3_8x(
                    rgb_group.add(wg_index(2 * x, 2 * y, rgb_depth, rgb_stride)),
                    fix_to_i16_8x!(_mm_add_epi16(sr_lo, y00), FIX6),
                    fix_to_i16_8x!(_mm_add_epi16(sg_lo, y00), FIX6),
                    fix_to_i16_8x!(_mm_add_epi16(sb_lo, y00), FIX6),
                );

                let y10 = _mm_mulhi_epu16(_mm_unpackhi_epi8(zero!(), y0), xxym);
                pack_i16x3_8x(
                    rgb_group.add(wg_index(2 * x + 1, 2 * y, rgb_depth, rgb_stride)),
                    fix_to_i16_8x!(_mm_add_epi16(sr_hi, y10), FIX6),
                    fix_to_i16_8x!(_mm_add_epi16(sg_hi, y10), FIX6),
                    fix_to_i16_8x!(_mm_add_epi16(sb_hi, y10), FIX6),
                );

                let y1 = _mm_loadu_si128(
                    y_group.add(wg_index(x, 2 * y + 1, nv12_depth, y_stride)) as *const __m128i
                );

                let y01 = _mm_mulhi_epu16(_mm_unpacklo_epi8(zero!(), y1), xxym);
                pack_i16x3_8x(
                    rgb_group.add(wg_index(2 * x, 2 * y + 1, rgb_depth, rgb_stride)),
                    fix_to_i16_8x!(_mm_add_epi16(sr_lo, y01), FIX6),
                    fix_to_i16_8x!(_mm_add_epi16(sg_lo, y01), FIX6),
                    fix_to_i16_8x!(_mm_add_epi16(sb_lo, y01), FIX6),
                );

                let y11 = _mm_mulhi_epu16(_mm_unpackhi_epi8(zero!(), y1), xxym);
                pack_i16x3_8x(
                    rgb_group.add(wg_index(2 * x + 1, 2 * y + 1, rgb_depth, rgb_stride)),
                    fix_to_i16_8x!(_mm_add_epi16(sr_hi, y11), FIX6),
                    fix_to_i16_8x!(_mm_add_epi16(sg_hi, y11), FIX6),
                    fix_to_i16_8x!(_mm_add_epi16(sb_hi, y11), FIX6),
                );
            }
        }
    }

    true
}

#[inline(always)]
fn i420_to_lrgb(
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
    if last_src_plane != 2
        || last_src_plane >= src_strides.len()
        || last_src_plane >= src_buffers.len()
        || dst_strides.is_empty()
        || dst_buffers.is_empty()
    {
        return false;
    }

    let depth = channels as usize;
    let col_count = width as usize;
    let line_count = height as usize;
    let packed_rgb_stride = depth * col_count;

    let y_stride = if src_strides[0] != 0 {
        src_strides[0]
    } else {
        col_count
    };

    let u_stride = if src_strides[1] != 0 {
        src_strides[1]
    } else {
        col_count / 2
    };

    let v_stride = if src_strides[2] != 0 {
        src_strides[2]
    } else {
        col_count / 2
    };

    let rgb_stride = if dst_strides[0] == 0 {
        packed_rgb_stride
    } else {
        dst_strides[0]
    };

    let rgb_plane = &mut dst_buffers[0];
    let (y_plane, u_plane, v_plane) = (src_buffers[0], src_buffers[1], src_buffers[2]);

    if line_count == 0 {
        return true;
    }

    let max_stride = usize::max_value() / line_count;
    if (y_stride > max_stride)
        || (u_stride > max_stride)
        || (v_stride > max_stride)
        || (rgb_stride > max_stride)
    {
        return false;
    }

    let wg_height = line_count / 2;
    if y_stride * line_count > y_plane.len()
        || u_stride * wg_height > u_plane.len()
        || v_stride * wg_height > v_plane.len()
        || rgb_stride * line_count > rgb_plane.len()
    {
        return false;
    }

    let col = colorimetry as usize;
    unsafe {
        let xxym = _mm_set1_epi16(BACKWARD_WEIGHTS[col][0]);
        let rcrm = _mm_set1_epi16(BACKWARD_WEIGHTS[col][1]);
        let gcrm = _mm_set1_epi16(BACKWARD_WEIGHTS[col][2]);
        let gcbm = _mm_set1_epi16(BACKWARD_WEIGHTS[col][3]);
        let bcbm = _mm_set1_epi16(BACKWARD_WEIGHTS[col][4]);
        let rn = _mm_set1_epi16(BACKWARD_WEIGHTS[col][5]);
        let gp = _mm_set1_epi16(BACKWARD_WEIGHTS[col][6]);
        let bn = _mm_set1_epi16(BACKWARD_WEIGHTS[col][7]);

        let y_group = y_plane.as_ptr();
        let u_group = u_plane.as_ptr();
        let v_group = v_plane.as_ptr();
        let rgb_group = rgb_plane.as_mut_ptr();
        let rgb_depth = 2 * YUV_TO_LRGB_WAVES;
        let i420_depth = YUV_TO_LRGB_WAVES;
        let wg_width = col_count / YUV_TO_LRGB_WAVES;

        for y in 0..wg_height {
            for x in 0..wg_width {
                let cb = unpack_ui8_i16be_8x(u_group.add(wg_index(x, y, i420_depth / 2, u_stride)));
                let cr = unpack_ui8_i16be_8x(v_group.add(wg_index(x, y, i420_depth / 2, v_stride)));

                let sb = _mm_sub_epi16(_mm_mulhi_epu16(cb, bcbm), bn);
                let sr = _mm_sub_epi16(_mm_mulhi_epu16(cr, rcrm), rn);
                let sg = _mm_sub_epi16(
                    gp,
                    _mm_add_epi16(_mm_mulhi_epu16(cb, gcbm), _mm_mulhi_epu16(cr, gcrm)),
                );

                let (sb_lo, sb_hi) = i16_to_i16x2_8x(sb);
                let (sr_lo, sr_hi) = i16_to_i16x2_8x(sr);
                let (sg_lo, sg_hi) = i16_to_i16x2_8x(sg);

                let y0 = _mm_loadu_si128(
                    y_group.add(wg_index(x, 2 * y, i420_depth, y_stride)) as *const __m128i
                );

                let y00 = _mm_mulhi_epu16(_mm_unpacklo_epi8(zero!(), y0), xxym);
                pack_i16x3_8x(
                    rgb_group.add(wg_index(2 * x, 2 * y, rgb_depth, rgb_stride)),
                    fix_to_i16_8x!(_mm_add_epi16(sr_lo, y00), FIX6),
                    fix_to_i16_8x!(_mm_add_epi16(sg_lo, y00), FIX6),
                    fix_to_i16_8x!(_mm_add_epi16(sb_lo, y00), FIX6),
                );

                let y10 = _mm_mulhi_epu16(_mm_unpackhi_epi8(zero!(), y0), xxym);
                pack_i16x3_8x(
                    rgb_group.add(wg_index(2 * x + 1, 2 * y, rgb_depth, rgb_stride)),
                    fix_to_i16_8x!(_mm_add_epi16(sr_hi, y10), FIX6),
                    fix_to_i16_8x!(_mm_add_epi16(sg_hi, y10), FIX6),
                    fix_to_i16_8x!(_mm_add_epi16(sb_hi, y10), FIX6),
                );

                let y1 = _mm_loadu_si128(
                    y_group.add(wg_index(x, 2 * y + 1, i420_depth, y_stride)) as *const __m128i
                );

                let y01 = _mm_mulhi_epu16(_mm_unpacklo_epi8(zero!(), y1), xxym);
                pack_i16x3_8x(
                    rgb_group.add(wg_index(2 * x, 2 * y + 1, rgb_depth, rgb_stride)),
                    fix_to_i16_8x!(_mm_add_epi16(sr_lo, y01), FIX6),
                    fix_to_i16_8x!(_mm_add_epi16(sg_lo, y01), FIX6),
                    fix_to_i16_8x!(_mm_add_epi16(sb_lo, y01), FIX6),
                );

                let y11 = _mm_mulhi_epu16(_mm_unpackhi_epi8(zero!(), y1), xxym);
                pack_i16x3_8x(
                    rgb_group.add(wg_index(2 * x + 1, 2 * y + 1, rgb_depth, rgb_stride)),
                    fix_to_i16_8x!(_mm_add_epi16(sr_hi, y11), FIX6),
                    fix_to_i16_8x!(_mm_add_epi16(sg_hi, y11), FIX6),
                    fix_to_i16_8x!(_mm_add_epi16(sb_hi, y11), FIX6),
                );
            }
        }
    }

    true
}

#[inline(always)]
fn i444_to_lrgb(
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
    if last_src_plane != 2
        || last_src_plane >= src_strides.len()
        || last_src_plane >= src_buffers.len()
        || dst_strides.is_empty()
        || dst_buffers.is_empty()
    {
        return false;
    }

    let depth = channels as usize;
    let col_count = width as usize;
    let line_count = height as usize;
    let packed_rgb_stride = depth * col_count;

    let y_stride = if src_strides[0] != 0 {
        src_strides[0]
    } else {
        col_count
    };

    let u_stride = if src_strides[1] != 0 {
        src_strides[1]
    } else {
        col_count
    };

    let v_stride = if src_strides[2] != 0 {
        src_strides[2]
    } else {
        col_count
    };

    let rgb_stride = if dst_strides[0] == 0 {
        packed_rgb_stride
    } else {
        dst_strides[0]
    };

    let rgb_plane = &mut dst_buffers[0];
    let (y_plane, u_plane, v_plane) = (src_buffers[0], src_buffers[1], src_buffers[2]);

    if line_count == 0 {
        return true;
    }

    let max_stride = usize::max_value() / line_count;
    if (y_stride > max_stride)
        || (u_stride > max_stride)
        || (v_stride > max_stride)
        || (rgb_stride > max_stride)
    {
        return false;
    }

    if y_stride * line_count > y_plane.len()
        || u_stride * line_count > u_plane.len()
        || v_stride * line_count > v_plane.len()
        || rgb_stride * line_count > rgb_plane.len()
    {
        return false;
    }

    let col = colorimetry as usize;
    unsafe {
        let xxym = _mm_set1_epi16(BACKWARD_WEIGHTS[col][0]);
        let rcrm = _mm_set1_epi16(BACKWARD_WEIGHTS[col][1]);
        let gcrm = _mm_set1_epi16(BACKWARD_WEIGHTS[col][2]);
        let gcbm = _mm_set1_epi16(BACKWARD_WEIGHTS[col][3]);
        let bcbm = _mm_set1_epi16(BACKWARD_WEIGHTS[col][4]);
        let rn = _mm_set1_epi16(BACKWARD_WEIGHTS[col][5]);
        let gp = _mm_set1_epi16(BACKWARD_WEIGHTS[col][6]);
        let bn = _mm_set1_epi16(BACKWARD_WEIGHTS[col][7]);

        let y_group = y_plane.as_ptr();
        let u_group = u_plane.as_ptr();
        let v_group = v_plane.as_ptr();
        let rgb_group = rgb_plane.as_mut_ptr();
        let rgb_depth = YUV_TO_LRGB_WAVES * 2;
        let group_width = YUV_TO_LRGB_WAVES / 2;
        let wg_width = col_count / group_width;

        for y in 0..line_count {
            for x in 0..wg_width {
                let cb0 = _mm_loadl_epi64(
                    u_group.add(wg_index(x, y, group_width, u_stride)) as *const __m128i
                );
                let cr0 = _mm_loadl_epi64(
                    v_group.add(wg_index(x, y, group_width, v_stride)) as *const __m128i
                );
                let y0 = _mm_loadl_epi64(
                    y_group.add(wg_index(x, y, group_width, y_stride)) as *const __m128i
                );

                let cb_lo = _mm_unpacklo_epi8(zero!(), cb0);
                let cr_lo = _mm_unpacklo_epi8(zero!(), cr0);
                let y_lo = _mm_mulhi_epu16(_mm_unpacklo_epi8(zero!(), y0), xxym);

                let sb_lo = _mm_sub_epi16(_mm_mulhi_epu16(cb_lo, bcbm), bn);
                let sr_lo = _mm_sub_epi16(_mm_mulhi_epu16(cr_lo, rcrm), rn);
                let sg_lo = _mm_sub_epi16(
                    gp,
                    _mm_add_epi16(_mm_mulhi_epu16(cb_lo, gcbm), _mm_mulhi_epu16(cr_lo, gcrm)),
                );

                pack_i16x3_8x(
                    rgb_group.add(wg_index(x, y, rgb_depth, rgb_stride)),
                    fix_to_i16_8x!(_mm_add_epi16(sr_lo, y_lo), FIX6),
                    fix_to_i16_8x!(_mm_add_epi16(sg_lo, y_lo), FIX6),
                    fix_to_i16_8x!(_mm_add_epi16(sb_lo, y_lo), FIX6),
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
    if is_wg_multiple(width, LRGB_TO_YUV_WAVES) {
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
    } else {
        x86::argb_lrgb_nv12_bt601(
            width,
            height,
            last_src_plane,
            src_strides,
            src_buffers,
            last_dst_plane,
            dst_strides,
            dst_buffers,
        )
    }
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
    if is_wg_multiple(width, LRGB_TO_YUV_WAVES) {
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
    } else {
        x86::argb_lrgb_nv12_bt709(
            width,
            height,
            last_src_plane,
            src_strides,
            src_buffers,
            last_dst_plane,
            dst_strides,
            dst_buffers,
        )
    }
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
    if is_wg_multiple(width, LRGB_TO_YUV_WAVES) {
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
    } else {
        x86::bgra_lrgb_nv12_bt601(
            width,
            height,
            last_src_plane,
            src_strides,
            src_buffers,
            last_dst_plane,
            dst_strides,
            dst_buffers,
        )
    }
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
    if is_wg_multiple(width, LRGB_TO_YUV_WAVES) {
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
    } else {
        x86::bgra_lrgb_nv12_bt709(
            width,
            height,
            last_src_plane,
            src_strides,
            src_buffers,
            last_dst_plane,
            dst_strides,
            dst_buffers,
        )
    }
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
    if is_wg_multiple(width, LRGB_TO_YUV_WAVES) {
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
    } else {
        x86::bgr_lrgb_nv12_bt601(
            width,
            height,
            last_src_plane,
            src_strides,
            src_buffers,
            last_dst_plane,
            dst_strides,
            dst_buffers,
        )
    }
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
    if is_wg_multiple(width, LRGB_TO_YUV_WAVES) {
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
    } else {
        x86::bgr_lrgb_nv12_bt709(
            width,
            height,
            last_src_plane,
            src_strides,
            src_buffers,
            last_dst_plane,
            dst_strides,
            dst_buffers,
        )
    }
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
    if is_wg_multiple(width, YUV_TO_LRGB_WAVES) {
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
    } else {
        x86::nv12_bt601_bgra_lrgb(
            width,
            height,
            last_src_plane,
            src_strides,
            src_buffers,
            last_dst_plane,
            dst_strides,
            dst_buffers,
        )
    }
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
    if is_wg_multiple(width, YUV_TO_LRGB_WAVES) {
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
    } else {
        x86::nv12_bt709_bgra_lrgb(
            width,
            height,
            last_src_plane,
            src_strides,
            src_buffers,
            last_dst_plane,
            dst_strides,
            dst_buffers,
        )
    }
}

pub fn rgb_lrgb_bgra_lrgb(
    width: u32,
    height: u32,
    last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    x86::rgb_lrgb_bgra_lrgb(
        width,
        height,
        last_src_plane,
        src_strides,
        src_buffers,
        last_dst_plane,
        dst_strides,
        dst_buffers,
    )
}

pub fn i420_bt601_bgra_lrgb(
    width: u32,
    height: u32,
    last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    if is_wg_multiple(width, YUV_TO_LRGB_WAVES) {
        i420_to_lrgb(
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
    } else {
        x86::i420_bt601_bgra_lrgb(
            width,
            height,
            last_src_plane,
            src_strides,
            src_buffers,
            last_dst_plane,
            dst_strides,
            dst_buffers,
        )
    }
}

pub fn i420_bt709_bgra_lrgb(
    width: u32,
    height: u32,
    last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    if is_wg_multiple(width, YUV_TO_LRGB_WAVES) {
        i420_to_lrgb(
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
    } else {
        x86::i420_bt709_bgra_lrgb(
            width,
            height,
            last_src_plane,
            src_strides,
            src_buffers,
            last_dst_plane,
            dst_strides,
            dst_buffers,
        )
    }
}

pub fn i444_bt601_bgra_lrgb(
    width: u32,
    height: u32,
    last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    if is_wg_multiple(width, YUV_TO_LRGB_WAVES / 2) {
        i444_to_lrgb(
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
    } else {
        x86::i444_bt601_bgra_lrgb(
            width,
            height,
            last_src_plane,
            src_strides,
            src_buffers,
            last_dst_plane,
            dst_strides,
            dst_buffers,
        )
    }
}

pub fn i444_bt709_bgra_lrgb(
    width: u32,
    height: u32,
    last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    if is_wg_multiple(width, YUV_TO_LRGB_WAVES / 2) {
        i444_to_lrgb(
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
    } else {
        x86::i444_bt709_bgra_lrgb(
            width,
            height,
            last_src_plane,
            src_strides,
            src_buffers,
            last_dst_plane,
            dst_strides,
            dst_buffers,
        )
    }
}
