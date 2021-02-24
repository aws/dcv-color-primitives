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
use crate::convert_image::x86;

use core::ptr::{read_unaligned as loadu, write_unaligned as storeu};

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m128i, _mm_add_epi16, _mm_add_epi32, _mm_and_si128, _mm_andnot_si128, _mm_cmpeq_epi32,
    _mm_cvtsi128_si32, _mm_madd_epi16, _mm_mulhi_epu16, _mm_or_si128, _mm_packs_epi32,
    _mm_packus_epi16, _mm_set1_epi16, _mm_set1_epi32, _mm_set1_epi64x, _mm_set_epi32,
    _mm_set_epi64x, _mm_setzero_si128, _mm_shuffle_epi32, _mm_shufflehi_epi16, _mm_shufflelo_epi16,
    _mm_slli_epi16, _mm_slli_epi32, _mm_slli_si128, _mm_srai_epi16, _mm_srai_epi32, _mm_srli_epi16,
    _mm_srli_epi32, _mm_srli_si128, _mm_sub_epi16, _mm_unpackhi_epi16, _mm_unpackhi_epi8,
    _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm_unpacklo_epi8,
};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, _mm_add_epi16, _mm_add_epi32, _mm_and_si128, _mm_andnot_si128, _mm_cmpeq_epi32,
    _mm_cvtsi128_si32, _mm_madd_epi16, _mm_mulhi_epu16, _mm_or_si128, _mm_packs_epi32,
    _mm_packus_epi16, _mm_set1_epi16, _mm_set1_epi32, _mm_set1_epi64x, _mm_set_epi32,
    _mm_set_epi64x, _mm_setzero_si128, _mm_shuffle_epi32, _mm_shufflehi_epi16, _mm_shufflelo_epi16,
    _mm_slli_epi16, _mm_slli_epi32, _mm_slli_si128, _mm_srai_epi16, _mm_srai_epi32, _mm_srli_epi16,
    _mm_srli_epi32, _mm_srli_si128, _mm_sub_epi16, _mm_unpackhi_epi16, _mm_unpackhi_epi8,
    _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm_unpacklo_epi8,
};

const LANE_COUNT: usize = 16;
const LRGB_TO_YUV_WG_SIZE: usize = 4;
const YUV_TO_LRGB_WG_SIZE: usize = 1;
const LRGB_TO_YUV_WAVES: usize = LANE_COUNT / LRGB_TO_YUV_WG_SIZE;
const YUV_TO_LRGB_WAVES: usize = LANE_COUNT / YUV_TO_LRGB_WG_SIZE;

#[cfg(not(tarpaulin_include))]
const fn mm_shuffle(z: i32, y: i32, x: i32, w: i32) -> i32 {
    (z << 6) | (y << 4) | (x << 2) | w
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
#[inline(always)]
unsafe fn i16_to_i16x2_8x(x: __m128i) -> (__m128i, __m128i) {
    (_mm_unpacklo_epi16(x, x), _mm_unpackhi_epi16(x, x))
}

/// Unpack 8 uchar samples into 8 short samples,
/// stored in big endian (8-wide)
///
/// image: g15g14g13g12 g11g10g9g18 g7g6g5g4 g3g2g1g0
/// res:   g7--g6-- g5--g4-- g3--g2-- g1--g0--
#[inline(always)]
unsafe fn unpack_ui8_i16be_8x(image: *const u8) -> __m128i {
    let x = _mm_set1_epi64x(loadu(image.cast()));
    _mm_unpacklo_epi8(zero!(), x)
}

/// Deinterleave 2 uchar samples into short samples,
/// stored in big endian (8-wide)
///
/// image: g7r7g6r6 g5r5g4r4 g3r3g2r2 g1r1g0r0
/// red:   r7--r6-- r5--r4-- r3--r2-- r1--r0--
/// green: g7--g6-- g5--g4-- g3--g2-- g1--g0--
#[inline(always)]
unsafe fn unpack_ui8x2_i16be_8x(image: *const u8) -> (__m128i, __m128i) {
    let x = loadu(image.cast());
    (
        _mm_slli_epi16(x, 8),
        _mm_slli_epi16(_mm_srli_epi16(x, 8), 8),
    )
}

/// Truncate and deinterleave 3 short samples into 4 uchar samples (8-wide)
/// Alpha set to `DEFAULT_ALPHA`
///
/// red:      --r7--r6 --r5--r4 --r3--r2 --r1--r0
/// green:    --r7--r6 --r5--r4 --r3--r2 --r1--r0
/// blue:     --r7--r6 --r5--r4 --r3--r2 --r1--r0
/// image[0]: ffr3g3b3 ffr2g2b2 ffr1g1b1 ffr0g0b0
/// image[1]: ffr7g7b7 ffr6g6b6 ffr5g5b5 ffr4g4b4
#[inline(always)]
unsafe fn pack_i16x3_8x(image: *mut u8, red: __m128i, green: __m128i, blue: __m128i) {
    let x = _mm_packus_epi16(blue, red);
    let y = _mm_packus_epi16(green, _mm_srli_epi16(_mm_cmpeq_epi32(zero!(), zero!()), 8));
    let z = _mm_unpacklo_epi8(x, y);
    let w = _mm_unpackhi_epi8(x, y);

    let rgba: *mut __m128i = image.cast();
    storeu(rgba, _mm_unpacklo_epi16(z, w));
    storeu(rgba.add(1), _mm_unpackhi_epi16(z, w));
}

/// Convert 3 deinterleaved uchar samples into 2 deinterleaved
/// short samples (4-wide)
///
/// image (sampler=0): a3r3g3b3 a2r2g2b2 a1r1g1b1 a0r0g0b0
/// image (sampler=1): b3g3r3a3 b2g2r2a2 b1g1r1a1 b0g0r0a0
/// image (sampler=2): ******** r3g3b3r2 g2b2r1g1 b1r0g0b0
/// image (sampler=3): -------- r3g3b3r2 g2b2r1g1 b1r0g0b0
///
/// `green_red`:       --g3--r3 --g2--r2 --g1--r1 --g0--r0
/// `green_blue`:      --g3--b3 --g2--b2 --g1--b1 --g0--b0
#[inline(always)]
unsafe fn unpack_ui8x3_i16x2_4x(image: *const u8, sampler: Sampler) -> (__m128i, __m128i) {
    let line = match sampler {
        Sampler::BgrOverflow => {
            let bgr: *const i32 = image.cast();
            _mm_set_epi32(0, loadu(bgr.add(2)), loadu(bgr.add(1)), loadu(bgr))
        }
        _ => loadu(image.cast()),
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

#[inline(always)]
unsafe fn rgb_to_bgra_4x(input: __m128i, output_buffer: *mut __m128i) {
    let alpha_mask = _mm_set1_epi32(0xff);

    // we have b3g3r3-- b2g2r2-- b1g1r1-- b0g0r0--
    let aligned_line = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(_mm_slli_si128(input, 1), _mm_srli_si128(input, 2)),
        _mm_unpacklo_epi32(_mm_srli_si128(input, 5), _mm_srli_si128(input, 8)),
    );

    let res = _mm_or_si128(aligned_line, alpha_mask);

    // Byte swap 128 bit
    let shr = _mm_srli_epi16(res, 8);
    let shl = _mm_slli_epi16(res, 8);
    let ored = _mm_or_si128(shl, shr);
    let pshuf32 = _mm_shuffle_epi32(ored, mm_shuffle(1, 0, 3, 2));
    let pshufl16 = _mm_shufflelo_epi16(pshuf32, mm_shuffle(0, 1, 2, 3));
    let pshufh16 = _mm_shufflehi_epi16(pshufl16, mm_shuffle(0, 1, 2, 3));

    storeu(
        output_buffer,
        _mm_shuffle_epi32(pshufh16, mm_shuffle(0, 1, 2, 3)),
    );
}

/// Truncate int to uchar (4-wide)
///
/// red:      ******r3 ******r2 ******r1 ******r0
/// image[0]: r3r2r1r0
#[inline(always)]
unsafe fn pack_i32_4x(image: *mut u8, red: __m128i) {
    let y = _mm_packs_epi32(red, red);
    let z = _mm_packus_epi16(y, y);
    storeu(image.cast(), _mm_cvtsi128_si32(z));
}

#[inline(always)]
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
#[inline(always)]
unsafe fn sum_i16x2_neighborhood_2x(xy0: __m128i, xy1: __m128i) -> __m128i {
    _mm_add_epi16(
        _mm_add_epi16(xy0, _mm_shuffle_epi32(xy0, xcgh_odd_even_words!())),
        _mm_add_epi16(xy1, _mm_shuffle_epi32(xy1, xcgh_odd_even_words!())),
    )
}

/// Convert linear rgb to yuv colorspace (4-wide)
#[inline(always)]
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
unsafe fn lrgb_to_i420_4x(
    rgb0: *const u8,
    rgb1: *const u8,
    y0: *mut u8,
    y1: *mut u8,
    u: *mut u8,
    v: *mut u8,
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

    // shuff: ******v1 ******v0 ******u1 ******u0
    let shuff = _mm_shuffle_epi32(
        fix_to_i32_4x!(affine_transform(srg, sbg, uv_weights), FIX18),
        mm_shuffle(3, 1, 2, 0),
    );

    // uv_res: v1v0u1u0
    let packed_to_32 = _mm_packs_epi32(shuff, shuff);
    let packed_to_16 = _mm_packus_epi16(packed_to_32, packed_to_32);

    // Checked: we want to reinterpret the bits
    #[allow(clippy::cast_sign_loss)]
    let uv_res = _mm_cvtsi128_si32(packed_to_16) as u32;

    // Checked: we are extracting the lower and upper part of a 32-bit integer
    #[allow(clippy::cast_possible_truncation)]
    {
        storeu(u.cast(), uv_res as u16);
        storeu(v.cast(), (uv_res >> 16) as u16);
    }
}

#[inline(always)]
unsafe fn lrgb_to_i444_4x(
    rgb: *const u8,
    y: *mut u8,
    u: *mut u8,
    v: *mut u8,
    sampler: Sampler,
    y_weights: &[__m128i; 3],
    u_weights: &[__m128i; 3],
    v_weights: &[__m128i; 3],
) {
    let (rg, bg) = unpack_ui8x3_i16x2_4x(rgb, sampler);
    pack_i32_4x(
        y,
        fix_to_i32_4x!(affine_transform(rg, bg, y_weights), FIX16),
    );

    pack_i32_4x(
        u,
        fix_to_i32_4x!(affine_transform(rg, bg, u_weights), FIX16),
    );

    pack_i32_4x(
        v,
        fix_to_i32_4x!(affine_transform(rg, bg, v_weights), FIX16),
    );
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn lrgb_to_yuv_sse2(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_strides: (usize, usize),
    dst_buffers: &mut (&mut [u8], &mut [u8]),
    depth: usize,
    weights: &[i32; 6],
    sampler: Sampler,
) {
    const DST_DEPTH: usize = LRGB_TO_YUV_WAVES;

    let (y_stride, uv_stride) = dst_strides;

    let y_weigths = [
        _mm_set1_epi32(weights[0]),
        _mm_set1_epi32(weights[1]),
        _mm_set1_epi32(Y_OFFSET),
    ];

    let uv_weights = [
        _mm_set_epi32(weights[2], weights[3], weights[2], weights[3]),
        _mm_set_epi32(weights[4], weights[5], weights[4], weights[5]),
        _mm_set1_epi32(C_OFFSET),
    ];

    let src_group = src_buffer.as_ptr();
    let y_group = dst_buffers.0.as_mut_ptr();
    let uv_group = dst_buffers.1.as_mut_ptr();

    let src_depth = depth * LRGB_TO_YUV_WAVES;
    let read_bytes_per_line = ((width - 1) / LRGB_TO_YUV_WAVES) * src_depth + LANE_COUNT;
    let y_start = if (depth == 4) || (read_bytes_per_line <= src_stride) {
        height
    } else {
        height - 2
    };

    let wg_width = width / LRGB_TO_YUV_WAVES;
    let wg_height = y_start / 2;

    for y in 0..wg_height {
        for x in 0..wg_width {
            lrgb_to_yuv_4x(
                src_group.add(wg_index(x, 2 * y, src_depth, src_stride)),
                src_group.add(wg_index(x, 2 * y + 1, src_depth, src_stride)),
                y_group.add(wg_index(x, 2 * y, DST_DEPTH, y_stride)),
                y_group.add(wg_index(x, 2 * y + 1, DST_DEPTH, y_stride)),
                uv_group.add(wg_index(x, y, DST_DEPTH, uv_stride)),
                sampler,
                &y_weigths,
                &uv_weights,
            );
        }
    }

    // Handle leftover line
    if y_start != height {
        let rem = (width - LRGB_TO_YUV_WAVES) / LRGB_TO_YUV_WAVES;
        for x in 0..rem {
            lrgb_to_yuv_4x(
                src_group.add(wg_index(x, y_start, src_depth, src_stride)),
                src_group.add(wg_index(x, y_start + 1, src_depth, src_stride)),
                y_group.add(wg_index(x, y_start, DST_DEPTH, y_stride)),
                y_group.add(wg_index(x, y_start + 1, DST_DEPTH, y_stride)),
                uv_group.add(wg_index(x, wg_height, DST_DEPTH, uv_stride)),
                sampler,
                &y_weigths,
                &uv_weights,
            );
        }

        // Handle leftover pixels
        lrgb_to_yuv_4x(
            src_group.add(wg_index(rem, y_start, src_depth, src_stride)),
            src_group.add(wg_index(rem, y_start + 1, src_depth, src_stride)),
            y_group.add(wg_index(rem, y_start, DST_DEPTH, y_stride)),
            y_group.add(wg_index(rem, y_start + 1, DST_DEPTH, y_stride)),
            uv_group.add(wg_index(rem, wg_height, DST_DEPTH, uv_stride)),
            Sampler::BgrOverflow,
            &y_weigths,
            &uv_weights,
        );
    }
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn lrgb_to_i420_sse2(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_strides: (usize, usize, usize),
    dst_buffers: &mut (&mut [u8], &mut [u8], &mut [u8]),
    depth: usize,
    weights: &[i32; 6],
    sampler: Sampler,
) {
    let (y_stride, u_stride, v_stride) = dst_strides;

    let y_weigths = [
        _mm_set1_epi32(weights[0]),
        _mm_set1_epi32(weights[1]),
        _mm_set1_epi32(Y_OFFSET),
    ];

    let uv_weights = [
        _mm_set_epi32(weights[2], weights[3], weights[2], weights[3]),
        _mm_set_epi32(weights[4], weights[5], weights[4], weights[5]),
        _mm_set1_epi32(C_OFFSET),
    ];

    let src_group = src_buffer.as_ptr();
    let y_group = dst_buffers.0.as_mut_ptr();
    let u_group = dst_buffers.1.as_mut_ptr();
    let v_group = dst_buffers.2.as_mut_ptr();

    let src_depth = depth * LRGB_TO_YUV_WAVES;
    let read_bytes_per_line = ((width - 1) / LRGB_TO_YUV_WAVES) * src_depth + LANE_COUNT;
    let y_start = if (depth == 4) || (read_bytes_per_line <= src_stride) {
        height
    } else {
        height - 2
    };

    let wg_width = width / LRGB_TO_YUV_WAVES;
    let wg_height = y_start / 2;

    for y in 0..wg_height {
        for x in 0..wg_width {
            lrgb_to_i420_4x(
                src_group.add(wg_index(x, 2 * y, src_depth, src_stride)),
                src_group.add(wg_index(x, 2 * y + 1, src_depth, src_stride)),
                y_group.add(wg_index(x, 2 * y, LRGB_TO_YUV_WAVES, y_stride)),
                y_group.add(wg_index(x, 2 * y + 1, LRGB_TO_YUV_WAVES, y_stride)),
                u_group.add(wg_index(x, y, LRGB_TO_YUV_WAVES / 2, u_stride)),
                v_group.add(wg_index(x, y, LRGB_TO_YUV_WAVES / 2, v_stride)),
                sampler,
                &y_weigths,
                &uv_weights,
            );
        }
    }

    // Handle leftover line
    if y_start != height {
        let rem = (width - LRGB_TO_YUV_WAVES) / LRGB_TO_YUV_WAVES;
        for x in 0..rem {
            lrgb_to_i420_4x(
                src_group.add(wg_index(x, y_start, src_depth, src_stride)),
                src_group.add(wg_index(x, y_start + 1, src_depth, src_stride)),
                y_group.add(wg_index(x, y_start, LRGB_TO_YUV_WAVES, y_stride)),
                y_group.add(wg_index(x, y_start + 1, LRGB_TO_YUV_WAVES, y_stride)),
                u_group.add(wg_index(x, wg_height, LRGB_TO_YUV_WAVES / 2, u_stride)),
                v_group.add(wg_index(x, wg_height, LRGB_TO_YUV_WAVES / 2, v_stride)),
                sampler,
                &y_weigths,
                &uv_weights,
            );
        }

        // Handle leftover pixels
        lrgb_to_i420_4x(
            src_group.add(wg_index(rem, y_start, src_depth, src_stride)),
            src_group.add(wg_index(rem, y_start + 1, src_depth, src_stride)),
            y_group.add(wg_index(rem, y_start, LRGB_TO_YUV_WAVES, y_stride)),
            y_group.add(wg_index(rem, y_start + 1, LRGB_TO_YUV_WAVES, y_stride)),
            u_group.add(wg_index(rem, wg_height, LRGB_TO_YUV_WAVES / 2, u_stride)),
            v_group.add(wg_index(rem, wg_height, LRGB_TO_YUV_WAVES / 2, v_stride)),
            Sampler::BgrOverflow,
            &y_weigths,
            &uv_weights,
        );
    }
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn lrgb_to_i444_sse2(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_strides: (usize, usize, usize),
    dst_buffers: &mut (&mut [u8], &mut [u8], &mut [u8]),
    depth: usize,
    weights: &[i32; 6],
    sampler: Sampler,
) {
    let (y_stride, u_stride, v_stride) = dst_strides;

    let y_weights = [
        _mm_set1_epi32(weights[0]),
        _mm_set1_epi32(weights[1]),
        _mm_set1_epi32(Y_OFFSET),
    ];

    let u_weights = [
        _mm_set1_epi32(weights[3]),
        _mm_set1_epi32(weights[5]),
        _mm_set1_epi32(C_OFFSET16),
    ];

    let v_weights = [
        _mm_set1_epi32(weights[2]),
        _mm_set1_epi32(weights[4]),
        _mm_set1_epi32(C_OFFSET16),
    ];

    let src_group = src_buffer.as_ptr();
    let y_group = dst_buffers.0.as_mut_ptr();
    let u_group = dst_buffers.1.as_mut_ptr();
    let v_group = dst_buffers.2.as_mut_ptr();

    let rgb_depth = depth * LRGB_TO_YUV_WAVES;
    let read_bytes_per_line = ((width - 1) / LRGB_TO_YUV_WAVES) * rgb_depth + LANE_COUNT;
    let y_start = if (depth == 4) || (read_bytes_per_line <= src_stride) {
        height
    } else {
        height - 1
    };

    let wg_width = width / LRGB_TO_YUV_WAVES;
    let wg_height = y_start;

    for y in 0..wg_height {
        for x in 0..wg_width {
            lrgb_to_i444_4x(
                src_group.add(wg_index(x, y, rgb_depth, src_stride)),
                y_group.add(wg_index(x, y, LRGB_TO_YUV_WAVES, y_stride)),
                u_group.add(wg_index(x, y, LRGB_TO_YUV_WAVES, u_stride)),
                v_group.add(wg_index(x, y, LRGB_TO_YUV_WAVES, v_stride)),
                sampler,
                &y_weights,
                &u_weights,
                &v_weights,
            );
        }
    }

    // Handle leftover line
    if y_start != height {
        let rem = (width - LRGB_TO_YUV_WAVES) / LRGB_TO_YUV_WAVES;
        for x in 0..rem {
            lrgb_to_i444_4x(
                src_group.add(wg_index(x, y_start, rgb_depth, src_stride)),
                y_group.add(wg_index(x, y_start, LRGB_TO_YUV_WAVES, y_stride)),
                u_group.add(wg_index(x, y_start, LRGB_TO_YUV_WAVES, u_stride)),
                v_group.add(wg_index(x, y_start, LRGB_TO_YUV_WAVES, v_stride)),
                sampler,
                &y_weights,
                &u_weights,
                &v_weights,
            );
        }

        // Handle leftover pixels
        lrgb_to_i444_4x(
            src_group.add(wg_index(rem, y_start, rgb_depth, src_stride)),
            y_group.add(wg_index(rem, y_start, LRGB_TO_YUV_WAVES, y_stride)),
            u_group.add(wg_index(rem, y_start, LRGB_TO_YUV_WAVES, u_stride)),
            v_group.add(wg_index(rem, y_start, LRGB_TO_YUV_WAVES, v_stride)),
            Sampler::BgrOverflow,
            &y_weights,
            &u_weights,
            &v_weights,
        );
    }
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn yuv_to_lrgb_sse2(
    width: usize,
    height: usize,
    src_strides: (usize, usize),
    src_buffers: (&[u8], &[u8]),
    dst_stride: usize,
    dst_buffer: &mut [u8],
    weights: &[i16; 8],
) {
    const SRC_DEPTH: usize = YUV_TO_LRGB_WAVES;
    const DST_DEPTH: usize = 2 * YUV_TO_LRGB_WAVES;

    let (y_stride, uv_stride) = src_strides;

    let xxym = _mm_set1_epi16(weights[0]);
    let rcrm = _mm_set1_epi16(weights[1]);
    let gcrm = _mm_set1_epi16(weights[2]);
    let gcbm = _mm_set1_epi16(weights[3]);
    let bcbm = _mm_set1_epi16(weights[4]);
    let rn = _mm_set1_epi16(weights[5]);
    let gp = _mm_set1_epi16(weights[6]);
    let bn = _mm_set1_epi16(weights[7]);

    let y_group = src_buffers.0.as_ptr();
    let uv_group = src_buffers.1.as_ptr();
    let dst_group = dst_buffer.as_mut_ptr();

    let wg_width = width / YUV_TO_LRGB_WAVES;
    let wg_height = height / 2;

    for y in 0..wg_height {
        for x in 0..wg_width {
            let (cb, cr) =
                unpack_ui8x2_i16be_8x(uv_group.add(wg_index(x, y, SRC_DEPTH, uv_stride)));

            let sb = _mm_sub_epi16(_mm_mulhi_epu16(cb, bcbm), bn);
            let sr = _mm_sub_epi16(_mm_mulhi_epu16(cr, rcrm), rn);
            let sg = _mm_sub_epi16(
                gp,
                _mm_add_epi16(_mm_mulhi_epu16(cb, gcbm), _mm_mulhi_epu16(cr, gcrm)),
            );

            let (sb_lo, sb_hi) = i16_to_i16x2_8x(sb);
            let (sr_lo, sr_hi) = i16_to_i16x2_8x(sr);
            let (sg_lo, sg_hi) = i16_to_i16x2_8x(sg);

            let y0 = loadu(y_group.add(wg_index(x, 2 * y, SRC_DEPTH, y_stride)).cast());

            let y00 = _mm_mulhi_epu16(_mm_unpacklo_epi8(zero!(), y0), xxym);
            pack_i16x3_8x(
                dst_group.add(wg_index(2 * x, 2 * y, DST_DEPTH, dst_stride)),
                fix_to_i16_8x!(_mm_add_epi16(sr_lo, y00), FIX6),
                fix_to_i16_8x!(_mm_add_epi16(sg_lo, y00), FIX6),
                fix_to_i16_8x!(_mm_add_epi16(sb_lo, y00), FIX6),
            );

            let y10 = _mm_mulhi_epu16(_mm_unpackhi_epi8(zero!(), y0), xxym);
            pack_i16x3_8x(
                dst_group.add(wg_index(2 * x + 1, 2 * y, DST_DEPTH, dst_stride)),
                fix_to_i16_8x!(_mm_add_epi16(sr_hi, y10), FIX6),
                fix_to_i16_8x!(_mm_add_epi16(sg_hi, y10), FIX6),
                fix_to_i16_8x!(_mm_add_epi16(sb_hi, y10), FIX6),
            );

            let y1 = loadu(
                y_group
                    .add(wg_index(x, 2 * y + 1, SRC_DEPTH, y_stride))
                    .cast(),
            );

            let y01 = _mm_mulhi_epu16(_mm_unpacklo_epi8(zero!(), y1), xxym);
            pack_i16x3_8x(
                dst_group.add(wg_index(2 * x, 2 * y + 1, DST_DEPTH, dst_stride)),
                fix_to_i16_8x!(_mm_add_epi16(sr_lo, y01), FIX6),
                fix_to_i16_8x!(_mm_add_epi16(sg_lo, y01), FIX6),
                fix_to_i16_8x!(_mm_add_epi16(sb_lo, y01), FIX6),
            );

            let y11 = _mm_mulhi_epu16(_mm_unpackhi_epi8(zero!(), y1), xxym);
            pack_i16x3_8x(
                dst_group.add(wg_index(2 * x + 1, 2 * y + 1, DST_DEPTH, dst_stride)),
                fix_to_i16_8x!(_mm_add_epi16(sr_hi, y11), FIX6),
                fix_to_i16_8x!(_mm_add_epi16(sg_hi, y11), FIX6),
                fix_to_i16_8x!(_mm_add_epi16(sb_hi, y11), FIX6),
            );
        }
    }
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn i420_to_lrgb_sse2(
    width: usize,
    height: usize,
    src_strides: (usize, usize, usize),
    src_buffers: (&[u8], &[u8], &[u8]),
    dst_stride: usize,
    dst_buffer: &mut [u8],
    weights: &[i16; 8],
) {
    const SRC_DEPTH: usize = YUV_TO_LRGB_WAVES;
    const DST_DEPTH: usize = 2 * YUV_TO_LRGB_WAVES;

    let (y_stride, u_stride, v_stride) = src_strides;

    let xxym = _mm_set1_epi16(weights[0]);
    let rcrm = _mm_set1_epi16(weights[1]);
    let gcrm = _mm_set1_epi16(weights[2]);
    let gcbm = _mm_set1_epi16(weights[3]);
    let bcbm = _mm_set1_epi16(weights[4]);
    let rn = _mm_set1_epi16(weights[5]);
    let gp = _mm_set1_epi16(weights[6]);
    let bn = _mm_set1_epi16(weights[7]);

    let y_group = src_buffers.0.as_ptr();
    let u_group = src_buffers.1.as_ptr();
    let v_group = src_buffers.2.as_ptr();
    let dst_group = dst_buffer.as_mut_ptr();

    let wg_width = width / YUV_TO_LRGB_WAVES;
    let wg_height = height / 2;

    for y in 0..wg_height {
        for x in 0..wg_width {
            let cb = unpack_ui8_i16be_8x(u_group.add(wg_index(x, y, SRC_DEPTH / 2, u_stride)));
            let cr = unpack_ui8_i16be_8x(v_group.add(wg_index(x, y, SRC_DEPTH / 2, v_stride)));

            let sb = _mm_sub_epi16(_mm_mulhi_epu16(cb, bcbm), bn);
            let sr = _mm_sub_epi16(_mm_mulhi_epu16(cr, rcrm), rn);
            let sg = _mm_sub_epi16(
                gp,
                _mm_add_epi16(_mm_mulhi_epu16(cb, gcbm), _mm_mulhi_epu16(cr, gcrm)),
            );

            let (sb_lo, sb_hi) = i16_to_i16x2_8x(sb);
            let (sr_lo, sr_hi) = i16_to_i16x2_8x(sr);
            let (sg_lo, sg_hi) = i16_to_i16x2_8x(sg);

            let y0 = loadu(y_group.add(wg_index(x, 2 * y, SRC_DEPTH, y_stride)).cast());

            let y00 = _mm_mulhi_epu16(_mm_unpacklo_epi8(zero!(), y0), xxym);
            pack_i16x3_8x(
                dst_group.add(wg_index(2 * x, 2 * y, DST_DEPTH, dst_stride)),
                fix_to_i16_8x!(_mm_add_epi16(sr_lo, y00), FIX6),
                fix_to_i16_8x!(_mm_add_epi16(sg_lo, y00), FIX6),
                fix_to_i16_8x!(_mm_add_epi16(sb_lo, y00), FIX6),
            );

            let y10 = _mm_mulhi_epu16(_mm_unpackhi_epi8(zero!(), y0), xxym);
            pack_i16x3_8x(
                dst_group.add(wg_index(2 * x + 1, 2 * y, DST_DEPTH, dst_stride)),
                fix_to_i16_8x!(_mm_add_epi16(sr_hi, y10), FIX6),
                fix_to_i16_8x!(_mm_add_epi16(sg_hi, y10), FIX6),
                fix_to_i16_8x!(_mm_add_epi16(sb_hi, y10), FIX6),
            );

            let y1 = loadu(
                y_group
                    .add(wg_index(x, 2 * y + 1, SRC_DEPTH, y_stride))
                    .cast(),
            );

            let y01 = _mm_mulhi_epu16(_mm_unpacklo_epi8(zero!(), y1), xxym);
            pack_i16x3_8x(
                dst_group.add(wg_index(2 * x, 2 * y + 1, DST_DEPTH, dst_stride)),
                fix_to_i16_8x!(_mm_add_epi16(sr_lo, y01), FIX6),
                fix_to_i16_8x!(_mm_add_epi16(sg_lo, y01), FIX6),
                fix_to_i16_8x!(_mm_add_epi16(sb_lo, y01), FIX6),
            );

            let y11 = _mm_mulhi_epu16(_mm_unpackhi_epi8(zero!(), y1), xxym);
            pack_i16x3_8x(
                dst_group.add(wg_index(2 * x + 1, 2 * y + 1, DST_DEPTH, dst_stride)),
                fix_to_i16_8x!(_mm_add_epi16(sr_hi, y11), FIX6),
                fix_to_i16_8x!(_mm_add_epi16(sg_hi, y11), FIX6),
                fix_to_i16_8x!(_mm_add_epi16(sb_hi, y11), FIX6),
            );
        }
    }
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn i444_to_lrgb_sse2(
    width: usize,
    height: usize,
    src_strides: (usize, usize, usize),
    src_buffers: (&[u8], &[u8], &[u8]),
    dst_stride: usize,
    dst_buffer: &mut [u8],
    weights: &[i16; 8],
) {
    const SRC_DEPTH: usize = YUV_TO_LRGB_WAVES / 2;
    const DST_DEPTH: usize = 2 * YUV_TO_LRGB_WAVES;

    let (y_stride, u_stride, v_stride) = src_strides;

    let xxym = _mm_set1_epi16(weights[0]);
    let rcrm = _mm_set1_epi16(weights[1]);
    let gcrm = _mm_set1_epi16(weights[2]);
    let gcbm = _mm_set1_epi16(weights[3]);
    let bcbm = _mm_set1_epi16(weights[4]);
    let rn = _mm_set1_epi16(weights[5]);
    let gp = _mm_set1_epi16(weights[6]);
    let bn = _mm_set1_epi16(weights[7]);

    let y_group = src_buffers.0.as_ptr();
    let u_group = src_buffers.1.as_ptr();
    let v_group = src_buffers.2.as_ptr();
    let dst_group = dst_buffer.as_mut_ptr();

    let wg_width = width / SRC_DEPTH;

    for y in 0..height {
        for x in 0..wg_width {
            let cb0 = _mm_set_epi64x(
                0,
                loadu(u_group.add(wg_index(x, y, SRC_DEPTH, u_stride)).cast()),
            );
            let cr0 = _mm_set_epi64x(
                0,
                loadu(v_group.add(wg_index(x, y, SRC_DEPTH, v_stride)).cast()),
            );
            let y0 = _mm_set_epi64x(
                0,
                loadu(y_group.add(wg_index(x, y, SRC_DEPTH, y_stride)).cast()),
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
                dst_group.add(wg_index(x, y, DST_DEPTH, dst_stride)),
                fix_to_i16_8x!(_mm_add_epi16(sr_lo, y_lo), FIX6),
                fix_to_i16_8x!(_mm_add_epi16(sg_lo, y_lo), FIX6),
                fix_to_i16_8x!(_mm_add_epi16(sb_lo, y_lo), FIX6),
            );
        }
    }
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn rgb_to_bgra_sse2(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_stride: usize,
    dst_buffer: &mut [u8],
) {
    const SRC_DEPTH: usize = 3;
    const DST_DEPTH: usize = 4;

    let first_pixel_mask = _mm_set_epi32(0, 0, 0, -1);
    let first_two_pixels_mask = _mm_set_epi32(0, 0, -1, -1);

    let src_group = src_buffer.as_ptr();
    let dst_group = dst_buffer.as_mut_ptr();

    let src_stride_diff = src_stride - (SRC_DEPTH * width);
    let dst_stride_diff = dst_stride - (DST_DEPTH * width);
    let mut src_offset = 0;
    let mut dst_offset = 0;

    for _ in 0..height {
        for _ in (0..width).step_by(LANE_COUNT) {
            let src_ptr: *const __m128i = src_group.add(src_offset).cast();
            let dst_ptr: *mut __m128i = dst_group.add(dst_offset).cast();

            let input0 = loadu(src_ptr);
            let input1 = loadu(src_ptr.add(1));
            let input2 = loadu(src_ptr.add(2));

            rgb_to_bgra_4x(input0, dst_ptr);

            // second iteration is with input0,input1
            // we should merge last 4 bytes of input0 with first 8 bytes of input1
            let last4 = _mm_shuffle_epi32(input0, mm_shuffle(0, 0, 0, 3));
            let first8 = _mm_shuffle_epi32(input1, mm_shuffle(0, 1, 0, 0));
            let input = _mm_or_si128(
                _mm_and_si128(last4, first_pixel_mask),
                _mm_andnot_si128(first_pixel_mask, first8),
            );

            rgb_to_bgra_4x(input, dst_ptr.add(1));

            // third iteration is with input1,input2
            // we should merge last 8 bytes of input1 with first 4 bytes of input2
            let last8 = _mm_shuffle_epi32(input1, mm_shuffle(0, 0, 3, 2));
            let first8 = _mm_shuffle_epi32(input2, mm_shuffle(1, 0, 0, 0));
            let input = _mm_or_si128(
                _mm_and_si128(last8, first_two_pixels_mask),
                _mm_andnot_si128(first_two_pixels_mask, first8),
            );

            rgb_to_bgra_4x(input, dst_ptr.add(2));

            // fourth iteration is with input2
            rgb_to_bgra_4x(
                _mm_shuffle_epi32(input2, mm_shuffle(0, 3, 2, 1)),
                dst_ptr.add(3),
            );

            src_offset += LANE_COUNT * SRC_DEPTH;
            dst_offset += LANE_COUNT * DST_DEPTH;
        }

        src_offset += src_stride_diff;
        dst_offset += dst_stride_diff;
    }
}

#[inline(never)]
fn nv12_bgra_lrgb(
    width: u32,
    height: u32,
    last_src_plane: usize,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
    colorimetry: usize,
) -> bool {
    const DST_DEPTH: usize = PixelFormatChannels::Four as usize;

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
    let rgb_stride = DST_DEPTH * w;

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

    let dst_buffer = &mut dst_buffers[0][..];
    if out_of_bounds(src_buffers.0.len(), src_strides.0, h - 1, w)
        || out_of_bounds(src_buffers.1.len(), src_strides.1, ch - 1, w)
        || out_of_bounds(dst_buffer.len(), dst_stride, h - 1, rgb_stride)
    {
        return false;
    }

    // Process vector part and scalar one
    let vector_part = lower_multiple_of_pot(w, YUV_TO_LRGB_WAVES);
    let scalar_part = w - vector_part;
    if vector_part > 0 {
        unsafe {
            yuv_to_lrgb_sse2(
                vector_part,
                h,
                src_strides,
                src_buffers,
                dst_stride,
                dst_buffer,
                &BACKWARD_WEIGHTS[colorimetry],
            );
        }
    }

    if scalar_part > 0 {
        let x = vector_part;
        let dx = x * DST_DEPTH;

        // The compiler is not smart here
        // This condition should never happen
        if x >= src_buffers.0.len() || x >= src_buffers.1.len() || dx >= dst_buffer.len() {
            return false;
        }

        x86::nv12_to_lrgb(
            scalar_part,
            h,
            src_strides,
            (&src_buffers.0[x..], &src_buffers.1[x..]),
            dst_stride,
            &mut dst_buffer[dx..],
            &x86::BACKWARD_WEIGHTS[colorimetry],
        );
    }

    true
}

#[inline(never)]
fn i420_bgra_lrgb(
    width: u32,
    height: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
    colorimetry: usize,
) -> bool {
    const DST_DEPTH: usize = PixelFormatChannels::Four as usize;

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
    let rgb_stride = DST_DEPTH * w;

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
    let dst_buffer = &mut dst_buffers[0][..];
    if out_of_bounds(src_buffers.0.len(), src_strides.0, h - 1, w)
        || out_of_bounds(src_buffers.1.len(), src_strides.1, ch - 1, cw)
        || out_of_bounds(src_buffers.2.len(), src_strides.2, ch - 1, cw)
        || out_of_bounds(dst_buffer.len(), dst_stride, h - 1, rgb_stride)
    {
        return false;
    }

    // Process vector part and scalar one
    let vector_part = lower_multiple_of_pot(w, YUV_TO_LRGB_WAVES);
    let scalar_part = w - vector_part;
    if vector_part > 0 {
        unsafe {
            i420_to_lrgb_sse2(
                vector_part,
                h,
                src_strides,
                src_buffers,
                dst_stride,
                dst_buffer,
                &BACKWARD_WEIGHTS[colorimetry],
            );
        }
    }

    if scalar_part > 0 {
        let x = vector_part;
        let cx = x / 2;
        let dx = x * DST_DEPTH;

        // The compiler is not smart here
        // This condition should never happen
        if x >= src_buffers.0.len()
            || cx >= src_buffers.1.len()
            || cx >= src_buffers.2.len()
            || dx >= dst_buffer.len()
        {
            return false;
        }

        x86::i420_to_lrgb(
            scalar_part,
            h,
            src_strides,
            (
                &src_buffers.0[x..],
                &src_buffers.1[cx..],
                &src_buffers.2[cx..],
            ),
            dst_stride,
            &mut dst_buffer[dx..],
            &x86::BACKWARD_WEIGHTS[colorimetry],
        );
    }

    true
}

#[inline(never)]
fn i444_bgra_lrgb(
    width: u32,
    height: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
    colorimetry: usize,
) -> bool {
    const DST_DEPTH: usize = PixelFormatChannels::Four as usize;

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
    let rgb_stride = DST_DEPTH * w;

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
    let dst_buffer = &mut dst_buffers[0][..];
    if out_of_bounds(src_buffers.0.len(), src_strides.0, h - 1, w)
        || out_of_bounds(src_buffers.1.len(), src_strides.1, h - 1, w)
        || out_of_bounds(src_buffers.2.len(), src_strides.2, h - 1, w)
        || out_of_bounds(dst_buffer.len(), dst_stride, h - 1, rgb_stride)
    {
        return false;
    }

    // Process vector part and scalar one
    let vector_part = lower_multiple_of_pot(w, YUV_TO_LRGB_WAVES / 2);
    let scalar_part = w - vector_part;
    if vector_part > 0 {
        unsafe {
            i444_to_lrgb_sse2(
                vector_part,
                h,
                src_strides,
                src_buffers,
                dst_stride,
                dst_buffer,
                &BACKWARD_WEIGHTS[colorimetry],
            );
        }
    }

    if scalar_part > 0 {
        let x = vector_part;
        let dx = x * DST_DEPTH;

        // The compiler is not smart here
        // This condition should never happen
        if x >= src_buffers.0.len()
            || x >= src_buffers.1.len()
            || x >= src_buffers.2.len()
            || dx >= dst_buffer.len()
        {
            return false;
        }

        x86::i444_to_lrgb(
            scalar_part,
            h,
            src_strides,
            (
                &src_buffers.0[x..],
                &src_buffers.1[x..],
                &src_buffers.2[x..],
            ),
            dst_stride,
            &mut dst_buffer[dx..],
            &x86::BACKWARD_WEIGHTS[colorimetry],
        );
    }

    true
}

#[inline(never)]
fn lrgb_i444(
    width: u32,
    height: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
    channels: PixelFormatChannels,
    colorimetry: usize,
    sampler: Sampler,
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
    let depth = channels as usize;
    let rgb_stride = depth * w;

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
    let (y_plane, u_plane, v_plane) = (
        &mut y_plane[0][..],
        &mut u_plane[0][..],
        &mut v_plane[0][..],
    );
    if out_of_bounds(src_buffer.len(), src_stride, h - 1, rgb_stride)
        || out_of_bounds(y_plane.len(), dst_strides.0, h - 1, w)
        || out_of_bounds(u_plane.len(), dst_strides.1, h - 1, w)
        || out_of_bounds(v_plane.len(), dst_strides.2, h - 1, w)
    {
        return false;
    }

    // Process vector part and scalar one
    let vector_part = lower_multiple_of_pot(w, LRGB_TO_YUV_WAVES);
    let scalar_part = w - vector_part;
    if vector_part > 0 {
        unsafe {
            lrgb_to_i444_sse2(
                vector_part,
                h,
                src_stride,
                src_buffer,
                dst_strides,
                &mut (y_plane, u_plane, v_plane),
                depth,
                &FORWARD_WEIGHTS[colorimetry],
                sampler,
            );
        }
    }

    if scalar_part > 0 {
        let x = vector_part;
        let sx = x * depth;

        // The compiler is not smart here
        // This condition should never happen
        if sx >= src_buffer.len() || x >= y_plane.len() || x >= u_plane.len() || x >= v_plane.len()
        {
            return false;
        }

        x86::lrgb_to_i444(
            scalar_part,
            h,
            src_stride,
            &src_buffer[sx..],
            dst_strides,
            &mut (&mut y_plane[x..], &mut u_plane[x..], &mut v_plane[x..]),
            depth,
            &x86::FORWARD_WEIGHTS[colorimetry],
            sampler,
        );
    }

    true
}

#[inline(never)]
fn lrgb_i420(
    width: u32,
    height: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
    channels: PixelFormatChannels,
    colorimetry: usize,
    sampler: Sampler,
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
    let depth = channels as usize;
    let rgb_stride = depth * w;

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
    let (y_plane, u_plane, v_plane) = (
        &mut y_plane[0][..],
        &mut u_plane[0][..],
        &mut v_plane[0][..],
    );
    if out_of_bounds(src_buffer.len(), src_stride, h - 1, rgb_stride)
        || out_of_bounds(y_plane.len(), dst_strides.0, h - 1, w)
        || out_of_bounds(u_plane.len(), dst_strides.1, ch - 1, cw)
        || out_of_bounds(v_plane.len(), dst_strides.2, ch - 1, cw)
    {
        return false;
    }

    // Process vector part and scalar one
    let vector_part = lower_multiple_of_pot(w, LRGB_TO_YUV_WAVES);
    let scalar_part = w - vector_part;
    if vector_part > 0 {
        unsafe {
            lrgb_to_i420_sse2(
                vector_part,
                h,
                src_stride,
                src_buffer,
                dst_strides,
                &mut (y_plane, u_plane, v_plane),
                depth,
                &FORWARD_WEIGHTS[colorimetry],
                sampler,
            );
        }
    }

    if scalar_part > 0 {
        let x = vector_part;
        let cx = x / 2;
        let sx = x * depth;

        // The compiler is not smart here
        // This condition should never happen
        if sx >= src_buffer.len()
            || x >= y_plane.len()
            || cx >= u_plane.len()
            || cx >= v_plane.len()
        {
            return false;
        }

        x86::lrgb_to_i420(
            scalar_part,
            h,
            src_stride,
            &src_buffer[sx..],
            dst_strides,
            &mut (&mut y_plane[x..], &mut u_plane[cx..], &mut v_plane[cx..]),
            depth,
            &x86::FORWARD_WEIGHTS[colorimetry],
            sampler,
        );
    }

    true
}

#[inline(never)]
fn lrgb_nv12(
    width: u32,
    height: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: usize,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
    channels: PixelFormatChannels,
    colorimetry: usize,
    sampler: Sampler,
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
    let depth = channels as usize;
    let rgb_stride = depth * w;

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

        (&mut y_plane[0][..], &mut uv_plane[0][..])
    };

    if out_of_bounds(src_buffer.len(), src_stride, h - 1, rgb_stride)
        || out_of_bounds(y_plane.len(), dst_strides.0, h - 1, w)
        || out_of_bounds(uv_plane.len(), dst_strides.1, ch - 1, w)
    {
        return false;
    }

    // Process vector part and scalar one
    let vector_part = lower_multiple_of_pot(w, LRGB_TO_YUV_WAVES);
    let scalar_part = w - vector_part;
    if vector_part > 0 {
        unsafe {
            lrgb_to_yuv_sse2(
                vector_part,
                h,
                src_stride,
                src_buffer,
                dst_strides,
                &mut (y_plane, uv_plane),
                depth,
                &FORWARD_WEIGHTS[colorimetry],
                sampler,
            );
        }
    }

    if scalar_part > 0 {
        let x = vector_part;
        let sx = x * depth;

        // The compiler is not smart here
        // This condition should never happen
        if sx >= src_buffer.len() || x >= y_plane.len() || x >= uv_plane.len() {
            return false;
        }

        x86::lrgb_to_nv12(
            scalar_part,
            h,
            src_stride,
            &src_buffer[sx..],
            dst_strides,
            &mut (&mut y_plane[x..], &mut uv_plane[x..]),
            depth,
            &x86::FORWARD_WEIGHTS[colorimetry],
            sampler,
        );
    }

    true
}

pub fn argb_lrgb_nv12_bt601(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_nv12(
        width,
        height,
        src_strides,
        src_buffers,
        last_dst_plane as usize,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Four,
        Colorimetry::Bt601 as usize,
        Sampler::Argb,
    )
}

pub fn argb_lrgb_nv12_bt709(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_nv12(
        width,
        height,
        src_strides,
        src_buffers,
        last_dst_plane as usize,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Four,
        Colorimetry::Bt709 as usize,
        Sampler::Argb,
    )
}

pub fn bgra_lrgb_nv12_bt601(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_nv12(
        width,
        height,
        src_strides,
        src_buffers,
        last_dst_plane as usize,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Four,
        Colorimetry::Bt601 as usize,
        Sampler::Bgra,
    )
}

pub fn bgra_lrgb_nv12_bt709(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_nv12(
        width,
        height,
        src_strides,
        src_buffers,
        last_dst_plane as usize,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Four,
        Colorimetry::Bt709 as usize,
        Sampler::Bgra,
    )
}

pub fn bgr_lrgb_nv12_bt601(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_nv12(
        width,
        height,
        src_strides,
        src_buffers,
        last_dst_plane as usize,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Three,
        Colorimetry::Bt601 as usize,
        Sampler::Bgr,
    )
}

pub fn bgr_lrgb_nv12_bt709(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_nv12(
        width,
        height,
        src_strides,
        src_buffers,
        last_dst_plane as usize,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Three,
        Colorimetry::Bt709 as usize,
        Sampler::Bgr,
    )
}

pub fn nv12_bt601_bgra_lrgb(
    width: u32,
    height: u32,
    last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    nv12_bgra_lrgb(
        width,
        height,
        last_src_plane as usize,
        src_strides,
        src_buffers,
        dst_strides,
        dst_buffers,
        Colorimetry::Bt601 as usize,
    )
}

pub fn nv12_bt709_bgra_lrgb(
    width: u32,
    height: u32,
    last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    nv12_bgra_lrgb(
        width,
        height,
        last_src_plane as usize,
        src_strides,
        src_buffers,
        dst_strides,
        dst_buffers,
        Colorimetry::Bt709 as usize,
    )
}

pub fn rgb_lrgb_bgra_lrgb(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    const SRC_DEPTH: usize = PixelFormatChannels::Three as usize;
    const DST_DEPTH: usize = PixelFormatChannels::Four as usize;

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
    let dst_buffer = &mut dst_buffers[0][..];
    if out_of_bounds(src_buffer.len(), src_stride, h - 1, w)
        || out_of_bounds(dst_buffer.len(), dst_stride, h - 1, w)
    {
        return false;
    }

    // Process vector part and scalar one
    let vector_part = lower_multiple_of_pot(w, LANE_COUNT);
    let scalar_part = w - vector_part;
    if vector_part > 0 {
        unsafe {
            rgb_to_bgra_sse2(
                vector_part,
                h,
                src_stride,
                src_buffer,
                dst_stride,
                dst_buffer,
            );
        }
    }

    if scalar_part > 0 {
        let x = vector_part;
        let sx = x * SRC_DEPTH;
        let dx = x * DST_DEPTH;

        // The compiler is not smart here
        // This condition should never happen
        if sx >= src_buffer.len() || dx >= dst_buffer.len() {
            return false;
        }

        x86::rgb_to_bgra(
            scalar_part,
            h,
            src_stride,
            &src_buffer[sx..],
            dst_stride,
            &mut dst_buffer[dx..],
        );
    }

    true
}

pub fn i420_bt601_bgra_lrgb(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    i420_bgra_lrgb(
        width,
        height,
        src_strides,
        src_buffers,
        dst_strides,
        dst_buffers,
        Colorimetry::Bt601 as usize,
    )
}

pub fn i420_bt709_bgra_lrgb(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    i420_bgra_lrgb(
        width,
        height,
        src_strides,
        src_buffers,
        dst_strides,
        dst_buffers,
        Colorimetry::Bt709 as usize,
    )
}

pub fn i444_bt601_bgra_lrgb(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    i444_bgra_lrgb(
        width,
        height,
        src_strides,
        src_buffers,
        dst_strides,
        dst_buffers,
        Colorimetry::Bt601 as usize,
    )
}

pub fn i444_bt709_bgra_lrgb(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    i444_bgra_lrgb(
        width,
        height,
        src_strides,
        src_buffers,
        dst_strides,
        dst_buffers,
        Colorimetry::Bt709 as usize,
    )
}

pub fn argb_lrgb_i420_bt601(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_i420(
        width,
        height,
        src_strides,
        src_buffers,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Four,
        Colorimetry::Bt601 as usize,
        Sampler::Argb,
    )
}

pub fn argb_lrgb_i420_bt709(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_i420(
        width,
        height,
        src_strides,
        src_buffers,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Four,
        Colorimetry::Bt709 as usize,
        Sampler::Argb,
    )
}

pub fn bgra_lrgb_i420_bt601(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_i420(
        width,
        height,
        src_strides,
        src_buffers,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Four,
        Colorimetry::Bt601 as usize,
        Sampler::Bgra,
    )
}

pub fn bgra_lrgb_i420_bt709(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_i420(
        width,
        height,
        src_strides,
        src_buffers,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Four,
        Colorimetry::Bt709 as usize,
        Sampler::Bgra,
    )
}

pub fn bgr_lrgb_i420_bt601(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_i420(
        width,
        height,
        src_strides,
        src_buffers,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Three,
        Colorimetry::Bt601 as usize,
        Sampler::Bgr,
    )
}

pub fn bgr_lrgb_i420_bt709(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_i420(
        width,
        height,
        src_strides,
        src_buffers,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Three,
        Colorimetry::Bt709 as usize,
        Sampler::Bgr,
    )
}

pub fn argb_lrgb_i444_bt601(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_i444(
        width,
        height,
        src_strides,
        src_buffers,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Four,
        Colorimetry::Bt601 as usize,
        Sampler::Argb,
    )
}

pub fn argb_lrgb_i444_bt709(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_i444(
        width,
        height,
        src_strides,
        src_buffers,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Four,
        Colorimetry::Bt709 as usize,
        Sampler::Argb,
    )
}

pub fn bgra_lrgb_i444_bt601(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_i444(
        width,
        height,
        src_strides,
        src_buffers,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Four,
        Colorimetry::Bt601 as usize,
        Sampler::Bgra,
    )
}

pub fn bgra_lrgb_i444_bt709(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_i444(
        width,
        height,
        src_strides,
        src_buffers,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Four,
        Colorimetry::Bt709 as usize,
        Sampler::Bgra,
    )
}

pub fn bgr_lrgb_i444_bt601(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_i444(
        width,
        height,
        src_strides,
        src_buffers,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Three,
        Colorimetry::Bt601 as usize,
        Sampler::Bgr,
    )
}

pub fn bgr_lrgb_i444_bt709(
    width: u32,
    height: u32,
    _last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    lrgb_i444(
        width,
        height,
        src_strides,
        src_buffers,
        dst_strides,
        dst_buffers,
        PixelFormatChannels::Three,
        Colorimetry::Bt709 as usize,
        Sampler::Bgr,
    )
}

pub fn bgra_lrgb_rgb_lrgb(
    width: u32,
    height: u32,
    last_src_plane: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: u32,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
) -> bool {
    x86::bgra_lrgb_rgb_lrgb(
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
