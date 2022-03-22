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

#![allow(clippy::wildcard_imports)]
use crate::convert_image::common::*; // We are importing everything
use crate::convert_image::x86;
use crate::{rgb_to_yuv_converter, yuv_to_rgb_converter};

#[cfg(target_arch = "x86_64")]
use core::arch::asm;

use core::ptr::{read_unaligned as loadu, write_unaligned as storeu};

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m256i, _mm256_add_epi16, _mm256_add_epi32, _mm256_and_si256, _mm256_cmpeq_epi32,
    _mm256_madd_epi16, _mm256_mulhi_epu16, _mm256_or_si256, _mm256_packs_epi32,
    _mm256_packus_epi16, _mm256_permute2x128_si256, _mm256_permute4x64_epi64,
    _mm256_permutevar8x32_epi32, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_set1_epi64x,
    _mm256_set_epi32, _mm256_set_epi64x, _mm256_set_m128i, _mm256_setr_epi32, _mm256_setr_epi8,
    _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_slli_epi16, _mm256_slli_epi32,
    _mm256_srai_epi16, _mm256_srai_epi32, _mm256_srli_epi16, _mm256_srli_epi32, _mm256_srli_si256,
    _mm256_sub_epi16, _mm256_unpackhi_epi16, _mm256_unpackhi_epi8, _mm256_unpacklo_epi16,
    _mm256_unpacklo_epi32, _mm256_unpacklo_epi64, _mm256_unpacklo_epi8, _mm_prefetch,
    _mm_setzero_si128, _MM_HINT_NTA,
};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256i, _mm256_add_epi16, _mm256_add_epi32, _mm256_and_si256, _mm256_cmpeq_epi32,
    _mm256_extract_epi64, _mm256_madd_epi16, _mm256_mulhi_epu16, _mm256_or_si256,
    _mm256_packs_epi32, _mm256_packus_epi16, _mm256_permute2x128_si256,
    _mm256_permutevar8x32_epi32, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_set1_epi64x,
    _mm256_set_epi32, _mm256_set_epi64x, _mm256_set_m128i, _mm256_setr_epi32, _mm256_setr_epi8,
    _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_slli_epi16, _mm256_slli_epi32,
    _mm256_srai_epi16, _mm256_srai_epi32, _mm256_srli_epi16, _mm256_srli_epi32, _mm256_srli_si256,
    _mm256_sub_epi16, _mm256_unpackhi_epi16, _mm256_unpackhi_epi8, _mm256_unpacklo_epi16,
    _mm256_unpacklo_epi32, _mm256_unpacklo_epi64, _mm256_unpacklo_epi8, _mm_prefetch,
    _mm_setzero_si128, _MM_HINT_NTA,
};

const LANE_COUNT: usize = 32;
const LRGB_TO_YUV_WG_SIZE: usize = 4;
const YUV_TO_LRGB_WG_SIZE: usize = 1;
const LRGB_TO_YUV_WAVES: usize = LANE_COUNT / LRGB_TO_YUV_WG_SIZE;
const YUV_TO_LRGB_WAVES: usize = LANE_COUNT / YUV_TO_LRGB_WG_SIZE;

const PACK_LO_DQWORD_2X256: i32 = 0x20;
const PACK_HI_DQWORD_2X256: i32 = 0x31;

macro_rules! zero {
    () => {
        _mm256_setzero_si256()
    };
}

macro_rules! pack_lo_dword_2x128 {
    () => {
        _mm256_set_epi32(0, 0, 0, 0, 0, 0, 4, 0)
    };
}

macro_rules! xcgh_odd_even_words {
    () => {
        _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1)
    };
}

macro_rules! align_dqword_2x96 {
    () => {
        _mm256_set_epi32(7, 5, 4, 3, 7, 2, 1, 0)
    };
}

const FORWARD_WEIGHTS: [[i32; 8]; Colorimetry::Length as usize] = [
    [
        i32x2_to_i32(XG_601 - SHORT_HALF, XR_601),
        i32x2_to_i32(SHORT_HALF, XB_601),
        i32x2_to_i32(ZG_601, ZR_601),
        i32x2_to_i32(YG_601, YR_601),
        i32x2_to_i32(0, ZB_601),
        i32x2_to_i32(0, YB_601),
        Y_OFFSET,
        0,
    ],
    [
        i32x2_to_i32(XG_709 - SHORT_HALF, XR_709),
        i32x2_to_i32(SHORT_HALF, XB_709),
        i32x2_to_i32(ZG_709, ZR_709),
        i32x2_to_i32(YG_709, YR_709),
        i32x2_to_i32(0, ZB_709),
        i32x2_to_i32(0, YB_709),
        Y_OFFSET,
        0,
    ],
    [
        i32x2_to_i32(XG_601FR - SHORT_HALF, XR_601FR),
        i32x2_to_i32(SHORT_HALF, XB_601FR),
        i32x2_to_i32(ZG_601FR, ZR_601FR - SHORT_HALF),
        i32x2_to_i32(YG_601FR, YR_601FR),
        i32x2_to_i32(0, ZB_601FR),
        i32x2_to_i32(0, YB_601FR - SHORT_HALF),
        FIX16_HALF,
        1,
    ],
    [
        i32x2_to_i32(XG_709FR - SHORT_HALF, XR_709FR),
        i32x2_to_i32(SHORT_HALF, XB_709FR),
        i32x2_to_i32(ZG_709FR, ZR_709FR - SHORT_HALF),
        i32x2_to_i32(YG_709FR, YR_709FR),
        i32x2_to_i32(0, ZB_709FR),
        i32x2_to_i32(0, YB_709FR - SHORT_HALF),
        FIX16_HALF,
        1,
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
    [
        i32_to_i16(XXYM_601FR),
        i32_to_i16(RCRM_601FR),
        i32_to_i16(GCRM_601FR),
        i32_to_i16(GCBM_601FR),
        i32_to_i16(BCBM_601FR),
        i32_to_i16(RN_601FR),
        i32_to_i16(GP_601FR),
        i32_to_i16(BN_601FR),
    ],
    [
        i32_to_i16(XXYM_709FR),
        i32_to_i16(RCRM_709FR),
        i32_to_i16(GCRM_709FR),
        i32_to_i16(GCBM_709FR),
        i32_to_i16(BCBM_709FR),
        i32_to_i16(RN_709FR),
        i32_to_i16(GP_709FR),
        i32_to_i16(BN_709FR),
    ],
];

/// Convert fixed point to int (8-wide)
macro_rules! fix_to_i32_8x {
    ($fix:expr, $frac_bits:expr) => {
        _mm256_srai_epi32($fix, $frac_bits)
    };
}

/// Convert fixed point to short (16-wide)
macro_rules! fix_to_i16_16x {
    ($fix:expr, $frac_bits:expr) => {
        _mm256_srai_epi16($fix, $frac_bits)
    };
}

#[cfg(target_arch = "x86")]
#[inline(always)]
unsafe fn _mm256_extract_epi64(a: __m256i, index: usize) -> i64 {
    let slice = std::mem::transmute::<__m256i, [i64; 4]>(a);

    slice[index]
}

/// Convert short to 2D short vector (16-wide)
#[inline(always)]
unsafe fn i16_to_i16x2_16x(x: __m256i) -> (__m256i, __m256i) {
    let y = _mm256_unpacklo_epi16(x, x);
    let z = _mm256_unpackhi_epi16(x, x);
    (
        _mm256_permute2x128_si256(y, z, PACK_LO_DQWORD_2X256),
        _mm256_permute2x128_si256(y, z, PACK_HI_DQWORD_2X256),
    )
}

/// Unpack 16 uchar samples into 16 short samples,
/// stored in big endian (16-wide)
#[inline(always)]
unsafe fn unpack_ui8_i16be_16x(image: *const u8) -> __m256i {
    let x = loadu(image.cast());
    let xx = _mm256_set_m128i(x, x);

    let hi = _mm256_unpackhi_epi8(zero!(), xx);
    let lo = _mm256_unpacklo_epi8(zero!(), xx);

    _mm256_permute2x128_si256(lo, hi, PACK_LO_DQWORD_2X256)
}

/// Deinterleave 2 uchar samples into short samples,
/// stored in big endian (16-wide)
#[inline(always)]
unsafe fn unpack_ui8x2_i16be_16x(image: *const u8) -> (__m256i, __m256i) {
    let x = loadu(image.cast());
    (
        _mm256_slli_epi16(x, 8),
        _mm256_slli_epi16(_mm256_srli_epi16(x, 8), 8),
    )
}

/// Truncate and deinterleave 3 short samples into 4 uchar samples (16-wide)
/// Alpha set to `DEFAULT_ALPHA`
#[inline(always)]
unsafe fn pack_i16x3_16x(image: *mut u8, red: __m256i, green: __m256i, blue: __m256i) {
    let blue_red = _mm256_packus_epi16(blue, red);
    let green_white = _mm256_packus_epi16(
        green,
        _mm256_srli_epi16(_mm256_cmpeq_epi32(zero!(), zero!()), 8),
    );
    let rgbw_lo = _mm256_unpacklo_epi8(blue_red, green_white);
    let rgbw_hi = _mm256_unpackhi_epi8(blue_red, green_white);

    let (rgbw_lo, rgbw_hi) = (
        _mm256_unpacklo_epi16(rgbw_lo, rgbw_hi),
        _mm256_unpackhi_epi16(rgbw_lo, rgbw_hi),
    );

    let rgba: *mut __m256i = image.cast();
    storeu(
        rgba,
        _mm256_permute2x128_si256(rgbw_lo, rgbw_hi, PACK_LO_DQWORD_2X256),
    );

    storeu(
        rgba.add(1),
        _mm256_permute2x128_si256(rgbw_lo, rgbw_hi, PACK_HI_DQWORD_2X256),
    );
}

/// Convert 3 deinterleaved uchar samples into 2 deinterleaved
/// short samples (8-wide)
#[inline(always)]
unsafe fn unpack_ui8x3_i16x2_8x(image: *const u8, sampler: Sampler) -> (__m256i, __m256i) {
    let line = match sampler {
        Sampler::BgrOverflow => {
            let bgr: *const i64 = image.cast();
            _mm256_set_epi64x(0, loadu(bgr.add(2)), loadu(bgr.add(1)), loadu(bgr))
        }
        _ => loadu(image.cast()),
    };

    let aligned_line = match sampler {
        Sampler::Bgr | Sampler::BgrOverflow => {
            let l = _mm256_permutevar8x32_epi32(line, align_dqword_2x96!());
            _mm256_unpacklo_epi64(
                _mm256_unpacklo_epi32(l, _mm256_srli_si256(l, 3)),
                _mm256_unpacklo_epi32(_mm256_srli_si256(l, 6), _mm256_srli_si256(l, 9)),
            )
        }
        _ => line,
    };

    let (red, blue, green) = match sampler {
        Sampler::Argb => (
            _mm256_srli_epi32(_mm256_slli_epi32(aligned_line, 16), 24),
            _mm256_srli_epi32(aligned_line, 24),
            _mm256_srli_epi32(
                _mm256_slli_epi32(_mm256_srli_epi32(aligned_line, 16), 24),
                8,
            ),
        ),
        _ => (
            _mm256_srli_epi32(_mm256_slli_epi32(aligned_line, 8), 24),
            _mm256_srli_epi32(_mm256_slli_epi32(aligned_line, 24), 24),
            _mm256_srli_epi32(_mm256_slli_epi32(_mm256_srli_epi32(aligned_line, 8), 24), 8),
        ),
    };

    (_mm256_or_si256(red, green), _mm256_or_si256(blue, green))
}

/// Truncate int to uchar (8-wide)
#[inline(always)]
unsafe fn pack_i32_8x(image: *mut u8, red: __m256i) {
    let x = _mm256_packs_epi32(red, red);
    let y = _mm256_packus_epi16(x, x);
    let z = _mm256_permutevar8x32_epi32(y, pack_lo_dword_2x128!());
    storeu(image.cast(), _mm256_extract_epi64(z, 0));
}

#[inline(always)]
unsafe fn affine_transform(xy: __m256i, zy: __m256i, weights: &[__m256i; 3]) -> __m256i {
    _mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_madd_epi16(xy, weights[0]),
            _mm256_madd_epi16(zy, weights[1]),
        ),
        weights[2],
    )
}

/// Sum 2x2 neighborhood of 2D short vectors (4-wide)
#[inline(always)]
unsafe fn sum_i16x2_neighborhood_4x(xy0: __m256i, xy1: __m256i) -> __m256i {
    _mm256_add_epi16(
        _mm256_add_epi16(
            xy0,
            _mm256_permutevar8x32_epi32(xy0, xcgh_odd_even_words!()),
        ),
        _mm256_add_epi16(
            xy1,
            _mm256_permutevar8x32_epi32(xy1, xcgh_odd_even_words!()),
        ),
    )
}

/// Convert linear rgb to yuv colorspace (8-wide)
#[inline(always)]
unsafe fn lrgb_to_yuv_8x(
    rgb0: *const u8,
    rgb1: *const u8,
    y0: *mut u8,
    y1: *mut u8,
    uv: *mut u8,
    sampler: Sampler,
    y_weigths: &[__m256i; 3],
    uv_weights: &[__m256i; 3],
    full_range: bool,
) {
    let (rg0, bg0) = unpack_ui8x3_i16x2_8x(rgb0, sampler);
    pack_i32_8x(
        y0,
        fix_to_i32_8x!(affine_transform(rg0, bg0, y_weigths), FIX16),
    );

    let (rg1, bg1) = unpack_ui8x3_i16x2_8x(rgb1, sampler);
    pack_i32_8x(
        y1,
        fix_to_i32_8x!(affine_transform(rg1, bg1, y_weigths), FIX16),
    );

    let srg = sum_i16x2_neighborhood_4x(rg0, rg1);
    let sbg = sum_i16x2_neighborhood_4x(bg0, bg1);
    let mut t = affine_transform(srg, sbg, uv_weights);
    if full_range {
        t = _mm256_add_epi32(
            t,
            _mm256_slli_epi32(
                _mm256_or_si256(
                    _mm256_and_si256(sbg, _mm256_set1_epi64x(0xFFFF_i64)),
                    _mm256_and_si256(srg, _mm256_set1_epi64x(0xFFFF_0000_0000_i64)),
                ),
                14,
            ),
        );
    }

    pack_i32_8x(uv, fix_to_i32_8x!(t, FIX18));
}

#[inline(always)]
unsafe fn lrgb_to_i420_8x(
    rgb0: *const u8,
    rgb1: *const u8,
    y0: *mut u8,
    y1: *mut u8,
    u: *mut u8,
    v: *mut u8,
    sampler: Sampler,
    y_weigths: &[__m256i; 3],
    uv_weights: &[__m256i; 3],
    full_range: bool,
) {
    let (rg0, bg0) = unpack_ui8x3_i16x2_8x(rgb0, sampler);
    pack_i32_8x(
        y0,
        fix_to_i32_8x!(affine_transform(rg0, bg0, y_weigths), FIX16),
    );

    let (rg1, bg1) = unpack_ui8x3_i16x2_8x(rgb1, sampler);
    pack_i32_8x(
        y1,
        fix_to_i32_8x!(affine_transform(rg1, bg1, y_weigths), FIX16),
    );

    let srg = sum_i16x2_neighborhood_4x(rg0, rg1);
    let sbg = sum_i16x2_neighborhood_4x(bg0, bg1);
    let mut t = affine_transform(srg, sbg, uv_weights);
    if full_range {
        t = _mm256_add_epi32(
            t,
            _mm256_slli_epi32(
                _mm256_or_si256(
                    _mm256_and_si256(sbg, _mm256_set1_epi64x(0xFFFF_i64)),
                    _mm256_and_si256(srg, _mm256_set1_epi64x(0xFFFF_0000_0000_i64)),
                ),
                14,
            ),
        );
    }

    let shuff = _mm256_permutevar8x32_epi32(
        fix_to_i32_8x!(t, FIX18),
        _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0),
    );

    let packed_to_32 = _mm256_packs_epi32(shuff, shuff);
    let packed_to_16 = _mm256_packus_epi16(packed_to_32, packed_to_32);
    let permuted = _mm256_permutevar8x32_epi32(packed_to_16, pack_lo_dword_2x128!());

    // Checked: we want to reinterpret the bits
    #[allow(clippy::cast_sign_loss)]
    let uv_res = _mm256_extract_epi64(permuted, 0) as u64;

    // Checked: we are extracting the lower and upper part of a 64-bit integer
    #[allow(clippy::cast_possible_truncation)]
    {
        storeu(u.cast(), uv_res as u32);
        storeu(v.cast(), (uv_res >> 32) as u32);
    }
}

#[inline(always)]
unsafe fn lrgb_to_i444_8x(
    rgb: *const u8,
    y: *mut u8,
    u: *mut u8,
    v: *mut u8,
    sampler: Sampler,
    y_weights: &[__m256i; 3],
    u_weights: &[__m256i; 3],
    v_weights: &[__m256i; 3],
    full_range: bool,
) {
    let (rg, bg) = unpack_ui8x3_i16x2_8x(rgb, sampler);
    pack_i32_8x(
        y,
        fix_to_i32_8x!(affine_transform(rg, bg, y_weights), FIX16),
    );

    let mut tu = affine_transform(rg, bg, u_weights);
    let mut tv = affine_transform(rg, bg, v_weights);
    if full_range {
        tu = _mm256_add_epi32(tu, _mm256_srli_epi32(_mm256_slli_epi32(bg, 16), 2));
        tv = _mm256_add_epi32(tv, _mm256_srli_epi32(_mm256_slli_epi32(rg, 16), 2));
    }

    pack_i32_8x(u, fix_to_i32_8x!(tu, FIX16));
    pack_i32_8x(v, fix_to_i32_8x!(tv, FIX16));
}

#[cfg(not(tarpaulin_include))]
#[cfg(target_arch = "x86")]
#[allow(clippy::cast_possible_wrap)]
const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    // Checked: we want to reinterpret the bits
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn lrgb_to_yuv_avx2(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_strides: (usize, usize),
    dst_buffers: &mut (&mut [u8], &mut [u8]),
    depth: usize,
    weights: &[i32; 8],
    sampler: Sampler,
) {
    const DST_DEPTH: usize = LRGB_TO_YUV_WAVES;

    let (y_stride, uv_stride) = dst_strides;

    let y_weigths = [
        _mm256_set1_epi32(weights[0]),
        _mm256_set1_epi32(weights[1]),
        _mm256_set1_epi32(weights[6]),
    ];

    let uv_weights = [
        _mm256_set_epi32(
            weights[2], weights[3], weights[2], weights[3], weights[2], weights[3], weights[2],
            weights[3],
        ),
        _mm256_set_epi32(
            weights[4], weights[5], weights[4], weights[5], weights[4], weights[5], weights[4],
            weights[5],
        ),
        _mm256_set1_epi32(C_OFFSET - FIX18_HALF * weights[7]),
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
            lrgb_to_yuv_8x(
                src_group.add(wg_index(x, 2 * y, src_depth, src_stride)),
                src_group.add(wg_index(x, 2 * y + 1, src_depth, src_stride)),
                y_group.add(wg_index(x, 2 * y, DST_DEPTH, y_stride)),
                y_group.add(wg_index(x, 2 * y + 1, DST_DEPTH, y_stride)),
                uv_group.add(wg_index(x, y, DST_DEPTH, uv_stride)),
                sampler,
                &y_weigths,
                &uv_weights,
                weights[7] == 1,
            );
        }
    }

    // Handle leftover line
    if y_start != height {
        let rem = (width - LRGB_TO_YUV_WAVES) / LRGB_TO_YUV_WAVES;
        for x in 0..rem {
            lrgb_to_yuv_8x(
                src_group.add(wg_index(x, y_start, src_depth, src_stride)),
                src_group.add(wg_index(x, y_start + 1, src_depth, src_stride)),
                y_group.add(wg_index(x, y_start, DST_DEPTH, y_stride)),
                y_group.add(wg_index(x, y_start + 1, DST_DEPTH, y_stride)),
                uv_group.add(wg_index(x, wg_height, DST_DEPTH, uv_stride)),
                sampler,
                &y_weigths,
                &uv_weights,
                weights[7] == 1,
            );
        }

        // Handle leftover pixels
        lrgb_to_yuv_8x(
            src_group.add(wg_index(rem, y_start, src_depth, src_stride)),
            src_group.add(wg_index(rem, y_start + 1, src_depth, src_stride)),
            y_group.add(wg_index(rem, y_start, DST_DEPTH, y_stride)),
            y_group.add(wg_index(rem, y_start + 1, DST_DEPTH, y_stride)),
            uv_group.add(wg_index(rem, wg_height, DST_DEPTH, uv_stride)),
            Sampler::BgrOverflow,
            &y_weigths,
            &uv_weights,
            weights[7] == 1,
        );
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn lrgb_to_i420_avx2(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_strides: (usize, usize, usize),
    dst_buffers: &mut (&mut [u8], &mut [u8], &mut [u8]),
    depth: usize,
    weights: &[i32; 8],
    sampler: Sampler,
) {
    let (y_stride, u_stride, v_stride) = dst_strides;

    let y_weigths = [
        _mm256_set1_epi32(weights[0]),
        _mm256_set1_epi32(weights[1]),
        _mm256_set1_epi32(weights[6]),
    ];

    let uv_weights = [
        _mm256_set_epi32(
            weights[2], weights[3], weights[2], weights[3], weights[2], weights[3], weights[2],
            weights[3],
        ),
        _mm256_set_epi32(
            weights[4], weights[5], weights[4], weights[5], weights[4], weights[5], weights[4],
            weights[5],
        ),
        _mm256_set1_epi32(C_OFFSET - FIX18_HALF * weights[7]),
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
            lrgb_to_i420_8x(
                src_group.add(wg_index(x, 2 * y, src_depth, src_stride)),
                src_group.add(wg_index(x, 2 * y + 1, src_depth, src_stride)),
                y_group.add(wg_index(x, 2 * y, LRGB_TO_YUV_WAVES, y_stride)),
                y_group.add(wg_index(x, 2 * y + 1, LRGB_TO_YUV_WAVES, y_stride)),
                u_group.add(wg_index(x, y, LRGB_TO_YUV_WAVES / 2, u_stride)),
                v_group.add(wg_index(x, y, LRGB_TO_YUV_WAVES / 2, v_stride)),
                sampler,
                &y_weigths,
                &uv_weights,
                weights[7] == 1,
            );
        }
    }

    // Handle leftover line
    if y_start != height {
        let rem = (width - LRGB_TO_YUV_WAVES) / LRGB_TO_YUV_WAVES;
        for x in 0..rem {
            lrgb_to_i420_8x(
                src_group.add(wg_index(x, y_start, src_depth, src_stride)),
                src_group.add(wg_index(x, y_start + 1, src_depth, src_stride)),
                y_group.add(wg_index(x, y_start, LRGB_TO_YUV_WAVES, y_stride)),
                y_group.add(wg_index(x, y_start + 1, LRGB_TO_YUV_WAVES, y_stride)),
                u_group.add(wg_index(x, wg_height, LRGB_TO_YUV_WAVES / 2, u_stride)),
                v_group.add(wg_index(x, wg_height, LRGB_TO_YUV_WAVES / 2, v_stride)),
                sampler,
                &y_weigths,
                &uv_weights,
                weights[7] == 1,
            );
        }

        // Handle leftover pixels
        lrgb_to_i420_8x(
            src_group.add(wg_index(rem, y_start, src_depth, src_stride)),
            src_group.add(wg_index(rem, y_start + 1, src_depth, src_stride)),
            y_group.add(wg_index(rem, y_start, LRGB_TO_YUV_WAVES, y_stride)),
            y_group.add(wg_index(rem, y_start + 1, LRGB_TO_YUV_WAVES, y_stride)),
            u_group.add(wg_index(rem, wg_height, LRGB_TO_YUV_WAVES / 2, u_stride)),
            v_group.add(wg_index(rem, wg_height, LRGB_TO_YUV_WAVES / 2, v_stride)),
            Sampler::BgrOverflow,
            &y_weigths,
            &uv_weights,
            weights[7] == 1,
        );
    }
}

#[inline]
#[cfg(target_arch = "x86")]
#[target_feature(enable = "avx2")]
unsafe fn bgra_to_rgb_avx2(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_stride: usize,
    dst_buffer: &mut [u8],
) {
    const SRC_DEPTH: usize = 4;
    const DST_DEPTH: usize = 3;

    let shf_mask = _mm256_setr_epi8(
        2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -128, -128, -128, -128, 2, 1, 0, 6, 5, 4, 10, 9, 8,
        14, 13, 12, -128, -128, -128, -128,
    );
    let pk_mask = _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 3, 7);
    let limit = lower_multiple_of_pot(width, LANE_COUNT);

    for i in 0..height {
        let src_group = src_buffer.as_ptr().add(src_stride * i);
        let dst_group = dst_buffer.as_mut_ptr().add(dst_stride * i);
        let mut y = 0;

        while y < limit {
            let bgra0 = loadu(src_group.add(SRC_DEPTH * y).cast());
            let bgra1 = loadu(src_group.add(SRC_DEPTH * y + LANE_COUNT).cast());
            let bgra2 = loadu(src_group.add(SRC_DEPTH * y + 2 * LANE_COUNT).cast());
            let bgra3 = loadu(src_group.add(SRC_DEPTH * y + 3 * LANE_COUNT).cast());

            let rgb0 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(bgra0, shf_mask), pk_mask);
            let rgb1 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(bgra1, shf_mask), pk_mask);
            let rgb2 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(bgra2, shf_mask), pk_mask);
            let rgb3 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(bgra3, shf_mask), pk_mask);

            storeu(
                dst_group.add(DST_DEPTH * y).cast(),
                _mm256_or_si256(rgb0, _mm256_permute4x64_epi64(rgb1, shuffle(0, 3, 3, 3))),
            );

            storeu(
                dst_group.add(DST_DEPTH * y + LANE_COUNT).cast(),
                _mm256_or_si256(
                    _mm256_permute4x64_epi64(rgb1, shuffle(3, 3, 2, 1)),
                    _mm256_permute4x64_epi64(rgb2, shuffle(1, 0, 3, 3)),
                ),
            );

            storeu(
                dst_group.add(DST_DEPTH * y + 2 * LANE_COUNT).cast(),
                _mm256_or_si256(
                    _mm256_permute4x64_epi64(rgb2, shuffle(3, 3, 3, 2)),
                    _mm256_permute4x64_epi64(rgb3, shuffle(2, 1, 0, 3)),
                ),
            );

            y += LANE_COUNT;
        }

        while y < width {
            *dst_group.add(DST_DEPTH * y) = *src_group.add(SRC_DEPTH * y + 2);
            *dst_group.add(DST_DEPTH * y + 1) = *src_group.add(SRC_DEPTH * y + 1);
            *dst_group.add(DST_DEPTH * y + 2) = *src_group.add(SRC_DEPTH * y);
            y += 1;
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn rgb_to_bgra_avx2(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_stride: usize,
    dst_buffer: &mut [u8],
) {
    const GROUP_SIZE: usize = 4;
    const SRC_DEPTH: usize = 3;
    const DST_DEPTH: usize = 4;

    let alpha_mask = _mm256_set1_epi32(-16_777_216); // 0xFF000000
    let shf_mask = _mm256_setr_epi8(
        2, 1, 0, -1, 5, 4, 3, -1, 8, 7, 6, -1, 11, 10, 9, -1, 2, 1, 0, -1, 5, 4, 3, -1, 8, 7, 6,
        -1, 11, 10, 9, -1,
    );

    let src_group = src_buffer.as_ptr();
    let dst_group = dst_buffer.as_mut_ptr();

    let src_stride_diff = src_stride - (SRC_DEPTH * width);
    let dst_stride_diff = dst_stride - (DST_DEPTH * width);
    let mut src_offset = 0;
    let mut dst_offset = 0;

    for _ in 0..height {
        _mm_prefetch(src_group.cast::<i8>(), _MM_HINT_NTA);

        for _ in (0..width).step_by(LANE_COUNT) {
            // In order to avoid out of bound read, 4 bytes are substracted from the offset
            // of last read which goes in the first lane of input3, can be seen from schema.
            // bFgFrFbE gErEbDgD rDbCgCrC bBgBrBbA gArAb9g9 r9b8g8r8 b7g7r7b6 g6r6b5g5 r5b4g4r4 b3g3r3b2 g2r2b1g1 r1b0g0r0 bFgFrFbE gErEbDgD rDbCgCrC bBgBrBbA gArAb9g9 r9b8g8r8 b7g7r7b6 g6r6b5g5 r5b4g4r4 b3g3r3b2 g2r2b1g1 r1b0g0r0
            //                                                                                                                                                                                     ^----------------0----------------^
            //                                                                                                                                                          ^---------------12----------------^
            //                                                                                                                               ^---------------24----------------^
            //                                                                                                    ^---------------36----------------^
            //                                                                         ^---------------48----------------^
            //                                              ^---------------60----------------^
            //                   ^---------------72----------------^
            // ^---------------80----------------^
            let src_ptr = src_group.add(src_offset);

            let input = _mm256_set_m128i(
                loadu(src_ptr.add(GROUP_SIZE * SRC_DEPTH).cast()),
                loadu(src_ptr.cast()),
            );

            let input1 = _mm256_set_m128i(
                loadu(src_ptr.add(GROUP_SIZE * SRC_DEPTH * 3).cast()),
                loadu(src_ptr.add(GROUP_SIZE * SRC_DEPTH * 2).cast()),
            );

            let input2 = _mm256_set_m128i(
                loadu(src_ptr.add(GROUP_SIZE * SRC_DEPTH * 5).cast()),
                loadu(src_ptr.add(GROUP_SIZE * SRC_DEPTH * 4).cast()),
            );

            let input3 = _mm256_set_m128i(
                loadu(src_ptr.add((GROUP_SIZE * SRC_DEPTH * 7) - 4).cast()),
                loadu(src_ptr.add(GROUP_SIZE * SRC_DEPTH * 6).cast()),
            );

            let input3 =
                _mm256_permutevar8x32_epi32(input3, _mm256_set_epi32(4, 7, 6, 5, 3, 2, 1, 0));

            let res = _mm256_or_si256(_mm256_shuffle_epi8(input, shf_mask), alpha_mask);
            let res1 = _mm256_or_si256(_mm256_shuffle_epi8(input1, shf_mask), alpha_mask);
            let res2 = _mm256_or_si256(_mm256_shuffle_epi8(input2, shf_mask), alpha_mask);
            let res3 = _mm256_or_si256(_mm256_shuffle_epi8(input3, shf_mask), alpha_mask);

            let dst_ptr: *mut __m256i = dst_group.add(dst_offset).cast();
            storeu(dst_ptr, res);
            storeu(dst_ptr.add(1), res1);
            storeu(dst_ptr.add(2), res2);
            storeu(dst_ptr.add(3), res3);

            src_offset += LANE_COUNT * SRC_DEPTH;
            dst_offset += LANE_COUNT * DST_DEPTH;
        }

        src_offset += src_stride_diff;
        dst_offset += dst_stride_diff;
    }
}

#[inline]
#[cfg(target_arch = "x86")]
#[target_feature(enable = "avx2")]
unsafe fn argb_to_rgb_avx2(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_stride: usize,
    dst_buffer: &mut [u8],
) {
    const SRC_DEPTH: usize = 4;
    const DST_DEPTH: usize = 3;

    let shf_mask = _mm256_setr_epi8(
        1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, -128, -128, -128, -128, 1, 2, 3, 5, 6, 7, 9, 10,
        11, 13, 14, 15, -128, -128, -128, -128,
    );
    let pk_mask = _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 3, 7);
    let limit = lower_multiple_of_pot(width, LANE_COUNT);

    for i in 0..height {
        let src_group = src_buffer.as_ptr().add(src_stride * i);
        let dst_group = dst_buffer.as_mut_ptr().add(dst_stride * i);
        let mut y = 0;

        while y < limit {
            let bgra0 = loadu(src_group.add(SRC_DEPTH * y).cast());
            let bgra1 = loadu(src_group.add(SRC_DEPTH * y + LANE_COUNT).cast());
            let bgra2 = loadu(src_group.add(SRC_DEPTH * y + 2 * LANE_COUNT).cast());
            let bgra3 = loadu(src_group.add(SRC_DEPTH * y + 3 * LANE_COUNT).cast());

            let rgb0 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(bgra0, shf_mask), pk_mask);
            let rgb1 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(bgra1, shf_mask), pk_mask);
            let rgb2 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(bgra2, shf_mask), pk_mask);
            let rgb3 = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(bgra3, shf_mask), pk_mask);

            storeu(
                dst_group.add(DST_DEPTH * y).cast(),
                _mm256_or_si256(rgb0, _mm256_permute4x64_epi64(rgb1, shuffle(0, 3, 3, 3))),
            );

            storeu(
                dst_group.add(DST_DEPTH * y + LANE_COUNT).cast(),
                _mm256_or_si256(
                    _mm256_permute4x64_epi64(rgb1, shuffle(3, 3, 2, 1)),
                    _mm256_permute4x64_epi64(rgb2, shuffle(1, 0, 3, 3)),
                ),
            );

            storeu(
                dst_group.add(DST_DEPTH * y + 2 * LANE_COUNT).cast(),
                _mm256_or_si256(
                    _mm256_permute4x64_epi64(rgb2, shuffle(3, 3, 3, 2)),
                    _mm256_permute4x64_epi64(rgb3, shuffle(2, 1, 0, 3)),
                ),
            );

            y += LANE_COUNT;
        }

        while y < width {
            *dst_group.add(DST_DEPTH * y) = *src_group.add(SRC_DEPTH * y + 1);
            *dst_group.add(DST_DEPTH * y + 1) = *src_group.add(SRC_DEPTH * y + 2);
            *dst_group.add(DST_DEPTH * y + 2) = *src_group.add(SRC_DEPTH * y + 3);
            y += 1;
        }
    }
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx", enable = "avx2")]
unsafe fn bgra_to_rgb_avx2(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_stride: usize,
    dst_buffer: &mut [u8],
) {
    let shuffle_mask = _mm256_setr_epi8(
        2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -128, -128, -128, -128, 2, 1, 0, 6, 5, 4, 10, 9, 8,
        14, 13, 12, -128, -128, -128, -128,
    );
    let perm_mask = _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 3, 7);

    asm!(
        "3:",
            "cmp {width:e},$32",
            "jl 4f",
            "mov eax,$32",
            "xor edx,edx",
            "2:",
                "vmovdqu ymm0,ymmword ptr [{src} + 4*rax - 128]",
                "vmovdqu ymm1,ymmword ptr [{src} + 4*rax - 96]",
                "vmovdqu ymm2,ymmword ptr [{src} + 4*rax - 64]",
                "vmovdqu ymm3,ymmword ptr [{src} + 4*rax - 32]",
                "add eax,$32",
                "vpshufb ymm0,ymm0,{shuffle_mask}",
                "vpshufb ymm1,ymm1,{shuffle_mask}",
                "vpshufb ymm2,ymm2,{shuffle_mask}",
                "vpshufb ymm3,ymm3,{shuffle_mask}",
                "vpermd ymm0,{perm_mask},ymm0",
                "vpermd ymm1,{perm_mask},ymm1",
                "vpermd ymm2,{perm_mask},ymm2",
                "vpermd ymm3,{perm_mask},ymm3",
                "vpermq ymm4,ymm1,$0x3f",
                "vpor ymm0,ymm0,ymm4",
                "vmovdqu ymmword ptr [{dst} + rdx],ymm0",
                "vpermq ymm1,ymm1,$0xf9",
                "vpermq ymm4,ymm2,$0x4f",
                "vpor ymm1,ymm1,ymm4",
                "vmovdqu ymmword ptr [{dst} + rdx + 32],ymm1",
                "vpermq ymm2,ymm2,$0xfe",
                "vpermq ymm3,ymm3,$0x93",
                "vpor ymm2,ymm2,ymm3",
                "vmovdqu ymmword ptr [{dst} + rdx + 64],ymm2",
                "add edx,$96",
                "cmp eax,{width:e}",
            "jng 2b",
            "4:",
            "mov eax,{width:e}",
            "and eax,$-32",
            "cmp eax,{width:e}",
            "jge 5f",
            "lea edx, [rax + 2*rax]",
            "6:",
                "mov ecx,dword ptr [{src} + 4*rax]",
                "add eax,$1",
                "bswap ecx",
                "sar ecx,8",
                "mov word ptr [{dst} + rdx],cx",
                "sar ecx,8",
                "mov word ptr [{dst} + rdx + 1],cx",
                "add edx,$3",
                "cmp eax,{width:e}",
            "jl 6b",
            "5:",
            "add {src},{src_stride}",
            "add {dst},{dst_stride}",
            "sub {height:e},$1",
        "jg 3b",

        src = in(reg) src_buffer.as_ptr(),
        dst = in(reg) dst_buffer.as_mut_ptr(),
        width = in(reg) width,
        height = in(reg) height,
        src_stride = in(reg) src_stride,
        dst_stride = in(reg) dst_stride,
        shuffle_mask = in(ymm_reg) shuffle_mask,
        perm_mask = in(ymm_reg) perm_mask,

        out("ecx") _,
        out("rax") _,
        out("rdx") _,
        out("ymm0") _,
        out("ymm1") _,
        out("ymm2") _,
        out("ymm3") _,
        out("ymm4") _,

        options(nostack),
    );
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx", enable = "avx2")]
unsafe fn argb_to_rgb_avx2(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_stride: usize,
    dst_buffer: &mut [u8],
) {
    let shuffle_mask = _mm256_setr_epi8(
        1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, -128, -128, -128, -128, 1, 2, 3, 5, 6, 7, 9, 10,
        11, 13, 14, 15, -128, -128, -128, -128,
    );
    let perm_mask = _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 3, 7);

    asm!(
        "3:",
            "cmp {width:e},$32",
            "jl 4f",
            "mov eax,$32",
            "xor edx,edx",
            "2:",
                "vmovdqu ymm0,ymmword ptr [{src} + 4*rax - 128]",
                "vmovdqu ymm1,ymmword ptr [{src} + 4*rax - 96]",
                "vmovdqu ymm2,ymmword ptr [{src} + 4*rax - 64]",
                "vmovdqu ymm3,ymmword ptr [{src} + 4*rax - 32]",
                "add eax,$32",
                "vpshufb ymm0,ymm0,{shuffle_mask}",
                "vpshufb ymm1,ymm1,{shuffle_mask}",
                "vpshufb ymm2,ymm2,{shuffle_mask}",
                "vpshufb ymm3,ymm3,{shuffle_mask}",
                "vpermd ymm0,{perm_mask},ymm0",
                "vpermd ymm1,{perm_mask},ymm1",
                "vpermd ymm2,{perm_mask},ymm2",
                "vpermd ymm3,{perm_mask},ymm3",
                "vpermq ymm4,ymm1,$0x3f",
                "vpor ymm0,ymm0,ymm4",
                "vmovdqu ymmword ptr [{dst} + rdx],ymm0",
                "vpermq ymm1,ymm1,$0xf9",
                "vpermq ymm4,ymm2,$0x4f",
                "vpor ymm1,ymm1,ymm4",
                "vmovdqu ymmword ptr [{dst} + rdx + 32],ymm1",
                "vpermq ymm2,ymm2,$0xfe",
                "vpermq ymm3,ymm3,$0x93",
                "vpor ymm2,ymm2,ymm3",
                "vmovdqu ymmword ptr [{dst} + rdx + 64],ymm2",
                "add edx,$96",
                "cmp eax,{width:e}",
            "jng 2b",
            "4:",
            "mov eax,{width:e}",
            "and eax,$-32",
            "cmp eax,{width:e}",
            "jge 5f",
            "lea edx, [rax + 2*rax]",
            "6:",
                "mov ecx,dword ptr [{src} + 4*rax]",
                "add eax,$1",
                "sar ecx,8",
                "mov word ptr [{dst} + rdx],cx",
                "sar ecx,8",
                "mov word ptr [{dst} + rdx + 1],cx",
                "add edx,$3",
                "cmp eax,{width:e}",
            "jl 6b",
            "5:",
            "add {src},{src_stride}",
            "add {dst},{dst_stride}",
            "sub {height:e},$1",
        "jg 3b",

        src = in(reg) src_buffer.as_ptr(),
        dst = in(reg) dst_buffer.as_mut_ptr(),
        width = in(reg) width,
        height = in(reg) height,
        src_stride = in(reg) src_stride,
        dst_stride = in(reg) dst_stride,
        shuffle_mask = in(ymm_reg) shuffle_mask,
        perm_mask = in(ymm_reg) perm_mask,

        out("ecx") _,
        out("rax") _,
        out("rdx") _,
        out("ymm0") _,
        out("ymm1") _,
        out("ymm2") _,
        out("ymm3") _,
        out("ymm4") _,

        options(nostack),
    );
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn lrgb_to_i444_avx2(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_strides: (usize, usize, usize),
    dst_buffers: &mut (&mut [u8], &mut [u8], &mut [u8]),
    depth: usize,
    weights: &[i32; 8],
    sampler: Sampler,
) {
    let (y_stride, u_stride, v_stride) = dst_strides;

    let y_weights = [
        _mm256_set1_epi32(weights[0]),
        _mm256_set1_epi32(weights[1]),
        _mm256_set1_epi32(weights[6]),
    ];

    let u_weights = [
        _mm256_set1_epi32(weights[3]),
        _mm256_set1_epi32(weights[5]),
        _mm256_set1_epi32(C_OFFSET16 - FIX16_HALF * weights[7]),
    ];

    let v_weights = [
        _mm256_set1_epi32(weights[2]),
        _mm256_set1_epi32(weights[4]),
        _mm256_set1_epi32(C_OFFSET16 - FIX16_HALF * weights[7]),
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
            lrgb_to_i444_8x(
                src_group.add(wg_index(x, y, rgb_depth, src_stride)),
                y_group.add(wg_index(x, y, LRGB_TO_YUV_WAVES, y_stride)),
                u_group.add(wg_index(x, y, LRGB_TO_YUV_WAVES, u_stride)),
                v_group.add(wg_index(x, y, LRGB_TO_YUV_WAVES, v_stride)),
                sampler,
                &y_weights,
                &u_weights,
                &v_weights,
                weights[7] == 1,
            );
        }
    }

    // Handle leftover line
    if y_start != height {
        let rem = (width - LRGB_TO_YUV_WAVES) / LRGB_TO_YUV_WAVES;
        for x in 0..rem {
            lrgb_to_i444_8x(
                src_group.add(wg_index(x, y_start, rgb_depth, src_stride)),
                y_group.add(wg_index(x, y_start, LRGB_TO_YUV_WAVES, y_stride)),
                u_group.add(wg_index(x, y_start, LRGB_TO_YUV_WAVES, u_stride)),
                v_group.add(wg_index(x, y_start, LRGB_TO_YUV_WAVES, v_stride)),
                sampler,
                &y_weights,
                &u_weights,
                &v_weights,
                weights[7] == 1,
            );
        }

        // Handle leftover pixels
        lrgb_to_i444_8x(
            src_group.add(wg_index(rem, y_start, rgb_depth, src_stride)),
            y_group.add(wg_index(rem, y_start, LRGB_TO_YUV_WAVES, y_stride)),
            u_group.add(wg_index(rem, y_start, LRGB_TO_YUV_WAVES, u_stride)),
            v_group.add(wg_index(rem, y_start, LRGB_TO_YUV_WAVES, v_stride)),
            Sampler::BgrOverflow,
            &y_weights,
            &u_weights,
            &v_weights,
            weights[7] == 1,
        );
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn yuv_to_lrgb_avx2(
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

    let xxym = _mm256_set1_epi16(weights[0]);
    let rcrm = _mm256_set1_epi16(weights[1]);
    let gcrm = _mm256_set1_epi16(weights[2]);
    let gcbm = _mm256_set1_epi16(weights[3]);
    let bcbm = _mm256_set1_epi16(weights[4]);
    let rn = _mm256_set1_epi16(weights[5]);
    let gp = _mm256_set1_epi16(weights[6]);
    let bn = _mm256_set1_epi16(weights[7]);

    let y_group = src_buffers.0.as_ptr();
    let uv_group = src_buffers.1.as_ptr();
    let dst_group = dst_buffer.as_mut_ptr();

    let wg_width = width / YUV_TO_LRGB_WAVES;
    let wg_height = height / 2;

    for y in 0..wg_height {
        for x in 0..wg_width {
            let (cb, cr) =
                unpack_ui8x2_i16be_16x(uv_group.add(wg_index(x, y, SRC_DEPTH, uv_stride)));

            let sb = _mm256_sub_epi16(_mm256_mulhi_epu16(cb, bcbm), bn);
            let sr = _mm256_sub_epi16(_mm256_mulhi_epu16(cr, rcrm), rn);
            let sg = _mm256_sub_epi16(
                gp,
                _mm256_add_epi16(_mm256_mulhi_epu16(cb, gcbm), _mm256_mulhi_epu16(cr, gcrm)),
            );

            let (sb_lo, sb_hi) = i16_to_i16x2_16x(sb);
            let (sr_lo, sr_hi) = i16_to_i16x2_16x(sr);
            let (sg_lo, sg_hi) = i16_to_i16x2_16x(sg);

            let y0 = loadu(y_group.add(wg_index(x, 2 * y, SRC_DEPTH, y_stride)).cast());

            let y00 = _mm256_mulhi_epu16(
                _mm256_permute2x128_si256(
                    _mm256_unpacklo_epi8(zero!(), y0),
                    _mm256_unpackhi_epi8(zero!(), y0),
                    PACK_LO_DQWORD_2X256,
                ),
                xxym,
            );
            pack_i16x3_16x(
                dst_group.add(wg_index(2 * x, 2 * y, DST_DEPTH, dst_stride)),
                fix_to_i16_16x!(_mm256_add_epi16(sr_lo, y00), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sg_lo, y00), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sb_lo, y00), FIX6),
            );

            let y10 = _mm256_mulhi_epu16(
                _mm256_permute2x128_si256(
                    _mm256_unpacklo_epi8(zero!(), y0),
                    _mm256_unpackhi_epi8(zero!(), y0),
                    PACK_HI_DQWORD_2X256,
                ),
                xxym,
            );
            pack_i16x3_16x(
                dst_group.add(wg_index(2 * x + 1, 2 * y, DST_DEPTH, dst_stride)),
                fix_to_i16_16x!(_mm256_add_epi16(sr_hi, y10), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sg_hi, y10), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sb_hi, y10), FIX6),
            );

            let y1 = loadu(
                y_group
                    .add(wg_index(x, 2 * y + 1, SRC_DEPTH, y_stride))
                    .cast(),
            );

            let y01 = _mm256_mulhi_epu16(
                _mm256_permute2x128_si256(
                    _mm256_unpacklo_epi8(zero!(), y1),
                    _mm256_unpackhi_epi8(zero!(), y1),
                    PACK_LO_DQWORD_2X256,
                ),
                xxym,
            );
            pack_i16x3_16x(
                dst_group.add(wg_index(2 * x, 2 * y + 1, DST_DEPTH, dst_stride)),
                fix_to_i16_16x!(_mm256_add_epi16(sr_lo, y01), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sg_lo, y01), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sb_lo, y01), FIX6),
            );

            let y11 = _mm256_mulhi_epu16(
                _mm256_permute2x128_si256(
                    _mm256_unpacklo_epi8(zero!(), y1),
                    _mm256_unpackhi_epi8(zero!(), y1),
                    PACK_HI_DQWORD_2X256,
                ),
                xxym,
            );
            pack_i16x3_16x(
                dst_group.add(wg_index(2 * x + 1, 2 * y + 1, DST_DEPTH, dst_stride)),
                fix_to_i16_16x!(_mm256_add_epi16(sr_hi, y11), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sg_hi, y11), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sb_hi, y11), FIX6),
            );
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn i420_to_lrgb_avx2(
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

    let xxym = _mm256_set1_epi16(weights[0]);
    let rcrm = _mm256_set1_epi16(weights[1]);
    let gcrm = _mm256_set1_epi16(weights[2]);
    let gcbm = _mm256_set1_epi16(weights[3]);
    let bcbm = _mm256_set1_epi16(weights[4]);
    let rn = _mm256_set1_epi16(weights[5]);
    let gp = _mm256_set1_epi16(weights[6]);
    let bn = _mm256_set1_epi16(weights[7]);

    let y_group = src_buffers.0.as_ptr();
    let u_group = src_buffers.1.as_ptr();
    let v_group = src_buffers.2.as_ptr();
    let dst_group = dst_buffer.as_mut_ptr();

    let wg_width = width / YUV_TO_LRGB_WAVES;
    let wg_height = height / 2;

    for y in 0..wg_height {
        for x in 0..wg_width {
            let cb = unpack_ui8_i16be_16x(u_group.add(wg_index(x, y, SRC_DEPTH / 2, u_stride)));
            let cr = unpack_ui8_i16be_16x(v_group.add(wg_index(x, y, SRC_DEPTH / 2, v_stride)));

            let sb = _mm256_sub_epi16(_mm256_mulhi_epu16(cb, bcbm), bn);
            let sr = _mm256_sub_epi16(_mm256_mulhi_epu16(cr, rcrm), rn);
            let sg = _mm256_sub_epi16(
                gp,
                _mm256_add_epi16(_mm256_mulhi_epu16(cb, gcbm), _mm256_mulhi_epu16(cr, gcrm)),
            );

            let (sb_lo, sb_hi) = i16_to_i16x2_16x(sb);
            let (sr_lo, sr_hi) = i16_to_i16x2_16x(sr);
            let (sg_lo, sg_hi) = i16_to_i16x2_16x(sg);

            let y0 = loadu(y_group.add(wg_index(x, 2 * y, SRC_DEPTH, y_stride)).cast());

            let y00 = _mm256_mulhi_epu16(
                _mm256_permute2x128_si256(
                    _mm256_unpacklo_epi8(zero!(), y0),
                    _mm256_unpackhi_epi8(zero!(), y0),
                    PACK_LO_DQWORD_2X256,
                ),
                xxym,
            );
            pack_i16x3_16x(
                dst_group.add(wg_index(2 * x, 2 * y, DST_DEPTH, dst_stride)),
                fix_to_i16_16x!(_mm256_add_epi16(sr_lo, y00), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sg_lo, y00), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sb_lo, y00), FIX6),
            );

            let y10 = _mm256_mulhi_epu16(
                _mm256_permute2x128_si256(
                    _mm256_unpacklo_epi8(zero!(), y0),
                    _mm256_unpackhi_epi8(zero!(), y0),
                    PACK_HI_DQWORD_2X256,
                ),
                xxym,
            );
            pack_i16x3_16x(
                dst_group.add(wg_index(2 * x + 1, 2 * y, DST_DEPTH, dst_stride)),
                fix_to_i16_16x!(_mm256_add_epi16(sr_hi, y10), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sg_hi, y10), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sb_hi, y10), FIX6),
            );

            let y1 = loadu(
                y_group
                    .add(wg_index(x, 2 * y + 1, SRC_DEPTH, y_stride))
                    .cast(),
            );

            let y01 = _mm256_mulhi_epu16(
                _mm256_permute2x128_si256(
                    _mm256_unpacklo_epi8(zero!(), y1),
                    _mm256_unpackhi_epi8(zero!(), y1),
                    PACK_LO_DQWORD_2X256,
                ),
                xxym,
            );
            pack_i16x3_16x(
                dst_group.add(wg_index(2 * x, 2 * y + 1, DST_DEPTH, dst_stride)),
                fix_to_i16_16x!(_mm256_add_epi16(sr_lo, y01), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sg_lo, y01), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sb_lo, y01), FIX6),
            );

            let y11 = _mm256_mulhi_epu16(
                _mm256_permute2x128_si256(
                    _mm256_unpacklo_epi8(zero!(), y1),
                    _mm256_unpackhi_epi8(zero!(), y1),
                    PACK_HI_DQWORD_2X256,
                ),
                xxym,
            );
            pack_i16x3_16x(
                dst_group.add(wg_index(2 * x + 1, 2 * y + 1, DST_DEPTH, dst_stride)),
                fix_to_i16_16x!(_mm256_add_epi16(sr_hi, y11), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sg_hi, y11), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sb_hi, y11), FIX6),
            );
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn i444_to_lrgb_avx2(
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

    let xxym = _mm256_set1_epi16(weights[0]);
    let rcrm = _mm256_set1_epi16(weights[1]);
    let gcrm = _mm256_set1_epi16(weights[2]);
    let gcbm = _mm256_set1_epi16(weights[3]);
    let bcbm = _mm256_set1_epi16(weights[4]);
    let rn = _mm256_set1_epi16(weights[5]);
    let gp = _mm256_set1_epi16(weights[6]);
    let bn = _mm256_set1_epi16(weights[7]);
    let zero_128 = _mm_setzero_si128();

    let y_group = src_buffers.0.as_ptr();
    let u_group = src_buffers.1.as_ptr();
    let v_group = src_buffers.2.as_ptr();
    let dst_group = dst_buffer.as_mut_ptr();

    let wg_width = width / SRC_DEPTH;

    for y in 0..height {
        for x in 0..wg_width {
            let cb = _mm256_set_m128i(
                zero_128,
                loadu(u_group.add(wg_index(x, y, SRC_DEPTH, u_stride)).cast()),
            );
            let cr = _mm256_set_m128i(
                zero_128,
                loadu(v_group.add(wg_index(x, y, SRC_DEPTH, v_stride)).cast()),
            );
            let y0 = _mm256_set_m128i(
                zero_128,
                loadu(y_group.add(wg_index(x, y, SRC_DEPTH, y_stride)).cast()),
            );

            let cb_lo = _mm256_permute2x128_si256(
                _mm256_unpacklo_epi8(zero!(), cb),
                _mm256_unpackhi_epi8(zero!(), cb),
                PACK_LO_DQWORD_2X256,
            );

            let cr_lo = _mm256_permute2x128_si256(
                _mm256_unpacklo_epi8(zero!(), cr),
                _mm256_unpackhi_epi8(zero!(), cr),
                PACK_LO_DQWORD_2X256,
            );

            let sb_lo = _mm256_sub_epi16(_mm256_mulhi_epu16(cb_lo, bcbm), bn);
            let sr_lo = _mm256_sub_epi16(_mm256_mulhi_epu16(cr_lo, rcrm), rn);
            let sg_lo = _mm256_sub_epi16(
                gp,
                _mm256_add_epi16(
                    _mm256_mulhi_epu16(cb_lo, gcbm),
                    _mm256_mulhi_epu16(cr_lo, gcrm),
                ),
            );

            let y_lo = _mm256_mulhi_epu16(
                _mm256_permute2x128_si256(
                    _mm256_unpacklo_epi8(zero!(), y0),
                    _mm256_unpackhi_epi8(zero!(), y0),
                    PACK_LO_DQWORD_2X256,
                ),
                xxym,
            );
            pack_i16x3_16x(
                dst_group.add(wg_index(x, y, DST_DEPTH, dst_stride)),
                fix_to_i16_16x!(_mm256_add_epi16(sr_lo, y_lo), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sg_lo, y_lo), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sb_lo, y_lo), FIX6),
            );
        }
    }
}

#[inline(never)]
fn nv12_bgra_lrgb(
    width: u32,
    height: u32,
    last_src_plane: usize,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: usize,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
    colorimetry: usize,
) -> bool {
    const DST_DEPTH: usize = 4;

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

    let dst_buffer = &mut *dst_buffers[0];
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
            yuv_to_lrgb_avx2(
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
    _last_src_plane: usize,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: usize,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
    colorimetry: usize,
) -> bool {
    const DST_DEPTH: usize = 4;

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
    let dst_buffer = &mut *dst_buffers[0];
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
            i420_to_lrgb_avx2(
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
    _last_src_plane: usize,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: usize,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
    colorimetry: usize,
) -> bool {
    const DST_DEPTH: usize = 4;

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
    let dst_buffer = &mut *dst_buffers[0];
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
            i444_to_lrgb_avx2(
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
    _last_src_plane: usize,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: usize,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
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
    let depth = match sampler {
        Sampler::Bgr => 3_usize,
        _ => 4_usize,
    };
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
    let (y_plane, u_plane, v_plane) = (&mut *y_plane[0], &mut *u_plane[0], &mut *v_plane[0]);
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
            lrgb_to_i444_avx2(
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
    _last_src_plane: usize,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    _last_dst_plane: usize,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
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
    let depth = match sampler {
        Sampler::Bgr => 3_usize,
        _ => 4_usize,
    };
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
    let (y_plane, u_plane, v_plane) = (&mut *y_plane[0], &mut *u_plane[0], &mut *v_plane[0]);
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
            lrgb_to_i420_avx2(
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
    _last_src_plane: usize,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: usize,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
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
    let depth = match sampler {
        Sampler::Bgr => 3_usize,
        _ => 4_usize,
    };
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

        (&mut *y_plane[0], &mut *uv_plane[0])
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
            lrgb_to_yuv_avx2(
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
yuv_to_rgb_converter!(I420, Bt601);
yuv_to_rgb_converter!(I420, Bt601FR);
yuv_to_rgb_converter!(I420, Bt709);
yuv_to_rgb_converter!(I420, Bt709FR);
yuv_to_rgb_converter!(I444, Bt601);
yuv_to_rgb_converter!(I444, Bt601FR);
yuv_to_rgb_converter!(I444, Bt709);
yuv_to_rgb_converter!(I444, Bt709FR);
yuv_to_rgb_converter!(Nv12, Bt601);
yuv_to_rgb_converter!(Nv12, Bt601FR);
yuv_to_rgb_converter!(Nv12, Bt709);
yuv_to_rgb_converter!(Nv12, Bt709FR);

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

    // Process vector part and scalar one
    let vector_part = lower_multiple_of_pot(w, LANE_COUNT);
    let scalar_part = w - vector_part;
    if vector_part > 0 {
        unsafe {
            rgb_to_bgra_avx2(
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

pub fn bgra_lrgb_rgb_lrgb(
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

    unsafe {
        bgra_to_rgb_avx2(w, h, src_stride, src_buffer, dst_stride, dst_buffer);
    }

    true
}

pub fn argb_lrgb_rgb_lrgb(
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

    unsafe {
        argb_to_rgb_avx2(w, h, src_stride, src_buffer, dst_stride, dst_buffer);
    }

    true
}
