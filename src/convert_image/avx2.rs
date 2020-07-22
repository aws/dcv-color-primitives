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
use crate::convert_image::sse2;
use crate::convert_image::x86;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m128i, __m256i, _mm256_add_epi16, _mm256_add_epi32, _mm256_cmpeq_epi32, _mm256_loadu2_m128i,
    _mm256_loadu_si256, _mm256_madd_epi16, _mm256_mulhi_epu16, _mm256_or_si256, _mm256_packs_epi32,
    _mm256_packus_epi16, _mm256_permute2x128_si256, _mm256_permutevar8x32_epi32, _mm256_set1_epi16,
    _mm256_set1_epi32, _mm256_set_epi32, _mm256_set_epi64x, _mm256_set_m128i, _mm256_setzero_si256,
    _mm256_slli_epi16, _mm256_slli_epi32, _mm256_srai_epi16, _mm256_srai_epi32, _mm256_srli_epi16,
    _mm256_srli_epi32, _mm256_srli_si256, _mm256_storeu_si256, _mm256_sub_epi16,
    _mm256_unpackhi_epi16, _mm256_unpackhi_epi8, _mm256_unpacklo_epi16, _mm256_unpacklo_epi32,
    _mm256_unpacklo_epi64, _mm256_unpacklo_epi8, _mm_loadu_si128, _mm_setzero_si128,
};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm256_add_epi16, _mm256_add_epi32, _mm256_cmpeq_epi32, _mm256_extract_epi64,
    _mm256_loadu2_m128i, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_mulhi_epu16,
    _mm256_or_si256, _mm256_packs_epi32, _mm256_packus_epi16, _mm256_permute2x128_si256,
    _mm256_permutevar8x32_epi32, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_set_epi32,
    _mm256_set_epi64x, _mm256_set_m128i, _mm256_setzero_si256, _mm256_slli_epi16,
    _mm256_slli_epi32, _mm256_srai_epi16, _mm256_srai_epi32, _mm256_srli_epi16, _mm256_srli_epi32,
    _mm256_srli_si256, _mm256_storeu_si256, _mm256_sub_epi16, _mm256_unpackhi_epi16,
    _mm256_unpackhi_epi8, _mm256_unpacklo_epi16, _mm256_unpacklo_epi32, _mm256_unpacklo_epi64,
    _mm256_unpacklo_epi8, _mm_loadu_si128, _mm_setzero_si128,
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

/// Convert fixed point to int (8-wide)
macro_rules! fix_to_i32_8x {
    ($fix:expr, $frac_bits:expr) => {
        _mm256_srai_epi32($fix, $frac_bits);
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
unsafe fn _mm256_extract_epi64(a: __m256i, index: i32) -> i64 {
    let slice = std::mem::transmute::<__m256i, [i64; 4]>(a);
    return slice[index as usize];
}

#[inline]
fn split_planes<'a>(
    last_src_plane: usize,
    interplane_split: usize,
    src_buffers: &[&'a [u8]],
) -> Option<(&'a [u8], &'a [u8])> {
    src_buffers
        .split_first()
        .and_then(|(y, uv)| match last_src_plane {
            0 => {
                if interplane_split <= y.len() {
                    Some(y.split_at(interplane_split))
                } else {
                    None
                }
            }
            _ => uv.split_first().map(|(uv, _)| (&y[..], &uv[..])),
        })
}

#[inline]
fn split_planes_mut<'a>(
    last_src_plane: usize,
    interplane_split: usize,
    src_buffers: &'a mut [&mut [u8]],
) -> Option<(&'a mut [u8], &'a mut [u8])> {
    src_buffers
        .split_first_mut()
        .and_then(|(y, uv)| match last_src_plane {
            0 => {
                if interplane_split <= y.len() {
                    Some(y.split_at_mut(interplane_split))
                } else {
                    None
                }
            }
            _ => uv
                .split_first_mut()
                .map(move |(uv, _)| (&mut y[..], &mut uv[..])),
        })
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
    let x = _mm_loadu_si128(image as *const __m128i);
    let xx = _mm256_set_m128i(x, x);

    let hi = _mm256_unpackhi_epi8(zero!(), xx);
    let lo = _mm256_unpacklo_epi8(zero!(), xx);

    _mm256_permute2x128_si256(lo, hi, PACK_LO_DQWORD_2X256)
}

/// Deinterleave 2 uchar samples into short samples,
/// stored in big endian (16-wide)
#[inline(always)]
unsafe fn unpack_ui8x2_i16be_16x(image: *const u8) -> (__m256i, __m256i) {
    let x = _mm256_loadu_si256(image as *const __m256i);
    (
        _mm256_slli_epi16(x, 8),
        _mm256_slli_epi16(_mm256_srli_epi16(x, 8), 8),
    )
}

/// Truncate and deinterleave 3 short samples into 4 uchar samples (16-wide)
/// Alpha set to DEFAULT_ALPHA
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

    _mm256_storeu_si256(
        image as *mut __m256i,
        _mm256_permute2x128_si256(rgbw_lo, rgbw_hi, PACK_LO_DQWORD_2X256),
    );

    _mm256_storeu_si256(
        image.add(LANE_COUNT) as *mut __m256i,
        _mm256_permute2x128_si256(rgbw_lo, rgbw_hi, PACK_HI_DQWORD_2X256),
    );
}

/// Convert 3 deinterleaved uchar samples into 2 deinterleaved
/// short samples (8-wide)
#[inline(always)]
unsafe fn unpack_ui8x3_i16x2_8x(image: *const u8, sampler: Sampler) -> (__m256i, __m256i) {
    let line = match sampler {
        Sampler::BgrOverflow => _mm256_set_epi64x(
            0,
            *(image.offset(16) as *const i64),
            *(image.offset(8) as *const i64),
            *(image as *const i64),
        ),
        _ => _mm256_loadu_si256(image as *const __m256i),
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
    *(image as *mut i64) = _mm256_extract_epi64(z, 0);
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
    pack_i32_8x(
        uv,
        fix_to_i32_8x!(affine_transform(srg, sbg, uv_weights), FIX18),
    );
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

    let shuff = _mm256_permutevar8x32_epi32(
        fix_to_i32_8x!(affine_transform(srg, sbg, uv_weights), FIX18),
        _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0),
    );

    let packed_to_32 = _mm256_packs_epi32(shuff, shuff);
    let packed_to_16 = _mm256_packus_epi16(packed_to_32, packed_to_32);
    let permuted = _mm256_permutevar8x32_epi32(packed_to_16, pack_lo_dword_2x128!());
    let uv_res = _mm256_extract_epi64(permuted, 0) as u64;

    *(u as *mut u32) = uv_res as u32;
    *(v as *mut u32) = (uv_res >> 32) as u32;
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
) {
    let (rg, bg) = unpack_ui8x3_i16x2_8x(rgb, sampler);
    pack_i32_8x(
        y,
        fix_to_i32_8x!(affine_transform(rg, bg, y_weights), FIX16),
    );

    pack_i32_8x(
        u,
        fix_to_i32_8x!(affine_transform(rg, bg, u_weights), FIX16),
    );

    pack_i32_8x(
        v,
        fix_to_i32_8x!(affine_transform(rg, bg, v_weights), FIX16),
    );
}

#[inline(always)]
fn lrgb_to_yuv(
    width: u32,
    height: u32,
    last_src_plane: usize,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: usize,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
    channels: PixelFormatChannels,
    colorimetry: Colorimetry,
    sampler: Sampler,
) -> bool {
    unsafe {
        lrgb_to_yuv_avx2(
            width,
            height,
            last_src_plane,
            src_strides,
            src_buffers,
            last_dst_plane,
            dst_strides,
            dst_buffers,
            channels,
            colorimetry,
            sampler,
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn lrgb_to_yuv_avx2(
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

    let yuv_planes = split_planes_mut(last_dst_plane, y_stride * line_count, dst_buffers);
    if yuv_planes.is_none() {
        return false;
    }

    if line_count == 0 || col_count == 0 {
        return true;
    }

    let max_stride = usize::max_value() / line_count;
    if (y_stride > max_stride) || (uv_stride > max_stride) || (rgb_stride > max_stride) {
        return false;
    }

    let (y_plane, uv_plane) = yuv_planes.unwrap();
    let wg_height = line_count / 2;
    if y_stride * line_count > y_plane.len()
        || uv_stride * wg_height > uv_plane.len()
        || rgb_stride * line_count > rgb_plane.len()
    {
        return false;
    }

    let col = colorimetry as usize;
    if col > 1 {
        return false;
    }

    let y_weigths = [
        _mm256_set1_epi32(FORWARD_WEIGHTS[col][0]),
        _mm256_set1_epi32(FORWARD_WEIGHTS[col][1]),
        _mm256_set1_epi32(Y_OFFSET),
    ];

    let uv_weights = [
        _mm256_set_epi32(
            FORWARD_WEIGHTS[col][2],
            FORWARD_WEIGHTS[col][3],
            FORWARD_WEIGHTS[col][2],
            FORWARD_WEIGHTS[col][3],
            FORWARD_WEIGHTS[col][2],
            FORWARD_WEIGHTS[col][3],
            FORWARD_WEIGHTS[col][2],
            FORWARD_WEIGHTS[col][3],
        ),
        _mm256_set_epi32(
            FORWARD_WEIGHTS[col][4],
            FORWARD_WEIGHTS[col][5],
            FORWARD_WEIGHTS[col][4],
            FORWARD_WEIGHTS[col][5],
            FORWARD_WEIGHTS[col][4],
            FORWARD_WEIGHTS[col][5],
            FORWARD_WEIGHTS[col][4],
            FORWARD_WEIGHTS[col][5],
        ),
        _mm256_set1_epi32(C_OFFSET),
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
            lrgb_to_yuv_8x(
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
            lrgb_to_yuv_8x(
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
        lrgb_to_yuv_8x(
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

    true
}

#[inline(always)]
fn lrgb_to_i420(
    width: u32,
    height: u32,
    last_src_plane: usize,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: usize,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
    channels: PixelFormatChannels,
    colorimetry: Colorimetry,
    sampler: Sampler,
) -> bool {
    unsafe {
        lrgb_to_i420_avx2(
            width,
            height,
            last_src_plane,
            src_strides,
            src_buffers,
            last_dst_plane,
            dst_strides,
            dst_buffers,
            channels,
            colorimetry,
            sampler,
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn lrgb_to_i420_avx2(
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
    if last_dst_plane != 2
        || (last_dst_plane >= dst_strides.len())
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

    let rgb_stride = if src_strides[0] == 0 {
        packed_rgb_stride
    } else {
        src_strides[0]
    };

    let y_stride = if dst_strides[0] == 0 {
        col_count
    } else {
        dst_strides[0]
    };

    let u_stride = if dst_strides[1] == 0 {
        col_count / 2
    } else {
        dst_strides[1]
    };

    let v_stride = if dst_strides[2] == 0 {
        col_count / 2
    } else {
        dst_strides[2]
    };

    let rgb_plane = &src_buffers[0];
    let (y_plane, uv_plane) = dst_buffers.split_at_mut(1);
    let (u_plane, v_plane) = uv_plane.split_at_mut(1);

    let y_plane = &mut y_plane[0][..];
    let u_plane = &mut u_plane[0][..];
    let v_plane = &mut v_plane[0][..];

    if line_count == 0 || col_count == 0 {
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
    if col > 1 {
        return false;
    }

    let y_weigths = [
        _mm256_set1_epi32(FORWARD_WEIGHTS[col][0]),
        _mm256_set1_epi32(FORWARD_WEIGHTS[col][1]),
        _mm256_set1_epi32(Y_OFFSET),
    ];

    let uv_weights = [
        _mm256_set_epi32(
            FORWARD_WEIGHTS[col][2],
            FORWARD_WEIGHTS[col][3],
            FORWARD_WEIGHTS[col][2],
            FORWARD_WEIGHTS[col][3],
            FORWARD_WEIGHTS[col][2],
            FORWARD_WEIGHTS[col][3],
            FORWARD_WEIGHTS[col][2],
            FORWARD_WEIGHTS[col][3],
        ),
        _mm256_set_epi32(
            FORWARD_WEIGHTS[col][4],
            FORWARD_WEIGHTS[col][5],
            FORWARD_WEIGHTS[col][4],
            FORWARD_WEIGHTS[col][5],
            FORWARD_WEIGHTS[col][4],
            FORWARD_WEIGHTS[col][5],
            FORWARD_WEIGHTS[col][4],
            FORWARD_WEIGHTS[col][5],
        ),
        _mm256_set1_epi32(C_OFFSET),
    ];

    let rgb_depth = depth * LRGB_TO_YUV_WAVES;
    let read_bytes_per_line = ((col_count - 1) / LRGB_TO_YUV_WAVES) * rgb_depth + LANE_COUNT;

    let y_start = if (depth == 4) || (read_bytes_per_line <= rgb_stride) {
        line_count
    } else {
        line_count - 2
    };

    let rgb_group = rgb_plane.as_ptr();
    let y_group = y_plane.as_mut_ptr();
    let u_group = u_plane.as_mut_ptr();
    let v_group = v_plane.as_mut_ptr();
    let wg_width = col_count / LRGB_TO_YUV_WAVES;
    let wg_height = y_start / 2;

    for y in 0..wg_height {
        for x in 0..wg_width {
            lrgb_to_i420_8x(
                rgb_group.add(wg_index(x, 2 * y, rgb_depth, rgb_stride)),
                rgb_group.add(wg_index(x, 2 * y + 1, rgb_depth, rgb_stride)),
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
    if y_start != line_count {
        let wg_width = (col_count - LRGB_TO_YUV_WAVES) / LRGB_TO_YUV_WAVES;
        for x in 0..wg_width {
            lrgb_to_i420_8x(
                rgb_group.add(wg_index(x, y_start, rgb_depth, rgb_stride)),
                rgb_group.add(wg_index(x, y_start + 1, rgb_depth, rgb_stride)),
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
        lrgb_to_i420_8x(
            rgb_group.add(wg_index(wg_width, y_start, rgb_depth, rgb_stride)),
            rgb_group.add(wg_index(wg_width, y_start + 1, rgb_depth, rgb_stride)),
            y_group.add(wg_index(wg_width, y_start, LRGB_TO_YUV_WAVES, y_stride)),
            y_group.add(wg_index(wg_width, y_start + 1, LRGB_TO_YUV_WAVES, y_stride)),
            u_group.add(wg_index(
                wg_width,
                wg_height,
                LRGB_TO_YUV_WAVES / 2,
                u_stride,
            )),
            v_group.add(wg_index(
                wg_width,
                wg_height,
                LRGB_TO_YUV_WAVES / 2,
                v_stride,
            )),
            Sampler::BgrOverflow,
            &y_weigths,
            &uv_weights,
        );
    }

    true
}

#[inline(always)]
fn lrgb_to_i444(
    width: u32,
    height: u32,
    last_src_plane: usize,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
    last_dst_plane: usize,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
    channels: PixelFormatChannels,
    colorimetry: Colorimetry,
    sampler: Sampler,
) -> bool {
    unsafe {
        lrgb_to_i444_avx2(
            width,
            height,
            last_src_plane,
            src_strides,
            src_buffers,
            last_dst_plane,
            dst_strides,
            dst_buffers,
            channels,
            colorimetry,
            sampler,
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn lrgb_to_i444_avx2(
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
    if last_dst_plane != 2
        || (last_dst_plane >= dst_strides.len())
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

    let rgb_stride = if src_strides[0] == 0 {
        packed_rgb_stride
    } else {
        src_strides[0]
    };

    let y_stride = if dst_strides[0] == 0 {
        col_count
    } else {
        dst_strides[0]
    };

    let u_stride = if dst_strides[1] == 0 {
        col_count
    } else {
        dst_strides[1]
    };

    let v_stride = if dst_strides[2] == 0 {
        col_count
    } else {
        dst_strides[2]
    };

    let rgb_plane = &src_buffers[0];
    let (y_plane, uv_plane) = dst_buffers.split_at_mut(1);
    let (u_plane, v_plane) = uv_plane.split_at_mut(1);

    let y_plane = &mut y_plane[0][..];
    let u_plane = &mut u_plane[0][..];
    let v_plane = &mut v_plane[0][..];

    if line_count == 0 || col_count == 0 {
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
    if col > 1 {
        return false;
    }

    let y_weights = [
        _mm256_set1_epi32(FORWARD_WEIGHTS[col][0]),
        _mm256_set1_epi32(FORWARD_WEIGHTS[col][1]),
        _mm256_set1_epi32(Y_OFFSET),
    ];

    let u_weights = [
        _mm256_set1_epi32(FORWARD_WEIGHTS[col][3]),
        _mm256_set1_epi32(FORWARD_WEIGHTS[col][5]),
        _mm256_set1_epi32(C_OFFSET16),
    ];

    let v_weights = [
        _mm256_set1_epi32(FORWARD_WEIGHTS[col][2]),
        _mm256_set1_epi32(FORWARD_WEIGHTS[col][4]),
        _mm256_set1_epi32(C_OFFSET16),
    ];

    let rgb_depth = depth * LRGB_TO_YUV_WAVES;
    let read_bytes_per_line = ((col_count - 1) / LRGB_TO_YUV_WAVES) * rgb_depth + LANE_COUNT;

    let y_start = if (depth == 4) || (read_bytes_per_line <= rgb_stride) {
        line_count
    } else {
        line_count - 1
    };

    let rgb_group = rgb_plane.as_ptr();
    let y_group = y_plane.as_mut_ptr();
    let u_group = u_plane.as_mut_ptr();
    let v_group = v_plane.as_mut_ptr();
    let wg_width = col_count / LRGB_TO_YUV_WAVES;
    let wg_height = y_start;

    for y in 0..wg_height {
        for x in 0..wg_width {
            lrgb_to_i444_8x(
                rgb_group.add(wg_index(x, y, rgb_depth, rgb_stride)),
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
    if y_start != line_count {
        let wg_width = (col_count - LRGB_TO_YUV_WAVES) / LRGB_TO_YUV_WAVES;
        for x in 0..wg_width {
            lrgb_to_i444_8x(
                rgb_group.add(wg_index(x, y_start, rgb_depth, rgb_stride)),
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
        lrgb_to_i444_8x(
            rgb_group.add(wg_index(wg_width, y_start, rgb_depth, rgb_stride)),
            y_group.add(wg_index(wg_width, y_start, LRGB_TO_YUV_WAVES, y_stride)),
            u_group.add(wg_index(wg_width, y_start, LRGB_TO_YUV_WAVES, u_stride)),
            v_group.add(wg_index(wg_width, y_start, LRGB_TO_YUV_WAVES, v_stride)),
            Sampler::BgrOverflow,
            &y_weights,
            &u_weights,
            &v_weights,
        );
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
    last_dst_plane: usize,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
    channels: PixelFormatChannels,
    colorimetry: Colorimetry,
) -> bool {
    unsafe {
        yuv_to_lrgb_avx2(
            width,
            height,
            last_src_plane,
            src_strides,
            src_buffers,
            last_dst_plane,
            dst_strides,
            dst_buffers,
            channels,
            colorimetry,
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn yuv_to_lrgb_avx2(
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

    let yuv_planes = split_planes(last_src_plane, y_stride * line_count, src_buffers);
    if yuv_planes.is_none() {
        return false;
    }

    if line_count == 0 || col_count == 0 {
        return true;
    }

    let max_stride = usize::max_value() / line_count;
    if (y_stride > max_stride) || (uv_stride > max_stride) || (rgb_stride > max_stride) {
        return false;
    }

    let (y_plane, uv_plane) = yuv_planes.unwrap();
    let wg_height = line_count / 2;
    if y_stride * line_count > y_plane.len()
        || uv_stride * wg_height > uv_plane.len()
        || rgb_stride * line_count > rgb_plane.len()
    {
        return false;
    }

    let col = colorimetry as usize;
    if col > 1 {
        return false;
    }

    let xxym = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][0]);
    let rcrm = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][1]);
    let gcrm = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][2]);
    let gcbm = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][3]);
    let bcbm = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][4]);
    let rn = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][5]);
    let gp = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][6]);
    let bn = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][7]);

    let y_group = y_plane.as_ptr();
    let uv_group = uv_plane.as_ptr();
    let rgb_group = rgb_plane.as_mut_ptr();
    let rgb_depth = 2 * YUV_TO_LRGB_WAVES;
    let nv12_depth = YUV_TO_LRGB_WAVES;
    let wg_width = col_count / YUV_TO_LRGB_WAVES;

    for y in 0..wg_height {
        for x in 0..wg_width {
            let (cb, cr) =
                unpack_ui8x2_i16be_16x(uv_group.add(wg_index(x, y, nv12_depth, uv_stride)));

            let sb = _mm256_sub_epi16(_mm256_mulhi_epu16(cb, bcbm), bn);
            let sr = _mm256_sub_epi16(_mm256_mulhi_epu16(cr, rcrm), rn);
            let sg = _mm256_sub_epi16(
                gp,
                _mm256_add_epi16(_mm256_mulhi_epu16(cb, gcbm), _mm256_mulhi_epu16(cr, gcrm)),
            );

            let (sb_lo, sb_hi) = i16_to_i16x2_16x(sb);
            let (sr_lo, sr_hi) = i16_to_i16x2_16x(sr);
            let (sg_lo, sg_hi) = i16_to_i16x2_16x(sg);

            let y0 = _mm256_loadu_si256(
                y_group.add(wg_index(x, 2 * y, nv12_depth, y_stride)) as *const __m256i
            );

            let y00 = _mm256_mulhi_epu16(
                _mm256_permute2x128_si256(
                    _mm256_unpacklo_epi8(zero!(), y0),
                    _mm256_unpackhi_epi8(zero!(), y0),
                    PACK_LO_DQWORD_2X256,
                ),
                xxym,
            );
            pack_i16x3_16x(
                rgb_group.add(wg_index(2 * x, 2 * y, rgb_depth, rgb_stride)),
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
                rgb_group.add(wg_index(2 * x + 1, 2 * y, rgb_depth, rgb_stride)),
                fix_to_i16_16x!(_mm256_add_epi16(sr_hi, y10), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sg_hi, y10), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sb_hi, y10), FIX6),
            );

            let y1 = _mm256_loadu_si256(
                y_group.add(wg_index(x, 2 * y + 1, nv12_depth, y_stride)) as *const __m256i
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
                rgb_group.add(wg_index(2 * x, 2 * y + 1, rgb_depth, rgb_stride)),
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
                rgb_group.add(wg_index(2 * x + 1, 2 * y + 1, rgb_depth, rgb_stride)),
                fix_to_i16_16x!(_mm256_add_epi16(sr_hi, y11), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sg_hi, y11), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sb_hi, y11), FIX6),
            );
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
    last_dst_plane: usize,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
    channels: PixelFormatChannels,
    colorimetry: Colorimetry,
) -> bool {
    unsafe {
        i420_to_lrgb_avx2(
            width,
            height,
            last_src_plane,
            src_strides,
            src_buffers,
            last_dst_plane,
            dst_strides,
            dst_buffers,
            channels,
            colorimetry,
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn i420_to_lrgb_avx2(
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

    if line_count == 0 || col_count == 0 {
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
    if col > 1 {
        return false;
    }

    let xxym = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][0]);
    let rcrm = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][1]);
    let gcrm = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][2]);
    let gcbm = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][3]);
    let bcbm = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][4]);
    let rn = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][5]);
    let gp = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][6]);
    let bn = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][7]);

    let y_group = y_plane.as_ptr();
    let u_group = u_plane.as_ptr();
    let v_group = v_plane.as_ptr();
    let rgb_group = rgb_plane.as_mut_ptr();
    let rgb_depth = 2 * YUV_TO_LRGB_WAVES;
    let i420_depth = YUV_TO_LRGB_WAVES;
    let wg_width = col_count / YUV_TO_LRGB_WAVES;

    for y in 0..wg_height {
        for x in 0..wg_width {
            let cb = unpack_ui8_i16be_16x(u_group.add(wg_index(x, y, i420_depth / 2, u_stride)));
            let cr = unpack_ui8_i16be_16x(v_group.add(wg_index(x, y, i420_depth / 2, v_stride)));

            let sb = _mm256_sub_epi16(_mm256_mulhi_epu16(cb, bcbm), bn);
            let sr = _mm256_sub_epi16(_mm256_mulhi_epu16(cr, rcrm), rn);
            let sg = _mm256_sub_epi16(
                gp,
                _mm256_add_epi16(_mm256_mulhi_epu16(cb, gcbm), _mm256_mulhi_epu16(cr, gcrm)),
            );

            let (sb_lo, sb_hi) = i16_to_i16x2_16x(sb);
            let (sr_lo, sr_hi) = i16_to_i16x2_16x(sr);
            let (sg_lo, sg_hi) = i16_to_i16x2_16x(sg);

            let y0 = _mm256_loadu_si256(
                y_group.add(wg_index(x, 2 * y, i420_depth, y_stride)) as *const __m256i
            );

            let y00 = _mm256_mulhi_epu16(
                _mm256_permute2x128_si256(
                    _mm256_unpacklo_epi8(zero!(), y0),
                    _mm256_unpackhi_epi8(zero!(), y0),
                    PACK_LO_DQWORD_2X256,
                ),
                xxym,
            );
            pack_i16x3_16x(
                rgb_group.add(wg_index(2 * x, 2 * y, rgb_depth, rgb_stride)),
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
                rgb_group.add(wg_index(2 * x + 1, 2 * y, rgb_depth, rgb_stride)),
                fix_to_i16_16x!(_mm256_add_epi16(sr_hi, y10), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sg_hi, y10), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sb_hi, y10), FIX6),
            );

            let y1 = _mm256_loadu_si256(
                y_group.add(wg_index(x, 2 * y + 1, i420_depth, y_stride)) as *const __m256i
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
                rgb_group.add(wg_index(2 * x, 2 * y + 1, rgb_depth, rgb_stride)),
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
                rgb_group.add(wg_index(2 * x + 1, 2 * y + 1, rgb_depth, rgb_stride)),
                fix_to_i16_16x!(_mm256_add_epi16(sr_hi, y11), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sg_hi, y11), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sb_hi, y11), FIX6),
            );
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
    last_dst_plane: usize,
    dst_strides: &[usize],
    dst_buffers: &mut [&mut [u8]],
    channels: PixelFormatChannels,
    colorimetry: Colorimetry,
) -> bool {
    unsafe {
        i444_to_lrgb_avx2(
            width,
            height,
            last_src_plane,
            src_strides,
            src_buffers,
            last_dst_plane,
            dst_strides,
            dst_buffers,
            channels,
            colorimetry,
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn i444_to_lrgb_avx2(
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

    if line_count == 0 || col_count == 0 {
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
    if col > 1 {
        return false;
    }

    let xxym = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][0]);
    let rcrm = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][1]);
    let gcrm = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][2]);
    let gcbm = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][3]);
    let bcbm = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][4]);
    let rn = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][5]);
    let gp = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][6]);
    let bn = _mm256_set1_epi16(BACKWARD_WEIGHTS[col][7]);
    let zero_128 = _mm_setzero_si128();

    let y_group = y_plane.as_ptr();
    let u_group = u_plane.as_ptr();
    let v_group = v_plane.as_ptr();
    let rgb_group = rgb_plane.as_mut_ptr();
    let rgb_depth = 2 * YUV_TO_LRGB_WAVES;
    let group_width = YUV_TO_LRGB_WAVES / 2;
    let wg_width = col_count / group_width;

    for y in 0..line_count {
        for x in 0..wg_width {
            let cb = _mm256_loadu2_m128i(
                &zero_128,
                u_group.add(wg_index(x, y, group_width, u_stride)) as *const __m128i,
            );
            let cr = _mm256_loadu2_m128i(
                &zero_128,
                v_group.add(wg_index(x, y, group_width, v_stride)) as *const __m128i,
            );
            let y0 = _mm256_loadu2_m128i(
                &zero_128,
                y_group.add(wg_index(x, y, group_width, y_stride)) as *const __m128i,
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
                rgb_group.add(wg_index(x, y, rgb_depth, rgb_stride)),
                fix_to_i16_16x!(_mm256_add_epi16(sr_lo, y_lo), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sg_lo, y_lo), FIX6),
                fix_to_i16_16x!(_mm256_add_epi16(sb_lo, y_lo), FIX6),
            );
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
        sse2::argb_lrgb_nv12_bt601(
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
        sse2::argb_lrgb_nv12_bt709(
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
        sse2::bgra_lrgb_nv12_bt601(
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
        sse2::bgra_lrgb_nv12_bt709(
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
        sse2::bgr_lrgb_nv12_bt601(
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
        sse2::bgr_lrgb_nv12_bt709(
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
        sse2::nv12_bt601_bgra_lrgb(
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
        sse2::nv12_bt709_bgra_lrgb(
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
        sse2::i420_bt601_bgra_lrgb(
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
        sse2::i420_bt709_bgra_lrgb(
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
        sse2::i444_bt601_bgra_lrgb(
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
        sse2::i444_bt709_bgra_lrgb(
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

pub fn argb_lrgb_i420_bt601(
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
        lrgb_to_i420(
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
        sse2::argb_lrgb_i420_bt601(
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

pub fn argb_lrgb_i420_bt709(
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
        lrgb_to_i420(
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
        sse2::argb_lrgb_i420_bt709(
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

pub fn bgra_lrgb_i420_bt601(
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
        lrgb_to_i420(
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
        sse2::bgra_lrgb_i420_bt601(
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

pub fn bgra_lrgb_i420_bt709(
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
        lrgb_to_i420(
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
        sse2::bgra_lrgb_i420_bt709(
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

pub fn bgr_lrgb_i420_bt601(
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
        lrgb_to_i420(
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
        sse2::bgr_lrgb_i420_bt601(
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

pub fn bgr_lrgb_i420_bt709(
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
        lrgb_to_i420(
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
        sse2::bgr_lrgb_i420_bt709(
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

pub fn argb_lrgb_i444_bt601(
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
        lrgb_to_i444(
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
        sse2::argb_lrgb_i444_bt601(
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

pub fn argb_lrgb_i444_bt709(
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
        lrgb_to_i444(
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
        sse2::argb_lrgb_i444_bt709(
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

pub fn bgra_lrgb_i444_bt601(
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
        lrgb_to_i444(
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
        sse2::bgra_lrgb_i444_bt601(
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

pub fn bgra_lrgb_i444_bt709(
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
        lrgb_to_i444(
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
        sse2::bgra_lrgb_i444_bt709(
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

pub fn bgr_lrgb_i444_bt601(
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
        lrgb_to_i444(
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
        sse2::bgr_lrgb_i444_bt601(
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

pub fn bgr_lrgb_i444_bt709(
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
        lrgb_to_i444(
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
        sse2::bgr_lrgb_i444_bt709(
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
