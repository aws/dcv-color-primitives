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
use crate::{rgb_to_yuv_converter, yuv_to_rgb_fallback_converter};

use core::arch::aarch64::*;

type YWeights = (u16, u16, u16, u16);
type UVWeights = (i16, i16, i16, i16, i16);
type Weights = (YWeights, UVWeights);

const LANE_COUNT: usize = 16;
const RGB_TO_YUV_WG_SIZE: usize = 2;
const RGB_TO_YUV_WAVES: usize = LANE_COUNT / RGB_TO_YUV_WG_SIZE;

const FORWARD_WEIGHTS: [Weights; Colorimetry::Length as usize] = [
    (
        (
            as_u16::<XR_601>(),
            as_u16::<XG_601>(),
            as_u16::<XB_601>(),
            as_u16::<Y_MIN>(),
        ),
        (
            as_i16::<YR_601>(),
            as_i16::<YG_601>(),
            as_i16::<ZG_601>(),
            as_i16::<{ -(YR_601 + YG_601) }>(),
            as_i16::<{ -ZG_601 + YR_601 + YG_601 }>(),
        ),
    ),
    (
        (
            as_u16::<XR_709>(),
            as_u16::<XG_709>(),
            as_u16::<XB_709>(),
            as_u16::<Y_MIN>(),
        ),
        (
            as_i16::<YR_709>(),
            as_i16::<YG_709>(),
            as_i16::<ZG_709>(),
            as_i16::<{ -(YR_709 + YG_709) }>(),
            as_i16::<{ -ZG_709 + YR_709 + YG_709 }>(),
        ),
    ),
    (
        (
            as_u16::<XR_601FR>(),
            as_u16::<XG_601FR>(),
            as_u16::<XB_601FR>(),
            0,
        ),
        (
            as_i16::<YR_601FR>(),
            as_i16::<YG_601FR>(),
            as_i16::<ZG_601FR>(),
            as_i16::<{ -(YR_601FR + YG_601FR) }>(),
            as_i16::<{ -ZG_601FR + YR_601FR + YG_601FR }>(),
        ),
    ),
    (
        (
            as_u16::<XR_709FR>(),
            as_u16::<XG_709FR>(),
            as_u16::<XB_709FR>(),
            0,
        ),
        (
            as_i16::<YR_709FR>(),
            as_i16::<YG_709FR>(),
            as_i16::<ZG_709FR>(),
            as_i16::<{ -(YR_709FR + YG_709FR) }>(),
            as_i16::<{ -ZG_709FR + YR_709FR + YG_709FR }>(),
        ),
    ),
];
const UV_BIAS: i16 = as_i16::<C_HALF>();

#[inline(always)]
unsafe fn load_rgb_8x<const SAMPLER: usize>(
    rgb: *const u8,
) -> (uint16x8_t, uint16x8_t, uint16x8_t) {
    if SAMPLER == Sampler::Bgr as usize {
        let src = vld3_u8(rgb.cast());
        (vmovl_u8(src.2), vmovl_u8(src.1), vmovl_u8(src.0))
    } else if SAMPLER == Sampler::Argb as usize {
        let src = vld4_u8(rgb.cast());
        (vmovl_u8(src.1), vmovl_u8(src.2), vmovl_u8(src.3))
    } else {
        let src = vld4_u8(rgb.cast());
        (vmovl_u8(src.2), vmovl_u8(src.1), vmovl_u8(src.0))
    }
}

#[inline(always)]
unsafe fn rgb_to_avg_8x(
    r_top: uint16x8_t,
    g_top: uint16x8_t,
    b_top: uint16x8_t,
    r_bot: uint16x8_t,
    g_bot: uint16x8_t,
    b_bot: uint16x8_t,
) -> (uint16x8_t, uint16x8_t, uint16x8_t) {
    let r = vaddq_u16(r_top, r_bot);
    let r = vpaddq_u16(r, r);
    let g = vaddq_u16(g_top, g_bot);
    let g = vpaddq_u16(g, g);
    let b = vaddq_u16(b_top, b_bot);
    let b = vpaddq_u16(b, b);

    (r, g, b)
}

#[inline(always)]
unsafe fn rgb_to_us_8x(
    rgb: (uint16x8_t, uint16x8_t, uint16x8_t),
    rgb_lo: (uint16x4_t, uint16x4_t, uint16x4_t),
    y_weights: (u16, u16, u16),
) -> uint16x8_t {
    let lo = vmull_n_u16(rgb_lo.0, y_weights.0);
    let hi = vmull_high_n_u16(rgb.0, y_weights.0);
    let lo = vmlal_n_u16(lo, rgb_lo.1, y_weights.1);
    let hi = vmlal_high_n_u16(hi, rgb.1, y_weights.1);
    let lo = vmlal_n_u16(lo, rgb_lo.2, y_weights.2);
    let hi = vmlal_high_n_u16(hi, rgb.2, y_weights.2);

    vcombine_u16(vrshrn_n_u32::<FIX16>(lo), vrshrn_n_u32::<FIX16>(hi))
}

#[inline(always)]
unsafe fn rgb_to_s_8x(
    rgb: (int16x8_t, int16x8_t, int16x8_t),
    rgb_lo: (int16x4_t, int16x4_t, int16x4_t),
    uv_weights: (i16, i16, i16),
) -> int16x8_t {
    let lo = vmull_n_s16(rgb_lo.0, uv_weights.0);
    let hi = vmull_high_n_s16(rgb.0, uv_weights.0);
    let lo = vmlal_n_s16(lo, rgb_lo.1, uv_weights.1);
    let hi = vmlal_high_n_s16(hi, rgb.1, uv_weights.1);
    let lo = vmlal_n_s16(lo, rgb_lo.2, uv_weights.2);
    let hi = vmlal_high_n_s16(hi, rgb.2, uv_weights.2);

    vcombine_s16(vrshrn_n_s32::<FIX16>(lo), vrshrn_n_s32::<FIX16>(hi))
}

#[inline(always)]
unsafe fn rgb_to_y_8x(
    rgb: (uint16x8_t, uint16x8_t, uint16x8_t),
    y_weights: &YWeights,
) -> uint8x8_t {
    let rgb_lo = (
        vget_low_u16(rgb.0),
        vget_low_u16(rgb.1),
        vget_low_u16(rgb.2),
    );

    let val = rgb_to_us_8x(rgb, rgb_lo, (y_weights.0, y_weights.1, y_weights.2));
    let val = vaddq_u16(val, vdupq_n_u16(y_weights.3));

    vqmovn_u16(val)
}

#[inline(always)]
unsafe fn rgb_to_uv_8x(
    r: uint16x8_t,
    g: uint16x8_t,
    b: uint16x8_t,
    uv_weights: &UVWeights,
) -> (int16x4_t, int16x4_t) {
    let r_lo = vreinterpretq_s16_u16(r);
    let g_lo = vreinterpretq_s16_u16(g);
    let b_lo = vreinterpretq_s16_u16(b);

    let lo = vmull_high_n_s16(r_lo, uv_weights.0);
    let hi = vmull_high_n_s16(r_lo, uv_weights.3);
    let lo = vmlal_high_n_s16(lo, g_lo, uv_weights.1);
    let hi = vmlal_high_n_s16(hi, g_lo, uv_weights.2);
    let lo = vmlal_high_n_s16(lo, b_lo, uv_weights.3);
    let hi = vmlal_high_n_s16(hi, b_lo, uv_weights.4);

    let lo = vshrq_n_s32::<{ FIX18 - FIX16 }>(lo);
    let hi = vshrq_n_s32::<{ FIX18 - FIX16 }>(hi);
    let lo = vrshrn_n_s32::<FIX16>(lo);
    let hi = vrshrn_n_s32::<FIX16>(hi);

    (lo, hi)
}

#[inline(always)]
unsafe fn rgb_to_y_and_avg_uv_8x<const SAMPLER: usize>(
    rgb_top: *const u8,
    rgb_bot: *const u8,
    y_top: *mut u8,
    y_bot: *mut u8,
    weights: &Weights,
) -> (int16x4_t, int16x4_t) {
    let (y_weights, uv_weights) = weights;

    let (r_top, g_top, b_top) = {
        let rgb = load_rgb_8x::<SAMPLER>(rgb_top);
        vst1_u8(y_top.cast(), rgb_to_y_8x(rgb, y_weights));
        rgb
    };
    let (r_bot, g_bot, b_bot) = {
        let rgb = load_rgb_8x::<SAMPLER>(rgb_bot);
        vst1_u8(y_bot.cast(), rgb_to_y_8x(rgb, y_weights));
        rgb
    };

    let (r, g, b) = rgb_to_avg_8x(r_top, g_top, b_top, r_bot, g_bot, b_bot);
    rgb_to_uv_8x(r, g, b, uv_weights)
}

#[inline(always)]
unsafe fn rgb_to_i444_8x<const SAMPLER: usize>(
    rgb: *const u8,
    y: *mut u8,
    u: *mut u8,
    v: *mut u8,
    weights: &Weights,
) {
    let (y_weights, uv_weights) = weights;

    let rgb = load_rgb_8x::<SAMPLER>(rgb);
    let rgb_lo = (
        vget_low_u16(rgb.0),
        vget_low_u16(rgb.1),
        vget_low_u16(rgb.2),
    );

    let y_val = rgb_to_us_8x(rgb, rgb_lo, (y_weights.0, y_weights.1, y_weights.2));
    let y_val = vaddq_u16(y_val, vdupq_n_u16(y_weights.3));
    let y_val = vqmovn_u16(y_val);
    vst1_u8(y.cast(), y_val);

    let rgb = (
        vreinterpretq_s16_u16(rgb.0),
        vreinterpretq_s16_u16(rgb.1),
        vreinterpretq_s16_u16(rgb.2),
    );
    let rgb_lo = (
        vreinterpret_s16_u16(rgb_lo.0),
        vreinterpret_s16_u16(rgb_lo.1),
        vreinterpret_s16_u16(rgb_lo.2),
    );
    let uv_bias = vdupq_n_s16(UV_BIAS);

    let u_val = rgb_to_s_8x(rgb, rgb_lo, (uv_weights.0, uv_weights.1, uv_weights.3));
    let u_val = vaddq_s16(u_val, uv_bias);
    let u_val = vqmovn_u16(vreinterpretq_u16_s16(u_val));
    vst1_u8(u.cast(), u_val);

    let v_val = rgb_to_s_8x(rgb, rgb_lo, (uv_weights.3, uv_weights.2, uv_weights.4));
    let v_val = vaddq_s16(v_val, uv_bias);
    let v_val = vqmovn_u16(vreinterpretq_u16_s16(v_val));
    vst1_u8(v.cast(), v_val);
}

#[inline(always)]
unsafe fn rgb_to_nv12_8x<const SAMPLER: usize>(
    rgb_top: *const u8,
    rgb_bot: *const u8,
    y_top: *mut u8,
    y_bot: *mut u8,
    uv: *mut u8,
    weights: &Weights,
) {
    let (lo, hi) = rgb_to_y_and_avg_uv_8x::<SAMPLER>(rgb_top, rgb_bot, y_top, y_bot, weights);

    let u_and_v = vcombine_s16(vzip1_s16(lo, hi), vzip2_s16(lo, hi));
    let u_and_v = vaddq_s16(u_and_v, vdupq_n_s16(UV_BIAS));
    let u_and_v = vqmovn_u16(vreinterpretq_u16_s16(u_and_v));

    vst1_u8(uv.cast(), u_and_v);
}

#[inline(always)]
unsafe fn rgb_to_i420_8x<const SAMPLER: usize>(
    rgb_top: *const u8,
    rgb_bot: *const u8,
    y_top: *mut u8,
    y_bot: *mut u8,
    u: *mut u8,
    v: *mut u8,
    weights: &Weights,
) {
    let (lo, hi) = rgb_to_y_and_avg_uv_8x::<SAMPLER>(rgb_top, rgb_bot, y_top, y_bot, weights);

    let u_then_v = vcombine_s16(lo, hi);
    let u_then_v = vaddq_s16(u_then_v, vdupq_n_s16(UV_BIAS));
    let u_then_v = vreinterpret_u32_u8(vqmovn_u16(vreinterpretq_u16_s16(u_then_v)));

    vst1_lane_u32(u.cast(), u_then_v, 0);
    vst1_lane_u32(v.cast(), u_then_v, 1);
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn rgb_to_nv12_neon<const SAMPLER: usize, const DEPTH: usize, const COLORIMETRY: usize>(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_strides: (usize, usize),
    dst_buffers: &mut (&mut [u8], &mut [u8]),
) {
    const DST_DEPTH: usize = RGB_TO_YUV_WAVES;

    let (y_stride, uv_stride) = dst_strides;

    let src_group = src_buffer.as_ptr();
    let y_group = dst_buffers.0.as_mut_ptr();
    let uv_group = dst_buffers.1.as_mut_ptr();

    let src_depth = DEPTH * RGB_TO_YUV_WAVES;
    let wg_width = width / RGB_TO_YUV_WAVES;
    let wg_height = height / 2;

    for y in 0..wg_height {
        for x in 0..wg_width {
            rgb_to_nv12_8x::<SAMPLER>(
                src_group.add(wg_index(x, 2 * y, src_depth, src_stride)),
                src_group.add(wg_index(x, 2 * y + 1, src_depth, src_stride)),
                y_group.add(wg_index(x, 2 * y, DST_DEPTH, y_stride)),
                y_group.add(wg_index(x, 2 * y + 1, DST_DEPTH, y_stride)),
                uv_group.add(wg_index(x, y, DST_DEPTH, uv_stride)),
                &FORWARD_WEIGHTS[COLORIMETRY],
            );
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn rgb_to_i420_neon<const SAMPLER: usize, const DEPTH: usize, const COLORIMETRY: usize>(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_strides: (usize, usize, usize),
    dst_buffers: &mut (&mut [u8], &mut [u8], &mut [u8]),
) {
    let (y_stride, u_stride, v_stride) = dst_strides;

    let src_group = src_buffer.as_ptr();
    let y_group = dst_buffers.0.as_mut_ptr();
    let u_group = dst_buffers.1.as_mut_ptr();
    let v_group = dst_buffers.2.as_mut_ptr();

    let src_depth = DEPTH * RGB_TO_YUV_WAVES;
    let wg_width = width / RGB_TO_YUV_WAVES;
    let wg_height = height / 2;

    for y in 0..wg_height {
        for x in 0..wg_width {
            rgb_to_i420_8x::<SAMPLER>(
                src_group.add(wg_index(x, 2 * y, src_depth, src_stride)),
                src_group.add(wg_index(x, 2 * y + 1, src_depth, src_stride)),
                y_group.add(wg_index(x, 2 * y, RGB_TO_YUV_WAVES, y_stride)),
                y_group.add(wg_index(x, 2 * y + 1, RGB_TO_YUV_WAVES, y_stride)),
                u_group.add(wg_index(x, y, RGB_TO_YUV_WAVES / 2, u_stride)),
                v_group.add(wg_index(x, y, RGB_TO_YUV_WAVES / 2, v_stride)),
                &FORWARD_WEIGHTS[COLORIMETRY],
            );
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn rgb_to_i444_neon<const SAMPLER: usize, const DEPTH: usize, const COLORIMETRY: usize>(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_strides: (usize, usize, usize),
    dst_buffers: &mut (&mut [u8], &mut [u8], &mut [u8]),
) {
    let (y_stride, u_stride, v_stride) = dst_strides;

    let src_group = src_buffer.as_ptr();
    let y_group = dst_buffers.0.as_mut_ptr();
    let u_group = dst_buffers.1.as_mut_ptr();
    let v_group = dst_buffers.2.as_mut_ptr();

    let rgb_depth = DEPTH * RGB_TO_YUV_WAVES;
    let wg_width = width / RGB_TO_YUV_WAVES;

    for y in 0..height {
        for x in 0..wg_width {
            rgb_to_i444_8x::<SAMPLER>(
                src_group.add(wg_index(x, y, rgb_depth, src_stride)),
                y_group.add(wg_index(x, y, RGB_TO_YUV_WAVES, y_stride)),
                u_group.add(wg_index(x, y, RGB_TO_YUV_WAVES, u_stride)),
                v_group.add(wg_index(x, y, RGB_TO_YUV_WAVES, v_stride)),
                &FORWARD_WEIGHTS[COLORIMETRY],
            );
        }
    }
}

#[inline(never)]
fn rgb_nv12<const SAMPLER: usize, const DEPTH: usize, const COLORIMETRY: usize>(
    width: u32,
    height: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
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
        || dst_strides.len() < 2
        || dst_buffers.len() < 2
    {
        return false;
    }

    let w = width as usize;
    let h = height as usize;
    let ch = h.div_ceil(2);
    let rgb_stride = DEPTH * w;
    let uv_stride = 2 * w.div_ceil(2);

    // Compute actual strides
    let src_stride = compute_stride(src_strides[0], rgb_stride);
    let dst_strides = (
        compute_stride(dst_strides[0], w),
        compute_stride(dst_strides[1], uv_stride),
    );

    // Ensure there is sufficient data in the buffers according
    // to the image dimensions and computed strides
    let src_buffer = &src_buffers[0];
    let (y_plane, uv_plane) = dst_buffers.split_at_mut(1);
    let (y_plane, uv_plane) = (&mut *y_plane[0], &mut *uv_plane[0]);

    if out_of_bounds(src_buffer.len(), src_stride, h - 1, rgb_stride)
        || out_of_bounds(y_plane.len(), dst_strides.0, h - 1, w)
        || out_of_bounds(uv_plane.len(), dst_strides.1, ch - 1, uv_stride)
    {
        return false;
    }

    // Process vector part and scalar one
    let vector_width = if DEPTH == 3 {
        (DEPTH * RGB_TO_YUV_WAVES) * (w / (DEPTH * RGB_TO_YUV_WAVES))
    } else {
        lower_multiple_of_pot(w, RGB_TO_YUV_WAVES)
    };
    let vector_height = lower_multiple_of_pot(h, 2);
    if vector_width > 0 && vector_height > 0 {
        unsafe {
            rgb_to_nv12_neon::<SAMPLER, DEPTH, COLORIMETRY>(
                vector_width,
                vector_height,
                src_stride,
                src_buffer,
                dst_strides,
                &mut (y_plane, uv_plane),
            );
        }
    }

    let scalar_width = w - vector_width;
    if scalar_width > 0 {
        let x = vector_width;
        let sx = x * DEPTH;

        x86::rgb_to_nv12::<SAMPLER, DEPTH, COLORIMETRY>(
            scalar_width,
            h,
            src_stride,
            &src_buffer[sx..],
            dst_strides,
            &mut (&mut y_plane[x..], &mut uv_plane[x..]),
        );
    }

    let scalar_height = h - vector_height;
    if scalar_height > 0 {
        x86::rgb_to_nv12::<SAMPLER, DEPTH, COLORIMETRY>(
            w,
            scalar_height,
            src_stride,
            &src_buffer[src_stride * vector_height..],
            dst_strides,
            &mut (
                &mut y_plane[dst_strides.0 * vector_height..],
                &mut uv_plane[dst_strides.1 * (vector_height >> 1)..],
            ),
        );
    }

    true
}

#[inline(never)]
fn rgb_i420<const SAMPLER: usize, const DEPTH: usize, const COLORIMETRY: usize>(
    width: u32,
    height: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
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
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
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

    // Process vector part and scalar one
    let vector_width = if DEPTH == 3 {
        (DEPTH * RGB_TO_YUV_WAVES) * (w / (DEPTH * RGB_TO_YUV_WAVES))
    } else {
        lower_multiple_of_pot(w, RGB_TO_YUV_WAVES)
    };
    let vector_height = lower_multiple_of_pot(h, 2);
    if vector_width > 0 {
        unsafe {
            rgb_to_i420_neon::<SAMPLER, DEPTH, COLORIMETRY>(
                vector_width,
                vector_height,
                src_stride,
                src_buffer,
                dst_strides,
                &mut (y_plane, u_plane, v_plane),
            );
        }
    }

    let scalar_width = w - vector_width;
    if scalar_width > 0 {
        let x = vector_width;
        let cx = x / 2;
        let sx = x * DEPTH;

        x86::rgb_to_i420::<SAMPLER, DEPTH, COLORIMETRY>(
            scalar_width,
            h,
            src_stride,
            &src_buffer[sx..],
            dst_strides,
            &mut (&mut y_plane[x..], &mut u_plane[cx..], &mut v_plane[cx..]),
        );
    }

    let scalar_height = h - vector_height;
    if scalar_height > 0 {
        x86::rgb_to_i420::<SAMPLER, DEPTH, COLORIMETRY>(
            w,
            scalar_height,
            src_stride,
            &src_buffer[src_stride * vector_height..],
            dst_strides,
            &mut (
                &mut y_plane[dst_strides.0 * vector_height..],
                &mut u_plane[dst_strides.1 * (vector_height >> 1)..],
                &mut v_plane[dst_strides.2 * (vector_height >> 1)..],
            ),
        );
    }

    true
}

#[inline(never)]
fn rgb_i444<const SAMPLER: usize, const DEPTH: usize, const COLORIMETRY: usize>(
    width: u32,
    height: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
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

    // Process vector part and scalar one
    let vector_part = lower_multiple_of_pot(w, RGB_TO_YUV_WAVES);
    let scalar_part = w - vector_part;
    if vector_part > 0 {
        unsafe {
            rgb_to_i444_neon::<SAMPLER, DEPTH, COLORIMETRY>(
                vector_part,
                h,
                src_stride,
                src_buffer,
                dst_strides,
                &mut (y_plane, u_plane, v_plane),
            );
        }
    }

    if scalar_part > 0 {
        let x = vector_part;
        let sx = x * DEPTH;

        x86::rgb_to_i444::<SAMPLER, DEPTH, COLORIMETRY>(
            scalar_part,
            h,
            src_stride,
            &src_buffer[sx..],
            dst_strides,
            &mut (&mut y_plane[x..], &mut u_plane[x..], &mut v_plane[x..]),
        );
    }

    true
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn bgr_to_rgb_neon(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_stride: usize,
    dst_buffer: &mut [u8],
) {
    const DEPTH: usize = 3;

    let src_group = src_buffer.as_ptr();
    let dst_group = dst_buffer.as_mut_ptr();

    for i in 0..height {
        let mut src_offset = src_stride * i;
        let mut dst_offset = dst_stride * i;

        for _ in (0..width).step_by(LANE_COUNT) {
            let src = vld3q_u8(src_group.add(src_offset).cast());
            let rgb = uint8x16x3_t(src.2, src.1, src.0);
            vst3q_u8(dst_group.add(dst_offset).cast(), rgb);

            src_offset += DEPTH * LANE_COUNT;
            dst_offset += DEPTH * LANE_COUNT;
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn bgra_to_rgb_neon(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_stride: usize,
    dst_buffer: &mut [u8],
) {
    const SRC_DEPTH: usize = 4;
    const DST_DEPTH: usize = 3;

    let src_group = src_buffer.as_ptr();
    let dst_group = dst_buffer.as_mut_ptr();

    for i in 0..height {
        let mut src_offset = src_stride * i;
        let mut dst_offset = dst_stride * i;

        for _ in (0..width).step_by(LANE_COUNT) {
            let src = vld4q_u8(src_group.add(src_offset).cast());
            let rgb = uint8x16x3_t(src.2, src.1, src.0);
            vst3q_u8(dst_group.add(dst_offset).cast(), rgb);

            src_offset += SRC_DEPTH * LANE_COUNT;
            dst_offset += DST_DEPTH * LANE_COUNT;
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn rgb_to_bgra_neon(
    width: usize,
    height: usize,
    src_stride: usize,
    src_buffer: &[u8],
    dst_stride: usize,
    dst_buffer: &mut [u8],
) {
    const SRC_DEPTH: usize = 3;
    const DST_DEPTH: usize = 4;

    let src_group = src_buffer.as_ptr();
    let dst_group = dst_buffer.as_mut_ptr();

    for i in 0..height {
        let mut src_offset = src_stride * i;
        let mut dst_offset = dst_stride * i;

        for _ in (0..width).step_by(LANE_COUNT) {
            let src = vld3q_u8(src_group.add(src_offset).cast());
            let rgba = uint8x16x4_t(src.2, src.1, src.0, vdupq_n_u8(DEFAULT_ALPHA));
            vst4q_u8(dst_group.add(dst_offset).cast(), rgba);

            src_offset += SRC_DEPTH * LANE_COUNT;
            dst_offset += DST_DEPTH * LANE_COUNT;
        }
    }
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
yuv_to_rgb_fallback_converter!(I420, Bt601, Bgra);
yuv_to_rgb_fallback_converter!(I420, Bt601, Rgba);
yuv_to_rgb_fallback_converter!(I420, Bt601FR, Bgra);
yuv_to_rgb_fallback_converter!(I420, Bt601FR, Rgba);
yuv_to_rgb_fallback_converter!(I420, Bt709, Bgra);
yuv_to_rgb_fallback_converter!(I420, Bt709, Rgba);
yuv_to_rgb_fallback_converter!(I420, Bt709FR, Bgra);
yuv_to_rgb_fallback_converter!(I420, Bt709FR, Rgba);
yuv_to_rgb_fallback_converter!(I444, Bt601, Bgra);
yuv_to_rgb_fallback_converter!(I444, Bt601, Rgba);
yuv_to_rgb_fallback_converter!(I444, Bt601FR, Bgra);
yuv_to_rgb_fallback_converter!(I444, Bt601FR, Rgba);
yuv_to_rgb_fallback_converter!(I444, Bt709, Bgra);
yuv_to_rgb_fallback_converter!(I444, Bt709, Rgba);
yuv_to_rgb_fallback_converter!(I444, Bt709FR, Bgra);
yuv_to_rgb_fallback_converter!(I444, Bt709FR, Rgba);
yuv_to_rgb_fallback_converter!(Nv12, Bt601, Bgra);
yuv_to_rgb_fallback_converter!(Nv12, Bt601, Rgb);
yuv_to_rgb_fallback_converter!(Nv12, Bt601, Rgba);
yuv_to_rgb_fallback_converter!(Nv12, Bt601FR, Bgra);
yuv_to_rgb_fallback_converter!(Nv12, Bt601FR, Rgb);
yuv_to_rgb_fallback_converter!(Nv12, Bt601FR, Rgba);
yuv_to_rgb_fallback_converter!(Nv12, Bt709, Bgra);
yuv_to_rgb_fallback_converter!(Nv12, Bt709, Rgb);
yuv_to_rgb_fallback_converter!(Nv12, Bt709, Rgba);
yuv_to_rgb_fallback_converter!(Nv12, Bt709FR, Bgra);
yuv_to_rgb_fallback_converter!(Nv12, Bt709FR, Rgb);
yuv_to_rgb_fallback_converter!(Nv12, Bt709FR, Rgba);

pub fn bgr_rgb(
    width: u32,
    height: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
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

    let vector_part = lower_multiple_of_pot(w, LANE_COUNT);
    let scalar_part = w - vector_part;
    if vector_part > 0 {
        unsafe {
            bgr_to_rgb_neon(
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
        let sx = x * DEPTH;
        let dx = x * DEPTH;

        x86::bgr_to_rgb(
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

pub fn bgra_rgb(
    width: u32,
    height: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
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

    // Process vector part and scalar one
    let vector_part = lower_multiple_of_pot(w, LANE_COUNT);
    let scalar_part = w - vector_part;
    if vector_part > 0 {
        unsafe {
            bgra_to_rgb_neon(
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

        x86::bgra_to_rgb(
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

pub fn rgb_bgra(
    width: u32,
    height: u32,
    src_strides: &[usize],
    src_buffers: &[&[u8]],
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
            rgb_to_bgra_neon(
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
