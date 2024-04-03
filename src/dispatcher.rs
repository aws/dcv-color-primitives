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
#![allow(clippy::crate_in_macro_def)]

use crate::color_space::ColorSpace;
use crate::pixel_format::PixelFormat;
use crate::static_assert;

#[doc(hidden)]
#[macro_export]
macro_rules! rgb_to_yuv_converter {
    ($src_pf:ident, $dst_pf:ident, $dst_cs:ident) => {
        paste::paste! {
            pub fn [<$src_pf:lower _ $dst_pf:lower _ $dst_cs:lower>](
                width: u32,
                height: u32,
                last_src_plane: u32,
                src_strides: &[usize],
                src_buffers: &[&[u8]],
                last_dst_plane: u32,
                dst_strides: &[usize],
                dst_buffers: &mut [&mut [u8]],
            ) -> bool {
                [<rgb _ $dst_pf:lower>]::<{ Sampler::$src_pf as usize }, { crate::pixel_format::PixelFormat::depth(crate::pixel_format::PixelFormat::$src_pf) }, { Colorimetry::$dst_cs as usize }>(
                    width,
                    height,
                    last_src_plane as usize,
                    src_strides,
                    src_buffers,
                    last_dst_plane as usize,
                    dst_strides,
                    dst_buffers,
                )
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! yuv_to_rgb_converter {
    ($src_pf:ident, $src_cs:ident, $dst_pf:ident) => {
        paste::paste! {
            pub fn [<$src_pf:lower _ $src_cs:lower _ $dst_pf:lower>] (
                width: u32,
                height: u32,
                last_src_plane: u32,
                src_strides: &[usize],
                src_buffers: &[&[u8]],
                last_dst_plane: u32,
                dst_strides: &[usize],
                dst_buffers: &mut [&mut [u8]],
            ) -> bool {
                [<$src_pf:lower _rgb>]::<
                    { Colorimetry::$src_cs as usize },
                    { crate::pixel_format::PixelFormat::depth(crate::pixel_format::PixelFormat::$dst_pf) },
                    { crate::pixel_format::PixelFormat::reversed(crate::pixel_format::PixelFormat::$dst_pf) }>(
                    width,
                    height,
                    last_src_plane as usize,
                    src_strides,
                    src_buffers,
                    last_dst_plane as usize,
                    dst_strides,
                    dst_buffers,
                )
            }
        }
    };
}

#[cfg_attr(coverage_nightly, coverage(off))]
const fn enum_count(lo: u32, hi: u32) -> u32 {
    hi - lo + 1
}

#[cfg_attr(coverage_nightly, coverage(off))]
const fn upper_power_of_two(x: u32) -> u32 {
    1 << (32 - (x - 1).leading_zeros())
}

const LO_RGB_PIXEL_FORMAT: u32 = PixelFormat::Argb as u32;
const HI_RGB_PIXEL_FORMAT: u32 = PixelFormat::Rgb as u32;
const LO_YUV_PIXEL_FORMAT: u32 = PixelFormat::I444 as u32;
const HI_YUV_PIXEL_FORMAT: u32 = PixelFormat::Nv12 as u32;
static_assert!(HI_RGB_PIXEL_FORMAT == LO_YUV_PIXEL_FORMAT - 1);

const LO_RGB_COLOR_SPACE: u32 = ColorSpace::Rgb as u32;
const HI_RGB_COLOR_SPACE: u32 = ColorSpace::Rgb as u32;
const LO_YUV_COLOR_SPACE: u32 = ColorSpace::Bt601 as u32;
const HI_YUV_COLOR_SPACE: u32 = ColorSpace::Bt709FR as u32;
static_assert!(HI_RGB_COLOR_SPACE == LO_YUV_COLOR_SPACE - 1);

const RGB_PIXEL_FORMAT_COUNT: u32 = enum_count(LO_RGB_PIXEL_FORMAT, HI_RGB_PIXEL_FORMAT);
const RGB_COLOR_SPACE_COUNT: u32 = enum_count(LO_RGB_COLOR_SPACE, HI_RGB_COLOR_SPACE);
const RGB_COUNT: u32 = RGB_PIXEL_FORMAT_COUNT * RGB_COLOR_SPACE_COUNT;
const YUV_PIXEL_FORMAT_COUNT: u32 = enum_count(LO_YUV_PIXEL_FORMAT, HI_YUV_PIXEL_FORMAT);
const YUV_COLOR_SPACE_COUNT: u32 = enum_count(LO_YUV_COLOR_SPACE, HI_YUV_COLOR_SPACE);

const ROWS: u32 = RGB_COUNT + (YUV_PIXEL_FORMAT_COUNT * YUV_COLOR_SPACE_COUNT);
const COLUMNS: u32 = upper_power_of_two(ROWS);

fn select_mode(a: u32, b: u32, cond: bool) -> u32 {
    let c = u32::from(cond);
    c * a + (1 - c) * b
}

pub const TABLE_SIZE: usize = (ROWS * COLUMNS) as usize;

pub fn get_pixel_format_mode(pixel_format: u32) -> bool {
    pixel_format <= HI_RGB_PIXEL_FORMAT
}

pub fn get_color_space_mode(color_space: u32) -> bool {
    static_assert!(HI_RGB_COLOR_SPACE == LO_RGB_COLOR_SPACE);
    color_space == HI_RGB_COLOR_SPACE
}

pub fn get_image_index(pixel_format: u32, color_space: u32, pixel_format_mode: bool) -> u32 {
    select_mode(
        // This never overflows
        RGB_PIXEL_FORMAT_COUNT * (color_space - LO_RGB_COLOR_SPACE)
            + (pixel_format - LO_RGB_PIXEL_FORMAT),
        // When pixel_format > HI_RGB_PIXEL_FORMAT, this is allowed to be garbage
        // because the value above will be selected
        RGB_COUNT
            .wrapping_add(
                YUV_PIXEL_FORMAT_COUNT.wrapping_mul(color_space.wrapping_sub(LO_YUV_COLOR_SPACE)),
            )
            .wrapping_add(pixel_format.wrapping_sub(LO_YUV_PIXEL_FORMAT)),
        pixel_format_mode,
    )
}

pub fn get_index(src_index: u32, dst_index: u32) -> usize {
    (src_index * COLUMNS + dst_index) as usize
}

#[cfg(not(feature = "test_instruction_sets"))]
#[cfg_attr(coverage_nightly, coverage(off))]
#[inline(never)]
pub fn is_pixel_format_valid(pixel_format: u32) -> bool {
    pixel_format.wrapping_sub(LO_RGB_PIXEL_FORMAT)
        <= HI_YUV_PIXEL_FORMAT.wrapping_sub(LO_RGB_PIXEL_FORMAT)
}

#[cfg(not(feature = "test_instruction_sets"))]
#[cfg_attr(coverage_nightly, coverage(off))]
#[inline(never)]
pub fn is_color_space_valid(color_space: u32) -> bool {
    color_space.wrapping_sub(LO_RGB_COLOR_SPACE)
        <= HI_YUV_COLOR_SPACE.wrapping_sub(LO_RGB_COLOR_SPACE)
}
