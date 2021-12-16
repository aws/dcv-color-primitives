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
pub const MAX_NUMBER_OF_PLANES: usize = 4;

/// An enumeration of supported pixel formats.
#[derive(Copy, Clone)]
#[repr(C)]
pub enum PixelFormat {
    /// RGB with alpha channel first.
    ///
    /// 32 bits per pixel
    Argb,
    /// Reverse RGB with alpha channel last.
    ///
    /// 32 bits per pixel
    Bgra,
    /// Reverse RGB packed into 24 bits without padding.
    ///
    /// 24 bits per pixel
    Bgr,
    /// RGB with alpha channel last.
    ///
    /// 32 bits per pixel
    Rgba,
    /// RGB packed into 24 bits without padding.
    ///
    /// 24 bits per pixel
    Rgb,
    /// YUV with one luma plane Y then 2 chroma planes U and V.
    /// Chroma planes are not sub-sampled.
    ///
    /// 24 bits per pixel
    I444,
    /// YUV with one luma plane Y then 2 chroma planes U, V.
    /// Chroma planes are sub-sampled in the horizontal dimension, by a factor of 2.
    ///
    /// 16 bits per pixel
    I422,
    /// YUV with one luma plane Y then U chroma plane and last the V chroma plane.
    /// The two chroma planes are sub-sampled in both the horizontal and vertical dimensions by a factor of 2.
    ///
    /// 12 bits per pixel
    I420,
    /// YUV with one luma plane Y then one plane with U and V values interleaved.
    /// Chroma planes are subsampled in both the horizontal and vertical dimensions by a factor of 2.
    ///
    /// 12 bits per pixel
    Nv12,
}

/// If a plane stride is assigned to this constant, the plane will be assumed to contain packed data
pub const STRIDE_AUTO: usize = 0;

pub const DEFAULT_STRIDES: [usize; MAX_NUMBER_OF_PLANES] = [STRIDE_AUTO; MAX_NUMBER_OF_PLANES];

#[cfg(not(tarpaulin_include))]
const fn make_pf_spec(planes: u32, width: u32, height: u32) -> u32 {
    (height << 3) | (width << 2) | planes
}

#[cfg(not(tarpaulin_include))]
const fn make_plane_spec(plane0: u32, plane1: u32, plane2: u32, plane3: u32) -> u32 {
    (plane3 << 18) | (plane2 << 12) | (plane1 << 6) | plane0
}

const INVALID_PLANE: u32 = 32;

const PF_SPECS: [u32; 9] = [
    make_pf_spec(0, 0, 0),
    make_pf_spec(0, 0, 0),
    make_pf_spec(0, 0, 0),
    make_pf_spec(0, 0, 0),
    make_pf_spec(0, 0, 0),
    make_pf_spec(2, 0, 0),
    make_pf_spec(2, 1, 0),
    make_pf_spec(2, 1, 1),
    make_pf_spec(1, 1, 1),
];

const STRIDE_SPECS: [u32; 9] = [
    make_plane_spec(0, 0, 0, 0),
    make_plane_spec(0, 0, 0, 0),
    make_plane_spec(0, 0, 0, INVALID_PLANE),
    make_plane_spec(0, 0, 0, 0),
    make_plane_spec(0, 0, 0, INVALID_PLANE),
    make_plane_spec(0, 0, 0, INVALID_PLANE),
    make_plane_spec(0, 1, 1, INVALID_PLANE),
    make_plane_spec(0, 1, 1, INVALID_PLANE),
    make_plane_spec(0, 0, INVALID_PLANE, INVALID_PLANE),
];

const HEIGHT_SPECS: [u32; 9] = [
    make_plane_spec(0, 0, 0, 0),
    make_plane_spec(0, 0, 0, 0),
    make_plane_spec(0, 0, 0, INVALID_PLANE),
    make_plane_spec(0, 0, 0, 0),
    make_plane_spec(0, 0, 0, INVALID_PLANE),
    make_plane_spec(0, 0, 0, INVALID_PLANE),
    make_plane_spec(0, 0, 0, INVALID_PLANE),
    make_plane_spec(0, 1, 1, INVALID_PLANE),
    make_plane_spec(0, 1, INVALID_PLANE, INVALID_PLANE),
];

fn get_pf_width(pf: u32) -> u32 {
    (pf >> 2) & 1
}

fn get_pf_height(pf: u32) -> u32 {
    pf >> 3
}

fn get_pf_planes(pf: u32) -> u32 {
    pf & 3
}

fn get_plane_value(bpp: u32, plane: usize) -> u32 {
    (bpp >> (6 * plane)) & 0x3F
}

fn get_plane_mask(bpp: u32, plane: usize) -> usize {
    usize::from(INVALID_PLANE != get_plane_value(bpp, plane))
}

fn get_plane_spec(dimension: u32, bpp: u32, plane: usize) -> usize {
    (dimension.wrapping_shr(get_plane_value(bpp, plane))) as usize
}

pub fn is_compatible(pixel_format: u32, width: u32, height: u32, last_plane: u32) -> bool {
    let spec = PF_SPECS[pixel_format as usize];
    let planes = get_pf_planes(spec);
    let matches_exact = last_plane.wrapping_sub(planes);
    let last_plane = if pixel_format == (PixelFormat::Nv12 as u32) {
        last_plane
    } else {
        1
    };

    ((width & get_pf_width(spec))
        | (height & get_pf_height(spec))
        | (last_plane.wrapping_mul(matches_exact)))
        == 0
}

pub fn get_buffers_size(
    pixel_format: u32,
    width: u32,
    height: u32,
    last_plane: u32,
    strides: &[usize],
    buffers_size: &mut [usize],
) -> bool {
    let last_plane = last_plane as usize;
    if last_plane >= MAX_NUMBER_OF_PLANES
        || last_plane >= strides.len()
        || last_plane >= buffers_size.len()
    {
        return false;
    }

    let stride = &mut [0_usize; MAX_NUMBER_OF_PLANES];

    let pixel_format = pixel_format as usize;
    let stride_spec = STRIDE_SPECS[pixel_format];
    for i in 0..MAX_NUMBER_OF_PLANES {
        stride[i] = if i >= strides.len() || strides[i] == STRIDE_AUTO {
            get_plane_mask(stride_spec, i) * get_plane_spec(width, stride_spec, i)
        } else {
            strides[i]
        };
    }

    let height_spec = HEIGHT_SPECS[pixel_format];
    if last_plane == 0 {
        buffers_size[0] = ((stride[0] * get_plane_spec(height, height_spec, 0))
            + (stride[1] * get_plane_spec(height, height_spec, 1)))
            + ((stride[2] * get_plane_spec(height, height_spec, 2))
                + (stride[3] * get_plane_spec(height, height_spec, 3)));
    } else {
        let buffer_array = &mut buffers_size[..=last_plane];
        let stride_array = &stride[..=last_plane];

        for (buffer_size, (i, stride)) in
            buffer_array.iter_mut().zip(stride_array.iter().enumerate())
        {
            *buffer_size = *stride * get_plane_spec(height, height_spec, i);
        }
    }

    true
}

#[cfg(not(feature = "test_instruction_sets"))]
pub fn are_planes_compatible(pixel_format: u32, num_planes: u32) -> bool {
    let last_plane = num_planes.wrapping_sub(1);
    let spec = PF_SPECS[pixel_format as usize];
    let matches_exact = last_plane.wrapping_sub(get_pf_planes(spec));
    let last_plane = if pixel_format == (PixelFormat::Nv12 as u32) {
        last_plane
    } else {
        1
    };

    last_plane.wrapping_mul(matches_exact) == 0
}
