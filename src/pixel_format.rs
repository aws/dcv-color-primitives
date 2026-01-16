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
use core::fmt;

pub const MAX_NUMBER_OF_PLANES: usize = 4;

/// An enumeration of supported pixel formats.
#[derive(Copy, Clone, Debug)]
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

impl PixelFormat {
    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(crate) const fn depth(pixel_format: PixelFormat) -> usize {
        match pixel_format {
            PixelFormat::Argb | PixelFormat::Bgra | PixelFormat::Rgba => 4,
            PixelFormat::Bgr | PixelFormat::Rgb => 3,
            _ => 0,
        }
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub(crate) const fn reversed(pixel_format: PixelFormat) -> bool {
        matches!(pixel_format, PixelFormat::Rgba)
    }
}

impl fmt::Display for PixelFormat {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PixelFormat::Argb => write!(f, "argb"),
            PixelFormat::Bgra => write!(f, "bgra"),
            PixelFormat::Bgr => write!(f, "bgr"),
            PixelFormat::Rgba => write!(f, "rgba"),
            PixelFormat::Rgb => write!(f, "rgb"),
            PixelFormat::I444 => write!(f, "i444"),
            PixelFormat::I422 => write!(f, "i422"),
            PixelFormat::I420 => write!(f, "i420"),
            PixelFormat::Nv12 => write!(f, "nv12"),
        }
    }
}

impl TryFrom<i32> for PixelFormat {
    type Error = ();

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(PixelFormat::Argb),
            1 => Ok(PixelFormat::Bgra),
            2 => Ok(PixelFormat::Bgr),
            3 => Ok(PixelFormat::Rgba),
            4 => Ok(PixelFormat::Rgb),
            5 => Ok(PixelFormat::I444),
            6 => Ok(PixelFormat::I422),
            7 => Ok(PixelFormat::I420),
            8 => Ok(PixelFormat::Nv12),
            _ => Err(()),
        }
    }
}

/// If a plane stride is assigned to this constant, the plane will be assumed to contain packed data
pub const STRIDE_AUTO: usize = 0;

pub const DEFAULT_STRIDES: [usize; MAX_NUMBER_OF_PLANES] = [STRIDE_AUTO; MAX_NUMBER_OF_PLANES];

const PF_PLANES: [u32; 9] = [1, 1, 1, 1, 1, 3, 3, 3, 2];

pub fn is_compatible(pixel_format: u32, num_planes: u32) -> bool {
    num_planes == PF_PLANES[pixel_format as usize]
}

pub fn get_buffers_size(
    pixel_format: PixelFormat,
    width: u32,
    height: u32,
    strides: &[usize],
    buffers_size: &mut [usize],
) -> bool {
    if strides.len() > MAX_NUMBER_OF_PLANES || buffers_size.len() > MAX_NUMBER_OF_PLANES {
        return false;
    }

    let min_plane_len = match pixel_format {
        PixelFormat::Argb
        | PixelFormat::Bgra
        | PixelFormat::Rgba
        | PixelFormat::Rgb
        | PixelFormat::Bgr => 1,
        PixelFormat::I444 | PixelFormat::I422 | PixelFormat::I420 => 3,
        PixelFormat::Nv12 => 2,
    };
    if strides.len() < min_plane_len || buffers_size.len() < min_plane_len {
        return false;
    }

    let width = width as usize;
    let height = height as usize;
    let main_default_stride = match pixel_format {
        PixelFormat::Argb | PixelFormat::Bgra | PixelFormat::Rgba => 4 * width,
        PixelFormat::Rgb | PixelFormat::Bgr => 3 * width,
        _ => width,
    };
    let main_stride = if strides[0] == STRIDE_AUTO {
        main_default_stride
    } else {
        strides[0]
    };
    buffers_size[0] = main_stride * height;

    if matches!(
        pixel_format,
        PixelFormat::I444 | PixelFormat::I422 | PixelFormat::I420 | PixelFormat::Nv12
    ) {
        let width_shift = usize::from(!matches!(pixel_format, PixelFormat::I444));
        let n_components = 1 + usize::from(matches!(pixel_format, PixelFormat::Nv12));
        let chroma_default_stride = n_components * width.div_ceil(1 + width_shift);
        let stride = if strides[1] == STRIDE_AUTO {
            chroma_default_stride
        } else {
            strides[1]
        };

        let height_shift = usize::from(!matches!(
            pixel_format,
            PixelFormat::I444 | PixelFormat::I422
        ));
        buffers_size[1] = stride * height.div_ceil(1 + height_shift);

        if !matches!(pixel_format, PixelFormat::Nv12) {
            let stride = if strides[2] == STRIDE_AUTO {
                chroma_default_stride
            } else {
                strides[2]
            };

            buffers_size[2] = stride * height.div_ceil(1 + height_shift);
        }
    }

    true
}

#[cfg(not(feature = "test_instruction_sets"))]
#[inline(always)]
pub fn are_planes_compatible(pixel_format: u32, num_planes: u32) -> bool {
    num_planes == PF_PLANES[pixel_format as usize]
}
