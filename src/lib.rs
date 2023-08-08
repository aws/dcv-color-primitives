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
#![warn(missing_docs)]
#![deny(trivial_casts)]
#![deny(trivial_numeric_casts)]
#![deny(unstable_features)]
#![deny(unused_import_braces)]
#![deny(
    clippy::complexity,
    clippy::correctness,
    clippy::perf,
    clippy::style,
    clippy::pedantic
)]
#![allow(
    clippy::too_many_arguments, // API design
    clippy::missing_safety_doc, // Until we add them...
    clippy::similar_names, // This requires effort to ensure
    // Due to vzeroupper use, compiler does not inline intrinsics
    // but rather creates a function for each one that wraps the operation followed
    // by vzeroupper().
    // This is detrimental to performance
    clippy::inline_always,
    // Yield false positives
    clippy::must_use_candidate,
)]

//! DCV color primitives is a library to perform image color model conversion.
//!
//! It is able to convert the following pixel formats:
//!
//! | Source pixel format  | Destination pixel formats  |
//! | -------------------- | -------------------------- |
//! | ARGB                 | I420, I444, NV12           |
//! | BGR                  | I420, I444, NV12, RGB      |
//! | BGRA                 | I420, I444, NV12, RGB      |
//! | I420                 | BGRA, RGBA                 |
//! | I444                 | BGRA, RGBA                 |
//! | NV12                 | BGRA, RGB, RGBA            |
//! | RGB                  | BGRA                       |
//!
//! The supported color models are:
//! * ycbcr, ITU-R Recommendation BT.601 (standard video system)
//! * ycbcr, ITU-R Recommendation BT.709 (CSC systems)
//!
//! Both standard range (0-235) and full range (0-255) are supported.
//!
//! # Examples
//!
//! Convert an image from bgra to nv12 (single plane) format, with Bt601 color space:
//! ```
//! use dcv_color_primitives as dcp;
//! use dcp::{convert_image, ColorSpace, ImageFormat, PixelFormat};
//!
//! fn convert() {
//!     const WIDTH: u32 = 640;
//!     const HEIGHT: u32 = 480;
//!
//!     let src_data = Box::new([0u8; 4 * (WIDTH as usize) * (HEIGHT as usize)]);
//!     let mut dst_data = Box::new([0u8; 3 * (WIDTH as usize) * (HEIGHT as usize) / 2]);
//!
//!     let src_format = ImageFormat {
//!         pixel_format: PixelFormat::Bgra,
//!         color_space: ColorSpace::Rgb,
//!         num_planes: 1,
//!     };
//!
//!     let dst_format = ImageFormat {
//!         pixel_format: PixelFormat::Nv12,
//!         color_space: ColorSpace::Bt601,
//!         num_planes: 1,
//!     };
//!
//!     convert_image(
//!         WIDTH,
//!         HEIGHT,
//!         &src_format,
//!         None,
//!         &[&*src_data],
//!         &dst_format,
//!         None,
//!         &mut [&mut *dst_data],
//!     );
//! }
//! ```
//!
//! Handle conversion errors:
//! ```
//! use dcv_color_primitives as dcp;
//! use dcp::{convert_image, ColorSpace, ImageFormat, PixelFormat};
//! use std::error;
//!
//! fn convert() -> Result<(), Box<dyn error::Error>> {
//!     const WIDTH: u32 = 640;
//!     const HEIGHT: u32 = 480;
//!
//!     let src_data = Box::new([0u8; 4 * (WIDTH as usize) * (HEIGHT as usize)]);
//!     let mut dst_data = Box::new([0u8; 3 * (WIDTH as usize) * (HEIGHT as usize) / 2]);
//!
//!     let src_format = ImageFormat {
//!         pixel_format: PixelFormat::Bgra,
//!         color_space: ColorSpace::Bt709,
//!         num_planes: 1,
//!     };
//!
//!     let dst_format = ImageFormat {
//!         pixel_format: PixelFormat::Nv12,
//!         color_space: ColorSpace::Bt601,
//!         num_planes: 1,
//!     };
//!
//!     convert_image(
//!         WIDTH,
//!         HEIGHT,
//!         &src_format,
//!         None,
//!         &[&*src_data],
//!         &dst_format,
//!         None,
//!         &mut [&mut *dst_data],
//!     )?;
//!
//!     Ok(())
//! }
//! ```
//!
//! Compute how many bytes are needed to store and image of a given format and size:
//! ```
//! use dcv_color_primitives as dcp;
//! use dcp::{get_buffers_size, ColorSpace, ImageFormat, PixelFormat};
//! use std::error;
//!
//! fn compute_size() -> Result<(), Box<dyn error::Error>> {
//!     const WIDTH: u32 = 640;
//!     const HEIGHT: u32 = 480;
//!     const NUM_PLANES: u32 = 1;
//!
//!     let format = ImageFormat {
//!         pixel_format: PixelFormat::Bgra,
//!         color_space: ColorSpace::Rgb,
//!         num_planes: NUM_PLANES,
//!     };
//!
//!     let sizes: &mut [usize] = &mut [0usize; NUM_PLANES as usize];
//!     get_buffers_size(WIDTH, HEIGHT, &format, None, sizes)?;
//!
//!     let buffer: Vec<_> = vec![0u8; sizes[0]];
//!
//!     // Do something with buffer
//!     // --snip--
//!
//!     Ok(())
//! }
//! ```
//!
//! Provide image planes to handle data scattered in multiple buffers that are not
//! necessarily contiguous:
//! ```
//! use dcv_color_primitives as dcp;
//! use dcp::{convert_image, get_buffers_size, ColorSpace, ImageFormat, PixelFormat};
//! use std::error;
//!
//! fn convert() -> Result<(), Box<dyn error::Error>> {
//!     const WIDTH: u32 = 640;
//!     const HEIGHT: u32 = 480;
//!     const NUM_SRC_PLANES: u32 = 2;
//!     const NUM_DST_PLANES: u32 = 1;
//!
//!     let src_format = ImageFormat {
//!         pixel_format: PixelFormat::Nv12,
//!         color_space: ColorSpace::Bt709,
//!         num_planes: NUM_SRC_PLANES,
//!     };
//!
//!     let src_sizes: &mut [usize] = &mut [0usize; NUM_SRC_PLANES as usize];
//!     get_buffers_size(WIDTH, HEIGHT, &src_format, None, src_sizes)?;
//!
//!     let src_y: Vec<_> = vec![0u8; src_sizes[0]];
//!     let src_uv: Vec<_> = vec![0u8; src_sizes[1]];
//!
//!     let dst_format = ImageFormat {
//!         pixel_format: PixelFormat::Bgra,
//!         color_space: ColorSpace::Rgb,
//!         num_planes: NUM_DST_PLANES,
//!     };
//!
//!     let dst_sizes: &mut [usize] = &mut [0usize; NUM_DST_PLANES as usize];
//!     get_buffers_size(WIDTH, HEIGHT, &dst_format, None, dst_sizes)?;
//!
//!     let mut dst_rgba: Vec<_> = vec![0u8; dst_sizes[0]];
//!
//!     convert_image(
//!         WIDTH,
//!         HEIGHT,
//!         &src_format,
//!         None,
//!         &[&src_y[..], &src_uv[..]],
//!         &dst_format,
//!         None,
//!         &mut [&mut dst_rgba[..]],
//!     )?;
//!
//!     Ok(())
//! }
//! ```
//!
//! Provide image strides to convert data which is not tightly packed:
//! ```
//! use dcv_color_primitives as dcp;
//! use dcp::{convert_image, get_buffers_size, ColorSpace, ImageFormat, PixelFormat};
//! use std::error;
//!
//! fn convert() -> Result<(), Box<dyn error::Error>> {
//!     const WIDTH: u32 = 640;
//!     const HEIGHT: u32 = 480;
//!     const NUM_SRC_PLANES: u32 = 1;
//!     const NUM_DST_PLANES: u32 = 2;
//!     const RGB_STRIDE: usize = 4 * (((3 * (WIDTH as usize)) + 3) / 4);
//!
//!     let src_format = ImageFormat {
//!         pixel_format: PixelFormat::Bgr,
//!         color_space: ColorSpace::Rgb,
//!         num_planes: NUM_SRC_PLANES,
//!     };
//!
//!     let src_strides: &[usize] = &[RGB_STRIDE];
//!
//!     let src_sizes: &mut [usize] = &mut [0usize; NUM_SRC_PLANES as usize];
//!     get_buffers_size(WIDTH, HEIGHT, &src_format, Some(src_strides), src_sizes)?;
//!
//!     let src_rgba: Vec<_> = vec![0u8; src_sizes[0]];
//!
//!     let dst_format = ImageFormat {
//!         pixel_format: PixelFormat::Nv12,
//!         color_space: ColorSpace::Bt709,
//!         num_planes: NUM_DST_PLANES,
//!     };
//!
//!     let dst_sizes: &mut [usize] = &mut [0usize; NUM_DST_PLANES as usize];
//!     get_buffers_size(WIDTH, HEIGHT, &dst_format, None, dst_sizes)?;
//!
//!     let mut dst_y: Vec<_> = vec![0u8; dst_sizes[0]];
//!     let mut dst_uv: Vec<_> = vec![0u8; dst_sizes[1]];
//!
//!     convert_image(
//!         WIDTH,
//!         HEIGHT,
//!         &src_format,
//!         Some(src_strides),
//!         &[&src_rgba[..]],
//!         &dst_format,
//!         None,
//!         &mut [&mut dst_y[..], &mut dst_uv[..]],
//!     )?;
//!
//!     Ok(())
//! }
//! ```
mod color_space;
mod convert_image;
mod cpu_info;
mod dispatcher;
mod pixel_format;
mod static_assert;

use cpu_info::{CpuManufacturer, InstructionSet};
use paste::paste;
use std::error;
use std::fmt;
#[cfg(feature = "test_instruction_sets")]
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::OnceLock;

pub use color_space::ColorSpace;
pub use pixel_format::{PixelFormat, STRIDE_AUTO};

/// An enumeration of errors.
#[derive(Debug)]
#[repr(C)]
pub enum ErrorKind {
    /// One or more parameters have invalid values for the called function
    InvalidValue,
    /// The combination of parameters is unsupported for the called function
    InvalidOperation,
    /// Not enough data was provided to the called function. Typically, provided
    /// arrays are not correctly sized
    NotEnoughData,
}

#[cfg(not(tarpaulin_include))]
impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ErrorKind::InvalidValue => write!(
                f,
                "One or more parameters have not legal values for the command"
            ),
            ErrorKind::InvalidOperation => write!(
                f,
                "The combination of parameters is not legal for the command"
            ),
            ErrorKind::NotEnoughData => write!(f, "Not enough data provided"),
        }
    }
}

#[cfg(not(tarpaulin_include))]
impl error::Error for ErrorKind {
    fn cause(&self) -> Option<&dyn error::Error> {
        None
    }
}

/// Describes how the image data is laid out in memory and its color space.
///
/// # Note
/// Not all combinations of pixel format, color space and number of planes
/// describe a valid image format.
///
/// Each pixel format has one or more compatible color spaces:
///
/// pixel format        | color space
/// --------------------|--------------------------------------
/// `PixelFormat::Argb` | `ColorSpace::Rgb`
/// `PixelFormat::Bgra` | `ColorSpace::Rgb`
/// `PixelFormat::Bgr`  | `ColorSpace::Rgb`
/// `PixelFormat::Rgba` | `ColorSpace::Rgb`
/// `PixelFormat::Rgb`  | `ColorSpace::Rgb`
/// `PixelFormat::I444` | `ColorSpace::Bt601(FR)`, `ColorSpace::Bt709(FR)`
/// `PixelFormat::I422` | `ColorSpace::Bt601(FR)`, `ColorSpace::Bt709(FR)`
/// `PixelFormat::I420` | `ColorSpace::Bt601(FR)`, `ColorSpace::Bt709(FR)`
/// `PixelFormat::Nv12` | `ColorSpace::Bt601(FR)`, `ColorSpace::Bt709(FR)`
///
/// Some pixel formats might impose additional restrictions on the accepted number of
/// planes and the image size:
///
/// pixel format        | subsampling | w   | h   | #planes | #1     | #2     | #3
/// --------------------|:-----------:|:---:|:---:|:-------:|:------:|:------:|:-------:
/// `PixelFormat::Argb` | 4:4:4       |     |     | 1       | argb:4 |        |
/// `PixelFormat::Bgra` | 4:4:4       |     |     | 1       | bgra:4 |        |
/// `PixelFormat::Bgr`  | 4:4:4       |     |     | 1       | bgr:3  |        |
/// `PixelFormat::Rgba` | 4:4:4       |     |     | 1       | rgba:4 |        |
/// `PixelFormat::Rgb`  | 4:4:4       |     |     | 1       | rgb:3  |        |
/// `PixelFormat::I444` | 4:4:4       |     |     | 3       | y:1    | u:1    | v:1
/// `PixelFormat::I422` | 4:2:2       |  2  |     | 1, 3    | y:1    | u:1/2  | v:1/2
/// `PixelFormat::I420` | 4:2:0       |  2  |  2  | 3       | y:1    | u:1/4  | v:1/4
/// `PixelFormat::Nv12` | 4:2:0       |  2  |  2  | 1, 2    | y:1    | uv:1/2 |
///
/// The values reported in columns `w` and `h`, when specified, indicate that the described
/// image should have width and height that are multiples of the specified values
#[repr(C)]
pub struct ImageFormat {
    /// Pixel format
    pub pixel_format: PixelFormat,
    /// Color space
    pub color_space: ColorSpace,
    /// Number of planes
    pub num_planes: u32,
}

type ConvertDispatcher =
    fn(u32, u32, u32, &[usize], &[&[u8]], u32, &[usize], &mut [&mut [u8]]) -> bool;

macro_rules! rgb_to_yuv {
    ($conv:expr, $set:ident, $src_pf:ident, $dst_pf:ident, $dst_cs:ident) => {
        paste! {
            $conv[dispatcher::get_index(
                dispatcher::get_image_index(
                    PixelFormat::$src_pf as u32,
                    ColorSpace::Rgb as u32,
                    dispatcher::get_pixel_format_mode(PixelFormat::$src_pf as u32),
                ),
                dispatcher::get_image_index(
                    PixelFormat::$dst_pf as u32,
                    ColorSpace::$dst_cs as u32,
                    dispatcher::get_pixel_format_mode(PixelFormat::$dst_pf as u32),
                ),
            )] = Some(convert_image::$set::[<$src_pf:lower _ $dst_pf:lower _ $dst_cs:lower>])
        }
    };
}

macro_rules! yuv_to_rgb {
    ($conv:expr, $set:ident, $src_pf:ident, $src_cs:ident, $dst_pf:ident) => {
        paste! {
            $conv[dispatcher::get_index(
                dispatcher::get_image_index(
                    PixelFormat::$src_pf as u32,
                    ColorSpace::$src_cs as u32,
                    dispatcher::get_pixel_format_mode(PixelFormat::$src_pf as u32),
                ),
                dispatcher::get_image_index(
                    PixelFormat::$dst_pf as u32,
                    ColorSpace::Rgb as u32,
                    dispatcher::get_pixel_format_mode(PixelFormat::$dst_pf as u32),
                ),
            )] = Some(convert_image::$set::[<$src_pf:lower _ $src_cs:lower _ $dst_pf:lower>])
        }
    };
}

macro_rules! rgb_to_rgb {
    ($conv:expr, $set:ident, $src_pf:ident, $dst_pf:ident) => {
        paste! {
            $conv[dispatcher::get_index(
                dispatcher::get_image_index(
                    PixelFormat::$src_pf as u32,
                    ColorSpace::Rgb as u32,
                    dispatcher::get_pixel_format_mode(PixelFormat::$src_pf as u32),
                ),
                dispatcher::get_image_index(
                    PixelFormat::$dst_pf as u32,
                    ColorSpace::Rgb as u32,
                    dispatcher::get_pixel_format_mode(PixelFormat::$dst_pf as u32),
                ),
            )] = Some(convert_image::$set::[<$src_pf:lower _ $dst_pf:lower>])
        }
    };
}

macro_rules! set_dispatch_table {
    ($conv:expr, $set:ident) => {
        rgb_to_rgb!($conv, $set, Bgr, Rgb);
        rgb_to_rgb!($conv, $set, Bgra, Rgb);
        rgb_to_rgb!($conv, $set, Rgb, Bgra);
        rgb_to_yuv!($conv, $set, Argb, I420, Bt601);
        rgb_to_yuv!($conv, $set, Argb, I420, Bt601FR);
        rgb_to_yuv!($conv, $set, Argb, I420, Bt709);
        rgb_to_yuv!($conv, $set, Argb, I420, Bt709FR);
        rgb_to_yuv!($conv, $set, Argb, I444, Bt601);
        rgb_to_yuv!($conv, $set, Argb, I444, Bt601FR);
        rgb_to_yuv!($conv, $set, Argb, I444, Bt709);
        rgb_to_yuv!($conv, $set, Argb, I444, Bt709FR);
        rgb_to_yuv!($conv, $set, Argb, Nv12, Bt601);
        rgb_to_yuv!($conv, $set, Argb, Nv12, Bt601FR);
        rgb_to_yuv!($conv, $set, Argb, Nv12, Bt709);
        rgb_to_yuv!($conv, $set, Argb, Nv12, Bt709FR);
        rgb_to_yuv!($conv, $set, Bgr, I420, Bt601);
        rgb_to_yuv!($conv, $set, Bgr, I420, Bt601FR);
        rgb_to_yuv!($conv, $set, Bgr, I420, Bt709);
        rgb_to_yuv!($conv, $set, Bgr, I420, Bt709FR);
        rgb_to_yuv!($conv, $set, Bgr, I444, Bt601);
        rgb_to_yuv!($conv, $set, Bgr, I444, Bt601FR);
        rgb_to_yuv!($conv, $set, Bgr, I444, Bt709);
        rgb_to_yuv!($conv, $set, Bgr, I444, Bt709FR);
        rgb_to_yuv!($conv, $set, Bgr, Nv12, Bt601);
        rgb_to_yuv!($conv, $set, Bgr, Nv12, Bt601FR);
        rgb_to_yuv!($conv, $set, Bgr, Nv12, Bt709);
        rgb_to_yuv!($conv, $set, Bgr, Nv12, Bt709FR);
        rgb_to_yuv!($conv, $set, Bgra, I420, Bt601);
        rgb_to_yuv!($conv, $set, Bgra, I420, Bt601FR);
        rgb_to_yuv!($conv, $set, Bgra, I420, Bt709);
        rgb_to_yuv!($conv, $set, Bgra, I420, Bt709FR);
        rgb_to_yuv!($conv, $set, Bgra, I444, Bt601);
        rgb_to_yuv!($conv, $set, Bgra, I444, Bt601FR);
        rgb_to_yuv!($conv, $set, Bgra, I444, Bt709);
        rgb_to_yuv!($conv, $set, Bgra, I444, Bt709FR);
        rgb_to_yuv!($conv, $set, Bgra, Nv12, Bt601);
        rgb_to_yuv!($conv, $set, Bgra, Nv12, Bt601FR);
        rgb_to_yuv!($conv, $set, Bgra, Nv12, Bt709);
        rgb_to_yuv!($conv, $set, Bgra, Nv12, Bt709FR);
        yuv_to_rgb!($conv, $set, I420, Bt601, Bgra);
        yuv_to_rgb!($conv, $set, I420, Bt601, Rgba);
        yuv_to_rgb!($conv, $set, I420, Bt601FR, Bgra);
        yuv_to_rgb!($conv, $set, I420, Bt601FR, Rgba);
        yuv_to_rgb!($conv, $set, I420, Bt709, Bgra);
        yuv_to_rgb!($conv, $set, I420, Bt709, Rgba);
        yuv_to_rgb!($conv, $set, I420, Bt709FR, Bgra);
        yuv_to_rgb!($conv, $set, I420, Bt709FR, Rgba);
        yuv_to_rgb!($conv, $set, I444, Bt601, Bgra);
        yuv_to_rgb!($conv, $set, I444, Bt601, Rgba);
        yuv_to_rgb!($conv, $set, I444, Bt601FR, Bgra);
        yuv_to_rgb!($conv, $set, I444, Bt601FR, Rgba);
        yuv_to_rgb!($conv, $set, I444, Bt709, Bgra);
        yuv_to_rgb!($conv, $set, I444, Bt709, Rgba);
        yuv_to_rgb!($conv, $set, I444, Bt709FR, Bgra);
        yuv_to_rgb!($conv, $set, I444, Bt709FR, Rgba);
        yuv_to_rgb!($conv, $set, Nv12, Bt601, Bgra);
        yuv_to_rgb!($conv, $set, Nv12, Bt601, Rgb);
        yuv_to_rgb!($conv, $set, Nv12, Bt601, Rgba);
        yuv_to_rgb!($conv, $set, Nv12, Bt601FR, Bgra);
        yuv_to_rgb!($conv, $set, Nv12, Bt601FR, Rgb);
        yuv_to_rgb!($conv, $set, Nv12, Bt601FR, Rgba);
        yuv_to_rgb!($conv, $set, Nv12, Bt709, Bgra);
        yuv_to_rgb!($conv, $set, Nv12, Bt709, Rgb);
        yuv_to_rgb!($conv, $set, Nv12, Bt709, Rgba);
        yuv_to_rgb!($conv, $set, Nv12, Bt709FR, Bgra);
        yuv_to_rgb!($conv, $set, Nv12, Bt709FR, Rgb);
        yuv_to_rgb!($conv, $set, Nv12, Bt709FR, Rgba);
    };
}

#[cfg(feature = "test_instruction_sets")]
static TEST_SET: AtomicI32 = AtomicI32::new(-1);

type DispatchTable = [Option<ConvertDispatcher>; dispatcher::TABLE_SIZE];

struct Context {
    manufacturer: CpuManufacturer,
    set: InstructionSet,
    converters: DispatchTable,
    #[cfg(feature = "test_instruction_sets")]
    test_converters: [Option<DispatchTable>; 3],
}

impl Context {
    pub fn global() -> &'static Context {
        static INSTANCE: OnceLock<Context> = OnceLock::new();
        INSTANCE.get_or_init(Context::new)
    }

    pub fn new() -> Self {
        let (manufacturer, set) = cpu_info::get();
        let mut context = Context {
            manufacturer,
            set,
            converters: [None; dispatcher::TABLE_SIZE],
            #[cfg(feature = "test_instruction_sets")]
            test_converters: [None; 3],
        };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        match context.set {
            InstructionSet::X86 => {
                set_dispatch_table!(context.converters, x86);
            }
            InstructionSet::Sse2 => {
                set_dispatch_table!(context.converters, sse2);
                #[cfg(feature = "test_instruction_sets")]
                {
                    let mut table: DispatchTable = [None; dispatcher::TABLE_SIZE];
                    set_dispatch_table!(table, x86);
                    context.test_converters[0] = Some(table);
                }
            }
            InstructionSet::Avx2 => {
                set_dispatch_table!(context.converters, avx2);

                #[cfg(feature = "test_instruction_sets")]
                {
                    let mut table: DispatchTable = [None; dispatcher::TABLE_SIZE];
                    set_dispatch_table!(table, sse2);
                    context.test_converters[1] = Some(table);

                    let mut table: DispatchTable = [None; dispatcher::TABLE_SIZE];
                    set_dispatch_table!(table, x86);
                    context.test_converters[0] = Some(table);
                }
            }
        }

        // This is the default for arm and wasm32 targets
        #[cfg(all(not(target_arch = "x86"), not(target_arch = "x86_64")))]
        {
            set_dispatch_table!(context.converters, x86);
        }

        context
    }
}

/// Returns a description of the algorithms that are best for the running cpu and
/// available instruction sets
///
/// # Examples
/// ```
/// use dcv_color_primitives as dcp;
/// println!("{}", dcp::describe_acceleration());
/// // => {cpu-manufacturer:Intel,instruction-set:Avx2}
pub fn describe_acceleration() -> String {
    let state = Context::global();

    format!(
        "{{cpu-manufacturer:{:?},instruction-set:{:?}}}",
        state.manufacturer, state.set
    )
}

/// Compute number of bytes required to store an image given its format, dimensions
/// and optionally its strides
///
/// # Arguments
/// * `width` - Width of the image in pixels
/// * `height` - Height of the image in pixels
/// * `format` - Image format
/// * `strides` - An array of distances in bytes between starts of consecutive lines
///               in each image planes
/// * `buffers_size` - An array describing the minimum number of bytes required in each
///                    image planes
///
/// # Examples
/// Compute how many bytes are needed to store and image of a given format and size
/// assuming *all planes contain data which is tightly packed*:
/// ```
/// use dcv_color_primitives as dcp;
/// use dcp::{get_buffers_size, ColorSpace, ImageFormat, PixelFormat};
/// use std::error;
///
/// fn compute_size_packed() -> Result<(), Box<dyn error::Error>> {
///     const WIDTH: u32 = 640;
///     const HEIGHT: u32 = 480;
///     const NUM_PLANES: u32 = 2;
///
///     let format = ImageFormat {
///         pixel_format: PixelFormat::Nv12,
///         color_space: ColorSpace::Bt601,
///         num_planes: NUM_PLANES,
///     };
///
///     let sizes: &mut [usize] = &mut [0usize; NUM_PLANES as usize];
///     get_buffers_size(WIDTH, HEIGHT, &format, None, sizes)?;
///
///     Ok(())
/// }
/// ```
///
/// Compute how many bytes are needed to store and image of a given format and size
/// in which *all planes have custom strides*:
/// ```
/// use dcv_color_primitives as dcp;
/// use dcp::{get_buffers_size, ColorSpace, ImageFormat, PixelFormat};
/// use std::error;
///
/// fn compute_size_custom_strides() -> Result<(), Box<dyn error::Error>> {
///     const WIDTH: u32 = 640;
///     const HEIGHT: u32 = 480;
///     const NUM_PLANES: u32 = 2;
///     const Y_STRIDE: usize = (WIDTH as usize) + 1;
///     const UV_STRIDE: usize = (WIDTH as usize) + 3;
///
///     let format = ImageFormat {
///         pixel_format: PixelFormat::Nv12,
///         color_space: ColorSpace::Bt601,
///         num_planes: NUM_PLANES,
///     };
///
///     let strides: &[usize] = &[ Y_STRIDE, UV_STRIDE, ];
///     let sizes: &mut [usize] = &mut [0usize; NUM_PLANES as usize];
///     get_buffers_size(WIDTH, HEIGHT, &format, Some(strides), sizes)?;
///
///     Ok(())
/// }
/// ```
///
/// Compute how many bytes are needed to store and image of a given format and size
/// in which *some planes have custom strides*, while *some other are assumed to
/// contain data which is tightly packed*:
/// ```
/// use dcv_color_primitives as dcp;
/// use dcp::{get_buffers_size, ColorSpace, ImageFormat, PixelFormat, STRIDE_AUTO};
/// use std::error;
///
/// fn compute_size_custom_strides() -> Result<(), Box<dyn error::Error>> {
///     const WIDTH: u32 = 640;
///     const HEIGHT: u32 = 480;
///     const NUM_PLANES: u32 = 2;
///     const Y_STRIDE: usize = (WIDTH as usize) + 1;
///
///     let format = ImageFormat {
///         pixel_format: PixelFormat::Nv12,
///         color_space: ColorSpace::Bt601,
///         num_planes: NUM_PLANES,
///     };
///
///     let strides: &[usize] = &[ Y_STRIDE, STRIDE_AUTO, ];
///     let sizes: &mut [usize] = &mut [0usize; NUM_PLANES as usize];
///     get_buffers_size(WIDTH, HEIGHT, &format, Some(strides), sizes)?;
///
///     Ok(())
/// }
/// ```
///
/// Default strides (e.g. the one you would set for tightly packed data) can be set
/// using the constant [`STRIDE_AUTO`]
///
/// # Errors
///
/// * [`InvalidValue`] if `width` or `height` violate the [`size constraints`] that might by
///   imposed by the image pixel format
///
/// * [`InvalidValue`] if the image format has a number of planes which is not compatible
///   with its pixel format
///
/// * [`NotEnoughData`] if the strides array is not `None` and its length is less than the
///   image format number of planes
///
/// * [`NotEnoughData`] if the buffers size array is not `None` and its length is less than the
///   image format number of planes
///
/// [`InvalidValue`]: ./enum.ErrorKind.html#variant.InvalidValue
/// [`NotEnoughData`]: ./enum.ErrorKind.html#variant.NotEnoughData
/// [`size constraints`]: ./struct.ImageFormat.html#note
/// [`STRIDE_AUTO`]: ./constant.STRIDE_AUTO.html
pub fn get_buffers_size(
    width: u32,
    height: u32,
    format: &ImageFormat,
    strides: Option<&[usize]>,
    buffers_size: &mut [usize],
) -> Result<(), ErrorKind> {
    let pixel_format = format.pixel_format as u32;
    let last_plane = format.num_planes.wrapping_sub(1);
    if !pixel_format::is_compatible(pixel_format, width, height, last_plane) {
        return Err(ErrorKind::InvalidValue);
    }

    if pixel_format::get_buffers_size(
        pixel_format,
        width,
        height,
        last_plane,
        strides.unwrap_or(&pixel_format::DEFAULT_STRIDES),
        buffers_size,
    ) {
        Ok(())
    } else {
        Err(ErrorKind::NotEnoughData)
    }
}

/// Converts from a color space to another one, applying downsampling/upsampling
/// to match destination image format.
///
/// # Arguments
/// * `width` - Width of the image to convert in pixels
/// * `height` - Height of the image to convert in pixels
/// * `src_format` - Source image format
/// * `src_strides` - An array of distances in bytes between starts of consecutive lines
///                   in each source image planes
/// * `src_buffers` - An array of image buffers in each source color plane
/// * `dst_format` - Destination image format
/// * `dst_strides` - An array of distances in bytes between starts of consecutive lines
///                   in each destination image planes
/// * `dst_buffers` - An array of image buffers in each destination color plane
///
/// # Errors
///
/// * [`InvalidValue`] if `width` or `height` violate the [`size constraints`]
///   that might by imposed by the source and destination image pixel formats
///
/// * [`InvalidValue`] if source or destination image formats have a number of planes
///   which is not compatible with their pixel formats
///
/// * [`InvalidOperation`] if there is no available method to convert the image with the
///   source pixel format to the image with the destination pixel format.
///
///   The list of available conversions is specified here:
///
///   Source image pixel format       | Supported destination image pixel formats
///   --------------------------------|------------------------------------------
///   `PixelFormat::Argb`             | `PixelFormat::I420` [`1`]
///   `PixelFormat::Argb`             | `PixelFormat::I444` [`1`]
///   `PixelFormat::Argb`             | `PixelFormat::Nv12` [`1`]
///   `PixelFormat::Bgra`             | `PixelFormat::I420` [`1`]
///   `PixelFormat::Bgra`             | `PixelFormat::I444` [`1`]
///   `PixelFormat::Bgra`             | `PixelFormat::Nv12` [`1`]
///   `PixelFormat::Bgra`             | `PixelFormat::Rgb`  [`4`]
///   `PixelFormat::Bgr`              | `PixelFormat::I420` [`1`]
///   `PixelFormat::Bgr`              | `PixelFormat::I444` [`1`]
///   `PixelFormat::Bgr`              | `PixelFormat::Nv12` [`1`]
///   `PixelFormat::Bgr`              | `PixelFormat::Rgb`  [`5`]
///   `PixelFormat::I420`             | `PixelFormat::Bgra`, `PixelFormat::Rgba` [`2`]
///   `PixelFormat::I444`             | `PixelFormat::Bgra`, `PixelFormat::Rgba` [`2`]
///   `PixelFormat::Nv12`             | `PixelFormat::Bgra`, `PixelFormat::Rgb`, `PixelFormat::Rgba` [`2`]
///   `PixelFormat::Rgb`              | `PixelFormat::Bgra` [`3`]
///
/// * [`NotEnoughData`] if the source stride array is not `None` and its length is less than the
///   source image format number of planes
///
/// * [`NotEnoughData`] if the destination stride array is not `None` and its length is less than the
///   destination image format number of planes
///
/// * [`NotEnoughData`] if one or more source/destination buffers does not provide enough data.
///
///   The minimum number of bytes to provide for each buffer depends from the image format, dimensions,
///   and strides (if they are not `None`).
///
///   You can compute the buffers' size using [`get_buffers_size`]
///
/// # Algorithm 1
/// Conversion from linear RGB model to ycbcr color model, with 4:2:0 downsampling
///
/// If the destination image color space is Bt601, the following formula is applied:
/// ```text
/// y  =  0.257 * r + 0.504 * g + 0.098 * b + 16
/// cb = -0.148 * r - 0.291 * g + 0.439 * b + 128
/// cr =  0.439 * r - 0.368 * g - 0.071 * b + 128
/// ```
///
/// If the destination image color space is Bt709, the following formula is applied:
/// ```text
/// y  =  0.213 * r + 0.715 * g + 0.072 * b + 16
/// cb = -0.117 * r - 0.394 * g + 0.511 * b + 128
/// cr =  0.511 * r - 0.464 * g - 0.047 * b + 128
///
/// If the destination image color space is Bt601FR, the following formula is applied:
/// ```text
/// y  =  0.299 * r + 0.587 * g + 0.114 * b
/// cb = -0.169 * r - 0.331 * g + 0.500 * b + 128
/// cr =  0.500 * r - 0.419 * g - 0.081 * b + 128
///
/// If the destination image color space is Bt709FR, the following formula is applied:
/// ```text
/// y  =  0.213 * r + 0.715 * g + 0.072 * b
/// cb = -0.115 * r - 0.385 * g + 0.500 * b + 128
/// cr =  0.500 * r - 0.454 * g - 0.046 * b + 128
/// ```
///
/// # Algorithm 2
/// Conversion from ycbcr model to linear RGB model, with 4:4:4 upsampling
///
/// If the destination image contains an alpha channel, each component will be set to 255
///
/// If the source image color space is Bt601, the following formula is applied:
/// ```text
/// r = 1.164 * (y - 16) + 1.596 * (cr - 128)
/// g = 1.164 * (y - 16) - 0.813 * (cr - 128) - 0.392 * (cb - 128)
/// b = 1.164 * (y - 16) + 2.017 * (cb - 128)
/// ```
///
/// If the source image color space is Bt709, the following formula is applied:
/// ```text
/// r = 1.164 * (y - 16) + 1.793 * (cr - 128)
/// g = 1.164 * (y - 16) - 0.534 * (cr - 128) - 0.213 * (cb - 128)
/// b = 1.164 * (y - 16) + 2.115 * (cb - 128)
///
/// If the source image color space is Bt601FR, the following formula is applied:
/// ```text
/// r = y + 1.402 * (cr - 128)
/// g = y - 0.714 * (cr - 128) - 0.344 * (cb - 128)
/// b = y + 1.772 * (cb - 128)
///
/// If the source image color space is Bt709FR, the following formula is applied:
/// ```text
/// r = y + 1.575 * (cr - 128)
/// g = y - 0.468 * (cr - 128) - 0.187 * (cb - 128)
/// b = y + 1.856 * (cb - 128)
/// ```
///
/// # Algorithm 3
/// Conversion from RGB to BGRA
///
/// # Algorithm 4
/// Conversion from BGRA to RGB
///
/// # Algorithm 5
/// Conversion from BGR to RGB
///
/// [`InvalidValue`]: ./enum.ErrorKind.html#variant.InvalidValue
/// [`InvalidOperation`]: ./enum.ErrorKind.html#variant.InvalidOperation
/// [`NotEnoughData`]: ./enum.ErrorKind.html#variant.NotEnoughData
/// [`size constraints`]: ./struct.ImageFormat.html#note
/// [`get_buffers_size`]: ./fn.get_buffers_size.html
/// [`1`]: ./fn.convert_image.html#algorithm-1
/// [`2`]: ./fn.convert_image.html#algorithm-2
/// [`3`]: ./fn.convert_image.html#algorithm-3
pub fn convert_image(
    width: u32,
    height: u32,
    src_format: &ImageFormat,
    src_strides: Option<&[usize]>,
    src_buffers: &[&[u8]],
    dst_format: &ImageFormat,
    dst_strides: Option<&[usize]>,
    dst_buffers: &mut [&mut [u8]],
) -> Result<(), ErrorKind> {
    let src_pixel_format = src_format.pixel_format as u32;
    let dst_pixel_format = dst_format.pixel_format as u32;
    let src_color_space = src_format.color_space as u32;
    let dst_color_space = dst_format.color_space as u32;

    // Cross-correlate pixel format with color space. Predicate handles Table 1
    let src_pf_mode = dispatcher::get_pixel_format_mode(src_pixel_format);
    let src_cs_mode = dispatcher::get_color_space_mode(src_color_space);
    let dst_pf_mode = dispatcher::get_pixel_format_mode(dst_pixel_format);
    let dst_cs_mode = dispatcher::get_color_space_mode(dst_color_space);
    let src_pf_cs_mismatch = src_pf_mode ^ src_cs_mode;
    let dst_pf_cs_mismatch = dst_pf_mode ^ dst_cs_mode;
    if src_pf_cs_mismatch | dst_pf_cs_mismatch {
        return Err(ErrorKind::InvalidValue);
    }

    // Cross-correlate pixel format with planes and alignment.
    // wrapping_sub is wanted. If num_planes is 0, this turns in a very big number that
    // still represents an invalid number of planes.
    let last_src_plane = src_format.num_planes.wrapping_sub(1);
    if !pixel_format::is_compatible(src_pixel_format, width, height, last_src_plane) {
        return Err(ErrorKind::InvalidValue);
    }

    let last_dst_plane = dst_format.num_planes.wrapping_sub(1);
    if !pixel_format::is_compatible(dst_pixel_format, width, height, last_dst_plane) {
        return Err(ErrorKind::InvalidValue);
    }

    // Cross-correlate modes.
    let src_index = dispatcher::get_image_index(src_pixel_format, src_color_space, src_pf_mode);
    let dst_index = dispatcher::get_image_index(dst_pixel_format, dst_color_space, dst_pf_mode);
    let index = dispatcher::get_index(src_index, dst_index);
    let converters = Context::global().converters;

    #[cfg(feature = "test_instruction_sets")]
    let converters = {
        let test_converters = Context::global().test_converters;
        #[allow(clippy::cast_sign_loss)]
        // Checked: we want the invalid value '-1' to be mapped outside the valid range
        test_converters
            .get(TEST_SET.load(Ordering::SeqCst) as usize)
            .map(|&x| x)
            .flatten()
            .unwrap_or(converters)
    };

    if index >= converters.len() {
        return Err(ErrorKind::InvalidOperation);
    }

    let converter = converters[index];
    match converter {
        None => Err(ErrorKind::InvalidOperation),
        Some(image_converter) => {
            if image_converter(
                width,
                height,
                last_src_plane,
                src_strides.unwrap_or(&pixel_format::DEFAULT_STRIDES),
                src_buffers,
                last_dst_plane,
                dst_strides.unwrap_or(&pixel_format::DEFAULT_STRIDES),
                dst_buffers,
            ) {
                Ok(())
            } else {
                Err(ErrorKind::NotEnoughData)
            }
        }
    }
}

/// This is for internal use only
#[cfg(feature = "test_instruction_sets")]
pub fn initialize_with_instruction_set(instruction_set: &str) {
    match instruction_set {
        "x86" => TEST_SET.store(0, Ordering::SeqCst),
        "sse2" => TEST_SET.store(1, Ordering::SeqCst),
        _ => TEST_SET.store(2, Ordering::SeqCst),
    };
}

#[doc(hidden)]
#[cfg(not(tarpaulin_include))]
#[cfg(not(feature = "test_instruction_sets"))]
pub mod c_api {
    #![allow(clippy::wildcard_imports)]
    use super::*; // We are importing everything
    use pixel_format::{are_planes_compatible, MAX_NUMBER_OF_PLANES};
    use std::cmp;
    use std::ffi::CString;
    use std::mem::{transmute, MaybeUninit};
    use std::os::raw::c_char;
    use std::ptr;
    use std::slice;

    const UNBOUNDED_C_ARRAY: usize = std::isize::MAX as usize;

    #[repr(C)]
    pub enum Result {
        Ok,
        Err,
    }

    unsafe fn set_error(error: *mut ErrorKind, value: ErrorKind) -> self::Result {
        if !error.is_null() {
            *error = value;
        }

        self::Result::Err
    }

    #[no_mangle]
    pub extern "C" fn dcp_describe_acceleration() -> *mut c_char {
        let acc = describe_acceleration();
        if let Ok(s) = CString::new(acc) {
            s.into_raw()
        } else {
            let p: *const c_char = ptr::null();
            p as *mut c_char
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn dcp_unref_string(string: *mut c_char) {
        if !string.is_null() {
            let _unused = CString::from_raw(string);
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn dcp_get_buffers_size(
        width: u32,
        height: u32,
        format: *const ImageFormat,
        strides: *const usize,
        buffers_size: *mut usize,
        error: *mut ErrorKind,
    ) -> self::Result {
        // Protect from C null pointers
        if format.is_null() || buffers_size.is_null() {
            return set_error(error, ErrorKind::InvalidValue);
        }

        // C enums are untrusted in the sense you can cast any value to an enum type
        let format = &*format;
        let pixel_format = format.pixel_format as u32;
        if !dispatcher::is_pixel_format_valid(pixel_format) {
            return set_error(error, ErrorKind::InvalidValue);
        }

        // We assume there is enough data in the buffers
        // If the assumption will not hold undefined behaviour occurs (like in C)
        let num_planes = format.num_planes as usize;
        if !are_planes_compatible(pixel_format, format.num_planes) {
            return set_error(error, ErrorKind::InvalidValue);
        }

        // Convert nullable type to Option
        let strides = if strides.is_null() {
            None
        } else {
            Some(slice::from_raw_parts(strides, num_planes))
        };

        let buffers_size = slice::from_raw_parts_mut(buffers_size, num_planes);
        match get_buffers_size(width, height, format, strides, buffers_size) {
            Ok(_) => self::Result::Ok,
            Err(error_kind) => set_error(error, error_kind),
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn dcp_convert_image(
        width: u32,
        height: u32,
        src_format: *const ImageFormat,
        src_strides: *const usize,
        src_buffers: *const *const u8,
        dst_format: *const ImageFormat,
        dst_strides: *const usize,
        dst_buffers: *const *mut u8,
        error: *mut ErrorKind,
    ) -> self::Result {
        // Protect from C null pointers
        if src_format.is_null()
            || dst_format.is_null()
            || src_buffers.is_null()
            || dst_buffers.is_null()
        {
            return set_error(error, ErrorKind::InvalidValue);
        }

        // C enums are untrusted in the sense you can cast any value to an enum type
        let src_format: &ImageFormat = &*src_format;
        let dst_format: &ImageFormat = &*dst_format;
        let src_pixel_format = src_format.pixel_format as u32;
        let dst_pixel_format = dst_format.pixel_format as u32;
        if !dispatcher::is_pixel_format_valid(src_pixel_format)
            || !dispatcher::is_pixel_format_valid(dst_pixel_format)
            || !dispatcher::is_color_space_valid(src_format.color_space as u32)
            || !dispatcher::is_color_space_valid(dst_format.color_space as u32)
        {
            return set_error(error, ErrorKind::InvalidValue);
        }

        // We assume there is enough data in the buffers
        // If the assumption will not hold undefined behaviour occurs (like in C)
        if !are_planes_compatible(src_pixel_format, src_format.num_planes)
            || !are_planes_compatible(dst_pixel_format, dst_format.num_planes)
        {
            return set_error(error, ErrorKind::InvalidValue);
        }

        let src_buffers = {
            let src_num_planes = src_format.num_planes as usize;
            let num_planes = cmp::min(src_num_planes, MAX_NUMBER_OF_PLANES);
            let mut src_buf: [MaybeUninit<&[u8]>; MAX_NUMBER_OF_PLANES] =
                [MaybeUninit::uninit().assume_init(); MAX_NUMBER_OF_PLANES];

            for (plane_index, item) in src_buf.iter_mut().enumerate().take(num_planes) {
                let ptr = *src_buffers.add(plane_index);
                if ptr.is_null() {
                    return set_error(error, ErrorKind::InvalidValue);
                }

                *item = MaybeUninit::new(slice::from_raw_parts(ptr, UNBOUNDED_C_ARRAY));
            }

            transmute::<_, [&[u8]; MAX_NUMBER_OF_PLANES]>(src_buf)
        };

        let mut dst_buffers = {
            let dst_num_planes = dst_format.num_planes as usize;
            let num_planes = cmp::min(dst_num_planes, MAX_NUMBER_OF_PLANES);
            let mut dst_buf: [MaybeUninit<&[u8]>; MAX_NUMBER_OF_PLANES] =
                [MaybeUninit::uninit().assume_init(); MAX_NUMBER_OF_PLANES];

            for (plane_index, item) in dst_buf.iter_mut().enumerate().take(num_planes) {
                let ptr = *dst_buffers.add(plane_index);
                if ptr.is_null() {
                    return set_error(error, ErrorKind::InvalidValue);
                }

                *item = MaybeUninit::new(slice::from_raw_parts_mut(ptr, UNBOUNDED_C_ARRAY));
            }

            transmute::<_, [&mut [u8]; MAX_NUMBER_OF_PLANES]>(dst_buf)
        };

        // Convert nullable type to Option
        let src_strides = if src_strides.is_null() {
            None
        } else {
            Some(slice::from_raw_parts(src_strides, UNBOUNDED_C_ARRAY))
        };

        let dst_strides = if dst_strides.is_null() {
            None
        } else {
            Some(slice::from_raw_parts(dst_strides, UNBOUNDED_C_ARRAY))
        };

        match convert_image(
            width,
            height,
            src_format,
            src_strides,
            &src_buffers[..],
            dst_format,
            dst_strides,
            &mut dst_buffers[..],
        ) {
            Ok(_) => self::Result::Ok,
            Err(error_kind) => set_error(error, error_kind),
        }
    }
}
