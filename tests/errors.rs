#![warn(unused)]
#![deny(trivial_casts)]
#![deny(trivial_numeric_casts)]
#![deny(unsafe_code)]
#![deny(unstable_features)]
#![deny(unused_import_braces)]
#![deny(
    clippy::complexity,
    clippy::correctness,
    clippy::perf,
    clippy::style,
    clippy::pedantic
)]
#![allow(clippy::too_many_lines)] // This requires effort to handle

use dcp::{ColorSpace, ErrorKind, ImageFormat, PixelFormat, convert_image};

use dcv_color_primitives as dcp;
use itertools::iproduct;

const PIXEL_FORMATS: &[PixelFormat; 9] = &[
    PixelFormat::Argb,
    PixelFormat::Bgra,
    PixelFormat::Bgr,
    PixelFormat::Rgba,
    PixelFormat::Rgb,
    PixelFormat::I444,
    PixelFormat::I422,
    PixelFormat::I420,
    PixelFormat::Nv12,
];

const COLOR_SPACES: &[ColorSpace; 5] = &[
    ColorSpace::Rgb,
    ColorSpace::Bt601,
    ColorSpace::Bt709,
    ColorSpace::Bt601FR,
    ColorSpace::Bt709FR,
];

const WIDTH_YUV: u32 = 33;
const HEIGHT_YUV: u32 = 3;

const PIXEL_FORMAT_I444: u32 = PixelFormat::I444 as u32;
const COLOR_SPACE_RGB: u32 = ColorSpace::Rgb as u32;
const PIXEL_FORMAT_ARGB: u32 = PixelFormat::Argb as u32;
const PIXEL_FORMAT_BGRA: u32 = PixelFormat::Bgra as u32;
const PIXEL_FORMAT_BGR: u32 = PixelFormat::Bgr as u32;
const PIXEL_FORMAT_NV12: u32 = PixelFormat::Nv12 as u32;
const PIXEL_FORMAT_RGB: u32 = PixelFormat::Rgb as u32;
const PIXEL_FORMAT_RGBA: u32 = PixelFormat::Rgba as u32;

macro_rules! set_expected {
    ($var:ident, $pred:expr, $status:path) => {
        if $var.is_ok() && $pred {
            $var = Err($status);
        }
    };
}

fn is_valid_format(format: &ImageFormat) -> bool {
    match format.pixel_format {
        PixelFormat::I444 | PixelFormat::I422 | PixelFormat::I420 => format.num_planes == 3,
        PixelFormat::Nv12 => format.num_planes == 2,
        _ => format.num_planes == 1,
    }
}

fn get_depth(pixel_format: PixelFormat) -> usize {
    match pixel_format {
        PixelFormat::Bgra | PixelFormat::Argb => 4,
        _ => 3,
    }
}

fn check_err(result: ErrorKind, err: ErrorKind) {
    assert_eq!(result as u32, err as u32);
}

fn check_bounds(
    width: u32,
    height: u32,
    src_format: &ImageFormat,
    src_buffers: &[&[u8]],
    dst_format: &ImageFormat,
    dst_buffers: &mut [&mut [u8]],
) {
    if width == 0 || height == 0 {
        return;
    }

    check_err(
        convert_image(
            width,
            height,
            src_format,
            None,
            &src_buffers[..src_format.num_planes as usize - 1],
            dst_format,
            None,
            dst_buffers,
        )
        .unwrap_err(),
        ErrorKind::NotEnoughData,
    );
    check_err(
        convert_image(
            width,
            height,
            src_format,
            None,
            src_buffers,
            dst_format,
            None,
            &mut dst_buffers[..dst_format.num_planes as usize - 1],
        )
        .unwrap_err(),
        ErrorKind::NotEnoughData,
    );

    // Overflow for source plane data
    for i in 0..src_format.num_planes as usize {
        let mut src_vec = Vec::new();

        for (c, v) in src_buffers.iter().enumerate() {
            if c == i {
                src_vec.push(&v[..0]);
            } else {
                src_vec.push(&v[..]);
            }
        }

        check_err(
            convert_image(
                width,
                height,
                src_format,
                None,
                &src_vec,
                dst_format,
                None,
                dst_buffers,
            )
            .unwrap_err(),
            ErrorKind::NotEnoughData,
        );
    }

    // Overflow for destination plane data
    for i in 0..dst_format.num_planes as usize {
        let mut dst_vec = Vec::new();

        for (c, v) in dst_buffers.iter_mut().enumerate() {
            if c == i {
                dst_vec.push(&mut v[..0]);
            } else {
                dst_vec.push(&mut v[..]);
            }
        }

        check_err(
            convert_image(
                width,
                height,
                src_format,
                None,
                src_buffers,
                dst_format,
                None,
                &mut dst_vec[..],
            )
            .unwrap_err(),
            ErrorKind::NotEnoughData,
        );
    }
}

fn rgb_conversion_errors(src_pixel_format: PixelFormat, dst_pixel_format: PixelFormat) {
    const WIDTH: u32 = 33;
    const HEIGHT: u32 = 1;

    let src_depth = get_depth(src_pixel_format);
    let dst_depth = get_depth(dst_pixel_format);
    let src_stride: usize = (WIDTH as usize) * src_depth;
    let dst_stride: usize = (WIDTH as usize) * dst_depth;
    let src_size: usize = src_stride * (HEIGHT as usize);
    let dst_size: usize = dst_stride * (HEIGHT as usize);

    let src_image = vec![0_u8; src_size];
    let mut dst_image = vec![0_u8; dst_size];

    for num_planes in 0..4 {
        let src_format = ImageFormat {
            pixel_format: src_pixel_format,
            color_space: ColorSpace::Rgb,
            num_planes: 1,
        };
        let dst_format = ImageFormat {
            pixel_format: dst_pixel_format,
            color_space: ColorSpace::Rgb,
            num_planes,
        };

        let src_buffers = &[&src_image[..]];
        let dst_buffers = &mut [&mut dst_image[..]];

        let mut expected = Ok(());
        set_expected!(
            expected,
            !is_valid_format(&src_format),
            ErrorKind::InvalidValue
        );
        set_expected!(
            expected,
            !is_valid_format(&dst_format),
            ErrorKind::InvalidValue
        );

        let status = convert_image(
            WIDTH,
            HEIGHT,
            &src_format,
            Some(&[src_stride; 1]),
            src_buffers,
            &dst_format,
            Some(&[dst_stride; 1]),
            dst_buffers,
        );

        assert_eq!(expected.is_ok(), status.is_ok());
        match status {
            Ok(()) => check_bounds(
                WIDTH,
                HEIGHT,
                &src_format,
                src_buffers,
                &dst_format,
                dst_buffers,
            ),
            Err(err) => check_err(err, expected.unwrap_err()),
        }
    }
}

fn rgb_to_yuv_errors(pixel_format: PixelFormat) {
    let cw = match pixel_format {
        PixelFormat::Nv12 => 2 * WIDTH_YUV.div_ceil(2),
        PixelFormat::I420 => WIDTH_YUV.div_ceil(2),
        _ => WIDTH_YUV,
    } as usize;
    let ch = match pixel_format {
        PixelFormat::I444 => HEIGHT_YUV,
        _ => HEIGHT_YUV.div_ceil(2),
    } as usize;
    let y_size = (WIDTH_YUV as usize) * (HEIGHT_YUV as usize);
    let uv_size = cw * ch;

    let slices = &[0, y_size + uv_size, y_size, 0];
    let mut y_plane = match pixel_format {
        PixelFormat::Nv12 => vec![0_u8; y_size + uv_size],
        _ => vec![0_u8; y_size],
    };
    let mut u_plane = vec![0_u8; uv_size];
    let mut v_plane = vec![0_u8; uv_size];

    for src_pixel_format in PIXEL_FORMATS {
        let src_stride = get_depth(*src_pixel_format) * (WIDTH_YUV as usize);
        let src_size = src_stride * (HEIGHT_YUV as usize);
        let src_image = vec![0_u8; src_size];

        for (num_planes, src_color_space, dst_color_space) in
            iproduct!(0..4, COLOR_SPACES, COLOR_SPACES)
        {
            let src_format = ImageFormat {
                pixel_format: *src_pixel_format,
                color_space: *src_color_space,
                num_planes: 1,
            };
            let dst_format = ImageFormat {
                pixel_format,
                color_space: *dst_color_space,
                num_planes,
            };

            let src_buffers = &[&src_image[..]];
            let mut dst_strides = Vec::with_capacity(1);
            let mut dst_buffers = Vec::with_capacity(1);

            dst_strides.push(WIDTH_YUV as usize);
            if let PixelFormat::Nv12 = pixel_format {
                dst_buffers.push(&mut y_plane[..slices[num_planes as usize]]);
                dst_buffers.push(&mut u_plane[..]);
                if num_planes > 1 {
                    dst_strides.push(cw);
                }
            } else {
                dst_strides.push(cw);
                dst_strides.push(cw);
                dst_buffers.push(&mut y_plane[..]);
                dst_buffers.push(&mut u_plane[..]);
                dst_buffers.push(&mut v_plane[..]);
            }

            let mut expected = Ok(());
            let src_pf = *src_pixel_format as u32;
            let src_cs = *src_color_space as u32;
            let dst_cs = *dst_color_space as u32;
            let src_pf_rgb = src_pf < PIXEL_FORMAT_I444;
            let src_cs_rgb = src_cs == COLOR_SPACE_RGB;
            let dst_cs_rgb = dst_cs == COLOR_SPACE_RGB;

            set_expected!(expected, !src_pf_rgb && src_cs_rgb, ErrorKind::InvalidValue);
            set_expected!(expected, src_pf_rgb && !src_cs_rgb, ErrorKind::InvalidValue);
            set_expected!(expected, dst_cs_rgb, ErrorKind::InvalidValue);
            set_expected!(
                expected,
                !is_valid_format(&src_format),
                ErrorKind::InvalidValue
            );
            set_expected!(
                expected,
                !is_valid_format(&dst_format),
                ErrorKind::InvalidValue
            );
            set_expected!(
                expected,
                src_pf != PIXEL_FORMAT_ARGB
                    && src_pf != PIXEL_FORMAT_BGRA
                    && src_pf != PIXEL_FORMAT_BGR,
                ErrorKind::InvalidOperation
            );
            set_expected!(
                expected,
                src_cs != COLOR_SPACE_RGB,
                ErrorKind::InvalidOperation
            );

            let status = convert_image(
                WIDTH_YUV,
                HEIGHT_YUV,
                &src_format,
                None,
                src_buffers,
                &dst_format,
                Some(&dst_strides),
                &mut dst_buffers,
            );

            assert_eq!(expected.is_ok(), status.is_ok());
            match status {
                Ok(()) => check_bounds(
                    WIDTH_YUV,
                    HEIGHT_YUV,
                    &src_format,
                    src_buffers,
                    &dst_format,
                    &mut dst_buffers,
                ),
                Err(err) => check_err(err, expected.unwrap_err()),
            }
        }
    }
}

fn yuv_to_rgb_errors(pixel_format: PixelFormat) {
    let cw = match pixel_format {
        PixelFormat::I420 => WIDTH_YUV.div_ceil(2),
        PixelFormat::Nv12 => 2 * WIDTH_YUV.div_ceil(2),
        _ => WIDTH_YUV,
    } as usize;
    let ch = match pixel_format {
        PixelFormat::I444 => HEIGHT_YUV,
        _ => HEIGHT_YUV.div_ceil(2),
    } as usize;
    let y_size = (WIDTH_YUV as usize) * (HEIGHT_YUV as usize);
    let uv_size = cw * ch;
    let dst_size = (WIDTH_YUV as usize) * (HEIGHT_YUV as usize) * 4;

    let slices = &[0, y_size + uv_size, y_size, 0];
    let y_plane = match pixel_format {
        PixelFormat::Nv12 => vec![0_u8; y_size + uv_size],
        _ => vec![0_u8; y_size],
    };
    let u_plane = vec![0_u8; uv_size];
    let v_plane = vec![0_u8; uv_size];
    let mut dst_image = vec![0_u8; dst_size];

    for (num_planes, dst_pixel_format, dst_color_space, src_color_space) in
        iproduct!(0..4, PIXEL_FORMATS, COLOR_SPACES, COLOR_SPACES)
    {
        let src_format = ImageFormat {
            pixel_format,
            color_space: *src_color_space,
            num_planes,
        };
        let dst_format = ImageFormat {
            pixel_format: *dst_pixel_format,
            color_space: *dst_color_space,
            num_planes: 1,
        };

        let mut src_strides = Vec::with_capacity(1);
        let mut src_buffers = Vec::with_capacity(1);
        let dst_buffers = &mut [&mut dst_image[..]];

        src_strides.push(WIDTH_YUV as usize);
        if let PixelFormat::Nv12 = pixel_format {
            src_buffers.push(&y_plane[..slices[num_planes as usize]]);
            src_buffers.push(&u_plane[..]);
            if num_planes > 1 {
                src_strides.push(cw);
            }
        } else {
            src_strides.push(cw);
            src_strides.push(cw);
            src_buffers.push(&y_plane[..]);
            src_buffers.push(&u_plane[..]);
            src_buffers.push(&v_plane[..]);
        }

        let mut expected = Ok(());
        let src_pf = pixel_format as u32;
        let dst_pf = *dst_pixel_format as u32;
        let dst_cs = *dst_color_space as u32;
        let src_cs = *src_color_space as u32;
        let dst_pf_rgb = dst_pf < PIXEL_FORMAT_I444;
        let dst_cs_rgb = dst_cs == COLOR_SPACE_RGB;
        let src_cs_rgb = src_cs == COLOR_SPACE_RGB;

        set_expected!(expected, src_cs_rgb, ErrorKind::InvalidValue);
        set_expected!(expected, !dst_pf_rgb && dst_cs_rgb, ErrorKind::InvalidValue);
        set_expected!(expected, dst_pf_rgb && !dst_cs_rgb, ErrorKind::InvalidValue);
        set_expected!(
            expected,
            !is_valid_format(&src_format),
            ErrorKind::InvalidValue
        );
        set_expected!(
            expected,
            !is_valid_format(&dst_format),
            ErrorKind::InvalidValue
        );
        set_expected!(
            expected,
            !(dst_pf == PIXEL_FORMAT_BGRA
                || dst_pf == PIXEL_FORMAT_RGBA
                || (src_pf == PIXEL_FORMAT_NV12 && dst_pf == PIXEL_FORMAT_RGB)),
            ErrorKind::InvalidOperation
        );
        set_expected!(
            expected,
            dst_cs != COLOR_SPACE_RGB,
            ErrorKind::InvalidOperation
        );

        let status = convert_image(
            WIDTH_YUV,
            HEIGHT_YUV,
            &src_format,
            Some(&src_strides),
            &src_buffers,
            &dst_format,
            None,
            dst_buffers,
        );

        assert_eq!(expected.is_ok(), status.is_ok());
        match status {
            Ok(()) => check_bounds(
                WIDTH_YUV,
                HEIGHT_YUV,
                &src_format,
                &src_buffers,
                &dst_format,
                dst_buffers,
            ),
            Err(err) => check_err(err, expected.unwrap_err()),
        }
    }
}

#[cfg(all(test, not(feature = "test_instruction_sets")))]
mod errors {
    use super::{PixelFormat, rgb_conversion_errors, rgb_to_yuv_errors, yuv_to_rgb_errors};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn nv12_to_rgb() {
        yuv_to_rgb_errors(PixelFormat::Nv12);
    }

    #[test]
    fn i420_to_rgb() {
        yuv_to_rgb_errors(PixelFormat::I420);
    }

    #[test]
    fn i444_to_rgb() {
        yuv_to_rgb_errors(PixelFormat::I444);
    }

    #[test]
    fn rgb_to_nv12() {
        rgb_to_yuv_errors(PixelFormat::Nv12);
    }

    #[test]
    fn rgb_to_i420() {
        rgb_to_yuv_errors(PixelFormat::I420);
    }

    #[test]
    fn rgb_to_i444() {
        rgb_to_yuv_errors(PixelFormat::I444);
    }

    #[test]
    fn rgb_to_bgra() {
        rgb_conversion_errors(PixelFormat::Rgb, PixelFormat::Bgra);
    }

    #[test]
    fn bgra_to_rgb() {
        rgb_conversion_errors(PixelFormat::Bgra, PixelFormat::Rgb);
    }

    #[test]
    fn bgr_to_rgb() {
        rgb_conversion_errors(PixelFormat::Bgr, PixelFormat::Rgb);
    }
}

#[cfg(all(test, feature = "test_instruction_sets"))]
mod errors {
    use super::*;
    use dcp::initialize_with_instruction_set;

    #[test]
    fn coverage() {
        const SETS: [&str; 3] = ["x86", "sse2", "avx2"];

        for set in &SETS {
            initialize_with_instruction_set(set);

            yuv_to_rgb_errors(PixelFormat::Nv12);
            yuv_to_rgb_errors(PixelFormat::I420);
            yuv_to_rgb_errors(PixelFormat::I444);
            rgb_to_yuv_errors(PixelFormat::Nv12);
            rgb_to_yuv_errors(PixelFormat::I420);
            rgb_to_yuv_errors(PixelFormat::I444);
            rgb_conversion_errors(PixelFormat::Rgb, PixelFormat::Bgra);
            rgb_conversion_errors(PixelFormat::Bgra, PixelFormat::Rgb);
            rgb_conversion_errors(PixelFormat::Bgr, PixelFormat::Rgb);
        }
    }
}
