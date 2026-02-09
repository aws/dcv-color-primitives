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
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]

use dcp::{ColorSpace, ImageFormat, PixelFormat, STRIDE_AUTO, convert_image};

use dcv_color_primitives as dcp;
use itertools::iproduct;
use rand::Rng;
use std::borrow::Cow;
use std::sync::OnceLock;

// Include utils functions directly
mod utils {
    include!("../utils.rs");
}
use utils::*;

const MAX_PLANE_WIDTH: usize = 8;
const MAX_PLANE_HEIGHT: usize = 8;
const MAX_UV_WIDTH: usize = MAX_PLANE_WIDTH >> 1;
const MAX_UV_HEIGHT: usize = MAX_PLANE_HEIGHT >> 1;

const SUPPORTED_COLOR_SPACES: &[ColorSpace] = &[
    ColorSpace::Bt601,
    ColorSpace::Bt709,
    ColorSpace::Bt2020,
    ColorSpace::Bt601FR,
    ColorSpace::Bt709FR,
    ColorSpace::Bt2020FR,
];

const YUV_RGB_MAX_WIDTH: usize = 33;
const YUV_RGB_MAX_HEIGHT: usize = 3;
const YUV_RGB_MAX_PAD: usize = 4;

static COEFFICIENTS_TABLE: OnceLock<[Coefficients; 6]> = OnceLock::new();

type FullPlane = [[u8; MAX_PLANE_WIDTH]; MAX_PLANE_HEIGHT];
type SubSampledPlane = [[u8; MAX_UV_WIDTH]; MAX_UV_HEIGHT];

// Generated tables are included from build.rs
include!(concat!(env!("OUT_DIR"), "/yuv_tables.rs"));

type PlaneData<'a> = Cow<'a, [[u8; MAX_PLANE_WIDTH]; MAX_PLANE_HEIGHT]>;
type SubSampledPlaneData<'a> = Cow<'a, [[u8; MAX_UV_WIDTH]; MAX_UV_HEIGHT]>;

enum PlaneRef<'a> {
    Full(PlaneData<'a>),
    SubSampled(SubSampledPlaneData<'a>),
}

fn fill_biplanar_chroma(
    plane: &mut [u8],
    u_src: [u8; 8],
    v_src: [u8; 8],
    width: usize,
    height: usize,
    stride: usize,
) {
    if stride == 0 {
        return;
    }

    for row in plane.chunks_exact_mut(stride).take(height) {
        for (pos, uv_val) in row.chunks_exact_mut(2).enumerate().take(width) {
            let index = pos & 0x7;
            uv_val[0] = u_src[index];
            uv_val[1] = v_src[index];
        }
    }
}

fn fill_planar_chroma(
    plane: &mut [u8],
    src: [u8; 8],
    width: usize,
    height: usize,
    stride: usize,
    shift: usize,
) {
    if stride == 0 {
        return;
    }

    for row in plane.chunks_exact_mut(stride).take(height) {
        for (pos, val) in row.iter_mut().enumerate().take(width) {
            *val = src[(pos >> shift) & 0x7];
        }
    }
}

fn fill_planar_luma(plane: &mut [u8], src: [u8; 8], width: usize, height: usize, stride: usize) {
    if stride == 0 {
        return;
    }

    for row in plane.chunks_exact_mut(stride).take(height) {
        for (pos, luma) in row.chunks_mut(2).enumerate().take(width) {
            luma.fill(src[pos & 7]);
        }
    }
}

fn get_color_space_index(color_space: ColorSpace) -> usize {
    match color_space {
        ColorSpace::Bt601 => 0,
        ColorSpace::Bt709 => 1,
        ColorSpace::Bt2020 => 2,
        ColorSpace::Bt601FR => 3,
        ColorSpace::Bt709FR => 4,
        _ => 5, // ColorSpace::Bt2020FR
    }
}

fn get_uv_stride(pixel_format: PixelFormat, w: usize, cw: usize, pad: usize) -> usize {
    match pixel_format {
        PixelFormat::Nv12 => 2 * cw + pad,
        PixelFormat::I420 => cw + pad,
        _ => w + pad, /* PixelFormat::I444 */
    }
}

fn get_expected_plane_data(
    color_space: ColorSpace,
    width: usize,
    height: usize,
) -> (SubSampledPlane, SubSampledPlane) {
    let coefficients = COEFFICIENTS_TABLE.get_or_init(|| {
        let mut coefficients = [((0, 0, 0), (0, 0), 0, 0); 6];

        for (index, coeff_row) in coefficients.iter_mut().enumerate() {
            let full_range = index >= 3;
            let model = index % 3;
            let (kr, kg, kb) = match model {
                0 => (0.299, 0.587, 0.114),    // BT.601
                1 => (0.2126, 0.7152, 0.0722), // BT.709
                _ => (0.2627, 0.6780, 0.0593), // BT.2020
            };
            let (y_min, y_max, c_min, c_max) = if full_range {
                (0, 255, 0, 255)
            } else {
                (16, 235, 16, 240)
            };
            let (y_scale, c_scale) = if full_range {
                (1f64, 1f64)
            } else {
                (
                    f64::from(y_max - y_min) / FULL_RANGE,
                    f64::from(c_max - c_min) / FULL_RANGE,
                )
            };

            *coeff_row = compute_coefficients::<false>((kr, kg, kb), y_min, y_scale, c_scale);
        }

        coefficients
    });

    let index = get_color_space_index(color_space);
    let (_, (yr, yg), zg, _) = coefficients[index];

    let yb = -(yr + yg);
    let zr = yb;
    let zb = -(zr + zg);

    let mut us = [[0u8; MAX_UV_WIDTH]; MAX_UV_HEIGHT];
    let mut vs = [[0u8; MAX_UV_WIDTH]; MAX_UV_HEIGHT];
    for (y, (u_row, v_row)) in us
        .iter_mut()
        .zip(vs.iter_mut())
        .take(height.div_ceil(2))
        .enumerate()
    {
        let y0 = 2 * y;
        let y1 = (y0 + 1).min(height - 1);
        let top_row = &RGB_SRC[y0];
        let bottom_row = &RGB_SRC[y1];

        for x in 0..width.div_ceil(2) {
            let x0 = 2 * x;
            let x1 = (x0 + 1).min(width - 1);
            let p00 = top_row[x0];
            let p10 = top_row[x1];
            let p01 = bottom_row[x0];
            let p11 = bottom_row[x1];

            let red = p00[0] + p10[0] + p01[0] + p11[0];
            let green = p00[1] + p10[1] + p01[1] + p11[1];
            let blue = p00[2] + p10[2] + p01[2] + p11[2];
            let u = ((yr * red + yg * green + yb * blue + UV_SHIFT_18) >> FIX18) as u8;
            let v = ((zr * red + zg * green + zb * blue + UV_SHIFT_18) >> FIX18) as u8;

            u_row[x] = u;
            v_row[x] = v;
        }
    }

    (us, vs)
}

fn get_depth(pixel_format: PixelFormat) -> usize {
    match pixel_format {
        PixelFormat::Bgra | PixelFormat::Argb => 4,
        _ => 3,
    }
}

fn check_plane(plane: &[u8], reference: &PlaneRef, width: usize, stride: usize) {
    if stride == 0 {
        return;
    }

    let PlaneRef::Full(reference) = reference else {
        return;
    };

    for (row, exp) in plane.chunks_exact(stride).zip(reference.iter()) {
        let (payload, pad) = row.split_at(width);
        assert!(payload.iter().zip(exp).all(|(&x, &y)| x == y));
        assert!(pad.iter().all(|&x| x == 0));
    }
}

fn check_subsampled_plane(plane: &[u8], reference: &PlaneRef, width: usize, stride: usize) {
    if stride == 0 {
        return;
    }

    let PlaneRef::SubSampled(reference) = reference else {
        return;
    };

    for (row, exp) in plane.chunks_exact(stride).zip(reference.iter()) {
        let (payload, pad) = row.split_at(width);
        assert!(payload.iter().zip(exp).all(|(&x, &y)| x == y));
        assert!(pad.iter().all(|&x| x == 0));
    }
}

fn rgb_to_yuv_size_mode_pad(
    image_size: (usize, usize),
    src_format: &ImageFormat,
    dst_format: &ImageFormat,
    pad: (usize, usize, usize, usize),
    plane_ref: &(PlaneRef, PlaneRef, PlaneRef),
) {
    let (src_pad, y_pad, u_pad, v_pad) = pad;
    let w = image_size.0;
    let h = image_size.1;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);

    // Allocate and initialize input
    let src_depth = get_depth(src_format.pixel_format);
    let src_stride = w * src_depth + src_pad;
    let src_size = src_stride * h;
    let mut src_image = vec![0_u8; src_size];
    if w > 0 && h > 0 {
        for (row, line) in src_image.chunks_exact_mut(src_stride).zip(RGB_SRC) {
            for (pixel_bytes, pixel) in
                &mut row[..src_depth * w].chunks_exact_mut(src_depth).zip(line)
            {
                match src_format.pixel_format {
                    PixelFormat::Argb => {
                        pixel_bytes[0] = pixel[3] as u8;
                        pixel_bytes[1] = pixel[0] as u8;
                        pixel_bytes[2] = pixel[1] as u8;
                        pixel_bytes[3] = pixel[2] as u8;
                    }
                    PixelFormat::Bgra => {
                        pixel_bytes[0] = pixel[2] as u8;
                        pixel_bytes[1] = pixel[1] as u8;
                        pixel_bytes[2] = pixel[0] as u8;
                        pixel_bytes[3] = pixel[3] as u8;
                    }
                    _ => {
                        pixel_bytes[0] = pixel[2] as u8;
                        pixel_bytes[1] = pixel[1] as u8;
                        pixel_bytes[2] = pixel[0] as u8;
                    }
                }
            }
        }
    }

    let src_stride = if src_pad == 0 {
        STRIDE_AUTO
    } else {
        src_stride
    };

    // Allocate output
    let y_stride = w + y_pad;
    let u_stride = get_uv_stride(dst_format.pixel_format, w, cw, u_pad);
    let v_stride = get_uv_stride(dst_format.pixel_format, w, cw, v_pad);
    let y_size = y_stride * h;
    let dst_size = y_size
        + match dst_format.pixel_format {
            PixelFormat::Nv12 => u_stride * ch,
            PixelFormat::I420 => (u_stride + v_stride) * ch,
            _ => (u_stride + v_stride) * h, /* PixelFormat::I444 */
        };
    let mut dst_image = vec![0_u8; dst_size];

    // Compute strides
    let mut dst_strides = Vec::with_capacity(3);
    let mut dst_buffers: Vec<&mut [u8]> = Vec::with_capacity(3);
    dst_strides.push(if y_pad == 0 { STRIDE_AUTO } else { y_stride });
    dst_strides.push(if u_pad == 0 { STRIDE_AUTO } else { u_stride });
    match dst_format.pixel_format {
        PixelFormat::Nv12 => {
            let (first, last) = dst_image.split_at_mut(y_size);
            dst_buffers.push(first);
            dst_buffers.push(last);
        }
        PixelFormat::I420 => {
            let (y_plane, chroma_planes) = dst_image.split_at_mut(y_size);
            let (u_plane, v_plane) = chroma_planes.split_at_mut(ch * u_stride);

            dst_buffers.push(y_plane);
            dst_buffers.push(u_plane);
            dst_buffers.push(v_plane);
            dst_strides.push(if v_pad == 0 { STRIDE_AUTO } else { v_stride });
        }
        _ => {
            /* PixelFormat::I444 */
            let (y_plane, chroma_planes) = dst_image.split_at_mut(y_size);
            let (u_plane, v_plane) = chroma_planes.split_at_mut(u_stride * h);

            dst_buffers.push(y_plane);
            dst_buffers.push(u_plane);
            dst_buffers.push(v_plane);
            dst_strides.push(if v_pad == 0 { STRIDE_AUTO } else { v_stride });
        }
    }

    assert!(
        convert_image(
            image_size.0 as u32,
            image_size.1 as u32,
            src_format,
            Some(&[src_stride]),
            &[&src_image[..]],
            dst_format,
            Some(&dst_strides[..]),
            &mut dst_buffers[..],
        )
        .is_ok()
    );

    if w == 0 || h == 0 {
        return;
    }

    check_plane(dst_buffers[0], &plane_ref.0, w, y_stride);
    match dst_format.pixel_format {
        PixelFormat::Nv12 => {
            let PlaneRef::SubSampled(u_ref) = &plane_ref.1 else {
                return;
            };
            let PlaneRef::SubSampled(v_ref) = &plane_ref.2 else {
                return;
            };

            for (uv_row, (u_exp, v_exp)) in dst_image[y_size..]
                .chunks_exact(u_stride)
                .zip(u_ref.iter().zip(v_ref.iter()))
            {
                let (payload, pad) = uv_row.split_at(2 * cw);
                assert!(
                    payload
                        .chunks_exact(2)
                        .zip(u_exp.iter().zip(v_exp))
                        .all(|(uv, (&u, &v))| uv[0] == u && uv[1] == v)
                );
                assert!(pad.iter().all(|&x| x == 0));
            }
        }
        PixelFormat::I420 => {
            let u_end = y_size + u_stride * ch;
            check_subsampled_plane(&dst_image[y_size..u_end], &plane_ref.1, cw, u_stride);
            check_subsampled_plane(&dst_image[u_end..], &plane_ref.2, cw, v_stride);
        }
        _ => {
            /* PixelFormat::I444 */
            check_plane(dst_buffers[1], &plane_ref.1, w, u_stride);
            check_plane(dst_buffers[2], &plane_ref.2, w, v_stride);
        }
    }
}

fn rgb_to_yuv_size_mode(
    image_size: (usize, usize),
    src_format: &ImageFormat,
    dst_format: &ImageFormat,
    plane_ref: &(PlaneRef, PlaneRef, PlaneRef),
) {
    const MAX_PAD: usize = 2;

    if matches!(dst_format.pixel_format, PixelFormat::Nv12) {
        for (src_pad, y_pad, uv_pad) in iproduct!(0..MAX_PAD, 0..MAX_PAD, 0..MAX_PAD) {
            rgb_to_yuv_size_mode_pad(
                image_size,
                src_format,
                dst_format,
                (src_pad, y_pad, uv_pad, uv_pad),
                plane_ref,
            );
        }
    } else {
        for pad in iproduct!(0..MAX_PAD, 0..MAX_PAD, 0..MAX_PAD, 0..MAX_PAD) {
            rgb_to_yuv_size_mode_pad(image_size, src_format, dst_format, pad, plane_ref);
        }
    }
}

fn rgb_to_yuv_size(
    image_size: (usize, usize),
    dst_image_format: &ImageFormat,
    plane_ref: &(PlaneRef, PlaneRef, PlaneRef),
) {
    const SUPPORTED_PIXEL_FORMATS: &[PixelFormat] =
        &[PixelFormat::Argb, PixelFormat::Bgra, PixelFormat::Bgr];

    for pixel_format in SUPPORTED_PIXEL_FORMATS {
        rgb_to_yuv_size_mode(
            image_size,
            &ImageFormat {
                pixel_format: *pixel_format,
                color_space: ColorSpace::Rgb,
            },
            dst_image_format,
            plane_ref,
        );
    }
}

fn rgb_to_yuv_ok(pixel_format: PixelFormat) {
    for color_space in SUPPORTED_COLOR_SPACES {
        let format = ImageFormat {
            pixel_format,
            color_space: *color_space,
        };

        for (width, height) in iproduct!(0..=MAX_PLANE_WIDTH, 0..=MAX_PLANE_HEIGHT) {
            let y_plane = match color_space {
                ColorSpace::Bt601 => &Y_BT601_REF,
                ColorSpace::Bt709 => &Y_BT709_REF,
                ColorSpace::Bt2020 => &Y_BT2020_REF,
                ColorSpace::Bt601FR => &Y_BT601FR_REF,
                ColorSpace::Bt709FR => &Y_BT709FR_REF,
                _ => &Y_BT2020FR_REF,
            };
            let (u_plane, v_plane) = if let PixelFormat::I444 = pixel_format {
                let (u_plane, v_plane) = match color_space {
                    ColorSpace::Bt601 => (&CB_BT601_REF, &CR_BT601_REF),
                    ColorSpace::Bt709 => (&CB_BT709_REF, &CR_BT709_REF),
                    ColorSpace::Bt2020 => (&CB_BT2020_REF, &CR_BT2020_REF),
                    ColorSpace::Bt601FR => (&CB_BT601FR_REF, &CR_BT601FR_REF),
                    ColorSpace::Bt709FR => (&CB_BT709FR_REF, &CR_BT709FR_REF),
                    _ => (&CB_BT2020FR_REF, &CR_BT2020FR_REF),
                };

                (
                    PlaneRef::Full(Cow::Borrowed(u_plane)),
                    PlaneRef::Full(Cow::Borrowed(v_plane)),
                )
            } else if (width & 1) == 0 && (height & 1) == 0 {
                let (u_plane, v_plane) = match color_space {
                    ColorSpace::Bt601 => (&CB2_BT601_REF, &CR2_BT601_REF),
                    ColorSpace::Bt709 => (&CB2_BT709_REF, &CR2_BT709_REF),
                    ColorSpace::Bt2020 => (&CB2_BT2020_REF, &CR2_BT2020_REF),
                    ColorSpace::Bt601FR => (&CB2_BT601FR_REF, &CR2_BT601FR_REF),
                    ColorSpace::Bt709FR => (&CB2_BT709FR_REF, &CR2_BT709FR_REF),
                    _ => (&CB2_BT2020FR_REF, &CR2_BT2020FR_REF),
                };
                (
                    PlaneRef::SubSampled(Cow::Borrowed(u_plane)),
                    PlaneRef::SubSampled(Cow::Borrowed(v_plane)),
                )
            } else {
                let uv_planes = get_expected_plane_data(*color_space, width, height);
                (
                    PlaneRef::SubSampled(Cow::Owned(uv_planes.0)),
                    PlaneRef::SubSampled(Cow::Owned(uv_planes.1)),
                )
            };
            let planes = (PlaneRef::Full(Cow::Borrowed(y_plane)), u_plane, v_plane);

            rgb_to_yuv_size((width, height), &format, &planes);
        }
    }
}

fn yuv_to_bgra_size_format_mode_stride(
    image_size: (usize, usize),
    src_format: &ImageFormat,
    dst_format: &ImageFormat,
    pad: (usize, usize, usize, usize),
) {
    let (y_pad, u_pad, v_pad, dst_pad) = pad;
    let w = image_size.0;
    let h = image_size.1;
    let cw = w.div_ceil(2);
    let ch = if let PixelFormat::I444 = src_format.pixel_format {
        h
    } else {
        h.div_ceil(2)
    };

    // Allocate and initialize input
    let y_stride = w + y_pad;
    let u_stride = get_uv_stride(src_format.pixel_format, w, cw, u_pad);
    let v_stride = get_uv_stride(src_format.pixel_format, w, cw, v_pad);
    let y_size = y_stride * h;
    let src_size = (y_size + u_stride * ch)
        + if let PixelFormat::Nv12 = src_format.pixel_format {
            0
        } else {
            v_stride * ch
        };

    let color_space_index = get_color_space_index(src_format.color_space);

    let mut src_image = vec![0_u8; src_size];
    if w > 0 && h > 0 {
        fill_planar_luma(&mut src_image, Y_SRC[color_space_index], cw, h, y_stride);
        let u_src = U_SRC[color_space_index];
        let v_src = V_SRC[color_space_index];
        match src_format.pixel_format {
            PixelFormat::Nv12 => {
                fill_biplanar_chroma(&mut src_image[y_size..], u_src, v_src, cw, ch, u_stride);
            }
            PixelFormat::I420 => {
                let v_begin = y_size + u_stride * ch;
                fill_planar_chroma(&mut src_image[y_size..v_begin], u_src, cw, ch, u_stride, 0);
                fill_planar_chroma(&mut src_image[v_begin..], v_src, cw, ch, v_stride, 0);
            }
            _ => {
                /* PixelFormat::I444 */
                let v_begin = y_size + u_stride * ch;
                fill_planar_chroma(&mut src_image[y_size..v_begin], u_src, w, ch, u_stride, 1);
                fill_planar_chroma(&mut src_image[v_begin..], v_src, w, ch, v_stride, 1);
            }
        }
    }

    let mut src_buffers: Vec<&[u8]> = Vec::with_capacity(3);
    let mut src_strides = Vec::with_capacity(3);
    src_strides.push(if y_pad == 0 { STRIDE_AUTO } else { y_stride });
    src_strides.push(if u_pad == 0 { STRIDE_AUTO } else { u_stride });
    if let PixelFormat::Nv12 = src_format.pixel_format {
        let (first, last) = src_image.split_at(y_size);
        src_buffers.push(first);
        src_buffers.push(last);
    } else {
        let u_size = u_stride * ch;
        src_strides.push(if v_pad == 0 { STRIDE_AUTO } else { v_stride });
        src_buffers.push(&src_image[..y_size]);
        src_buffers.push(&src_image[y_size..y_size + u_size]);
        src_buffers.push(&src_image[y_size + u_size..]);
    }

    // Allocate output
    let dst_stride = w * 4 + dst_pad;
    let dst_size = dst_stride * h;
    let mut dst_image = vec![0_u8; dst_size];
    let dst_stride = if dst_pad == 0 {
        STRIDE_AUTO
    } else {
        dst_stride
    };

    assert!(
        convert_image(
            image_size.0 as u32,
            image_size.1 as u32,
            src_format,
            Some(&src_strides[..]),
            &src_buffers[..],
            dst_format,
            Some(&[dst_stride]),
            &mut [&mut dst_image[..]],
        )
        .is_ok()
    );

    if w == 0 || h == 0 {
        return;
    }

    let mut expected_row = vec![0_i32; 3 * w];
    for (x, pixel) in expected_row.chunks_exact_mut(3).enumerate() {
        let index = (x >> 1) & 7;

        // Expected blue
        pixel[0] = if ((index >> 2) & 1) == 0 { 0 } else { 255 };
        // Expected green
        pixel[1] = if ((index >> 1) & 1) == 0 { 0 } else { 255 };
        // Expected red
        pixel[2] = if (index & 1) == 0 { 0 } else { 255 };
    }

    let (r_offset, b_offset) = match dst_format.pixel_format {
        PixelFormat::Bgra => (0, 2),
        _ => (2, 0),
    };

    let pack_stride = w * 4;
    let dst_stride = pack_stride + dst_pad;
    for row in dst_image.chunks_exact(dst_stride).take(h) {
        let (pixels, pad) = row.split_at(pack_stride);
        assert!(
            pixels
                .chunks_exact(4)
                .zip(expected_row.chunks_exact(3))
                .all(|(pixel, expected)| {
                    (i32::from(pixel[r_offset]) - expected[0]).abs() <= 2
                        && (i32::from(pixel[1]) - expected[1]).abs() <= 2
                        && (i32::from(pixel[b_offset]) - expected[2]).abs() <= 2
                        && pixel[3] == 255
                })
        );
        assert!(pad.iter().all(|&x| x == 0));
    }
}

fn yuv_to_bgra_size_format_mode(
    image_size: (usize, usize),
    src_format: &ImageFormat,
    dst_format: &ImageFormat,
) {
    if matches!(src_format.pixel_format, PixelFormat::Nv12) {
        for (y_pad, uv_pad, dst_pad) in
            iproduct!(0..YUV_RGB_MAX_PAD, 0..YUV_RGB_MAX_PAD, 0..YUV_RGB_MAX_PAD)
        {
            yuv_to_bgra_size_format_mode_stride(
                image_size,
                src_format,
                dst_format,
                (y_pad, uv_pad, uv_pad, dst_pad),
            );
        }
    } else {
        for pad in iproduct!(
            0..YUV_RGB_MAX_PAD,
            0..YUV_RGB_MAX_PAD,
            0..YUV_RGB_MAX_PAD,
            0..YUV_RGB_MAX_PAD
        ) {
            yuv_to_bgra_size_format_mode_stride(image_size, src_format, dst_format, pad);
        }
    }
}

fn yuv_to_bgra_ok(pixel_format: PixelFormat) {
    const SUPPORTED_FORMATS: &[PixelFormat] = &[PixelFormat::Bgra, PixelFormat::Rgba];

    for (color_space, format) in iproduct!(SUPPORTED_COLOR_SPACES, SUPPORTED_FORMATS) {
        let src_format = ImageFormat {
            pixel_format,
            color_space: *color_space,
        };
        let dst_format = ImageFormat {
            pixel_format: *format,
            color_space: ColorSpace::Rgb,
        };

        for (width, height) in iproduct!(0..=YUV_RGB_MAX_WIDTH, 0..=YUV_RGB_MAX_HEIGHT) {
            yuv_to_bgra_size_format_mode((width, height), &src_format, &dst_format);
        }
    }
}

fn yuv_to_rgb_size_format_mode_stride(
    image_size: (usize, usize),
    src_format: &ImageFormat,
    dst_format: &ImageFormat,
    pad: (usize, usize, usize, usize),
) {
    let (y_pad, u_pad, v_pad, dst_pad) = pad;
    let w = image_size.0;
    let h = image_size.1;
    let cw = w.div_ceil(2);
    let ch = if let PixelFormat::I444 = src_format.pixel_format {
        h
    } else {
        h.div_ceil(2)
    };

    let y_stride = w + y_pad;
    let y_size = y_stride * h;
    let (u_stride, v_stride, src_size) = match src_format.pixel_format {
        PixelFormat::Nv12 => {
            let u_stride = 2 * cw + u_pad;
            (u_stride, u_stride, y_size + u_stride * ch)
        }
        PixelFormat::I420 => {
            let u_stride = cw + u_pad;
            let v_stride = cw + v_pad;
            let u_size = u_stride * ch;
            (u_stride, v_stride, y_size + u_size + v_stride * ch)
        }
        _ => {
            // PixelFormat::I444
            let u_stride = w + u_pad;
            let v_stride = w + v_pad;
            let u_size = u_stride * ch;
            (u_stride, v_stride, y_size + u_size + v_stride * ch)
        }
    };

    let color_space_index = get_color_space_index(src_format.color_space);

    let mut src_image = vec![0_u8; src_size];
    if w > 0 && h > 0 {
        fill_planar_luma(&mut src_image, Y_SRC[color_space_index], cw, h, y_stride);
        match src_format.pixel_format {
            PixelFormat::Nv12 => {
                fill_biplanar_chroma(
                    &mut src_image[y_size..],
                    U_SRC[color_space_index],
                    V_SRC[color_space_index],
                    cw,
                    ch,
                    u_stride,
                );
            }
            PixelFormat::I420 => {
                let u_size = u_stride * ch;
                fill_planar_chroma(
                    &mut src_image[y_size..y_size + u_size],
                    U_SRC[color_space_index],
                    cw,
                    ch,
                    u_stride,
                    0,
                );
                fill_planar_chroma(
                    &mut src_image[y_size + u_size..],
                    V_SRC[color_space_index],
                    cw,
                    ch,
                    v_stride,
                    0,
                );
            }
            _ => {
                // PixelFormat::I444
                let u_size = u_stride * ch;
                fill_planar_chroma(
                    &mut src_image[y_size..y_size + u_size],
                    U_SRC[color_space_index],
                    w,
                    ch,
                    u_stride,
                    1,
                );
                fill_planar_chroma(
                    &mut src_image[y_size + u_size..],
                    V_SRC[color_space_index],
                    w,
                    ch,
                    v_stride,
                    1,
                );
            }
        }
    }

    let dst_stride = w * 3 + dst_pad;
    let dst_size = dst_stride * h;
    let mut dst_image = vec![0_u8; dst_size];
    let dst_stride = if dst_pad == 0 {
        STRIDE_AUTO
    } else {
        dst_stride
    };

    let y_size = y_stride * h;
    let (src_strides, src_buffers): (Vec<usize>, Vec<&[u8]>) =
        if let PixelFormat::Nv12 = src_format.pixel_format {
            let (first, last) = src_image.split_at(y_size);
            (
                vec![
                    if y_pad == 0 { STRIDE_AUTO } else { y_stride },
                    if u_pad == 0 { STRIDE_AUTO } else { u_stride },
                ],
                vec![first, last],
            )
        } else {
            let u_size = u_stride * ch;
            (
                vec![
                    if y_pad == 0 { STRIDE_AUTO } else { y_stride },
                    if u_pad == 0 { STRIDE_AUTO } else { u_stride },
                    if v_pad == 0 { STRIDE_AUTO } else { v_stride },
                ],
                vec![
                    &src_image[..y_size],
                    &src_image[y_size..y_size + u_size],
                    &src_image[y_size + u_size..],
                ],
            )
        };

    assert!(
        convert_image(
            image_size.0 as u32,
            image_size.1 as u32,
            src_format,
            Some(&src_strides[..]),
            &src_buffers[..],
            dst_format,
            Some(&[dst_stride]),
            &mut [&mut dst_image[..]],
        )
        .is_ok()
    );

    if w == 0 || h == 0 {
        return;
    }

    let pack_stride = w * 3;
    let mut expected_row = vec![0_i32; pack_stride];

    let (r_offset, g_offset, b_offset) = if let PixelFormat::Rgb = dst_format.pixel_format {
        (0, 1, 2)
    } else {
        (2, 1, 0)
    };

    for (x, pixel) in expected_row.chunks_exact_mut(3).enumerate() {
        let index = (x >> 1) & 7;

        pixel[r_offset] = if (index & 1) == 0 { 0 } else { 255 };
        pixel[g_offset] = if ((index >> 1) & 1) == 0 { 0 } else { 255 };
        pixel[b_offset] = if ((index >> 2) & 1) == 0 { 0 } else { 255 };
    }

    let dst_stride = pack_stride + dst_pad;
    for row in dst_image.chunks_exact(dst_stride).take(h) {
        let (pixels, pad) = row.split_at(pack_stride);
        assert!(
            pixels
                .chunks_exact(3)
                .zip(expected_row.chunks_exact(3))
                .all(|(pixel, expected)| {
                    (i32::from(pixel[0]) - expected[0]).abs() <= 2
                        && (i32::from(pixel[1]) - expected[1]).abs() <= 2
                        && (i32::from(pixel[2]) - expected[2]).abs() <= 2
                })
        );
        assert!(pad.iter().all(|&x| x == 0));
    }
}

fn yuv_to_rgb_size_format_mode(
    image_size: (usize, usize),
    src_format: &ImageFormat,
    dst_format: &ImageFormat,
) {
    if matches!(src_format.pixel_format, PixelFormat::Nv12) {
        for (y_pad, uv_pad, dst_pad) in
            iproduct!(0..YUV_RGB_MAX_PAD, 0..YUV_RGB_MAX_PAD, 0..YUV_RGB_MAX_PAD)
        {
            yuv_to_rgb_size_format_mode_stride(
                image_size,
                src_format,
                dst_format,
                (y_pad, uv_pad, uv_pad, dst_pad),
            );
        }
    } else {
        for pad in iproduct!(
            0..YUV_RGB_MAX_PAD,
            0..YUV_RGB_MAX_PAD,
            0..YUV_RGB_MAX_PAD,
            0..YUV_RGB_MAX_PAD
        ) {
            yuv_to_rgb_size_format_mode_stride(image_size, src_format, dst_format, pad);
        }
    }
}

fn yuv_to_rgb_ok(pixel_format: PixelFormat, dst_pixel_format: PixelFormat) {
    let dst_format = ImageFormat {
        pixel_format: dst_pixel_format,
        color_space: ColorSpace::Rgb,
    };

    for color_space in SUPPORTED_COLOR_SPACES {
        let src_format = ImageFormat {
            pixel_format,
            color_space: *color_space,
        };

        for (width, height) in iproduct!(0..=YUV_RGB_MAX_WIDTH, 0..=YUV_RGB_MAX_HEIGHT) {
            yuv_to_rgb_size_format_mode((width, height), &src_format, &dst_format);
        }
    }
}

fn rgb_ok(src_pixel_format: PixelFormat, dst_pixel_format: PixelFormat) {
    const MAX_WIDTH: u32 = 49;
    const MAX_HEIGHT: u32 = 8;
    const MAX_PAD: usize = 3;

    let src_depth = get_depth(src_pixel_format);
    let dst_depth = get_depth(dst_pixel_format);

    let src_format = ImageFormat {
        pixel_format: src_pixel_format,
        color_space: ColorSpace::Rgb,
    };
    let dst_format = ImageFormat {
        pixel_format: dst_pixel_format,
        color_space: ColorSpace::Rgb,
    };
    let mut rng = rand::rng();

    for (width, height, src_pad, dst_pad) in
        iproduct!(0..=MAX_WIDTH, 0..=MAX_HEIGHT, 0..MAX_PAD, 0..MAX_PAD)
    {
        let w = width as usize;
        let h = height as usize;
        let src_stride = src_depth * w + src_pad;
        let dst_stride = dst_depth * w + dst_pad;

        let src_strides = if src_pad == 0 {
            STRIDE_AUTO
        } else {
            src_stride
        };
        let dst_strides = if dst_pad == 0 {
            STRIDE_AUTO
        } else {
            dst_stride
        };

        let (b_offset, g_offset, r_offset, a_offset) = if let PixelFormat::Argb = src_pixel_format {
            (3, 2, 1, 0)
        } else {
            (0, 1, 2, 3)
        };

        let mut src_image = vec![0_u8; src_stride * h];
        let mut dst_image = vec![0_u8; dst_stride * h];
        for y in 0..h {
            for x in 0..w {
                let offset = y * src_stride + x * src_depth;

                src_image[offset + b_offset] = rng.random::<u8>(); // b
                src_image[offset + g_offset] = rng.random::<u8>(); // g
                src_image[offset + r_offset] = rng.random::<u8>(); // r
                if src_depth == 4 {
                    src_image[offset + a_offset] = 255; // a
                }
            }
        }

        assert!(
            convert_image(
                width,
                height,
                &src_format,
                Some(&[src_strides]),
                &[&src_image[..]],
                &dst_format,
                Some(&[dst_strides]),
                &mut [&mut dst_image[..]],
            )
            .is_ok()
        );

        for y in 0..h {
            for x in 0..w {
                let input_index = y * src_stride + x * src_depth;
                let output_index = y * dst_stride + x * dst_depth;

                assert_eq!(dst_image[output_index], src_image[input_index + r_offset]);
                assert_eq!(
                    dst_image[output_index + 1],
                    src_image[input_index + g_offset]
                );
                assert_eq!(
                    dst_image[output_index + 2],
                    src_image[input_index + b_offset]
                );
                if dst_depth == 4 {
                    assert_eq!(dst_image[output_index + a_offset], 255);
                }
            }
        }
    }
}

#[cfg(all(test, not(feature = "test_instruction_sets")))]
mod conversions {
    use super::{PixelFormat, rgb_ok, rgb_to_yuv_ok, yuv_to_bgra_ok, yuv_to_rgb_ok};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn nv12_to_rgbx() {
        yuv_to_bgra_ok(PixelFormat::Nv12);
    }

    #[test]
    fn i420_to_rgbx() {
        yuv_to_bgra_ok(PixelFormat::I420);
    }

    #[test]
    fn i444_to_rgbx() {
        yuv_to_bgra_ok(PixelFormat::I444);
    }

    #[test]
    fn rgb_to_nv12() {
        rgb_to_yuv_ok(PixelFormat::Nv12);
    }

    #[test]
    fn rgb_to_i420() {
        rgb_to_yuv_ok(PixelFormat::I420);
    }

    #[test]
    fn rgb_to_i444() {
        rgb_to_yuv_ok(PixelFormat::I444);
    }

    #[test]
    fn nv12_to_rgb() {
        yuv_to_rgb_ok(PixelFormat::Nv12, PixelFormat::Rgb);
    }

    #[test]
    fn i420_to_rgb() {
        yuv_to_rgb_ok(PixelFormat::I420, PixelFormat::Rgb);
    }

    #[test]
    fn i444_to_rgb() {
        yuv_to_rgb_ok(PixelFormat::I444, PixelFormat::Rgb);
    }

    #[test]
    fn nv12_to_bgr() {
        yuv_to_rgb_ok(PixelFormat::Nv12, PixelFormat::Bgr);
    }

    #[test]
    fn i420_to_bgr() {
        yuv_to_rgb_ok(PixelFormat::I420, PixelFormat::Bgr);
    }

    #[test]
    fn i444_to_bgr() {
        yuv_to_rgb_ok(PixelFormat::I444, PixelFormat::Bgr);
    }

    #[test]
    fn rgb_to_bgra() {
        rgb_ok(PixelFormat::Rgb, PixelFormat::Bgra);
    }

    #[test]
    fn bgra_to_rgb() {
        rgb_ok(PixelFormat::Bgra, PixelFormat::Rgb);
    }

    #[test]
    fn bgr_to_rgb() {
        rgb_ok(PixelFormat::Bgr, PixelFormat::Rgb);
    }

    #[test]
    fn argb_to_rgb() {
        rgb_ok(PixelFormat::Argb, PixelFormat::Rgb);
    }
}

#[cfg(all(test, feature = "test_instruction_sets"))]
mod conversions {
    use super::*;
    use dcp::initialize_with_instruction_set;

    #[test]
    fn coverage() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        const SETS: [&str; 3] = ["x86", "sse2", "avx2"];
        #[cfg(target_arch = "aarch64")]
        const SETS: [&str; 2] = ["x86", "neon"];
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        const SETS: [&str; 1] = ["x86"];

        for set in &SETS {
            initialize_with_instruction_set(set);

            rgb_ok(PixelFormat::Bgr, PixelFormat::Rgb);
            rgb_ok(PixelFormat::Bgra, PixelFormat::Rgb);
            rgb_ok(PixelFormat::Rgb, PixelFormat::Bgra);
            rgb_ok(PixelFormat::Argb, PixelFormat::Rgb);
            rgb_to_yuv_ok(PixelFormat::I420);
            rgb_to_yuv_ok(PixelFormat::I444);
            rgb_to_yuv_ok(PixelFormat::Nv12);
            yuv_to_bgra_ok(PixelFormat::I420);
            yuv_to_bgra_ok(PixelFormat::I444);
            yuv_to_bgra_ok(PixelFormat::Nv12);
            yuv_to_rgb_ok(PixelFormat::Nv12, PixelFormat::Rgb);
            yuv_to_rgb_ok(PixelFormat::I420, PixelFormat::Rgb);
            yuv_to_rgb_ok(PixelFormat::I444, PixelFormat::Rgb);
            yuv_to_rgb_ok(PixelFormat::Nv12, PixelFormat::Bgr);
            yuv_to_rgb_ok(PixelFormat::I420, PixelFormat::Bgr);
            yuv_to_rgb_ok(PixelFormat::I444, PixelFormat::Bgr);
        }
    }
}
