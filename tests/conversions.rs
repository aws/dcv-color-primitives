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

mod common;

use common::*;
use dcp::{ColorSpace, ImageFormat, PixelFormat, STRIDE_AUTO, convert_image};

use dcv_color_primitives as dcp;
use itertools::iproduct;
use rand::Rng;
use std::borrow::Cow;
use std::sync::OnceLock;

const MAX_PLANE_WIDTH: usize = 8;
const MAX_PLANE_HEIGHT: usize = 8;
const MAX_UV_WIDTH: usize = MAX_PLANE_WIDTH >> 1;
const MAX_UV_HEIGHT: usize = MAX_PLANE_HEIGHT >> 1;

const SUPPORTED_COLOR_SPACES: &[ColorSpace] = &[
    ColorSpace::Bt601,
    ColorSpace::Bt709,
    ColorSpace::Bt601FR,
    ColorSpace::Bt709FR,
];

const YUV_RGB_MAX_WIDTH: usize = 33;
const YUV_RGB_MAX_HEIGHT: usize = 3;
const YUV_RGB_MAX_PAD: usize = 4;

static COEFFICIENTS_TABLE: OnceLock<[Coefficients; 4]> = OnceLock::new();

type FullPlane = [[u8; MAX_PLANE_WIDTH]; MAX_PLANE_HEIGHT];
type SubSampledPlane = [[u8; MAX_UV_WIDTH]; MAX_UV_HEIGHT];

const Y_BT601_REF: FullPlane = [
    [74, 78, 101, 133, 118, 135, 87, 206],
    [93, 171, 116, 94, 102, 100, 171, 122],
    [141, 74, 200, 214, 98, 132, 65, 147],
    [141, 54, 186, 175, 154, 185, 81, 93],
    [118, 92, 125, 151, 40, 164, 181, 176],
    [139, 93, 110, 112, 132, 114, 157, 64],
    [113, 188, 101, 88, 105, 93, 128, 153],
    [131, 65, 48, 76, 129, 162, 117, 141],
];

const CB_BT601_REF: FullPlane = [
    [116, 118, 204, 91, 136, 159, 97, 100],
    [99, 51, 179, 130, 161, 99, 117, 75],
    [63, 120, 143, 89, 147, 155, 184, 64],
    [118, 174, 134, 154, 133, 104, 185, 178],
    [192, 101, 191, 102, 192, 131, 81, 43],
    [60, 87, 198, 81, 175, 86, 61, 108],
    [95, 59, 122, 204, 83, 196, 124, 150],
    [184, 203, 185, 98, 140, 47, 196, 92],
];

const CR_BT601_REF: FullPlane = [
    [187, 105, 177, 198, 208, 103, 119, 135],
    [229, 162, 147, 87, 79, 221, 118, 111],
    [187, 86, 115, 119, 141, 163, 166, 169],
    [164, 126, 89, 171, 84, 83, 81, 91],
    [183, 103, 75, 82, 121, 178, 97, 165],
    [83, 210, 212, 158, 183, 60, 88, 118],
    [135, 155, 93, 182, 70, 132, 176, 107],
    [157, 133, 120, 122, 186, 153, 65, 155],
];

const CB2_BT601_REF: SubSampledPlane = [
    [96, 151, 139, 97],
    [119, 130, 135, 153],
    [110, 143, 146, 73],
    [135, 152, 116, 140],
];

const CR2_BT601_REF: SubSampledPlane = [
    [171, 152, 153, 121],
    [141, 124, 118, 127],
    [145, 132, 135, 117],
    [145, 129, 135, 126],
];

const Y_BT709_REF: FullPlane = [
    [63, 84, 82, 123, 101, 137, 93, 208],
    [75, 173, 106, 103, 108, 84, 174, 132],
    [136, 84, 201, 221, 93, 121, 51, 146],
    [134, 49, 193, 163, 163, 197, 84, 95],
    [99, 101, 129, 164, 34, 154, 193, 179],
    [157, 80, 85, 111, 115, 133, 173, 68],
    [116, 191, 109, 68, 122, 84, 118, 155],
    [118, 56, 43, 80, 116, 166, 122, 139],
];

const CB_BT709_REF: FullPlane = [
    [123, 115, 211, 98, 145, 156, 95, 100],
    [110, 53, 182, 126, 156, 109, 116, 72],
    [68, 115, 141, 87, 149, 160, 189, 67],
    [122, 174, 130, 160, 128, 98, 181, 174],
    [200, 97, 186, 96, 192, 136, 77, 45],
    [53, 95, 209, 84, 182, 78, 56, 107],
    [95, 60, 118, 212, 76, 197, 129, 148],
    [189, 205, 185, 97, 147, 48, 190, 94],
];

const CR_BT709_REF: FullPlane = [
    [187, 103, 184, 197, 211, 104, 116, 133],
    [229, 157, 151, 86, 80, 221, 117, 106],
    [184, 84, 116, 115, 143, 166, 171, 165],
    [164, 130, 89, 174, 83, 80, 84, 94],
    [189, 100, 79, 78, 125, 180, 92, 160],
    [77, 209, 219, 155, 188, 56, 82, 117],
    [132, 151, 92, 189, 66, 137, 176, 108],
    [162, 139, 124, 120, 189, 148, 69, 153],
];

const CB2_BT709_REF: SubSampledPlane = [
    [100, 154, 142, 96],
    [120, 130, 134, 153],
    [112, 144, 147, 71],
    [137, 153, 117, 140],
];

const CR2_BT709_REF: SubSampledPlane = [
    [169, 154, 154, 118],
    [140, 124, 118, 129],
    [144, 133, 137, 113],
    [146, 131, 135, 127],
];

const Y_BT601FR_REF: FullPlane = [
    [67, 72, 99, 136, 119, 138, 83, 222],
    [90, 181, 116, 91, 100, 98, 180, 123],
    [146, 68, 214, 231, 95, 135, 57, 153],
    [145, 45, 198, 185, 161, 196, 75, 90],
    [119, 89, 127, 157, 28, 173, 192, 187],
    [144, 89, 110, 112, 135, 114, 165, 55],
    [113, 201, 99, 84, 104, 90, 130, 159],
    [134, 57, 37, 69, 132, 170, 118, 145],
];

const CB_BT601FR_REF: FullPlane = [
    [115, 116, 214, 86, 137, 163, 92, 96],
    [95, 40, 186, 131, 166, 95, 116, 68],
    [54, 119, 145, 84, 149, 159, 192, 55],
    [117, 180, 135, 158, 133, 101, 193, 185],
    [201, 97, 199, 99, 201, 131, 75, 31],
    [50, 81, 208, 75, 182, 81, 52, 106],
    [90, 49, 121, 215, 77, 205, 123, 153],
    [192, 213, 192, 94, 142, 36, 206, 87],
];

const CR_BT601FR_REF: FullPlane = [
    [195, 102, 183, 208, 219, 99, 117, 136],
    [243, 167, 149, 81, 72, 234, 116, 108],
    [195, 80, 113, 117, 143, 167, 172, 174],
    [169, 126, 84, 177, 78, 77, 75, 86],
    [191, 99, 68, 75, 119, 185, 92, 170],
    [77, 221, 224, 162, 190, 51, 83, 117],
    [136, 159, 88, 189, 63, 133, 182, 104],
    [161, 134, 119, 121, 194, 157, 56, 159],
];

const CB2_BT601FR_REF: SubSampledPlane = [
    [92, 154, 140, 93],
    [118, 130, 136, 156],
    [107, 145, 149, 66],
    [136, 156, 115, 142],
];

const CR2_BT601FR_REF: SubSampledPlane = [
    [177, 155, 156, 120],
    [142, 123, 116, 127],
    [147, 132, 136, 116],
    [147, 129, 137, 125],
];

const Y_BT709FR_REF: FullPlane = [
    [55, 79, 77, 124, 99, 140, 90, 224],
    [69, 183, 105, 101, 108, 80, 184, 135],
    [140, 79, 215, 239, 89, 123, 40, 152],
    [138, 39, 206, 171, 171, 210, 79, 92],
    [97, 98, 131, 172, 21, 160, 206, 189],
    [164, 75, 80, 111, 115, 136, 183, 60],
    [116, 204, 108, 60, 124, 80, 119, 162],
    [119, 46, 31, 75, 116, 175, 124, 143],
];

const CB_BT709FR_REF: FullPlane = [
    [122, 113, 222, 94, 148, 160, 91, 96],
    [108, 43, 189, 125, 160, 107, 114, 65],
    [60, 114, 143, 82, 152, 164, 198, 59],
    [122, 181, 130, 164, 128, 94, 188, 181],
    [210, 93, 194, 92, 201, 138, 70, 34],
    [43, 91, 221, 77, 190, 71, 45, 104],
    [90, 51, 116, 224, 69, 207, 130, 150],
    [197, 215, 193, 92, 149, 37, 199, 90],
];

const CR_BT709FR_REF: FullPlane = [
    [196, 100, 191, 206, 222, 101, 114, 134],
    [243, 161, 154, 80, 74, 234, 115, 103],
    [191, 78, 114, 114, 145, 171, 177, 170],
    [169, 130, 83, 181, 77, 74, 78, 89],
    [198, 96, 72, 72, 125, 187, 87, 164],
    [70, 220, 232, 159, 196, 46, 76, 115],
    [133, 154, 86, 198, 57, 138, 183, 105],
    [167, 140, 124, 119, 197, 151, 60, 157],
];

const CB2_BT709FR_REF: SubSampledPlane = [
    [97, 158, 144, 91],
    [119, 130, 134, 156],
    [109, 146, 150, 63],
    [138, 156, 116, 142],
];

const CR2_BT709FR_REF: SubSampledPlane = [
    [175, 158, 158, 117],
    [142, 123, 117, 129],
    [146, 134, 138, 111],
    [148, 132, 136, 126],
];

// Largest group that uses neither avx2 nor sse2 is 62x64.
// We can arrange the image as blocks:
// y0 y0 | y1 y1 | ...
// y0 y0 | y1 y1 | ...
// u0 v0 | u1 v1 | ...
//
// Min width to represent 8 colors below: 2 * 8 = 16x16
//
// Color table (bt601):
//             r    g    b
// black    (  0,   0,   0):    16, 128, 128
// red      (255,   0,   0):    82,  90, 240
// green    (  0, 255,   0):   145,  54,  34
// yellow   (255, 255,   0):   210,  16, 146
// blue     (  0,   0, 255):    41, 240, 110
// magenta  (255,   0, 255):   107, 202, 222
// cyan     (  0, 255, 255):   169, 166,  16
// white    (255, 255, 255):   235, 128, 128
//
// Color table (bt709):
//             r    g    b
// black    (  0,   0,   0):    16, 128, 128
// red      (255,   0,   0):    63, 102, 240
// green    (  0, 255,   0):   173,  42,  26
// yellow   (255, 255,   0):   219,  16, 138
// blue     (  0,   0, 255):    32, 240, 118
// magenta  (255,   0, 255):    78, 214, 230
// cyan     (  0, 255, 255):   188, 154,  16
// white    (255, 255, 255):   235, 128, 128
//
// Color table (bt601 full range):
//             r    g    b
// black    (  0,   0,   0):     0, 128, 128
// red      (255,   0,   0):    76,  84, 255
// green    (  0, 255,   0):   149,  43,  21
// yellow   (255, 255,   0):   225,   0, 148
// blue     (  0,   0, 255):    29, 255, 107
// magenta  (255,   0, 255):   105, 212, 234
// cyan     (  0, 255, 255):   178, 171,   0
// white    (255, 255, 255):   255, 128, 128
//
// Color table (bt709 full range):
//             r    g    b
// black    (  0,   0,   0):     0, 128, 128
// red      (255,   0,   0):    54,  98, 255
// green    (  0, 255,   0):   182,  29,  12
// yellow   (255, 255,   0):   237,   0, 139
// blue     (  0,   0, 255):    18, 255, 116
// magenta  (255,   0, 255):    73, 226, 243
// cyan     (  0, 255, 255):   201, 157,   0
// white    (255, 255, 255):   255, 128, 128
const Y_SRC: [[u8; 8]; 4] = [
    [16, 82, 145, 210, 41, 107, 169, 235],
    [16, 63, 173, 219, 32, 78, 188, 235],
    [0, 76, 149, 225, 29, 105, 178, 255],
    [0, 54, 182, 237, 18, 73, 201, 255],
];

const U_SRC: [[u8; 8]; 4] = [
    [128, 90, 54, 16, 240, 202, 166, 128],
    [128, 102, 42, 16, 240, 214, 154, 128],
    [128, 84, 43, 0, 255, 212, 171, 128],
    [128, 98, 29, 0, 255, 226, 157, 128],
];

const V_SRC: [[u8; 8]; 4] = [
    [128, 240, 34, 146, 110, 222, 16, 128],
    [128, 240, 26, 138, 118, 230, 16, 128],
    [128, 255, 21, 148, 107, 234, 0, 128],
    [128, 255, 12, 139, 116, 243, 0, 128],
];

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
    for row in plane.chunks_exact_mut(stride).take(height) {
        for (pos, val) in row.iter_mut().enumerate().take(width) {
            *val = src[(pos >> shift) & 0x7];
        }
    }
}

fn fill_planar_luma(plane: &mut [u8], src: [u8; 8], width: usize, height: usize, stride: usize) {
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
        ColorSpace::Bt601FR => 2,
        _ => 3,
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
        let mut coefficients = [((0, 0, 0), (0, 0), 0, 0); 4];

        for (index, coeff_row) in coefficients.iter_mut().enumerate() {
            let full_range = index >> 1;
            let model = index & 1;
            let (kr, kg, kb) = if model == 0 {
                (0.299, 0.587, 0.114)
            } else {
                (0.2126, 0.7152, 0.0722)
            };
            let (y_min, y_max, c_min, c_max) = if full_range == 1 {
                (0, 255, 0, 255)
            } else {
                (16, 235, 16, 240)
            };
            let (y_scale, c_scale) = if full_range == 1 {
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

            let r = p00[0] + p10[0] + p01[0] + p11[0];
            let g = p00[1] + p10[1] + p01[1] + p11[1];
            let b = p00[2] + p10[2] + p01[2] + p11[2];
            let u = ((yr * r + yg * g + yb * b + UV_SHIFT_18) >> FIX18) as u8;
            let v = ((zr * r + zg * g + zb * b + UV_SHIFT_18) >> FIX18) as u8;

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
    let PlaneRef::Full(reference) = reference else {
        return;
    };

    for (row, exp) in plane.chunks(stride).zip(reference.iter()) {
        let (payload, pad) = row.split_at(width);
        assert!(payload.iter().zip(exp).all(|(&x, &y)| x == y));
        assert!(pad.iter().all(|&x| x == 0));
    }
}

fn check_subsampled_plane(plane: &[u8], reference: &PlaneRef, width: usize, stride: usize) {
    let PlaneRef::SubSampled(reference) = reference else {
        return;
    };

    for (row, exp) in plane.chunks(stride).zip(reference.iter()) {
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
                .chunks(u_stride)
                .zip(u_ref.iter().zip(v_ref.iter()))
            {
                let (payload, pad) = uv_row.split_at(2 * cw);
                assert!(
                    payload
                        .chunks(2)
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

    if dst_format.num_planes == 2 {
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
        assert_eq!(dst_format.num_planes, 3);
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
                num_planes: 1,
            },
            dst_image_format,
            plane_ref,
        );
    }
}

fn rgb_to_yuv_ok(pixel_format: PixelFormat, num_planes: u32) {
    for color_space in SUPPORTED_COLOR_SPACES {
        let format = ImageFormat {
            pixel_format,
            color_space: *color_space,
            num_planes,
        };

        for (width, height) in iproduct!(0..=MAX_PLANE_WIDTH, 0..=MAX_PLANE_HEIGHT) {
            let y_plane = match color_space {
                ColorSpace::Bt601 => &Y_BT601_REF,
                ColorSpace::Bt709 => &Y_BT709_REF,
                ColorSpace::Bt601FR => &Y_BT601FR_REF,
                _ => &Y_BT709FR_REF,
            };
            let (u_plane, v_plane) = if let PixelFormat::I444 = pixel_format {
                let (u_plane, v_plane) = match color_space {
                    ColorSpace::Bt601 => (&CB_BT601_REF, &CR_BT601_REF),
                    ColorSpace::Bt709 => (&CB_BT709_REF, &CR_BT709_REF),
                    ColorSpace::Bt601FR => (&CB_BT601FR_REF, &CR_BT601FR_REF),
                    _ => (&CB_BT709FR_REF, &CR_BT709FR_REF),
                };

                (
                    PlaneRef::Full(Cow::Borrowed(u_plane)),
                    PlaneRef::Full(Cow::Borrowed(v_plane)),
                )
            } else if (width & 1) == 0 && (height & 1) == 0 {
                let (u_plane, v_plane) = match color_space {
                    ColorSpace::Bt601 => (&CB2_BT601_REF, &CR2_BT601_REF),
                    ColorSpace::Bt709 => (&CB2_BT709_REF, &CR2_BT709_REF),
                    ColorSpace::Bt601FR => (&CB2_BT601FR_REF, &CR2_BT601FR_REF),
                    _ => (&CB2_BT709FR_REF, &CR2_BT709FR_REF),
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
    if src_format.num_planes == 2 {
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
        assert_eq!(src_format.num_planes, 3);
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

fn yuv_to_bgra_ok(pixel_format: PixelFormat, num_planes: u32) {
    const SUPPORTED_FORMATS: &[PixelFormat] = &[PixelFormat::Bgra, PixelFormat::Rgba];

    for (color_space, format) in iproduct!(SUPPORTED_COLOR_SPACES, SUPPORTED_FORMATS) {
        let src_format = ImageFormat {
            pixel_format,
            color_space: *color_space,
            num_planes,
        };
        let dst_format = ImageFormat {
            pixel_format: *format,
            color_space: ColorSpace::Rgb,
            num_planes: 1,
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
    pad: (usize, usize, usize),
) {
    let (y_pad, u_pad, dst_pad) = pad;
    let w = image_size.0;
    let h = image_size.1;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);

    let y_stride = w + y_pad;
    let u_stride = 2 * cw + u_pad;
    let y_size = y_stride * h;
    let src_size = y_size + u_stride * ch;

    let color_space_index = get_color_space_index(src_format.color_space);

    let mut src_image = vec![0_u8; src_size];
    if w > 0 && h > 0 {
        fill_planar_luma(&mut src_image, Y_SRC[color_space_index], cw, h, y_stride);
        fill_biplanar_chroma(
            &mut src_image[y_size..],
            U_SRC[color_space_index],
            V_SRC[color_space_index],
            cw,
            ch,
            u_stride,
        );
    }

    let dst_stride = w * 3 + dst_pad;
    let dst_size = dst_stride * h;
    let mut dst_image = vec![0_u8; dst_size];
    let dst_stride = if dst_pad == 0 {
        STRIDE_AUTO
    } else {
        dst_stride
    };

    let (first, last) = src_image.split_at(y_size);
    let src_strides = [
        if y_pad == 0 { STRIDE_AUTO } else { y_stride },
        if u_pad == 0 { STRIDE_AUTO } else { u_stride },
    ];
    let src_buffers = [first, last];

    let mut expected_row = vec![0_i32; 3 * w];
    for (x, pixel) in expected_row.chunks_exact_mut(3).enumerate() {
        let index = (x >> 1) & 7;

        pixel[0] = if (index & 1) == 0 { 0 } else { 255 };
        pixel[1] = if ((index >> 1) & 1) == 0 { 0 } else { 255 };
        pixel[2] = if ((index >> 2) & 1) == 0 { 0 } else { 255 };
    }

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
    for (y_pad, uv_pad, dst_pad) in
        iproduct!(0..YUV_RGB_MAX_PAD, 0..YUV_RGB_MAX_PAD, 0..YUV_RGB_MAX_PAD)
    {
        yuv_to_rgb_size_format_mode_stride(
            image_size,
            src_format,
            dst_format,
            (y_pad, uv_pad, dst_pad),
        );
    }
}

fn yuv_to_rgb_ok() {
    let dst_format = ImageFormat {
        pixel_format: PixelFormat::Rgb,
        color_space: ColorSpace::Rgb,
        num_planes: 1,
    };

    for color_space in SUPPORTED_COLOR_SPACES {
        let src_format = ImageFormat {
            pixel_format: PixelFormat::Nv12,
            color_space: *color_space,
            num_planes: 2,
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
        num_planes: 1,
    };
    let dst_format = ImageFormat {
        pixel_format: dst_pixel_format,
        color_space: ColorSpace::Rgb,
        num_planes: 1,
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

        let mut src_image = vec![0_u8; src_stride * h];
        let mut dst_image = vec![0_u8; dst_stride * h];
        for y in 0..h {
            for x in 0..w {
                let offset = y * src_stride + x * src_depth;

                src_image[offset] = rng.random::<u8>();
                src_image[offset + 1] = rng.random::<u8>();
                src_image[offset + 2] = rng.random::<u8>();
                if src_depth == 4 {
                    src_image[offset + 3] = 255;
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

                assert_eq!(dst_image[output_index], src_image[input_index + 2]);
                assert_eq!(dst_image[output_index + 1], src_image[input_index + 1]);
                assert_eq!(dst_image[output_index + 2], src_image[input_index]);
                if dst_depth == 4 {
                    assert_eq!(dst_image[output_index + 3], 255);
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
        yuv_to_bgra_ok(PixelFormat::Nv12, 2);
    }

    #[test]
    fn i420_to_rgbx() {
        yuv_to_bgra_ok(PixelFormat::I420, 3);
    }

    #[test]
    fn i444_to_rgbx() {
        yuv_to_bgra_ok(PixelFormat::I444, 3);
    }

    #[test]
    fn rgb_to_nv12() {
        rgb_to_yuv_ok(PixelFormat::Nv12, 2);
    }

    #[test]
    fn rgb_to_i420() {
        rgb_to_yuv_ok(PixelFormat::I420, 3);
    }

    #[test]
    fn rgb_to_i444() {
        rgb_to_yuv_ok(PixelFormat::I444, 3);
    }

    #[test]
    fn nv12_to_rgb() {
        yuv_to_rgb_ok();
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
            rgb_to_yuv_ok(PixelFormat::I420, 3);
            rgb_to_yuv_ok(PixelFormat::I444, 3);
            rgb_to_yuv_ok(PixelFormat::Nv12, 2);
            yuv_to_bgra_ok(PixelFormat::I420, 3);
            yuv_to_bgra_ok(PixelFormat::I444, 3);
            yuv_to_bgra_ok(PixelFormat::Nv12, 2);
            yuv_to_rgb_ok();
        }
    }
}
