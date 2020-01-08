#![warn(unused)]
#![deny(trivial_casts)]
#![deny(trivial_numeric_casts)]
#![deny(unsafe_code)]
#![deny(unstable_features)]
#![deny(unused_import_braces)]

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::slice::*;

use dcp::*;
use dcv_color_primitives as dcp;
use rand::random;

const MAX_NUMBER_OF_PLANES: u32 = 3;

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

const COLOR_SPACES: &[ColorSpace; 3] = &[ColorSpace::Lrgb, ColorSpace::Bt601, ColorSpace::Bt709];

const PIXEL_FORMAT_I444: u32 = PixelFormat::I444 as u32;
const PIXEL_FORMAT_I422: u32 = PixelFormat::I422 as u32;
const PIXEL_FORMAT_I420: u32 = PixelFormat::I420 as u32;
const COLOR_SPACE_LRGB: u32 = ColorSpace::Lrgb as u32;
const PIXEL_FORMAT_ARGB: u32 = PixelFormat::Argb as u32;
const PIXEL_FORMAT_BGRA: u32 = PixelFormat::Bgra as u32;
const PIXEL_FORMAT_BGR: u32 = PixelFormat::Bgr as u32;

const RGB_TO_YUV_INPUT: [[[u8; 4]; 8]; 8] = [
    [
        [161, 24, 44, 58],
        [35, 95, 51, 205],
        [177, 30, 252, 158],
        [248, 94, 62, 28],
        [247, 51, 135, 38],
        [98, 147, 200, 127],
        [68, 103, 20, 124],
        [233, 227, 165, 0],
    ],
    [
        [251, 19, 32, 170],
        [235, 183, 25, 77],
        [146, 81, 218, 161],
        [25, 124, 96, 56],
        [22, 127, 167, 179],
        [247, 34, 40, 53],
        [164, 193, 159, 24],
        [96, 158, 17, 223],
    ],
    [
        [240, 123, 14, 108],
        [0, 105, 52, 116],
        [194, 219, 244, 47],
        [216, 254, 153, 84],
        [116, 77, 133, 68],
        [190, 96, 190, 133],
        [118, 4, 170, 115],
        [218, 145, 23, 50],
    ],
    [
        [202, 120, 126, 231],
        [42, 28, 137, 40],
        [136, 227, 210, 177],
        [254, 140, 238, 88],
        [90, 195, 170, 67],
        [125, 242, 148, 88],
        [1, 91, 190, 245],
        [31, 100, 190, 225],
    ],
    [
        [207, 49, 249, 131],
        [48, 120, 34, 82],
        [43, 145, 253, 141],
        [83, 205, 105, 44],
        [16, 9, 157, 22],
        [253, 131, 178, 148],
        [142, 236, 98, 6],
        [246, 190, 15, 213],
    ],
    [
        [72, 207, 6, 168],
        [220, 39, 6, 219],
        [244, 14, 252, 45],
        [159, 106, 17, 184],
        [222, 72, 230, 39],
        [6, 185, 30, 35],
        [101, 223, 30, 14],
        [40, 71, 16, 244],
    ],
    [
        [124, 121, 46, 190],
        [244, 206, 61, 169],
        [43, 130, 87, 247],
        [170, 10, 238, 229],
        [12, 168, 14, 220],
        [96, 60, 226, 235],
        [206, 93, 122, 117],
        [126, 168, 203, 39],
    ],
    [
        [181, 88, 248, 45],
        [65, 24, 208, 166],
        [24, 21, 151, 85],
        [60, 86, 9, 153],
        [225, 80, 156, 159],
        [210, 181, 6, 214],
        [17, 142, 255, 163],
        [189, 137, 72, 87],
    ],
];

const RGB_TO_YUV_BT601_OUTPUT: [[u8; 8]; 12] = [
    [74, 78, 101, 133, 118, 135, 87, 206],
    [93, 171, 116, 94, 102, 100, 171, 122],
    [141, 74, 200, 214, 98, 132, 65, 147],
    [141, 54, 186, 175, 154, 185, 81, 93],
    [118, 92, 125, 151, 40, 164, 181, 176],
    [139, 93, 110, 112, 132, 114, 157, 64],
    [113, 188, 101, 88, 105, 93, 128, 153],
    [131, 65, 48, 76, 129, 162, 117, 141],
    [96, 171, 151, 152, 139, 153, 97, 121],
    [119, 141, 130, 124, 135, 118, 153, 127],
    [110, 145, 143, 132, 146, 135, 73, 117],
    [135, 145, 152, 129, 116, 135, 140, 126],
];

const RGB_TO_YUV_BT709_OUTPUT: [[u8; 8]; 12] = [
    [63, 84, 82, 123, 101, 137, 93, 208],
    [75, 173, 106, 103, 108, 84, 174, 132],
    [136, 84, 201, 221, 93, 121, 51, 146],
    [134, 49, 193, 163, 163, 197, 84, 95],
    [99, 101, 129, 164, 34, 154, 193, 179],
    [157, 80, 85, 111, 115, 133, 173, 68],
    [116, 191, 109, 68, 122, 84, 118, 155],
    [118, 56, 43, 80, 116, 166, 122, 139],
    [100, 169, 154, 154, 142, 154, 96, 118],
    [120, 140, 130, 124, 134, 118, 153, 129],
    [112, 144, 144, 133, 147, 137, 71, 113],
    [137, 146, 153, 131, 117, 135, 140, 127],
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
const Y_TO_RGB_INPUT: [[u8; 8]; 2] = [
    [16, 82, 145, 210, 41, 107, 169, 235],
    [16, 63, 173, 219, 32, 78, 188, 235],
];

const CB_TO_RGB_INPUT: [[u8; 8]; 2] = [
    [128, 90, 54, 16, 240, 202, 166, 128],
    [128, 102, 42, 16, 240, 214, 154, 128],
];

const CR_TO_RGB_INPUT: [[u8; 8]; 2] = [
    [128, 240, 34, 146, 110, 222, 16, 128],
    [128, 240, 26, 138, 118, 230, 16, 128],
];

const NUM_LOG2_DEN: [[usize; 2]; 9] = [
    [4, 0],
    [4, 0],
    [3, 0],
    [4, 0],
    [3, 0],
    [3, 0],
    [2, 0],
    [3, 1],
    [3, 1],
];

const NUM_LOG2_DEN_PER_PLANE: [[usize; (2 * MAX_NUMBER_OF_PLANES) as usize]; 9] = [
    [4, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 0],
    [3, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 0],
    [3, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 1, 1, 1, 1],
    [1, 0, 1, 2, 1, 2],
    [1, 0, 1, 1, 0, 0],
];

macro_rules! set_expected {
    ($var:ident, $pred:expr, $status:path) => {
        if $var.is_ok() && $pred {
            $var = Err($status);
        }
    };
}

fn bootstrap() {
    let desc = describe_acceleration();
    if desc.is_err() {
        initialize();

        let desc = describe_acceleration();
        assert_eq!(desc.is_ok(), true);
        match desc {
            Ok(s) => println!("{}", s),
            Err(_) => assert!(false),
        }
    }
}

#[test]
fn init() {
    bootstrap();
}

fn rgb_to_yuv_size_mode_stride(
    num_planes: u32,
    width: u32,
    height: u32,
    color_space: ColorSpace,
    pixel_format: PixelFormat,
    src_stride: usize,
    luma_stride: usize,
    chroma_stride: usize,
) {
    let depth: usize = match pixel_format {
        PixelFormat::Bgr => 3,
        _ => 4,
    };

    let w = width as usize;
    let h = height as usize;

    let chroma_height = h / 2;
    let in_size = src_stride * h;
    let out_size = (luma_stride * h) + (chroma_stride * chroma_height);
    let min_src_stride = w * depth;
    let src_fill_bytes = src_stride - min_src_stride;
    let luma_fill_bytes = luma_stride - w;
    let chroma_fill_bytes = chroma_stride - w;

    let src_format = ImageFormat {
        pixel_format,
        color_space: ColorSpace::Lrgb,
        num_planes: 1,
    };

    let dst_format = ImageFormat {
        pixel_format: PixelFormat::Nv12,
        color_space,
        num_planes,
    };

    // Allocate and initialize input
    let mut test_input: Box<[u8]> = vec![0u8; in_size].into_boxed_slice();
    for y in 0..h {
        for x in 0..w {
            let p = y * src_stride + x * depth;
            match pixel_format {
                PixelFormat::Argb => {
                    test_input[p] = RGB_TO_YUV_INPUT[y][x][3];
                    test_input[p + 1] = RGB_TO_YUV_INPUT[y][x][0];
                    test_input[p + 2] = RGB_TO_YUV_INPUT[y][x][1];
                    test_input[p + 3] = RGB_TO_YUV_INPUT[y][x][2];
                }
                PixelFormat::Bgra => {
                    test_input[p] = RGB_TO_YUV_INPUT[y][x][2];
                    test_input[p + 1] = RGB_TO_YUV_INPUT[y][x][1];
                    test_input[p + 2] = RGB_TO_YUV_INPUT[y][x][0];
                    test_input[p + 3] = RGB_TO_YUV_INPUT[y][x][3];
                }
                _ => {
                    test_input[p] = RGB_TO_YUV_INPUT[y][x][2];
                    test_input[p + 1] = RGB_TO_YUV_INPUT[y][x][1];
                    test_input[p + 2] = RGB_TO_YUV_INPUT[y][x][0];
                }
            }
        }
    }

    // Allocate output
    let mut test_output: Box<[u8]> = vec![0u8; out_size].into_boxed_slice();

    // Compute strides
    let src_stride = if src_fill_bytes == 0 {
        STRIDE_AUTO
    } else {
        src_stride
    };

    let mut dst_strides: Vec<usize> = Vec::with_capacity(2);
    dst_strides.push(if luma_fill_bytes == 0 {
        STRIDE_AUTO
    } else {
        luma_stride
    });

    let mut dst_buffers: Vec<&mut [u8]> = Vec::with_capacity(2);

    if num_planes == 1 {
        dst_buffers.push(&mut test_output);
    } else {
        dst_strides.push(if chroma_fill_bytes == 0 {
            STRIDE_AUTO
        } else {
            chroma_stride
        });

        let (first, last) = test_output.split_at_mut(luma_stride * h);
        dst_buffers.push(first);
        dst_buffers.push(last);
    }

    match convert_image(
        width,
        height,
        &src_format,
        Some(&[src_stride]),
        &[&test_input[..]],
        &dst_format,
        Some(&dst_strides[..]),
        &mut dst_buffers[..],
    ) {
        Err(_) => assert!(false),
        Ok(_) => {
            let mut i = 0usize;

            let reference = match color_space {
                ColorSpace::Bt601 => &RGB_TO_YUV_BT601_OUTPUT,
                _ => &RGB_TO_YUV_BT709_OUTPUT,
            };

            // Check all luma samples are correct
            for y in 0..h {
                for x in 0..w {
                    assert!(test_output[i] == reference[y][x]);
                    i += 1;
                }

                for _x in 0..luma_fill_bytes {
                    assert!(test_output[i] == 0);
                    i += 1;
                }
            }

            // Check all chroma samples are correct
            for y in 0..chroma_height {
                for x in 0..w {
                    assert!(test_output[i] == reference[8 + y][x]);
                    i += 1;
                }

                for _x in 0..chroma_fill_bytes {
                    assert!(test_output[i] == 0);
                    i += 1;
                }
            }

            // Rest must be identically null
            while i < out_size {
                assert!(test_output[i] == 0);
                i += 1;
            }
        }
    }
}

fn rgb_to_yuv_size_mode(
    num_planes: u32,
    width: u32,
    height: u32,
    color_space: ColorSpace,
    pixel_format: PixelFormat,
) {
    const MAX_FILL_BYTES: usize = 15;

    let w = width as usize;
    let depth: usize = match pixel_format {
        PixelFormat::Bgr => 3,
        _ => 4,
    };

    let min_src_stride = w * depth;
    let max_src_stride = min_src_stride + MAX_FILL_BYTES;
    let min_dst_stride = w;
    let max_dst_stride = min_dst_stride + MAX_FILL_BYTES;

    if num_planes == 1 {
        for luma_stride in min_dst_stride..=max_dst_stride {
            for src_stride in min_src_stride..=max_src_stride {
                rgb_to_yuv_size_mode_stride(
                    num_planes,
                    width,
                    height,
                    color_space,
                    pixel_format,
                    src_stride,
                    luma_stride,
                    luma_stride,
                );
            }
        }
    } else {
        for luma_stride in min_dst_stride..=max_dst_stride {
            for chroma_stride in min_dst_stride..=max_dst_stride {
                for src_stride in min_src_stride..=max_src_stride {
                    rgb_to_yuv_size_mode_stride(
                        num_planes,
                        width,
                        height,
                        color_space,
                        pixel_format,
                        src_stride,
                        luma_stride,
                        chroma_stride,
                    );
                }
            }
        }
    }
}

fn rgb_to_yuv_size(num_planes: u32, width: u32, height: u32) {
    const SUPPORTED_PIXEL_FORMATS: &[PixelFormat] =
        &[PixelFormat::Argb, PixelFormat::Bgra, PixelFormat::Bgr];

    const SUPPORTED_COLOR_SPACES: &[ColorSpace] = &[ColorSpace::Bt601, ColorSpace::Bt709];

    for color_space in SUPPORTED_COLOR_SPACES.iter() {
        for pixel_format in SUPPORTED_PIXEL_FORMATS.iter() {
            rgb_to_yuv_size_mode(num_planes, width, height, *color_space, *pixel_format);
        }
    }
}

fn yuv_to_rgb_size_mode_stride(
    num_planes: u32,
    width: u32,
    height: u32,
    color_space: ColorSpace,
    luma_stride: usize,
    chroma_stride: usize,
    dst_stride: usize,
) {
    let w = width as usize;
    let h = height as usize;

    let chroma_height = h / 2;
    let in_size = (luma_stride * h) + (chroma_stride * chroma_height);
    let out_size = dst_stride * h;
    let luma_fill_bytes = luma_stride - w;
    let chroma_fill_bytes = chroma_stride - w;
    let min_dst_stride = w * 4;
    let dst_fill_bytes = dst_stride - min_dst_stride;

    let color_space_index: usize = match color_space {
        ColorSpace::Bt601 => 0,
        _ => 1,
    };

    let src_format = ImageFormat {
        pixel_format: PixelFormat::Nv12,
        color_space,
        num_planes,
    };

    let dst_format = ImageFormat {
        pixel_format: PixelFormat::Bgra,
        color_space: ColorSpace::Lrgb,
        num_planes: 1,
    };

    // Allocate and initialize input
    let mut test_input: Box<[u8]> = vec![0u8; in_size].into_boxed_slice();
    let mut i = 0usize;

    for _y in 0..h {
        for x in (0..w).step_by(2) {
            let index = (x >> 1) & 0x7;
            let luma = Y_TO_RGB_INPUT[color_space_index][index];
            test_input[i] = luma;
            test_input[i + 1] = luma;
            i += 2;
        }

        i += luma_fill_bytes;
    }

    for _y in 0..chroma_height {
        for x in (0..w).step_by(2) {
            let index = (x >> 1) & 0x7;
            test_input[i] = CB_TO_RGB_INPUT[color_space_index][index];
            test_input[i + 1] = CR_TO_RGB_INPUT[color_space_index][index];
            i += 2;
        }

        i += chroma_fill_bytes;
    }

    // Allocate output
    let mut test_output: Box<[u8]> = vec![0u8; out_size].into_boxed_slice();

    // Compute strides
    let dst_stride = if dst_fill_bytes == 0 {
        STRIDE_AUTO
    } else {
        dst_stride
    };

    let mut src_strides: Vec<usize> = Vec::with_capacity(2);
    src_strides.push(if luma_fill_bytes == 0 {
        STRIDE_AUTO
    } else {
        luma_stride
    });

    let mut src_buffers: Vec<&[u8]> = Vec::with_capacity(2);

    if num_planes == 1 {
        src_buffers.push(&test_input);
    } else {
        src_strides.push(if chroma_fill_bytes == 0 {
            STRIDE_AUTO
        } else {
            chroma_stride
        });

        let (first, last) = test_input.split_at(luma_stride * h);
        src_buffers.push(first);
        src_buffers.push(last);
    }

    match convert_image(
        width,
        height,
        &src_format,
        Some(&src_strides[..]),
        &src_buffers[..],
        &dst_format,
        Some(&[dst_stride]),
        &mut [&mut test_output[..]],
    ) {
        Err(_) => assert!(false),
        Ok(_) => {
            let mut i = 0usize;

            for _y in 0..h {
                for x in 0..w {
                    let index = (x >> 1) & 7;
                    let exp_b = if ((index >> 2) & 1) == 0 { 0 } else { 255 };
                    let exp_g = if ((index >> 1) & 1) == 0 { 0 } else { 255 };
                    let exp_r = if (index & 1) == 0 { 0 } else { 255 };

                    assert!(((test_output[i] as i32) - exp_b).abs() <= 2);
                    assert!(((test_output[i + 1] as i32) - exp_g).abs() <= 2);
                    assert!(((test_output[i + 2] as i32) - exp_r).abs() <= 2);
                    assert_eq!(test_output[i + 3], 255);

                    i += 4;
                }

                for _x in 0..dst_fill_bytes {
                    assert_eq!(test_output[i], 0);
                    i += 1;
                }
            }

            // Rest must be identically null
            while i < out_size {
                assert_eq!(test_output[i], 0);
                i += 1;
            }
        }
    }
}

fn yuv_to_rgb_size_mode(num_planes: u32, width: u32, height: u32, color_space: ColorSpace) {
    const MAX_FILL_BYTES: usize = 15;

    let w = width as usize;
    let min_src_stride = w;
    let max_src_stride = min_src_stride + MAX_FILL_BYTES;
    let min_dst_stride = w * 4;
    let max_dst_stride = min_dst_stride + MAX_FILL_BYTES;

    if num_planes == 1 {
        for luma_stride in min_src_stride..=max_src_stride {
            for dst_stride in min_dst_stride..=max_dst_stride {
                yuv_to_rgb_size_mode_stride(
                    num_planes,
                    width,
                    height,
                    color_space,
                    luma_stride,
                    luma_stride,
                    dst_stride,
                );
            }
        }
    } else {
        for luma_stride in min_src_stride..=max_src_stride {
            for chroma_stride in min_src_stride..=max_src_stride {
                for dst_stride in min_dst_stride..=max_dst_stride {
                    yuv_to_rgb_size_mode_stride(
                        num_planes,
                        width,
                        height,
                        color_space,
                        luma_stride,
                        chroma_stride,
                        dst_stride,
                    );
                }
            }
        }
    }
}

fn yuv_to_rgb_size(num_planes: u32, width: u32, height: u32) {
    const SUPPORTED_COLOR_SPACES: &[ColorSpace] = &[ColorSpace::Bt601, ColorSpace::Bt709];

    for color_space in SUPPORTED_COLOR_SPACES.iter() {
        yuv_to_rgb_size_mode(num_planes, width, height, *color_space);
    }
}

#[test]
fn rgb_to_yuv_ok() {
    bootstrap();

    const MAX_WIDTH: u32 = 8;
    const MAX_HEIGHT: u32 = 8;

    for planes in 1..2 {
        for width in (2..=MAX_WIDTH).step_by(2) {
            for height in (2..=MAX_HEIGHT).step_by(2) {
                rgb_to_yuv_size(planes, width, height);
            }
        }
    }
}

#[test]
fn yuv_to_rgb_ok() {
    const MAX_WIDTH: u32 = 64;
    const MAX_HEIGHT: u32 = 64;

    for num_planes in 1..2 {
        for width in (2..MAX_WIDTH).step_by(2) {
            for height in (2..MAX_HEIGHT).step_by(2) {
                yuv_to_rgb_size(num_planes, width, height);
            }
        }
    }
}

#[test]
fn rgb_to_yuv_errors() {
    bootstrap();

    const WIDTH: u32 = 2;
    const HEIGHT: u32 = 2;
    const CHROMA_HEIGHT: u32 = HEIGHT / 2;
    const SRC_STRIDE: usize = (WIDTH as usize) * 4;
    const IN_SIZE: usize = SRC_STRIDE * (HEIGHT as usize);
    const OUT_SIZE: usize = (WIDTH as usize) * ((HEIGHT as usize) + (CHROMA_HEIGHT as usize));

    // Allocate and initialize input
    let test_input: &[u8] = &[0u8; IN_SIZE];

    // Allocate and initialize output
    let test_output_p0: &mut [u8] = &mut [0u8; OUT_SIZE];
    let test_output_p1: &mut [u8] = &mut [0u8; OUT_SIZE];

    for num_planes in 0u32..=3 {
        // Only 1 and 2 are valid values
        for src_pixel_format in PIXEL_FORMATS.iter() {
            for src_color_space in COLOR_SPACES.iter() {
                for dst_color_space in COLOR_SPACES.iter() {
                    let src_format = ImageFormat {
                        pixel_format: *src_pixel_format,
                        color_space: *src_color_space,
                        num_planes: 1,
                    };

                    let dst_format = ImageFormat {
                        pixel_format: PixelFormat::Nv12,
                        color_space: *dst_color_space,
                        num_planes: num_planes,
                    };

                    // Compute strides
                    let src_buffers: &[&[u8]] = &[test_input];
                    let dst_buffers: &mut [&mut [u8]] = &mut [test_output_p0, test_output_p1];
                    let dst_strides: &[usize] = &[WIDTH as usize; 2];

                    // Test image convert
                    let src_pf = *src_pixel_format as u32;
                    let src_cs = *src_color_space as u32;
                    let dst_cs = *dst_color_space as u32;
                    let src_pf_rgb = src_pf < PIXEL_FORMAT_I444;
                    let src_cs_rgb = src_cs <= COLOR_SPACE_LRGB;
                    let dst_cs_rgb = dst_cs <= COLOR_SPACE_LRGB;

                    let mut expected: Result<(), ErrorKind> = Ok(());

                    set_expected!(expected, (WIDTH & 1) != 0, ErrorKind::InvalidValue);
                    set_expected!(expected, (HEIGHT & 1) != 0, ErrorKind::InvalidValue);

                    set_expected!(expected, !src_pf_rgb && src_cs_rgb, ErrorKind::InvalidValue);
                    set_expected!(expected, src_pf_rgb && !src_cs_rgb, ErrorKind::InvalidValue);

                    set_expected!(expected, dst_cs_rgb, ErrorKind::InvalidValue);

                    set_expected!(expected, num_planes < 1, ErrorKind::InvalidValue);
                    set_expected!(expected, num_planes > 2, ErrorKind::InvalidValue);

                    set_expected!(
                        expected,
                        src_pf != PIXEL_FORMAT_ARGB
                            && src_pf != PIXEL_FORMAT_BGRA
                            && src_pf != PIXEL_FORMAT_BGR,
                        ErrorKind::InvalidOperation
                    );

                    set_expected!(
                        expected,
                        src_cs != COLOR_SPACE_LRGB,
                        ErrorKind::InvalidOperation
                    );

                    let status = convert_image(
                        WIDTH,
                        HEIGHT,
                        &src_format,
                        None,
                        src_buffers,
                        &dst_format,
                        Some(dst_strides),
                        dst_buffers,
                    );

                    assert!(expected.is_ok() == status.is_ok());
                    match status {
                        Ok(_) => assert!(expected.is_ok()),
                        Err(s) => assert!((s as u32) == (expected.err().unwrap() as u32)),
                    }
                }
            }
        }
    }
}

#[test]
fn yuv_to_rgb_errors() {
    bootstrap();

    const WIDTH: u32 = 2;
    const HEIGHT: u32 = 2;
    const CHROMA_HEIGHT: u32 = HEIGHT / 2;
    const DST_STRIDE: usize = (WIDTH as usize) * 4;
    const IN_SIZE: usize = (WIDTH as usize) * ((HEIGHT as usize) + (CHROMA_HEIGHT as usize));
    const OUT_SIZE: usize = DST_STRIDE * (HEIGHT as usize);

    // Allocate and initialize input
    let test_input_p0: &[u8] = &[0u8; IN_SIZE];
    let test_input_p1: &[u8] = &[0u8; IN_SIZE];

    // Allocate and initialize output
    let test_output: &mut [u8] = &mut [0u8; OUT_SIZE];

    for num_planes in 0u32..=3 {
        // Only 1 and 2 are valid values
        for dst_pixel_format in PIXEL_FORMATS.iter() {
            for dst_color_space in COLOR_SPACES.iter() {
                for src_color_space in COLOR_SPACES.iter() {
                    let src_format = ImageFormat {
                        pixel_format: PixelFormat::Nv12,
                        color_space: *src_color_space,
                        num_planes: num_planes,
                    };

                    let dst_format = ImageFormat {
                        pixel_format: *dst_pixel_format,
                        color_space: *dst_color_space,
                        num_planes: 1,
                    };

                    // Compute strides
                    let src_buffers: &[&[u8]] = &[test_input_p0, test_input_p1];
                    let src_strides: &[usize] = &[WIDTH as usize; 2];
                    let dst_buffers: &mut [&mut [u8]] = &mut [test_output];

                    // Test image convert
                    let dst_pf = *dst_pixel_format as u32;
                    let dst_cs = *dst_color_space as u32;
                    let src_cs = *src_color_space as u32;
                    let dst_pf_rgb = dst_pf < PIXEL_FORMAT_I444;
                    let dst_cs_rgb = dst_cs <= COLOR_SPACE_LRGB;
                    let src_cs_rgb = src_cs <= COLOR_SPACE_LRGB;

                    let mut expected: Result<(), ErrorKind> = Ok(());

                    set_expected!(expected, (WIDTH & 1) != 0, ErrorKind::InvalidValue);
                    set_expected!(expected, (HEIGHT & 1) != 0, ErrorKind::InvalidValue);

                    set_expected!(expected, src_cs_rgb, ErrorKind::InvalidValue);

                    set_expected!(expected, !dst_pf_rgb && dst_cs_rgb, ErrorKind::InvalidValue);
                    set_expected!(expected, dst_pf_rgb && !dst_cs_rgb, ErrorKind::InvalidValue);

                    set_expected!(expected, num_planes < 1, ErrorKind::InvalidValue);
                    set_expected!(expected, num_planes > 2, ErrorKind::InvalidValue);

                    set_expected!(
                        expected,
                        dst_pf != PIXEL_FORMAT_BGRA,
                        ErrorKind::InvalidOperation
                    );

                    set_expected!(
                        expected,
                        dst_cs != COLOR_SPACE_LRGB,
                        ErrorKind::InvalidOperation
                    );

                    let status = convert_image(
                        WIDTH,
                        HEIGHT,
                        &src_format,
                        Some(src_strides),
                        src_buffers,
                        &dst_format,
                        None,
                        dst_buffers,
                    );

                    assert!(expected.is_ok() == status.is_ok());
                    match status {
                        Ok(_) => assert!(expected.is_ok()),
                        Err(s) => assert!((s as u32) == (expected.err().unwrap() as u32)),
                    }
                }
            }
        }
    }
}

#[test]
fn buffers_size() {
    bootstrap();

    const WIDTH: u32 = 4098;
    const HEIGHT: u32 = 258;
    let buffers_size = &mut [0usize; MAX_NUMBER_OF_PLANES as usize];

    for num_planes in 0..=MAX_NUMBER_OF_PLANES + 1 {
        for pixel_format in PIXEL_FORMATS.iter() {
            let pf = *pixel_format as u32;

            let format = ImageFormat {
                pixel_format: *pixel_format,
                color_space: ColorSpace::Lrgb,
                num_planes,
            };

            // Compute valid number of planes
            let mut max_number_of_planes = 0;
            while max_number_of_planes < MAX_NUMBER_OF_PLANES {
                if NUM_LOG2_DEN_PER_PLANE[pf as usize][(2 * max_number_of_planes) as usize] == 0 {
                    break;
                }
                max_number_of_planes = max_number_of_planes + 1;
            }

            // Invalid width
            let mut expected: Result<(), ErrorKind> = Ok(());

            set_expected!(expected, pf >= PIXEL_FORMAT_I422, ErrorKind::InvalidValue);
            set_expected!(
                expected,
                num_planes != 1 && num_planes != max_number_of_planes,
                ErrorKind::InvalidValue
            );

            let status = get_buffers_size(1, HEIGHT, &format, None, buffers_size);
            match status {
                Ok(_) => assert!(expected.is_ok()),
                Err(s) => assert!((s as u32) == (expected.err().unwrap() as u32)),
            }

            // Invalid height
            let mut expected: Result<(), ErrorKind> = Ok(());

            set_expected!(expected, pf >= PIXEL_FORMAT_I420, ErrorKind::InvalidValue);
            set_expected!(
                expected,
                num_planes != 1 && num_planes != max_number_of_planes,
                ErrorKind::InvalidValue
            );

            let status = get_buffers_size(WIDTH, 1, &format, None, buffers_size);
            match status {
                Ok(_) => assert!(expected.is_ok()),
                Err(s) => assert!((s as u32) == (expected.err().unwrap() as u32)),
            }

            // Test size is valid
            let mut expected: Result<(), ErrorKind> = Ok(());

            set_expected!(
                expected,
                num_planes != 1 && num_planes != max_number_of_planes,
                ErrorKind::InvalidValue
            );

            let status = get_buffers_size(WIDTH, HEIGHT, &format, None, buffers_size);

            match status {
                Ok(_) => {
                    let pf = pf as usize;
                    let num_planes = num_planes as usize;
                    let area = (WIDTH as usize) * (HEIGHT as usize);

                    if num_planes == 1 {
                        assert_eq!(
                            buffers_size[0],
                            (area * NUM_LOG2_DEN[pf][0]) >> NUM_LOG2_DEN[pf][1]
                        );
                    } else {
                        for i in 0..num_planes {
                            assert_eq!(
                                buffers_size[i],
                                (area * NUM_LOG2_DEN_PER_PLANE[pf][2 * i])
                                    >> NUM_LOG2_DEN_PER_PLANE[pf][2 * i + 1]
                            );
                        }
                    }
                }
                Err(s) => assert!((s as u32) == (expected.err().unwrap() as u32)),
            }
        }
    }
}

#[test]
fn over_4gb() {
    bootstrap();

    const WIDTH: u32 = 0x40000000;
    const HEIGHT: u32 = 2;

    const LUMA_SIZE: usize = (WIDTH as usize) * (HEIGHT as usize);
    const EXPECTED_SRC_BUFFER_SIZE: usize = LUMA_SIZE * 4;
    const EXPECTED_DST_BUFFER_SIZE: usize = (LUMA_SIZE * 3) / 2;

    let src_format = ImageFormat {
        pixel_format: PixelFormat::Argb,
        color_space: ColorSpace::Lrgb,
        num_planes: 1,
    };

    let dst_format = ImageFormat {
        pixel_format: PixelFormat::Nv12,
        color_space: ColorSpace::Bt601,
        num_planes: 1,
    };

    let src_buffers_size = &mut [0usize; 1];
    match get_buffers_size(WIDTH, HEIGHT, &src_format, None, src_buffers_size) {
        Ok(()) => assert_eq!(src_buffers_size[0], EXPECTED_SRC_BUFFER_SIZE),
        Err(_) => assert!(false),
    }

    let dst_buffers_size = &mut [0usize; 1];
    match get_buffers_size(WIDTH, HEIGHT, &dst_format, None, dst_buffers_size) {
        Ok(()) => assert_eq!(dst_buffers_size[0], EXPECTED_DST_BUFFER_SIZE),
        Err(_) => assert!(false),
    }

    #[allow(unsafe_code)]
    unsafe {
        let src_layout = Layout::from_size_align_unchecked(EXPECTED_SRC_BUFFER_SIZE, 1);

        let src_ptr = alloc_zeroed(src_layout);
        if src_ptr.is_null() {
            return;
        }

        let dst_layout = Layout::from_size_align_unchecked(EXPECTED_DST_BUFFER_SIZE, 1);
        let dst_ptr = alloc_zeroed(dst_layout);
        if !dst_ptr.is_null() {
            let src_image: &[u8] = from_raw_parts(src_ptr, EXPECTED_SRC_BUFFER_SIZE);
            let mut dst_image: &mut [u8] = from_raw_parts_mut(dst_ptr, EXPECTED_DST_BUFFER_SIZE);

            let src_buffers: &[&[u8]] = &[&src_image];
            let dst_buffers: &mut [&mut [u8]] = &mut [&mut dst_image];

            let status = convert_image(
                WIDTH,
                HEIGHT,
                &src_format,
                None,
                src_buffers,
                &dst_format,
                None,
                dst_buffers,
            );

            assert_eq!(status.is_ok(), true);

            // Check all luma samples are correct
            for i in 0..LUMA_SIZE {
                assert_eq!(dst_image[i], 16);
            }

            // Check all chroma samples are correct
            for i in LUMA_SIZE..EXPECTED_DST_BUFFER_SIZE {
                assert_eq!(dst_image[i], 128);
            }

            dealloc(dst_ptr, dst_layout);
        }

        dealloc(src_ptr, src_layout);
    }
}

#[test]
fn rgb_bgra_ok() {
    bootstrap();

    const MAX_WIDTH: usize = 32;
    const MAX_HEIGHT: usize = 32;
    const MAX_FILL_BYTES: usize = 7;
    const SRC_BPP: usize = 3;
    const DST_BPP: usize = 4;

    let src_format = ImageFormat {
        pixel_format: PixelFormat::Rgb,
        color_space: ColorSpace::Lrgb,
        num_planes: 1,
    };

    let dst_format = ImageFormat {
        pixel_format: PixelFormat::Bgra,
        color_space: ColorSpace::Lrgb,
        num_planes: 1,
    };

    for width in 0..=MAX_WIDTH {
        for height in 0..=MAX_HEIGHT {
            for src_stride_fill in 0..=MAX_FILL_BYTES {
                for dst_stride_fill in 0..=MAX_FILL_BYTES {
                    let src_stride = (SRC_BPP * width) + src_stride_fill;
                    let dst_stride = (DST_BPP * width) + dst_stride_fill;

                    let src_stride_param = if src_stride_fill == 0 {
                        STRIDE_AUTO
                    } else {
                        src_stride
                    };

                    let dst_stride_param = if dst_stride_fill == 0 {
                        STRIDE_AUTO
                    } else {
                        dst_stride
                    };

                    let mut src_buffers: Vec<&[u8]> = Vec::with_capacity(1);
                    let mut test_input: Box<[u8]> =
                        vec![0u8; src_stride * height].into_boxed_slice();
                    let mut test_output: Box<[u8]> =
                        vec![0u8; dst_stride * height].into_boxed_slice();

                    for h_iter in 0..height {
                        for w_iter in 0..width {
                            test_input[(h_iter * src_stride) + (w_iter * SRC_BPP) + 0] =
                                random::<u8>();
                            test_input[(h_iter * src_stride) + (w_iter * SRC_BPP) + 1] =
                                random::<u8>();
                            test_input[(h_iter * src_stride) + (w_iter * SRC_BPP) + 2] =
                                random::<u8>();
                        }
                    }
                    src_buffers.push(&test_input);

                    match convert_image(
                        width as u32,
                        height as u32,
                        &src_format,
                        Some(&[src_stride_param]),
                        &src_buffers[..],
                        &dst_format,
                        Some(&[dst_stride_param]),
                        &mut [&mut test_output[..]],
                    ) {
                        Err(e) => {
                            println!("{}", e);
                            assert!(false)
                        }
                        Ok(_) => {
                            for h_iter in 0..height {
                                for w_iter in 0..width {
                                    let input_index: usize =
                                        (h_iter * src_stride) + (w_iter * SRC_BPP);
                                    let output_index: usize =
                                        (h_iter * dst_stride) + (w_iter * DST_BPP);
                                    assert_eq!(
                                        test_output[output_index + 0],
                                        test_input[input_index + 2]
                                    );
                                    assert_eq!(
                                        test_output[output_index + 1],
                                        test_input[input_index + 1]
                                    );
                                    assert_eq!(
                                        test_output[output_index + 2],
                                        test_input[input_index + 0]
                                    );
                                    assert_eq!(test_output[output_index + 3], 255);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
