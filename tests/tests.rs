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

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

#[cfg(target_arch = "x86_64")]
use std::alloc::{alloc, alloc_zeroed, dealloc, Layout};
#[cfg(target_arch = "x86_64")]
use std::ptr::write_bytes;
#[cfg(target_arch = "x86_64")]
use std::slice::{from_raw_parts, from_raw_parts_mut};

use dcp::{
    convert_image, describe_acceleration, get_buffers_size, ColorSpace, ErrorKind, ImageFormat,
    PixelFormat, STRIDE_AUTO,
};

#[cfg(not(feature = "test_instruction_sets"))]
use dcp::initialize;
#[cfg(feature = "test_instruction_sets")]
use dcp::initialize_with_instruction_set;

use dcv_color_primitives as dcp;
use itertools::iproduct;
use rand::Rng;

#[cfg(feature = "test_instruction_sets")]
const SETS: [&str; 3] = ["x86", "sse2", "avx2"];
#[cfg(not(feature = "test_instruction_sets"))]
const SETS: [&str; 1] = [""];

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

const COLOR_SPACES: &[ColorSpace; 5] = &[
    ColorSpace::Rgb,
    ColorSpace::Bt601,
    ColorSpace::Bt709,
    ColorSpace::Bt601FR,
    ColorSpace::Bt709FR,
];

const PIXEL_FORMAT_I444: u32 = PixelFormat::I444 as u32;
const PIXEL_FORMAT_I422: u32 = PixelFormat::I422 as u32;
const PIXEL_FORMAT_I420: u32 = PixelFormat::I420 as u32;
const COLOR_SPACE_RGB: u32 = ColorSpace::Rgb as u32;
const PIXEL_FORMAT_ARGB: u32 = PixelFormat::Argb as u32;
const PIXEL_FORMAT_BGRA: u32 = PixelFormat::Bgra as u32;
const PIXEL_FORMAT_BGR: u32 = PixelFormat::Bgr as u32;
const RGB_SRC: [[[u8; 4]; 8]; 8] = [
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

const Y_BT601_REF: &[&[u8]] = &[
    &[74, 78, 101, 133, 118, 135, 87, 206],
    &[93, 171, 116, 94, 102, 100, 171, 122],
    &[141, 74, 200, 214, 98, 132, 65, 147],
    &[141, 54, 186, 175, 154, 185, 81, 93],
    &[118, 92, 125, 151, 40, 164, 181, 176],
    &[139, 93, 110, 112, 132, 114, 157, 64],
    &[113, 188, 101, 88, 105, 93, 128, 153],
    &[131, 65, 48, 76, 129, 162, 117, 141],
];

const CB_BT601_REF: &[&[u8]] = &[
    &[116, 118, 204, 91, 136, 159, 97, 100],
    &[99, 51, 179, 130, 161, 99, 117, 75],
    &[63, 120, 143, 89, 147, 155, 184, 64],
    &[118, 174, 134, 154, 133, 104, 185, 178],
    &[192, 101, 191, 102, 192, 131, 81, 43],
    &[60, 87, 198, 81, 175, 86, 61, 108],
    &[95, 59, 122, 204, 83, 196, 124, 150],
    &[184, 203, 185, 98, 140, 47, 196, 92],
];

const CR_BT601_REF: &[&[u8]] = &[
    &[187, 105, 177, 198, 208, 103, 119, 135],
    &[229, 162, 147, 87, 79, 221, 118, 111],
    &[187, 86, 115, 119, 141, 163, 166, 169],
    &[164, 126, 89, 171, 84, 83, 81, 91],
    &[183, 103, 75, 82, 121, 178, 97, 165],
    &[83, 210, 212, 158, 183, 60, 88, 118],
    &[135, 155, 93, 182, 70, 132, 176, 107],
    &[157, 133, 120, 122, 186, 153, 65, 155],
];

const CB2_BT601_REF: &[&[u8]] = &[
    &[96, 151, 139, 97],
    &[119, 130, 135, 153],
    &[110, 143, 146, 73],
    &[135, 152, 116, 140],
];

const CR2_BT601_REF: &[&[u8]] = &[
    &[171, 152, 153, 121],
    &[141, 124, 118, 127],
    &[145, 132, 135, 117],
    &[145, 129, 135, 126],
];

const Y_BT709_REF: &[&[u8]] = &[
    &[63, 84, 82, 123, 101, 137, 93, 208],
    &[75, 173, 106, 103, 108, 84, 174, 132],
    &[136, 84, 201, 221, 93, 121, 51, 146],
    &[134, 49, 193, 163, 163, 197, 84, 95],
    &[99, 101, 129, 164, 34, 154, 193, 179],
    &[157, 80, 85, 111, 115, 133, 173, 68],
    &[116, 191, 109, 68, 122, 84, 118, 155],
    &[118, 56, 43, 80, 116, 166, 122, 139],
];

const CB_BT709_REF: &[&[u8]] = &[
    &[123, 115, 211, 98, 145, 156, 95, 100],
    &[110, 53, 182, 126, 156, 109, 116, 72],
    &[68, 115, 141, 87, 149, 160, 189, 67],
    &[122, 174, 130, 160, 128, 98, 181, 174],
    &[200, 97, 186, 96, 192, 136, 77, 46],
    &[53, 95, 209, 84, 182, 78, 56, 107],
    &[95, 60, 118, 212, 76, 197, 129, 148],
    &[189, 205, 185, 97, 147, 48, 190, 94],
];

const CR_BT709_REF: &[&[u8]] = &[
    &[187, 103, 184, 197, 211, 104, 116, 133],
    &[229, 157, 151, 86, 80, 221, 117, 106],
    &[184, 84, 116, 115, 143, 166, 171, 165],
    &[164, 130, 89, 174, 83, 80, 84, 94],
    &[189, 100, 79, 78, 125, 180, 92, 160],
    &[77, 209, 219, 155, 188, 56, 82, 117],
    &[132, 151, 92, 189, 66, 137, 176, 108],
    &[162, 139, 124, 120, 189, 148, 69, 153],
];

const CB2_BT709_REF: &[&[u8]] = &[
    &[100, 154, 142, 96],
    &[120, 130, 134, 153],
    &[112, 144, 147, 71],
    &[137, 153, 117, 140],
];

const CR2_BT709_REF: &[&[u8]] = &[
    &[169, 154, 154, 118],
    &[140, 124, 118, 129],
    &[144, 133, 137, 113],
    &[146, 131, 135, 127],
];

const Y_BT601FR_REF: &[&[u8]] = &[
    &[67, 72, 99, 136, 119, 138, 83, 222],
    &[90, 181, 116, 91, 100, 98, 180, 123],
    &[146, 68, 214, 231, 95, 135, 57, 153],
    &[145, 45, 198, 185, 161, 196, 75, 90],
    &[119, 89, 127, 157, 28, 173, 192, 187],
    &[144, 89, 110, 112, 135, 114, 165, 55],
    &[113, 201, 99, 84, 104, 90, 130, 159],
    &[134, 57, 37, 69, 132, 170, 118, 145],
];

const CB_BT601FR_REF: &[&[u8]] = &[
    &[114, 116, 214, 86, 136, 162, 92, 95],
    &[95, 40, 185, 130, 165, 95, 115, 67],
    &[53, 119, 144, 83, 149, 159, 191, 54],
    &[117, 180, 134, 157, 133, 100, 192, 184],
    &[201, 97, 199, 98, 200, 130, 74, 31],
    &[50, 80, 208, 74, 181, 80, 52, 105],
    &[89, 49, 121, 215, 77, 204, 123, 152],
    &[192, 213, 192, 93, 141, 35, 205, 86],
];

const CR_BT601FR_REF: &[&[u8]] = &[
    &[194, 101, 183, 207, 219, 99, 117, 136],
    &[242, 166, 149, 80, 72, 234, 116, 108],
    &[195, 79, 113, 117, 142, 167, 171, 174],
    &[168, 126, 83, 177, 77, 77, 74, 86],
    &[190, 98, 68, 75, 119, 185, 92, 170],
    &[76, 221, 223, 161, 190, 51, 82, 116],
    &[135, 158, 87, 189, 62, 132, 182, 104],
    &[161, 133, 118, 121, 194, 156, 56, 159],
];

const CB2_BT601FR_REF: &[&[u8]] = &[
    &[91, 154, 140, 93],
    &[117, 130, 135, 155],
    &[107, 145, 148, 65],
    &[136, 155, 114, 142],
];

const CR2_BT601FR_REF: &[&[u8]] = &[
    &[176, 155, 156, 119],
    &[142, 122, 116, 126],
    &[146, 132, 136, 115],
    &[147, 129, 136, 125],
];

const Y_BT709FR_REF: &[&[u8]] = &[
    &[55, 79, 77, 124, 99, 140, 90, 224],
    &[69, 183, 105, 101, 108, 80, 184, 135],
    &[140, 79, 215, 239, 89, 123, 40, 152],
    &[138, 39, 206, 171, 171, 210, 79, 92],
    &[97, 98, 131, 172, 21, 160, 206, 189],
    &[164, 75, 80, 111, 115, 136, 183, 60],
    &[116, 204, 108, 60, 124, 80, 119, 162],
    &[119, 46, 31, 75, 116, 175, 124, 143],
];

const CB_BT709FR_REF: &[&[u8]] = &[
    &[122, 112, 222, 94, 147, 160, 90, 96],
    &[107, 43, 189, 125, 160, 106, 114, 64],
    &[60, 113, 143, 81, 151, 164, 197, 58],
    &[121, 180, 129, 163, 127, 94, 187, 180],
    &[209, 93, 193, 91, 201, 137, 69, 34],
    &[42, 90, 220, 77, 189, 71, 45, 104],
    &[90, 51, 116, 223, 68, 206, 129, 150],
    &[197, 215, 192, 92, 149, 37, 198, 89],
];

const CR_BT709FR_REF: &[&[u8]] = &[
    &[195, 100, 191, 206, 222, 101, 114, 133],
    &[243, 161, 154, 79, 73, 234, 115, 103],
    &[191, 77, 114, 113, 144, 170, 177, 170],
    &[168, 130, 83, 180, 76, 73, 78, 89],
    &[197, 95, 72, 71, 124, 186, 87, 164],
    &[69, 220, 232, 158, 195, 45, 75, 115],
    &[132, 153, 86, 197, 57, 138, 183, 105],
    &[167, 140, 123, 118, 197, 150, 60, 156],
];

const CB2_BT709FR_REF: &[&[u8]] = &[
    &[96, 157, 143, 91],
    &[119, 129, 134, 156],
    &[109, 145, 149, 63],
    &[138, 156, 115, 142],
];

const CR2_BT709FR_REF: &[&[u8]] = &[
    &[175, 157, 157, 116],
    &[142, 122, 116, 128],
    &[145, 133, 138, 110],
    &[148, 131, 135, 126],
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

type PlaneRef<'a> = &'a [&'a [u8]];

macro_rules! set_expected {
    ($var:ident, $pred:expr, $status:path) => {
        if $var.is_ok() && $pred {
            $var = Err($status);
        }
    };
}

fn is_valid_format(format: &ImageFormat, width: u32, height: u32) -> bool {
    match format.pixel_format {
        PixelFormat::I444 => format.num_planes != 3,
        PixelFormat::I422 | PixelFormat::I420 => {
            format.num_planes != 3 || (width & 1) == 1 || (height & 1) == 1
        }
        PixelFormat::Nv12 => {
            (format.num_planes < 1 || format.num_planes > 2)
                || (width & 1) == 1
                || (height & 1) == 1
        }
        _ => format.num_planes != 1,
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

fn check_plane(plane: &[u8], reference: PlaneRef, width: usize, height: usize, stride: usize) {
    for (row, exp) in plane.chunks(stride).zip(reference.iter().take(height)) {
        let (payload, pad) = row.split_at(width);
        assert!(payload
            .iter()
            .zip(exp.iter().take(width))
            .all(|(&x, &y)| x == y));
        assert!(pad.iter().all(|&x| x == 0));
    }
}

fn bootstrap(set: &str) {
    #[cfg(not(feature = "test_instruction_sets"))]
    {
        let _ = set;
        let desc = describe_acceleration();
        if desc.is_ok() {
            return;
        }

        initialize();
    }

    #[cfg(feature = "test_instruction_sets")]
    {
        let desc = describe_acceleration();
        match desc {
            Err(ErrorKind::NotInitialized) => {
                const WIDTH: u32 = 640;
                const HEIGHT: u32 = 480;
                const PIXELS: usize = (WIDTH as usize) * (HEIGHT as usize);

                let src_buffers: &[&[u8]] = &[&[0_u8; 4 * PIXELS]];
                let dst_buffers: &mut [&mut [u8]] = &mut [&mut [0_u8; 3 * PIXELS]];
                let src_format = ImageFormat {
                    pixel_format: PixelFormat::Bgra,
                    color_space: ColorSpace::Rgb,
                    num_planes: 1,
                };
                let dst_format = ImageFormat {
                    pixel_format: PixelFormat::Rgb,
                    color_space: ColorSpace::Rgb,
                    num_planes: 1,
                };
                let sizes = &mut [0_usize; 1];

                assert!(get_buffers_size(WIDTH, HEIGHT, &src_format, None, sizes).is_ok());
                check_err(
                    convert_image(
                        WIDTH,
                        HEIGHT,
                        &src_format,
                        None,
                        src_buffers,
                        &dst_format,
                        None,
                        dst_buffers,
                    )
                    .unwrap_err(),
                    ErrorKind::NotInitialized,
                );
            }
            Err(_) => panic!(),
            Ok(_) => (),
        }

        initialize_with_instruction_set(set);
        assert!(describe_acceleration().is_ok());
    }
}

fn rgb_to_yuv_size_mode_pad(
    image_size: (u32, u32),
    src_format: &ImageFormat,
    dst_format: &ImageFormat,
    pad: (usize, usize, usize, usize),
    plane_ref: (PlaneRef, PlaneRef, PlaneRef),
) {
    let (src_pad, y_pad, u_pad, v_pad) = pad;
    let w = image_size.0 as usize;
    let h = image_size.1 as usize;
    let cw = w / 2;
    let ch = h / 2;

    let y_stride = w + y_pad;
    let u_stride = match dst_format.pixel_format {
        PixelFormat::I444 | PixelFormat::Nv12 => w + u_pad,
        PixelFormat::I420 => cw + u_pad,
        _ => unreachable!(),
    };
    let v_stride = match dst_format.pixel_format {
        PixelFormat::Nv12 => u_stride,
        PixelFormat::I420 => cw + v_pad,
        PixelFormat::I444 => w + v_pad,
        _ => unreachable!(),
    };
    let src_depth = get_depth(src_format.pixel_format);

    let src_stride = w * src_depth + src_pad;
    let src_size = src_stride * h;
    let dst_size = match dst_format.pixel_format {
        PixelFormat::Nv12 => y_stride * h + u_stride * ch,
        PixelFormat::I420 => y_stride * h + (u_stride + v_stride) * ch,
        PixelFormat::I444 => (y_stride + u_stride + v_stride) * h,
        _ => unreachable!(),
    };

    // Allocate and initialize input
    let mut src_image = vec![0_u8; src_size].into_boxed_slice();
    for (y, line) in RGB_SRC.iter().enumerate().take(h) {
        for (x, pixel) in line.iter().enumerate().take(w) {
            let p = y * src_stride + x * src_depth;
            match src_format.pixel_format {
                PixelFormat::Argb => {
                    src_image[p] = pixel[3];
                    src_image[p + 1] = pixel[0];
                    src_image[p + 2] = pixel[1];
                    src_image[p + 3] = pixel[2];
                }
                PixelFormat::Bgra => {
                    src_image[p] = pixel[2];
                    src_image[p + 1] = pixel[1];
                    src_image[p + 2] = pixel[0];
                    src_image[p + 3] = pixel[3];
                }
                _ => {
                    src_image[p] = pixel[2];
                    src_image[p + 1] = pixel[1];
                    src_image[p + 2] = pixel[0];
                }
            }
        }
    }

    // Allocate output
    let mut dst_image = vec![0_u8; dst_size].into_boxed_slice();

    // Compute strides
    let src_stride = if src_pad == 0 {
        STRIDE_AUTO
    } else {
        src_stride
    };

    let mut dst_strides = Vec::with_capacity(3);
    let mut dst_buffers: Vec<&mut [u8]> = Vec::with_capacity(3);
    dst_strides.push(if y_pad == 0 { STRIDE_AUTO } else { y_stride });
    match dst_format.pixel_format {
        PixelFormat::Nv12 => {
            if dst_format.num_planes == 1 {
                dst_buffers.push(&mut dst_image);
            } else if dst_format.num_planes == 2 {
                dst_strides.push(if u_pad == 0 { STRIDE_AUTO } else { u_stride });

                let (first, last) = dst_image.split_at_mut(y_stride * h);
                dst_buffers.push(first);
                dst_buffers.push(last);
            }
        }
        PixelFormat::I420 => {
            let (y_plane, chroma_planes) = dst_image.split_at_mut(y_stride * h);
            let (u_plane, v_plane) = chroma_planes.split_at_mut(ch * u_stride);

            dst_buffers.push(y_plane);
            dst_buffers.push(u_plane);
            dst_buffers.push(v_plane);
            dst_strides.push(if u_pad == 0 { STRIDE_AUTO } else { u_stride });
            dst_strides.push(if v_pad == 0 { STRIDE_AUTO } else { v_stride });
        }
        PixelFormat::I444 => {
            let (y_plane, chroma_planes) = dst_image.split_at_mut(y_stride * h);
            let (u_plane, v_plane) = chroma_planes.split_at_mut(u_stride * h);

            dst_buffers.push(y_plane);
            dst_buffers.push(u_plane);
            dst_buffers.push(v_plane);
            dst_strides.push(if u_pad == 0 { STRIDE_AUTO } else { u_stride });
            dst_strides.push(if v_pad == 0 { STRIDE_AUTO } else { v_stride });
        }
        _ => unreachable!(),
    }

    assert!(convert_image(
        image_size.0,
        image_size.1,
        src_format,
        Some(&[src_stride]),
        &[&src_image[..]],
        dst_format,
        Some(&dst_strides[..]),
        &mut dst_buffers[..],
    )
    .is_ok());

    if w == 0 || h == 0 {
        return;
    }

    check_plane(dst_buffers[0], plane_ref.0, w, h, y_stride);
    match dst_format.pixel_format {
        PixelFormat::Nv12 => {
            for (uv_row, (u_exp, v_exp)) in dst_image[y_stride * h..]
                .chunks(u_stride)
                .zip(plane_ref.1.iter().zip(plane_ref.2).take(ch))
            {
                let (payload, pad) = uv_row.split_at(w);
                assert!(payload
                    .chunks(2)
                    .zip(u_exp.iter().zip(v_exp.iter().take(cw)))
                    .all(|(uv, (&u, &v))| uv[0] == u && uv[1] == v));
                assert!(pad.iter().all(|&x| x == 0));
            }
        }
        PixelFormat::I420 => {
            check_plane(dst_buffers[1], plane_ref.1, cw, ch, u_stride);
            check_plane(dst_buffers[2], plane_ref.2, cw, ch, v_stride);
        }
        PixelFormat::I444 => {
            check_plane(dst_buffers[1], plane_ref.1, w, h, u_stride);
            check_plane(dst_buffers[2], plane_ref.2, w, h, v_stride);
        }
        _ => unreachable!(),
    }
}

fn rgb_to_yuv_size_mode(
    image_size: (u32, u32),
    src_format: &ImageFormat,
    dst_format: &ImageFormat,
    plane_ref: (PlaneRef, PlaneRef, PlaneRef),
) {
    const MAX_PAD: usize = 2;

    if dst_format.num_planes == 1 {
        for (src_pad, y_pad) in iproduct!(0..MAX_PAD, 0..MAX_PAD) {
            rgb_to_yuv_size_mode_pad(
                image_size,
                src_format,
                dst_format,
                (src_pad, y_pad, y_pad, y_pad),
                plane_ref,
            );
        }
    } else if dst_format.num_planes == 2 {
        for (src_pad, y_pad, uv_pad) in iproduct!(0..MAX_PAD, 0..MAX_PAD, 0..MAX_PAD) {
            rgb_to_yuv_size_mode_pad(
                image_size,
                src_format,
                dst_format,
                (src_pad, y_pad, uv_pad, uv_pad),
                plane_ref,
            );
        }
    } else if dst_format.num_planes == 3 {
        for pad in iproduct!(0..MAX_PAD, 0..MAX_PAD, 0..MAX_PAD, 0..MAX_PAD) {
            rgb_to_yuv_size_mode_pad(image_size, src_format, dst_format, pad, plane_ref);
        }
    }
}

fn rgb_to_yuv_size(
    image_size: (u32, u32),
    dst_image_format: &ImageFormat,
    plane_ref: (PlaneRef, PlaneRef, PlaneRef),
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
    const SUPPORTED_COLOR_SPACES: &[ColorSpace] = &[
        ColorSpace::Bt601,
        ColorSpace::Bt709,
        ColorSpace::Bt601FR,
        ColorSpace::Bt709FR,
    ];
    const MAX_WIDTH: u32 = 8;
    const MAX_HEIGHT: u32 = 8;

    let step = match pixel_format {
        PixelFormat::I444 => 1,
        _ => 2,
    };

    for color_space in SUPPORTED_COLOR_SPACES {
        let format = ImageFormat {
            pixel_format,
            color_space: *color_space,
            num_planes,
        };

        let plane_ref = if let PixelFormat::I444 = pixel_format {
            match color_space {
                ColorSpace::Bt601 => (Y_BT601_REF, CB_BT601_REF, CR_BT601_REF),
                ColorSpace::Bt709 => (Y_BT709_REF, CB_BT709_REF, CR_BT709_REF),
                ColorSpace::Bt601FR => (Y_BT601FR_REF, CB_BT601FR_REF, CR_BT601FR_REF),
                _ => (Y_BT709FR_REF, CB_BT709FR_REF, CR_BT709FR_REF),
            }
        } else {
            match color_space {
                ColorSpace::Bt601 => (Y_BT601_REF, CB2_BT601_REF, CR2_BT601_REF),
                ColorSpace::Bt709 => (Y_BT709_REF, CB2_BT709_REF, CR2_BT709_REF),
                ColorSpace::Bt601FR => (Y_BT601FR_REF, CB2_BT601FR_REF, CR2_BT601FR_REF),
                _ => (Y_BT709FR_REF, CB2_BT709FR_REF, CR2_BT709FR_REF),
            }
        };

        for width in (0..=MAX_WIDTH).step_by(step) {
            for height in (0..=MAX_HEIGHT).step_by(step) {
                rgb_to_yuv_size((width, height), &format, plane_ref);
            }
        }
    }
}

fn yuv_to_rgb_size_format_mode_stride(
    image_size: (u32, u32),
    src_format: &ImageFormat,
    dst_format: &ImageFormat,
    pad: (usize, usize, usize, usize),
) {
    let (y_pad, u_pad, v_pad, dst_pad) = pad;
    let w = image_size.0 as usize;
    let h = image_size.1 as usize;

    let y_stride = w + y_pad;
    let cw = w / 2;
    let ch = match src_format.pixel_format {
        PixelFormat::Nv12 | PixelFormat::I420 => h / 2,
        PixelFormat::I444 => h,
        _ => unreachable!(),
    };
    let u_stride = match src_format.pixel_format {
        PixelFormat::Nv12 | PixelFormat::I444 => w + u_pad,
        PixelFormat::I420 => cw + u_pad,
        _ => unreachable!(),
    };
    let v_stride = match src_format.pixel_format {
        PixelFormat::Nv12 => u_stride,
        PixelFormat::I420 => cw + v_pad,
        PixelFormat::I444 => w + v_pad,
        _ => unreachable!(),
    };
    let src_size = match src_format.pixel_format {
        PixelFormat::Nv12 => y_stride * h + u_stride * ch,
        PixelFormat::I444 | PixelFormat::I420 => y_stride * h + (u_stride + v_stride) * ch,
        _ => unreachable!(),
    };

    let dst_stride = w * 4 + dst_pad;
    let dst_size = dst_stride * h;
    let color_space_index = match src_format.color_space {
        ColorSpace::Bt601 => 0,
        ColorSpace::Bt709 => 1,
        ColorSpace::Bt601FR => 2,
        _ => 3,
    };

    // Allocate and initialize input
    let mut src_image = vec![0_u8; src_size].into_boxed_slice();
    let mut i = 0;
    while i < y_stride * h {
        for x in (0..w).step_by(2) {
            let index = (x >> 1) & 7;
            let luma = Y_SRC[color_space_index][index];

            src_image[i + x] = luma;
            src_image[i + x + 1] = luma;
        }

        i += y_stride;
    }

    match src_format.pixel_format {
        PixelFormat::Nv12 => {
            i = y_stride * h;
            while i < src_size {
                for x in (0..w).step_by(2) {
                    let index = (x >> 1) & 0x7;
                    src_image[i + x] = U_SRC[color_space_index][index];
                    src_image[i + x + 1] = V_SRC[color_space_index][index];
                }

                i += u_stride;
            }
        }
        PixelFormat::I420 => {
            let p0 = y_stride * h;
            let p1 = p0 + u_stride * ch;

            i = p0;
            while i < p1 {
                for x in (0..w).step_by(2) {
                    let pos = x >> 1;
                    let index = pos & 0x7;
                    src_image[i + pos] = U_SRC[color_space_index][index];
                }

                i += u_stride;
            }

            i = p1;
            while i < src_size {
                for x in (0..w).step_by(2) {
                    let pos = x >> 1;
                    let index = pos & 0x7;
                    src_image[i + pos] = V_SRC[color_space_index][index];
                }

                i += v_stride;
            }
        }
        PixelFormat::I444 => {
            let p0 = y_stride * h;
            let p1 = p0 + u_stride * ch;

            i = p0;
            while i < p1 {
                for x in 0..w {
                    let index = (x >> 1) & 0x7;
                    src_image[i + x] = U_SRC[color_space_index][index];
                }

                i += u_stride;
            }

            i = p1;
            while i < src_size {
                for x in 0..w {
                    let index = (x >> 1) & 0x7;
                    src_image[i + x] = V_SRC[color_space_index][index];
                }

                i += v_stride;
            }
        }
        _ => unreachable!(),
    }

    // Allocate output
    let mut dst_image = vec![0_u8; dst_size].into_boxed_slice();

    // Compute strides
    let dst_stride = if dst_pad == 0 {
        STRIDE_AUTO
    } else {
        dst_stride
    };

    let mut src_buffers: Vec<&[u8]> = Vec::with_capacity(3);
    let mut src_strides = Vec::with_capacity(3);
    src_strides.push(if y_pad == 0 { STRIDE_AUTO } else { y_stride });
    match src_format.pixel_format {
        PixelFormat::Nv12 => {
            if src_format.num_planes == 1 {
                src_buffers.push(&src_image);
            } else if src_format.num_planes == 2 {
                src_strides.push(if u_pad == 0 { STRIDE_AUTO } else { u_stride });

                let (first, last) = src_image.split_at(y_stride * h);
                src_buffers.push(first);
                src_buffers.push(last);
            }
        }
        PixelFormat::I420 | PixelFormat::I444 => {
            if src_format.num_planes == 3 {
                let y_size = y_stride * h;
                let u_size = u_stride * ch;

                src_strides.push(if u_pad == 0 { STRIDE_AUTO } else { u_stride });
                src_strides.push(if v_pad == 0 { STRIDE_AUTO } else { v_stride });
                src_buffers.push(&src_image[0..y_size]);
                src_buffers.push(&src_image[y_size..y_size + u_size]);
                src_buffers.push(&src_image[y_size + u_size..]);
            }
        }
        _ => unreachable!(),
    }

    let mut expected_row = vec![0_i32; 3 * w].into_boxed_slice();
    for (x, pixel) in expected_row.chunks_exact_mut(3).enumerate() {
        let index = (x >> 1) & 7;

        // Expected blue
        pixel[0] = if ((index >> 2) & 1) == 0 { 0 } else { 255 };
        // Expected green
        pixel[1] = if ((index >> 1) & 1) == 0 { 0 } else { 255 };
        // Expected red
        pixel[2] = if (index & 1) == 0 { 0 } else { 255 };
    }

    assert!(convert_image(
        image_size.0,
        image_size.1,
        src_format,
        Some(&src_strides[..]),
        &src_buffers[..],
        dst_format,
        Some(&[dst_stride]),
        &mut [&mut dst_image[..]],
    )
    .is_ok());

    i = 0;
    while i < dst_size {
        let pack_stride = 4 * w;

        for col in 0..w {
            let src = 4 * col;
            let dst = 3 * col;
            assert!(
                (i32::from(dst_image[i + src]) - expected_row[dst]).abs() <= 2
                    && (i32::from(dst_image[i + src + 1]) - expected_row[dst + 1]).abs() <= 2
                    && (i32::from(dst_image[i + src + 2]) - expected_row[dst + 2]).abs() <= 2
                    && dst_image[i + src + 3] == 255
            );
        }

        for col in 0..dst_pad {
            assert_eq!(dst_image[i + pack_stride + col], 0);
        }

        i += pack_stride + dst_pad;
    }
}

fn yuv_to_rgb_size_format_mode(
    image_size: (u32, u32),
    src_format: &ImageFormat,
    dst_format: &ImageFormat,
) {
    const MAX_PAD: usize = 4;

    if src_format.num_planes == 1 {
        for (y_pad, dst_pad) in iproduct!(0..MAX_PAD, 0..MAX_PAD) {
            yuv_to_rgb_size_format_mode_stride(
                image_size,
                src_format,
                dst_format,
                (y_pad, y_pad, y_pad, dst_pad),
            );
        }
    } else if src_format.num_planes == 2 {
        for (y_pad, uv_pad, dst_pad) in iproduct!(0..MAX_PAD, 0..MAX_PAD, 0..MAX_PAD) {
            yuv_to_rgb_size_format_mode_stride(
                image_size,
                src_format,
                dst_format,
                (y_pad, uv_pad, uv_pad, dst_pad),
            );
        }
    } else if src_format.num_planes == 3 {
        for pad in iproduct!(0..MAX_PAD, 0..MAX_PAD, 0..MAX_PAD, 0..MAX_PAD) {
            yuv_to_rgb_size_format_mode_stride(image_size, src_format, dst_format, pad);
        }
    }
}

fn yuv_to_rgb_ok(pixel_format: PixelFormat, num_planes: u32) {
    const SUPPORTED_COLOR_SPACES: &[ColorSpace] = &[
        ColorSpace::Bt601,
        ColorSpace::Bt709,
        ColorSpace::Bt601FR,
        ColorSpace::Bt709FR,
    ];
    const MAX_WIDTH: u32 = 34;
    const MAX_HEIGHT: u32 = 4;

    let step = match pixel_format {
        PixelFormat::I444 => 1,
        _ => 2,
    };
    let dst_format = ImageFormat {
        pixel_format: PixelFormat::Bgra,
        color_space: ColorSpace::Rgb,
        num_planes: 1,
    };

    for color_space in SUPPORTED_COLOR_SPACES {
        let src_format = ImageFormat {
            pixel_format,
            color_space: *color_space,
            num_planes,
        };

        for width in (0..=MAX_WIDTH).step_by(step) {
            for height in (0..=MAX_HEIGHT).step_by(step) {
                yuv_to_rgb_size_format_mode((width, height), &src_format, &dst_format);
            }
        }
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

    let src_image = vec![0_u8; src_size].into_boxed_slice();
    let mut dst_image = vec![0_u8; dst_size].into_boxed_slice();

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
            is_valid_format(&src_format, WIDTH, HEIGHT),
            ErrorKind::InvalidValue
        );
        set_expected!(
            expected,
            is_valid_format(&dst_format, WIDTH, HEIGHT),
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
            Ok(_) => check_bounds(
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

fn rgb_conversion_ok(src_pixel_format: PixelFormat, dst_pixel_format: PixelFormat) {
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
    let mut rng = rand::thread_rng();

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

        let mut src_image = vec![0_u8; src_stride * h].into_boxed_slice();
        let mut dst_image = vec![0_u8; dst_stride * h].into_boxed_slice();
        for y in 0..h {
            for x in 0..w {
                let offset = y * src_stride + x * src_depth;

                src_image[offset] = rng.gen::<u8>();
                src_image[offset + 1] = rng.gen::<u8>();
                src_image[offset + 2] = rng.gen::<u8>();
                if src_depth == 4 {
                    src_image[offset + 3] = 255;
                }
            }
        }

        assert!(convert_image(
            width,
            height,
            &src_format,
            Some(&[src_strides]),
            &[&src_image[..]],
            &dst_format,
            Some(&[dst_strides]),
            &mut [&mut dst_image[..]],
        )
        .is_ok());

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

fn rgb_to_nv12_ok() {
    rgb_to_yuv_ok(PixelFormat::Nv12, 1);
    rgb_to_yuv_ok(PixelFormat::Nv12, 2);
}

fn rgb_to_i420_ok() {
    rgb_to_yuv_ok(PixelFormat::I420, 3);
}

fn rgb_to_i444_ok() {
    rgb_to_yuv_ok(PixelFormat::I444, 3);
}

fn i420_to_rgb_ok() {
    yuv_to_rgb_ok(PixelFormat::I420, 3);
}

fn nv12_to_rgb_ok() {
    yuv_to_rgb_ok(PixelFormat::Nv12, 1);
    yuv_to_rgb_ok(PixelFormat::Nv12, 2);
}

fn i444_to_rgb_ok() {
    yuv_to_rgb_ok(PixelFormat::I444, 3);
}

fn rgb_ok() {
    rgb_conversion_ok(PixelFormat::Rgb, PixelFormat::Bgra);
    rgb_conversion_ok(PixelFormat::Bgra, PixelFormat::Rgb);
    rgb_conversion_ok(PixelFormat::Bgr, PixelFormat::Rgb);
}

fn rgb_errors() {
    rgb_conversion_errors(PixelFormat::Rgb, PixelFormat::Bgra);
    rgb_conversion_errors(PixelFormat::Bgra, PixelFormat::Rgb);
    rgb_conversion_errors(PixelFormat::Bgr, PixelFormat::Rgb);
}

fn rgb_to_yuv_errors(pixel_format: PixelFormat) {
    let (w, h) = match pixel_format {
        PixelFormat::I444 => (33, 3),
        _ => (34, 2),
    };
    let cw = match pixel_format {
        PixelFormat::I420 => w / 2,
        _ => w,
    } as usize;
    let ch = match pixel_format {
        PixelFormat::I444 => h,
        _ => h / 2,
    } as usize;
    let y_size = (w as usize) * (h as usize);
    let uv_size = cw * ch;

    let slices = &[0, y_size + uv_size, y_size, 0];
    let mut y_plane = match pixel_format {
        PixelFormat::Nv12 => vec![0_u8; y_size + uv_size].into_boxed_slice(),
        _ => vec![0_u8; y_size].into_boxed_slice(),
    };
    let mut u_plane = vec![0_u8; uv_size].into_boxed_slice();
    let mut v_plane = vec![0_u8; uv_size].into_boxed_slice();

    for src_pixel_format in PIXEL_FORMATS {
        let src_stride = get_depth(*src_pixel_format) * (w as usize);
        let src_size = src_stride * (h as usize);
        let src_image = vec![0_u8; src_size].into_boxed_slice();

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

            dst_strides.push(w as usize);
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
                is_valid_format(&src_format, w, h),
                ErrorKind::InvalidValue
            );
            set_expected!(
                expected,
                is_valid_format(&dst_format, w, h),
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
                w,
                h,
                &src_format,
                None,
                src_buffers,
                &dst_format,
                Some(&dst_strides),
                &mut dst_buffers,
            );

            assert_eq!(expected.is_ok(), status.is_ok());
            match status {
                Ok(_) => check_bounds(
                    w,
                    h,
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
    let (w, h) = match pixel_format {
        PixelFormat::I444 => (33, 3),
        _ => (34, 2),
    };
    let cw = match pixel_format {
        PixelFormat::I420 => w / 2,
        _ => w,
    } as usize;
    let ch = match pixel_format {
        PixelFormat::I444 => h,
        _ => h / 2,
    } as usize;
    let y_size = (w as usize) * (h as usize);
    let uv_size = cw * ch;
    let dst_size = (w as usize) * (h as usize) * 4;

    let slices = &[0, y_size + uv_size, y_size, 0];
    let y_plane = match pixel_format {
        PixelFormat::Nv12 => vec![0_u8; y_size + uv_size].into_boxed_slice(),
        _ => vec![0_u8; y_size].into_boxed_slice(),
    };
    let u_plane = vec![0_u8; uv_size].into_boxed_slice();
    let v_plane = vec![0_u8; uv_size].into_boxed_slice();
    let mut dst_image = vec![0_u8; dst_size].into_boxed_slice();

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

        src_strides.push(w as usize);
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
            is_valid_format(&src_format, w, h),
            ErrorKind::InvalidValue
        );
        set_expected!(
            expected,
            is_valid_format(&dst_format, w, h),
            ErrorKind::InvalidValue
        );
        set_expected!(
            expected,
            dst_pf != PIXEL_FORMAT_BGRA,
            ErrorKind::InvalidOperation
        );
        set_expected!(
            expected,
            dst_cs != COLOR_SPACE_RGB,
            ErrorKind::InvalidOperation
        );

        let status = convert_image(
            w,
            h,
            &src_format,
            Some(&src_strides),
            &src_buffers,
            &dst_format,
            None,
            dst_buffers,
        );

        assert_eq!(expected.is_ok(), status.is_ok());
        match status {
            Ok(_) => check_bounds(w, h, &src_format, &src_buffers, &dst_format, dst_buffers),
            Err(err) => check_err(err, expected.unwrap_err()),
        }
    }
}

fn buffers_size() {
    const WIDTH: u32 = 4098;
    const HEIGHT: u32 = 258;
    let buffers_size = &mut [0_usize; MAX_NUMBER_OF_PLANES as usize];

    for (num_planes, pixel_format) in iproduct!(0..=MAX_NUMBER_OF_PLANES + 1, PIXEL_FORMATS) {
        let pf = *pixel_format as u32;
        let format = ImageFormat {
            pixel_format: *pixel_format,
            color_space: ColorSpace::Rgb,
            num_planes,
        };

        // Invalid width
        let mut expected = Ok(());
        set_expected!(expected, pf >= PIXEL_FORMAT_I422, ErrorKind::InvalidValue);
        set_expected!(
            expected,
            is_valid_format(&format, WIDTH, HEIGHT),
            ErrorKind::InvalidValue
        );

        let status = get_buffers_size(1, HEIGHT, &format, None, buffers_size);
        match status {
            Ok(_) => assert!(expected.is_ok()),
            Err(err) => check_err(err, expected.err().unwrap()),
        }

        // Invalid height
        let mut expected = Ok(());
        set_expected!(expected, pf >= PIXEL_FORMAT_I420, ErrorKind::InvalidValue);
        set_expected!(
            expected,
            is_valid_format(&format, WIDTH, HEIGHT),
            ErrorKind::InvalidValue
        );

        let status = get_buffers_size(WIDTH, 1, &format, None, buffers_size);
        match status {
            Ok(_) => assert!(expected.is_ok()),
            Err(err) => check_err(err, expected.err().unwrap()),
        }

        // Test size is valid
        let mut expected = Ok(());
        set_expected!(
            expected,
            is_valid_format(&format, WIDTH, HEIGHT),
            ErrorKind::InvalidValue
        );

        let status = get_buffers_size(WIDTH, HEIGHT, &format, None, buffers_size);
        assert_eq!(expected.is_ok(), status.is_ok());

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
                    let mut strides = Vec::new();

                    for (i, buffer_size) in buffers_size.iter().enumerate().take(num_planes) {
                        let mul = NUM_LOG2_DEN_PER_PLANE[pf][2 * i];
                        let shf = NUM_LOG2_DEN_PER_PLANE[pf][2 * i + 1];
                        assert_eq!(*buffer_size, (area * mul) >> shf);

                        let row = match pixel_format {
                            PixelFormat::I422 | PixelFormat::I420 => {
                                if i > 0 {
                                    1
                                } else {
                                    0
                                }
                            }
                            _ => 0,
                        };

                        strides.push(((WIDTH as usize) * mul) >> row);
                    }

                    assert!(get_buffers_size(
                        WIDTH,
                        HEIGHT,
                        &format,
                        Some(&strides[..]),
                        buffers_size
                    )
                    .is_ok());

                    for (i, buffer_size) in buffers_size.iter().enumerate().take(num_planes) {
                        assert_eq!(
                            *buffer_size,
                            (area * NUM_LOG2_DEN_PER_PLANE[pf][2 * i])
                                >> NUM_LOG2_DEN_PER_PLANE[pf][2 * i + 1]
                        );
                    }
                }

                // empty buffer vector should return not enough data
                assert!(
                    get_buffers_size(WIDTH, HEIGHT, &format, None, &mut buffers_size[..0]).is_err()
                );
            }
            Err(err) => check_err(err, expected.err().unwrap()),
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[cfg(not(tarpaulin))]
fn over_4gb() {
    // In this test the output image will larger than 4GB:
    // width = 536870944 * 2 * 4 = 4294967552
    const WIDTH: u32 = 0x2000_0020;
    const HEIGHT: u32 = 2;
    const LUMA_SIZE: usize = (WIDTH as usize) * (HEIGHT as usize);
    const EXPECTED_SRC_BUFFER_SIZE: usize = (LUMA_SIZE * 3) / 2;
    const EXPECTED_DST_BUFFER_SIZE: usize = LUMA_SIZE * 4;
    const PAGE_SIZE: usize = 4096;
    const MAX_LUMA: u8 = 235;
    const HALF_CHROMA: u8 = 128;

    let src_format = ImageFormat {
        pixel_format: PixelFormat::Nv12,
        color_space: ColorSpace::Bt601,
        num_planes: 1,
    };
    let dst_format = ImageFormat {
        pixel_format: PixelFormat::Bgra,
        color_space: ColorSpace::Rgb,
        num_planes: 1,
    };

    let src_size = &mut [0_usize; 1];
    assert!(get_buffers_size(WIDTH, HEIGHT, &src_format, None, src_size).is_ok());
    assert_eq!(src_size[0], EXPECTED_SRC_BUFFER_SIZE);

    let dst_size = &mut [0_usize; 1];
    assert!(get_buffers_size(WIDTH, HEIGHT, &dst_format, None, dst_size).is_ok());
    assert_eq!(dst_size[0], EXPECTED_DST_BUFFER_SIZE);

    #[allow(unsafe_code)]
    unsafe {
        let src_layout = Layout::from_size_align_unchecked(EXPECTED_SRC_BUFFER_SIZE, 1);
        let src_ptr = alloc(src_layout);
        if src_ptr.is_null() {
            return;
        }

        write_bytes::<u8>(src_ptr, MAX_LUMA, LUMA_SIZE);
        write_bytes::<u8>(
            src_ptr.add(LUMA_SIZE),
            HALF_CHROMA,
            EXPECTED_SRC_BUFFER_SIZE - LUMA_SIZE,
        );

        let dst_layout = Layout::from_size_align_unchecked(EXPECTED_DST_BUFFER_SIZE, 1);
        let dst_ptr = alloc_zeroed(dst_layout);
        if !dst_ptr.is_null() {
            let src_image: &[u8] = from_raw_parts_mut(src_ptr, EXPECTED_SRC_BUFFER_SIZE);
            let dst_image = from_raw_parts_mut(dst_ptr, EXPECTED_DST_BUFFER_SIZE);

            // Touch output
            for i in (0..EXPECTED_DST_BUFFER_SIZE).step_by(PAGE_SIZE) {
                dst_image[i] = 0;
            }

            dst_image[dst_image.len() - 1] = 0;

            let src_buffers = &[src_image];
            let dst_buffers = &mut [&mut *dst_image];

            assert!(convert_image(
                WIDTH,
                HEIGHT,
                &src_format,
                None,
                src_buffers,
                &dst_format,
                None,
                dst_buffers,
            )
            .is_ok());

            // Check all samples are correct
            #[allow(clippy::cast_ptr_alignment)]
            let dst_image_as_u64: &[u64] =
                from_raw_parts(dst_ptr as *const u64, EXPECTED_DST_BUFFER_SIZE / 8);
            assert!(dst_image_as_u64.iter().all(|&x| x == std::u64::MAX));

            dealloc(dst_ptr, dst_layout);
        }

        dealloc(src_ptr, src_layout);
    }
}

#[test]
fn functional_tests() {
    for set in &SETS {
        bootstrap(set);

        buffers_size();

        yuv_to_rgb_errors(PixelFormat::Nv12);
        yuv_to_rgb_errors(PixelFormat::I420);
        yuv_to_rgb_errors(PixelFormat::I444);
        rgb_to_yuv_errors(PixelFormat::Nv12);
        rgb_to_yuv_errors(PixelFormat::I420);
        rgb_to_yuv_errors(PixelFormat::I444);
        rgb_errors();

        i444_to_rgb_ok();
        i420_to_rgb_ok();
        nv12_to_rgb_ok();
        rgb_to_i444_ok();
        rgb_to_i420_ok();
        rgb_to_nv12_ok();

        rgb_ok();
    }

    #[cfg(target_arch = "x86_64")]
    #[cfg(not(tarpaulin))]
    over_4gb();
}
