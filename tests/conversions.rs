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

use dcp::{ColorSpace, ImageFormat, PixelFormat, STRIDE_AUTO, convert_image};

use dcv_color_primitives as dcp;
use itertools::iproduct;
use rand::Rng;

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
    &[200, 97, 186, 96, 192, 136, 77, 45],
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
    &[115, 116, 214, 86, 137, 163, 92, 96],
    &[95, 40, 186, 131, 166, 95, 116, 68],
    &[54, 119, 145, 84, 149, 159, 192, 55],
    &[117, 180, 135, 158, 133, 101, 193, 185],
    &[201, 97, 199, 99, 201, 131, 75, 31],
    &[50, 81, 208, 75, 182, 81, 52, 106],
    &[90, 49, 121, 215, 77, 205, 123, 153],
    &[192, 213, 192, 94, 142, 36, 206, 87],
];

const CR_BT601FR_REF: &[&[u8]] = &[
    &[195, 102, 183, 208, 219, 99, 117, 136],
    &[243, 167, 149, 81, 72, 234, 116, 108],
    &[195, 80, 113, 117, 143, 167, 172, 174],
    &[169, 126, 84, 177, 78, 77, 75, 86],
    &[191, 99, 68, 75, 119, 185, 92, 170],
    &[77, 221, 224, 162, 190, 51, 83, 117],
    &[136, 159, 88, 189, 63, 133, 182, 104],
    &[161, 134, 119, 121, 194, 157, 56, 159],
];

const CB2_BT601FR_REF: &[&[u8]] = &[
    &[92, 154, 140, 93],
    &[118, 130, 136, 156],
    &[107, 145, 149, 66],
    &[136, 156, 115, 142],
];

const CR2_BT601FR_REF: &[&[u8]] = &[
    &[177, 155, 156, 120],
    &[142, 123, 116, 127],
    &[147, 132, 136, 116],
    &[147, 129, 137, 125],
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
    &[122, 113, 222, 94, 148, 160, 91, 96],
    &[108, 43, 189, 125, 160, 107, 114, 65],
    &[60, 114, 143, 82, 152, 164, 198, 59],
    &[122, 181, 130, 164, 128, 94, 188, 181],
    &[210, 93, 194, 92, 201, 138, 70, 34],
    &[43, 91, 221, 77, 190, 71, 45, 104],
    &[90, 51, 116, 224, 69, 207, 130, 150],
    &[197, 215, 193, 92, 149, 37, 199, 90],
];

const CR_BT709FR_REF: &[&[u8]] = &[
    &[196, 100, 191, 206, 222, 101, 114, 134],
    &[243, 161, 154, 80, 74, 234, 115, 103],
    &[191, 78, 114, 114, 145, 171, 177, 170],
    &[169, 130, 83, 181, 77, 74, 78, 89],
    &[198, 96, 72, 72, 125, 187, 87, 164],
    &[70, 220, 232, 159, 196, 46, 76, 115],
    &[133, 154, 86, 198, 57, 138, 183, 105],
    &[167, 140, 124, 119, 197, 151, 60, 157],
];

const CB2_BT709FR_REF: &[&[u8]] = &[
    &[97, 158, 144, 91],
    &[119, 130, 134, 156],
    &[109, 146, 150, 63],
    &[138, 156, 116, 142],
];

const CR2_BT709FR_REF: &[&[u8]] = &[
    &[175, 158, 158, 117],
    &[142, 123, 117, 129],
    &[146, 134, 138, 111],
    &[148, 132, 136, 126],
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

type PlaneRef<'a> = &'a [&'a [u8]];

fn get_depth(pixel_format: PixelFormat) -> usize {
    match pixel_format {
        PixelFormat::Bgra | PixelFormat::Argb => 4,
        _ => 3,
    }
}

fn check_plane(plane: &[u8], reference: PlaneRef, width: usize, height: usize, stride: usize) {
    for (row, exp) in plane.chunks(stride).zip(reference.iter().take(height)) {
        let (payload, pad) = row.split_at(width);
        assert!(
            payload
                .iter()
                .zip(exp.iter().take(width))
                .all(|(&x, &y)| x == y)
        );
        assert!(pad.iter().all(|&x| x == 0));
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

    assert!(
        convert_image(
            image_size.0,
            image_size.1,
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

    check_plane(dst_buffers[0], plane_ref.0, w, h, y_stride);
    match dst_format.pixel_format {
        PixelFormat::Nv12 => {
            for (uv_row, (u_exp, v_exp)) in dst_image[y_stride * h..]
                .chunks(u_stride)
                .zip(plane_ref.1.iter().zip(plane_ref.2).take(ch))
            {
                let (payload, pad) = uv_row.split_at(w);
                assert!(
                    payload
                        .chunks(2)
                        .zip(u_exp.iter().zip(v_exp.iter().take(cw)))
                        .all(|(uv, (&u, &v))| uv[0] == u && uv[1] == v)
                );
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

fn yuv_to_bgra_size_format_mode_stride(
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

    assert!(
        convert_image(
            image_size.0,
            image_size.1,
            src_format,
            Some(&src_strides[..]),
            &src_buffers[..],
            dst_format,
            Some(&[dst_stride]),
            &mut [&mut dst_image[..]],
        )
        .is_ok()
    );

    let (r_offset, b_offset) = match dst_format.pixel_format {
        PixelFormat::Bgra => (0, 2),
        _ => (2, 0),
    };

    i = 0;
    while i < dst_size {
        let pack_stride = 4 * w;

        for col in 0..w {
            let src = 4 * col;
            let dst = 3 * col;
            assert!(
                (i32::from(dst_image[i + src + r_offset]) - expected_row[dst]).abs() <= 2
                    && (i32::from(dst_image[i + src + 1]) - expected_row[dst + 1]).abs() <= 2
                    && (i32::from(dst_image[i + src + b_offset]) - expected_row[dst + 2]).abs()
                        <= 2
                    && dst_image[i + src + 3] == 255
            );
        }

        for col in 0..dst_pad {
            assert_eq!(dst_image[i + pack_stride + col], 0);
        }

        i += pack_stride + dst_pad;
    }
}

fn yuv_to_bgra_size_format_mode(
    image_size: (u32, u32),
    src_format: &ImageFormat,
    dst_format: &ImageFormat,
) {
    const MAX_PAD: usize = 4;

    if src_format.num_planes == 1 {
        for (y_pad, dst_pad) in iproduct!(0..MAX_PAD, 0..MAX_PAD) {
            yuv_to_bgra_size_format_mode_stride(
                image_size,
                src_format,
                dst_format,
                (y_pad, y_pad, y_pad, dst_pad),
            );
        }
    } else if src_format.num_planes == 2 {
        for (y_pad, uv_pad, dst_pad) in iproduct!(0..MAX_PAD, 0..MAX_PAD, 0..MAX_PAD) {
            yuv_to_bgra_size_format_mode_stride(
                image_size,
                src_format,
                dst_format,
                (y_pad, uv_pad, uv_pad, dst_pad),
            );
        }
    } else if src_format.num_planes == 3 {
        for pad in iproduct!(0..MAX_PAD, 0..MAX_PAD, 0..MAX_PAD, 0..MAX_PAD) {
            yuv_to_bgra_size_format_mode_stride(image_size, src_format, dst_format, pad);
        }
    }
}

fn yuv_to_bgra_ok(pixel_format: PixelFormat, num_planes: u32) {
    const SUPPORTED_COLOR_SPACES: &[ColorSpace] = &[
        ColorSpace::Bt601,
        ColorSpace::Bt709,
        ColorSpace::Bt601FR,
        ColorSpace::Bt709FR,
    ];
    const SUPPORTED_FORMATS: &[PixelFormat] = &[PixelFormat::Bgra, PixelFormat::Rgba];
    const MAX_WIDTH: u32 = 34;
    const MAX_HEIGHT: u32 = 4;

    let step = match pixel_format {
        PixelFormat::I444 => 1,
        _ => 2,
    };

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

        for width in (0..=MAX_WIDTH).step_by(step) {
            for height in (0..=MAX_HEIGHT).step_by(step) {
                yuv_to_bgra_size_format_mode((width, height), &src_format, &dst_format);
            }
        }
    }
}

fn yuv_to_rgb_size_format_mode_stride(
    image_size: (u32, u32),
    src_format: &ImageFormat,
    dst_format: &ImageFormat,
    pad: (usize, usize, usize),
) {
    let (y_pad, u_pad, dst_pad) = pad;
    let w = image_size.0 as usize;
    let h = image_size.1 as usize;

    let y_stride = w + y_pad;
    let u_stride = w + u_pad;
    let ch = h / 2;
    let src_size = y_stride * h + u_stride * ch;

    let dst_stride = w * 3 + dst_pad;
    let dst_size = dst_stride * h;
    let color_space_index = match src_format.color_space {
        ColorSpace::Bt601 => 0,
        ColorSpace::Bt709 => 1,
        ColorSpace::Bt601FR => 2,
        _ => 3,
    };

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

    while i < src_size {
        for x in (0..w).step_by(2) {
            let index = (x >> 1) & 0x7;
            src_image[i + x] = U_SRC[color_space_index][index];
            src_image[i + x + 1] = V_SRC[color_space_index][index];
        }

        i += u_stride;
    }

    let mut dst_image = vec![0_u8; dst_size].into_boxed_slice();
    let dst_stride = if dst_pad == 0 {
        STRIDE_AUTO
    } else {
        dst_stride
    };

    let mut src_buffers: Vec<&[u8]> = Vec::with_capacity(2);
    let mut src_strides = Vec::with_capacity(2);
    src_strides.push(if y_pad == 0 { STRIDE_AUTO } else { y_stride });
    if src_format.num_planes == 1 {
        src_buffers.push(&src_image);
    } else if src_format.num_planes == 2 {
        src_strides.push(if u_pad == 0 { STRIDE_AUTO } else { u_stride });

        let (first, last) = src_image.split_at(y_stride * h);
        src_buffers.push(first);
        src_buffers.push(last);
    }

    let mut expected_row = vec![0_i32; 3 * w].into_boxed_slice();
    for (x, pixel) in expected_row.chunks_exact_mut(3).enumerate() {
        let index = (x >> 1) & 7;

        pixel[0] = if (index & 1) == 0 { 0 } else { 255 };
        pixel[1] = if ((index >> 1) & 1) == 0 { 0 } else { 255 };
        pixel[2] = if ((index >> 2) & 1) == 0 { 0 } else { 255 };
    }

    assert!(
        convert_image(
            image_size.0,
            image_size.1,
            src_format,
            Some(&src_strides[..]),
            &src_buffers[..],
            dst_format,
            Some(&[dst_stride]),
            &mut [&mut dst_image[..]],
        )
        .is_ok()
    );

    let pack_stride = 3 * w;
    i = 0;
    while i < dst_size {
        for col in 0..w {
            let off = 3 * col;
            assert!(
                (i32::from(dst_image[i + off]) - expected_row[off]).abs() <= 2
                    && (i32::from(dst_image[i + off + 1]) - expected_row[off + 1]).abs() <= 2
                    && (i32::from(dst_image[i + off + 2]) - expected_row[off + 2]).abs() <= 2
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
                (y_pad, y_pad, dst_pad),
            );
        }
    } else {
        for (y_pad, uv_pad, dst_pad) in iproduct!(0..MAX_PAD, 0..MAX_PAD, 0..MAX_PAD) {
            yuv_to_rgb_size_format_mode_stride(
                image_size,
                src_format,
                dst_format,
                (y_pad, uv_pad, dst_pad),
            );
        }
    }
}

fn yuv_to_rgb_ok(num_planes: u32) {
    const SUPPORTED_COLOR_SPACES: &[ColorSpace] = &[
        ColorSpace::Bt601,
        ColorSpace::Bt709,
        ColorSpace::Bt601FR,
        ColorSpace::Bt709FR,
    ];
    const MAX_WIDTH: u32 = 34;
    const MAX_HEIGHT: u32 = 4;

    let dst_format = ImageFormat {
        pixel_format: PixelFormat::Rgb,
        color_space: ColorSpace::Rgb,
        num_planes: 1,
    };

    for color_space in SUPPORTED_COLOR_SPACES {
        let src_format = ImageFormat {
            pixel_format: PixelFormat::Nv12,
            color_space: *color_space,
            num_planes,
        };

        for width in (0..=MAX_WIDTH).step_by(2) {
            for height in (0..=MAX_HEIGHT).step_by(2) {
                yuv_to_rgb_size_format_mode((width, height), &src_format, &dst_format);
            }
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
    fn nv12_to_rgbx_1() {
        yuv_to_bgra_ok(PixelFormat::Nv12, 1);
    }

    #[test]
    fn nv12_to_rgbx_2() {
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
    fn rgb_to_nv12_1() {
        rgb_to_yuv_ok(PixelFormat::Nv12, 1);
    }

    #[test]
    fn rgb_to_nv12_2() {
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
    fn nv12_to_rgb_1() {
        yuv_to_rgb_ok(1);
    }

    #[test]
    fn nv12_to_rgb_2() {
        yuv_to_rgb_ok(2);
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
            rgb_to_yuv_ok(PixelFormat::Nv12, 1);
            rgb_to_yuv_ok(PixelFormat::Nv12, 2);
            yuv_to_bgra_ok(PixelFormat::I420, 3);
            yuv_to_bgra_ok(PixelFormat::I444, 3);
            yuv_to_bgra_ok(PixelFormat::Nv12, 1);
            yuv_to_bgra_ok(PixelFormat::Nv12, 2);
            yuv_to_rgb_ok(1);
            yuv_to_rgb_ok(2);
        }
    }
}
