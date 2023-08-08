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

#[cfg(all(
    target_arch = "x86_64",
    not(tarpaulin),
    not(feature = "test_instruction_sets")
))]
use std::{
    alloc::{alloc, alloc_zeroed, dealloc, Layout},
    ptr::write_bytes,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use dcp::{
    describe_acceleration, get_buffers_size, ColorSpace, ErrorKind, ImageFormat, PixelFormat,
};
use dcv_color_primitives as dcp;
use itertools::iproduct;

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

const PIXEL_FORMAT_I422: u32 = PixelFormat::I422 as u32;
const PIXEL_FORMAT_I420: u32 = PixelFormat::I420 as u32;

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

fn check_err(result: ErrorKind, err: ErrorKind) {
    assert_eq!(result as u32, err as u32);
}

#[test]
fn bootstrap() {
    println!("{}", describe_acceleration());
}

#[test]
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
                            PixelFormat::I422 | PixelFormat::I420 => i32::from(i > 0),
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

#[cfg(all(
    target_arch = "x86_64",
    not(tarpaulin),
    not(feature = "test_instruction_sets")
))]
#[test]
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

            assert!(dcp::convert_image(
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
