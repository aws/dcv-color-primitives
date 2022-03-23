use affinity::set_process_affinity;
use criterion::*;
use std::alloc::{alloc, dealloc, Layout};
use std::error;
use std::fmt;
use std::fs::{remove_file, OpenOptions};
use std::io::{Cursor, Read, Write};
use std::path::Path;
use std::slice::from_raw_parts_mut;
use std::time::Duration;
use std::time::Instant;
use thread_priority::{set_current_thread_priority, ThreadPriority};

use dcp::*;
use dcv_color_primitives as dcp;

const BGRA_INPUT: &[u8] = include_bytes!("input.bgra");
const RGB_INPUT: &[u8] = include_bytes!("input.rgb");

const BGRA_I420_OUTPUT: &str = "./benches/output_bgra.i420";
const BGRA_I444_OUTPUT: &str = "./benches/output_bgra.i444";
const BGR_I420_OUTPUT: &str = "./benches/output_bgr.i420";
const BGR_I444_OUTPUT: &str = "./benches/output_bgr.i444";
const I420_BGRA_OUTPUT: &str = "./benches/output_i420.bgra";
const I444_BGRA_OUTPUT: &str = "./benches/output_i444.bgra";
const BGRA_RGB_OUTPUT: &str = "./benches/output_bgra.rgb";
const RGB_BGRA_OUTPUT: &str = "./benches/output_rgb.bgra";

const SAMPLE_SIZE: usize = 22;
const PAGE_SIZE: usize = 4096;

struct BenchmarkInput<'a> {
    width: usize,
    height: usize,
    name: &'a str,
}

const INPUTS: &[BenchmarkInput] = &[
    BenchmarkInput {
        width: 640,
        height: 480,
        name: "480",
    },
    BenchmarkInput {
        width: 1280,
        height: 720,
        name: "720",
    },
    BenchmarkInput {
        width: 1920,
        height: 1080,
        name: "1080",
    },
    BenchmarkInput {
        width: 2560,
        height: 1440,
        name: "2k",
    },
    BenchmarkInput {
        width: 3840,
        height: 2160,
        name: "4k",
    },
    BenchmarkInput {
        width: 4096,
        height: 4096,
        name: "max",
    },
];

#[derive(Debug, Clone)]
struct BenchmarkError;

impl fmt::Display for BenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Generic benchmark error")
    }
}

impl error::Error for BenchmarkError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}

type BenchmarkResult<T> = std::result::Result<T, Box<dyn error::Error>>;
type Converter = fn(output_path: &str, width: usize, height: usize) -> BenchmarkResult<Duration>;
type InputConverter =
    fn(width: usize, height: usize, output_buffer: &mut [u8]) -> BenchmarkResult<()>;
type OutputConverter =
    fn(output_path: &str, width: usize, height: usize, source: &[u8]) -> BenchmarkResult<Duration>;

fn skip_line(file: &mut Cursor<&[u8]>) -> BenchmarkResult<()> {
    let mut byte = [0; 1];
    while byte[0] != 0xA {
        file.read_exact(&mut byte)?;
    }

    Ok(())
}

fn alloc_buffer<'a>(len: usize, commit: bool) -> &'a mut [u8] {
    #[allow(unsafe_code)]
    unsafe {
        let layout = Layout::from_size_align_unchecked(len, 64);
        let ptr = alloc(layout);

        if commit {
            for i in (0..len).step_by(PAGE_SIZE) {
                *ptr.add(i) = 0;
            }

            *ptr.add(len - 1) = 0;
        }

        from_raw_parts_mut(ptr, len)
    }
}

fn dealloc_buffer(buf: &mut [u8], len: usize) {
    #[allow(unsafe_code)]
    unsafe {
        let layout = Layout::from_size_align_unchecked(len, 64);
        let ptr = buf.as_mut_ptr();

        dealloc(ptr, layout)
    }
}

fn load_buffer(buf: &mut [u8], from: &[u8], skip_header: bool) -> BenchmarkResult<()> {
    let mut input_file = Cursor::new(from);

    if skip_header {
        skip_line(&mut input_file)?;
        skip_line(&mut input_file)?;
    }

    input_file.read_exact(buf)?;

    Ok(())
}

fn save_buffer(buf: &[u8], to: &str, width: usize, height: usize) -> BenchmarkResult<()> {
    if !Path::new(to).exists() {
        let mut file = OpenOptions::new().write(true).create(true).open(to)?;

        write!(file, "P5\n{} {}\n255\n", width, height)?;
        file.write_all(buf)?;
    }

    Ok(())
}

fn convert_buffer(
    width: usize,
    height: usize,
    src_format: &ImageFormat,
    input_data: &[&[u8]],
    dst_format: &ImageFormat,
    output_data: &mut [&mut [u8]],
) -> BenchmarkResult<Duration> {
    let start = Instant::now();

    convert_image(
        width as u32,
        height as u32,
        src_format,
        None,
        input_data,
        dst_format,
        None,
        output_data,
    )?;

    Ok(start.elapsed())
}

fn bgra_i420_input(width: usize, height: usize, output_buffer: &mut [u8]) -> BenchmarkResult<()> {
    let src_size = 4 * width * height;

    let input_buffer = alloc_buffer(src_size, false);
    load_buffer(input_buffer, BGRA_INPUT, true)?;

    let input_data: &[&[u8]] = &[&input_buffer];
    let (y_data, uv_data) = output_buffer.split_at_mut(width * height);
    let (u_data, v_data) = uv_data.split_at_mut(width * height / 4);
    let output_data: &mut [&mut [u8]] = &mut [&mut *y_data, &mut *u_data, &mut *v_data];

    let src_format = ImageFormat {
        pixel_format: PixelFormat::Bgra,
        color_space: ColorSpace::Lrgb,
        num_planes: 1,
    };
    let dst_format = ImageFormat {
        pixel_format: PixelFormat::I420,
        color_space: ColorSpace::Bt601,
        num_planes: 3,
    };

    convert_image(
        width as u32,
        height as u32,
        &src_format,
        None,
        input_data,
        &dst_format,
        None,
        output_data,
    )?;

    dealloc_buffer(input_buffer, src_size);

    Ok(())
}

fn bgra_i444_input(width: usize, height: usize, output_buffer: &mut [u8]) -> BenchmarkResult<()> {
    let src_size = 4 * width * height;

    let input_buffer = alloc_buffer(src_size, false);
    load_buffer(input_buffer, BGRA_INPUT, true)?;

    let input_data: &[&[u8]] = &[&input_buffer];
    let (y_data, uv_data) = output_buffer.split_at_mut(width * height);
    let (u_data, v_data) = uv_data.split_at_mut(width * height);
    let output_data: &mut [&mut [u8]] = &mut [&mut *y_data, &mut *u_data, &mut *v_data];

    let src_format = ImageFormat {
        pixel_format: PixelFormat::Bgra,
        color_space: ColorSpace::Lrgb,
        num_planes: 1,
    };
    let dst_format = ImageFormat {
        pixel_format: PixelFormat::I444,
        color_space: ColorSpace::Bt601,
        num_planes: 3,
    };

    convert_image(
        width as u32,
        height as u32,
        &src_format,
        None,
        input_data,
        &dst_format,
        None,
        output_data,
    )?;

    dealloc_buffer(input_buffer, src_size);

    Ok(())
}

fn bgra_i420(output_path: &str, width: usize, height: usize) -> BenchmarkResult<Duration> {
    let src_size = 4 * width * height;
    let dst_size = 3 * width * height / 2;

    let input_buffer = alloc_buffer(src_size, false);
    load_buffer(input_buffer, BGRA_INPUT, true)?;

    let output_buffer = alloc_buffer(dst_size, true);

    let input_data: &[&[u8]] = &[&input_buffer];
    let (y_data, uv_data) = output_buffer.split_at_mut(width * height);
    let (u_data, v_data) = uv_data.split_at_mut(width * height / 4);
    let output_data: &mut [&mut [u8]] = &mut [&mut *y_data, &mut *u_data, &mut *v_data];

    let src_format = ImageFormat {
        pixel_format: PixelFormat::Bgra,
        color_space: ColorSpace::Lrgb,
        num_planes: 1,
    };
    let dst_format = ImageFormat {
        pixel_format: PixelFormat::I420,
        color_space: ColorSpace::Bt601,
        num_planes: 3,
    };

    let elapsed = convert_buffer(
        width,
        height,
        &src_format,
        input_data,
        &dst_format,
        output_data,
    )?;

    save_buffer(output_buffer, output_path, width, height + height / 2)?;
    dealloc_buffer(input_buffer, src_size);
    dealloc_buffer(output_buffer, dst_size);

    Ok(elapsed)
}

fn bgra_i444(output_path: &str, width: usize, height: usize) -> BenchmarkResult<Duration> {
    let src_size = 4 * width * height;
    let dst_size = 3 * width * height;

    let input_buffer = alloc_buffer(src_size, false);
    load_buffer(input_buffer, BGRA_INPUT, true)?;

    let output_buffer = alloc_buffer(dst_size, true);

    let input_data: &[&[u8]] = &[&input_buffer];
    let (y_data, uv_data) = output_buffer.split_at_mut(width * height);
    let (u_data, v_data) = uv_data.split_at_mut(width * height);
    let output_data: &mut [&mut [u8]] = &mut [&mut *y_data, &mut *u_data, &mut *v_data];

    let src_format = ImageFormat {
        pixel_format: PixelFormat::Bgra,
        color_space: ColorSpace::Lrgb,
        num_planes: 1,
    };
    let dst_format = ImageFormat {
        pixel_format: PixelFormat::I444,
        color_space: ColorSpace::Bt601,
        num_planes: 3,
    };

    let elapsed = convert_buffer(
        width,
        height,
        &src_format,
        input_data,
        &dst_format,
        output_data,
    )?;

    save_buffer(output_buffer, output_path, width, 3 * height)?;
    dealloc_buffer(input_buffer, src_size);
    dealloc_buffer(output_buffer, dst_size);

    Ok(elapsed)
}

fn bgr_i420(output_path: &str, width: usize, height: usize) -> BenchmarkResult<Duration> {
    let src_size = 3 * width * height;
    let dst_size = 3 * width * height / 2;

    let input_buffer = alloc_buffer(src_size, false);
    load_buffer(input_buffer, RGB_INPUT, true)?;

    let output_buffer = alloc_buffer(dst_size, true);

    let input_data: &[&[u8]] = &[&input_buffer];
    let (y_data, uv_data) = output_buffer.split_at_mut(width * height);
    let (u_data, v_data) = uv_data.split_at_mut(width * height / 4);
    let output_data: &mut [&mut [u8]] = &mut [&mut *y_data, &mut *u_data, &mut *v_data];

    let src_format = ImageFormat {
        pixel_format: PixelFormat::Bgr,
        color_space: ColorSpace::Lrgb,
        num_planes: 1,
    };
    let dst_format = ImageFormat {
        pixel_format: PixelFormat::I420,
        color_space: ColorSpace::Bt601,
        num_planes: 3,
    };

    let elapsed = convert_buffer(
        width,
        height,
        &src_format,
        input_data,
        &dst_format,
        output_data,
    )?;

    save_buffer(output_buffer, output_path, width, height + height / 2)?;
    dealloc_buffer(input_buffer, src_size);
    dealloc_buffer(output_buffer, dst_size);

    Ok(elapsed)
}

fn bgr_i444(output_path: &str, width: usize, height: usize) -> BenchmarkResult<Duration> {
    let src_size = 3 * width * height;
    let dst_size = 3 * width * height;

    let input_buffer = alloc_buffer(src_size, false);
    load_buffer(input_buffer, RGB_INPUT, true)?;

    let output_buffer = alloc_buffer(dst_size, true);

    let input_data: &[&[u8]] = &[&input_buffer];
    let (y_data, uv_data) = output_buffer.split_at_mut(width * height);
    let (u_data, v_data) = uv_data.split_at_mut(width * height);
    let output_data: &mut [&mut [u8]] = &mut [&mut *y_data, &mut *u_data, &mut *v_data];

    let src_format = ImageFormat {
        pixel_format: PixelFormat::Bgr,
        color_space: ColorSpace::Lrgb,
        num_planes: 1,
    };
    let dst_format = ImageFormat {
        pixel_format: PixelFormat::I444,
        color_space: ColorSpace::Bt601,
        num_planes: 3,
    };

    let elapsed = convert_buffer(
        width,
        height,
        &src_format,
        input_data,
        &dst_format,
        output_data,
    )?;

    save_buffer(output_buffer, output_path, width, 3 * height)?;
    dealloc_buffer(input_buffer, src_size);
    dealloc_buffer(output_buffer, dst_size);

    Ok(elapsed)
}

fn i420_bgra(
    output_path: &str,
    width: usize,
    height: usize,
    source: &[u8],
) -> BenchmarkResult<Duration> {
    let src_size = 3 * width * height / 2;
    let dst_size = 4 * width * height;

    let input_buffer = alloc_buffer(src_size, false);
    load_buffer(input_buffer, source, false)?;

    let output_buffer = alloc_buffer(dst_size, true);

    let y_plane_size = width * height;
    let u_plane_size = y_plane_size / 4;
    let input_data: &[&[u8]] = &[
        &input_buffer[0..y_plane_size],
        &input_buffer[y_plane_size..(y_plane_size + u_plane_size)],
        &input_buffer[(y_plane_size + u_plane_size)..],
    ];
    let output_data: &mut [&mut [u8]] = &mut [&mut output_buffer[..]];

    let src_format = ImageFormat {
        pixel_format: PixelFormat::I420,
        color_space: ColorSpace::Bt601,
        num_planes: 3,
    };
    let dst_format = ImageFormat {
        pixel_format: PixelFormat::Bgra,
        color_space: ColorSpace::Lrgb,
        num_planes: 1,
    };

    let elapsed = convert_buffer(
        width,
        height,
        &src_format,
        input_data,
        &dst_format,
        output_data,
    )?;

    save_buffer(output_buffer, output_path, 4 * width, height)?;
    dealloc_buffer(input_buffer, src_size);
    dealloc_buffer(output_buffer, dst_size);

    Ok(elapsed)
}

fn i444_bgra(
    output_path: &str,
    width: usize,
    height: usize,
    source: &[u8],
) -> BenchmarkResult<Duration> {
    let src_size = 3 * width * height;
    let dst_size = 4 * width * height;

    let input_buffer = alloc_buffer(src_size, false);
    load_buffer(input_buffer, source, false)?;

    let output_buffer = alloc_buffer(dst_size, true);

    let y_plane_size = width * height;
    let u_plane_size = y_plane_size;
    let input_data: &[&[u8]] = &[
        &input_buffer[0..y_plane_size],
        &input_buffer[y_plane_size..(y_plane_size + u_plane_size)],
        &input_buffer[(y_plane_size + u_plane_size)..],
    ];
    let output_data: &mut [&mut [u8]] = &mut [&mut output_buffer[..]];

    let src_format = ImageFormat {
        pixel_format: PixelFormat::I444,
        color_space: ColorSpace::Bt601,
        num_planes: 3,
    };
    let dst_format = ImageFormat {
        pixel_format: PixelFormat::Bgra,
        color_space: ColorSpace::Lrgb,
        num_planes: 1,
    };

    let elapsed = convert_buffer(
        width,
        height,
        &src_format,
        input_data,
        &dst_format,
        output_data,
    )?;

    save_buffer(output_buffer, output_path, 4 * width, height)?;
    dealloc_buffer(input_buffer, src_size);
    dealloc_buffer(output_buffer, dst_size);

    Ok(elapsed)
}

fn bgra_rgb(output_path: &str, width: usize, height: usize) -> BenchmarkResult<Duration> {
    let src_size = 4 * width * height;
    let dst_size = 3 * width * height;

    let input_buffer = alloc_buffer(src_size, false);
    load_buffer(input_buffer, BGRA_INPUT, true)?;

    let output_buffer = alloc_buffer(dst_size, true);

    let input_data: &[&[u8]] = &[&input_buffer];
    let output_data: &mut [&mut [u8]] = &mut [&mut output_buffer[..]];

    let src_format = ImageFormat {
        pixel_format: PixelFormat::Bgra,
        color_space: ColorSpace::Lrgb,
        num_planes: 1,
    };
    let dst_format = ImageFormat {
        pixel_format: PixelFormat::Rgb,
        color_space: ColorSpace::Lrgb,
        num_planes: 1,
    };

    let elapsed = convert_buffer(
        width,
        height,
        &src_format,
        input_data,
        &dst_format,
        output_data,
    )?;

    save_buffer(output_buffer, output_path, 3 * width, height)?;
    dealloc_buffer(input_buffer, src_size);
    dealloc_buffer(output_buffer, dst_size);

    Ok(elapsed)
}

fn rgb_bgra(output_path: &str, width: usize, height: usize) -> BenchmarkResult<Duration> {
    let src_size = 3 * width * height;
    let dst_size = 4 * width * height;

    let input_buffer = alloc_buffer(src_size, false);
    load_buffer(input_buffer, RGB_INPUT, true)?;

    let output_buffer = alloc_buffer(dst_size, true);

    let input_data: &[&[u8]] = &[&input_buffer];
    let output_data: &mut [&mut [u8]] = &mut [&mut output_buffer[..]];

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

    let elapsed = convert_buffer(
        width,
        height,
        &src_format,
        input_data,
        &dst_format,
        output_data,
    )?;

    save_buffer(output_buffer, output_path, 4 * width, height)?;
    dealloc_buffer(input_buffer, src_size);
    dealloc_buffer(output_buffer, dst_size);

    Ok(elapsed)
}

fn configure_process() {
    let cores: Vec<usize> = (0..1).collect();

    set_process_affinity(&cores).unwrap();
    set_current_thread_priority(ThreadPriority::Max).unwrap();
}

fn convert_to(
    group: &mut BenchmarkGroup<measurement::WallTime>,
    name: &str,
    input: &BenchmarkInput,
    output_path: &str,
    func: Converter,
) {
    if Path::new(output_path).exists() {
        remove_file(Path::new(output_path)).expect("Unable to delete benchmark output");
    }

    group.bench_with_input(BenchmarkId::new(name, input.name), input, |b, i| {
        b.iter_custom(|iters| {
            let mut total = Duration::new(0, 0);
            for _i in 0..iters {
                total += func(output_path, i.width, i.height).expect("Benchmark iteration failed");
            }

            total
        });
    });
}

fn convert_from_to(
    group: &mut BenchmarkGroup<measurement::WallTime>,
    name: &str,
    input: &BenchmarkInput,
    output_path: &str,
    input_func: InputConverter,
    output_func: OutputConverter,
    num: u64,
    den: u64,
) {
    if Path::new(output_path).exists() {
        remove_file(Path::new(output_path)).expect("Unable to delete benchmark output");
    }

    let src_size = input.width * input.height * (num as usize) / (den as usize);
    let src_data = alloc_buffer(src_size, false);

    input_func(input.width, input.height, src_data).expect("Benchmark iteration failed");

    group.throughput(Throughput::Bytes(src_size as u64));
    group.bench_with_input(BenchmarkId::new(name, input.name), input, |b, i| {
        b.iter_custom(|iters| {
            let mut total = Duration::new(0, 0);
            for _i in 0..iters {
                total += output_func(output_path, i.width, i.height, src_data)
                    .expect("Benchmark iteration failed");
            }

            total
        });
    });

    dealloc_buffer(src_data, src_size);
}

fn from_bgra(c: &mut Criterion) {
    configure_process();
    initialize();

    let mut group = c.benchmark_group("bgra");

    group
        .sample_size(SAMPLE_SIZE)
        .warm_up_time(Duration::from_millis(64))
        .sampling_mode(SamplingMode::Flat);

    for input in INPUTS {
        group.throughput(Throughput::Bytes(
            4 * (input.width as u64) * (input.height as u64),
        ));

        convert_to(&mut group, "i420", input, BGRA_I420_OUTPUT, bgra_i420);
        convert_to(&mut group, "i444", input, BGRA_I444_OUTPUT, bgra_i444);
        convert_to(&mut group, "rgb", input, BGRA_RGB_OUTPUT, bgra_rgb);
    }

    group.finish();
}

fn from_bgr(c: &mut Criterion) {
    configure_process();
    initialize();

    let mut group = c.benchmark_group("bgr");

    group
        .sample_size(SAMPLE_SIZE)
        .warm_up_time(Duration::from_millis(64))
        .sampling_mode(SamplingMode::Flat);

    for input in INPUTS {
        group.throughput(Throughput::Bytes(
            3 * (input.width as u64) * (input.height as u64),
        ));

        convert_to(&mut group, "i420", input, BGR_I420_OUTPUT, bgr_i420);
        convert_to(&mut group, "i444", input, BGR_I444_OUTPUT, bgr_i444);
        convert_to(&mut group, "bgra", input, RGB_BGRA_OUTPUT, rgb_bgra);
    }

    group.finish();
}

fn from_i420(c: &mut Criterion) {
    configure_process();
    initialize();

    let mut group = c.benchmark_group("i420");

    group
        .sample_size(SAMPLE_SIZE)
        .warm_up_time(Duration::from_millis(64))
        .sampling_mode(SamplingMode::Flat);

    for input in INPUTS {
        group.throughput(Throughput::Bytes(
            3 * (input.width as u64) * (input.height as u64) / 2,
        ));

        convert_from_to(
            &mut group,
            "bgra",
            input,
            I420_BGRA_OUTPUT,
            bgra_i420_input,
            i420_bgra,
            3,
            2,
        );
    }

    group.finish();
}

fn from_i444(c: &mut Criterion) {
    configure_process();
    initialize();

    let mut group = c.benchmark_group("i444");

    group
        .sample_size(SAMPLE_SIZE)
        .warm_up_time(Duration::from_millis(64))
        .sampling_mode(SamplingMode::Flat);

    for input in INPUTS {
        group.throughput(Throughput::Bytes(
            3 * (input.width as u64) * (input.height as u64),
        ));

        convert_from_to(
            &mut group,
            "bgra",
            input,
            I444_BGRA_OUTPUT,
            bgra_i444_input,
            i444_bgra,
            3,
            1,
        );
    }

    group.finish();
}

criterion_group!(benches, from_bgra, from_bgr, from_i420, from_i444);
criterion_main!(benches);
