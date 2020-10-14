use criterion::*;
use std::error;
use std::fmt;
use std::fs::{remove_file, OpenOptions};
use std::io::BufRead;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::Duration;
use std::time::Instant;

use dcp::*;
use dcv_color_primitives as dcp;

const NV12_OUTPUT: &str = &"./output.nv12";
const BGRA_OUTPUT: &str = &"./output.bgra";
const RGB_BGRA_OUTPUT: &str = &"./rgb_output.bgra";
const BGRA_RGB_OUTPUT: &str = &"./bgra_rgb_output.rgb";
const I420_OUTPUT: &str = &"./i420_output.bgra";
const I444_OUTPUT: &str = &"./i444_output.bgra";
const BGRA_I420_OUTPUT: &str = &"./bgra_i420_output.i420";
const BGRA_I444_OUTPUT: &str = &"./bgra_i444_output.i444";

const SAMPLE_SIZE: usize = 22;
const PAGE_SIZE: usize = 4096;

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

fn skip_line(file: &mut Cursor<&[u8]>) -> BenchmarkResult<()> {
    let mut byte = [0; 1];
    while byte[0] != 0xA {
        file.read(&mut byte)?;
    }

    Ok(())
}

fn read_line(file: &mut Cursor<&[u8]>) -> BenchmarkResult<String> {
    let mut string = String::new();
    let written_chars = file.read_line(&mut string)?;

    let result = match written_chars {
        0 => Err(BenchmarkError),
        _ => Ok(string),
    };

    result.map_err(|e| e.into())
}

fn pnm_size(file: &mut Cursor<&[u8]>) -> BenchmarkResult<(u32, u32)> {
    file.seek(SeekFrom::Start(0))?;
    skip_line(file)?;

    let dimensions: Vec<_> = read_line(file)?
        .split_whitespace()
        .map(|s| s.parse::<u32>().unwrap())
        .collect();

    let width = dimensions.get(0).ok_or(BenchmarkError)?;
    let height = dimensions.get(1).ok_or(BenchmarkError)?;
    Ok((*width, *height))
}

fn pnm_data(file: &mut Cursor<&[u8]>) -> BenchmarkResult<(u32, u32, Vec<u8>)> {
    file.seek(SeekFrom::Start(0))?;
    let (width, height) = pnm_size(file)?;

    let size: usize = (width as usize) * (height as usize);
    let mut x: Vec<u8> = Vec::with_capacity(size);
    skip_line(file)?;
    file.read_to_end(&mut x)?;

    Ok((width, height, x))
}

fn bgra_nv12(mut input_file: &mut Cursor<&[u8]>, output_path: &str) -> BenchmarkResult<Duration> {
    let (mut width, height, input_buffer) = { pnm_data(&mut input_file)? };
    width /= 4;

    // Allocate output
    let dst_size: usize = 3 * (width as usize) * (height as usize) / 2;
    let mut output_buffer: Vec<u8> = vec![0; dst_size];
    for i in (0..dst_size).step_by(PAGE_SIZE) {
        output_buffer[i] = 0;
    }

    let input_data: &[&[u8]] = &[&input_buffer];
    let output_data: &mut [&mut [u8]] = &mut [&mut output_buffer[..]];

    let src_format = ImageFormat {
        pixel_format: PixelFormat::Bgra,
        color_space: ColorSpace::Lrgb,
        num_planes: 1,
    };

    let dst_format = ImageFormat {
        pixel_format: PixelFormat::Nv12,
        color_space: ColorSpace::Bt601,
        num_planes: 1,
    };

    let start = Instant::now();
    convert_image(
        width,
        height,
        &src_format,
        None,
        input_data,
        &dst_format,
        None,
        output_data,
    )?;

    let elapsed = start.elapsed();

    if !Path::new(output_path).exists() {
        let mut buffer = OpenOptions::new()
            .write(true)
            .create(true)
            .open(output_path)?;
        write!(buffer, "P5\n{} {}\n255\n", width, height + height / 2)?;
        buffer.write(&output_buffer)?;
    }

    Ok(elapsed)
}

fn bgra_i420(mut input_file: &mut Cursor<&[u8]>, output_path: &str) -> BenchmarkResult<Duration> {
    let (mut width, height, input_buffer) = { pnm_data(&mut input_file)? };
    width /= 4;
    let w: usize = width as usize;
    let h: usize = height as usize;

    // Allocate output
    let dst_size: usize = 3 * (width as usize) * (height as usize) / 2;
    let mut output_buffer: Vec<u8> = vec![0; dst_size];
    for i in (0..dst_size).step_by(PAGE_SIZE) {
        output_buffer[i] = 0;
    }

    let input_data: &[&[u8]] = &[&input_buffer];
    let (y_data, uv_data) = output_buffer.split_at_mut(w * h);
    let (u_data, v_data) = uv_data.split_at_mut(w * h / 4);
    let output_data: &mut [&mut [u8]] = &mut [&mut y_data[..], &mut u_data[..], &mut v_data[..]];

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

    let start = Instant::now();
    convert_image(
        width,
        height,
        &src_format,
        None,
        input_data,
        &dst_format,
        None,
        output_data,
    )?;

    let elapsed = start.elapsed();

    if !Path::new(output_path).exists() {
        let mut buffer = OpenOptions::new()
            .write(true)
            .create(true)
            .open(output_path)?;
        write!(buffer, "P5\n{} {}\n255\n", width, height + height / 2)?;
        buffer.write(&output_buffer)?;
    }

    Ok(elapsed)
}

fn bgra_i444(mut input_file: &mut Cursor<&[u8]>, output_path: &str) -> BenchmarkResult<Duration> {
    let (mut width, height, input_buffer) = { pnm_data(&mut input_file)? };
    width /= 4;
    let w: usize = width as usize;
    let h: usize = height as usize;

    // Allocate output
    let dst_size: usize = 3 * (width as usize) * (height as usize);
    let mut output_buffer: Vec<u8> = vec![0; dst_size];
    for i in (0..dst_size).step_by(PAGE_SIZE) {
        output_buffer[i] = 0;
    }

    let input_data: &[&[u8]] = &[&input_buffer];
    let (y_data, uv_data) = output_buffer.split_at_mut(w * h);
    let (u_data, v_data) = uv_data.split_at_mut(w * h);
    let output_data: &mut [&mut [u8]] = &mut [&mut y_data[..], &mut u_data[..], &mut v_data[..]];

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

    let start = Instant::now();
    convert_image(
        width,
        height,
        &src_format,
        None,
        input_data,
        &dst_format,
        None,
        output_data,
    )?;

    let elapsed = start.elapsed();

    if !Path::new(output_path).exists() {
        let mut buffer = OpenOptions::new()
            .write(true)
            .create(true)
            .open(output_path)?;
        write!(buffer, "P5\n{} {}\n255\n", width, height + height + height)?;
        buffer.write(&output_buffer)?;
    }

    Ok(elapsed)
}

fn nv12_bgra(mut input_file: &mut Cursor<&[u8]>, output_path: &str) -> BenchmarkResult<Duration> {
    let (width, mut height, input_buffer) = { pnm_data(&mut input_file)? };
    height = 2 * height / 3;

    // Allocate output
    let dst_size: usize = 4 * (width as usize) * (height as usize);
    let mut output_buffer: Vec<u8> = vec![0; dst_size];
    for i in (0..dst_size).step_by(PAGE_SIZE) {
        output_buffer[i] = 0;
    }

    let input_data: &[&[u8]] = &[&input_buffer];
    let output_data: &mut [&mut [u8]] = &mut [&mut output_buffer[..]];

    let src_format = ImageFormat {
        pixel_format: PixelFormat::Nv12,
        color_space: ColorSpace::Bt601,
        num_planes: 1,
    };

    let dst_format = ImageFormat {
        pixel_format: PixelFormat::Bgra,
        color_space: ColorSpace::Lrgb,
        num_planes: 1,
    };

    let start = Instant::now();
    convert_image(
        width,
        height,
        &src_format,
        None,
        input_data,
        &dst_format,
        None,
        output_data,
    )?;

    let elapsed = start.elapsed();

    // Write to file
    if !Path::new(output_path).exists() {
        let mut buffer = OpenOptions::new()
            .write(true)
            .create(true)
            .open(output_path)?;
        write!(buffer, "P5\n{} {}\n255\n", 4 * width, height)?;
        buffer.write(&output_buffer)?;
    }

    Ok(elapsed)
}

fn rgb_bgra(mut input_file: &mut Cursor<&[u8]>, output_path: &str) -> BenchmarkResult<Duration> {
    let (mut width, height, input_buffer) = { pnm_data(&mut input_file)? };
    width /= 3;

    // Allocate output
    let dst_size: usize = 4 * (width as usize) * (height as usize);
    let mut output_buffer: Vec<u8> = vec![0; dst_size];
    for i in (0..dst_size).step_by(PAGE_SIZE) {
        output_buffer[i] = 0;
    }

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

    let start = Instant::now();
    convert_image(
        width,
        height,
        &src_format,
        None,
        input_data,
        &dst_format,
        None,
        output_data,
    )?;

    let elapsed = start.elapsed();

    // Write to file
    if !Path::new(output_path).exists() {
        let mut buffer = OpenOptions::new()
            .write(true)
            .create(true)
            .open(output_path)?;
        write!(buffer, "P5\n{} {}\n255\n", 4 * width, height)?;
        buffer.write(&output_buffer)?;
    }

    Ok(elapsed)
}

fn bgra_rgb(mut input_file: &mut Cursor<&[u8]>, output_path: &str) -> BenchmarkResult<Duration> {
    let (mut width, height, input_buffer) = { pnm_data(&mut input_file)? };
    width /= 4;

    // Allocate output
    let dst_size: usize = 3 * (width as usize) * (height as usize);
    let mut output_buffer: Vec<u8> = vec![0; dst_size];
    for i in (0..dst_size).step_by(PAGE_SIZE) {
        output_buffer[i] = 0;
    }

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

    let start = Instant::now();
    convert_image(
        width,
        height,
        &src_format,
        None,
        input_data,
        &dst_format,
        None,
        output_data,
    )?;

    let elapsed = start.elapsed();

    // Write to file
    if !Path::new(output_path).exists() {
        let mut buffer = OpenOptions::new()
            .write(true)
            .create(true)
            .open(output_path)?;
        write!(buffer, "P5\n{} {}\n255\n", 3 * width, height)?;
        buffer.write(&output_buffer)?;
    }

    Ok(elapsed)
}

fn i420_bgra(mut input_file: &mut Cursor<&[u8]>, output_path: &str) -> BenchmarkResult<Duration> {
    let (width, mut height, input_buffer) = { pnm_data(&mut input_file)? };
    height = 2 * height / 3;

    // Allocate output
    let dst_size: usize = 4 * (width as usize) * (height as usize);
    let y_plane_size: usize = (width as usize) * (height as usize);
    let u_plane_size: usize = y_plane_size / 4;
    let mut output_buffer: Vec<u8> = vec![0; dst_size];
    for i in (0..dst_size).step_by(PAGE_SIZE) {
        output_buffer[i] = 0;
    }

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

    let start = Instant::now();
    convert_image(
        width,
        height,
        &src_format,
        None,
        input_data,
        &dst_format,
        None,
        output_data,
    )?;

    let elapsed = start.elapsed();

    // Write to file
    if !Path::new(output_path).exists() {
        let mut buffer = OpenOptions::new()
            .write(true)
            .create(true)
            .open(output_path)?;
        write!(buffer, "P5\n{} {}\n255\n", 4 * width, height)?;
        buffer.write(&output_buffer)?;
    }

    Ok(elapsed)
}

fn i444_bgra(mut input_file: &mut Cursor<&[u8]>, output_path: &str) -> BenchmarkResult<Duration> {
    let (width, mut height, input_buffer) = { pnm_data(&mut input_file)? };
    height = height / 3;

    // Allocate output
    let dst_size: usize = 4 * (width as usize) * (height as usize);
    let mut output_buffer: Vec<u8> = vec![0; dst_size];
    for i in (0..dst_size).step_by(PAGE_SIZE) {
        output_buffer[i] = 0;
    }

    let y_size = (width as usize) * (height as usize);
    let input_data: &[&[u8]] = &[
        &input_buffer[0..y_size],
        &input_buffer[y_size..(2 * y_size)],
        &input_buffer[(2 * y_size)..],
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

    let start = Instant::now();
    convert_image(
        width,
        height,
        &src_format,
        None,
        input_data,
        &dst_format,
        None,
        output_data,
    )?;

    let elapsed = start.elapsed();

    // Write to file
    if !Path::new(output_path).exists() {
        let mut buffer = OpenOptions::new()
            .write(true)
            .create(true)
            .open(output_path)?;
        write!(buffer, "P5\n{} {}\n255\n", 4 * width, height)?;
        buffer.write(&output_buffer)?;
    }

    Ok(elapsed)
}

fn bench(c: &mut Criterion) {
    initialize();

    let mut group = c.benchmark_group("dcv-color-primitives");
    group.sample_size(SAMPLE_SIZE);

    {
        let output_path = &NV12_OUTPUT;
        if Path::new(output_path).exists() {
            remove_file(Path::new(output_path)).expect("Unable to delete benchmark output");
        }

        let mut input_file: Cursor<&[u8]> = Cursor::new(include_bytes!("input.bgra"));
        let (width, height) =
            { pnm_size(&mut input_file).expect("Malformed benchmark input file") };
        group.throughput(Throughput::Elements((width as u64) * (height as u64)));
        group.bench_function("bgra>nv12", move |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::new(0, 0);
                for _i in 0..iters {
                    total += bgra_nv12(&mut input_file, output_path)
                        .expect("Benchmark iteration failed");
                }

                total
            });
        });
    }

    {
        let output_path = &BGRA_I420_OUTPUT;
        if Path::new(output_path).exists() {
            remove_file(Path::new(output_path)).expect("Unable to delete benchmark output");
        }

        let mut input_file: Cursor<&[u8]> = Cursor::new(include_bytes!("input.bgra"));
        let (width, height) =
            { pnm_size(&mut input_file).expect("Malformed benchmark input file") };
        group.throughput(Throughput::Elements((width as u64) * (height as u64)));
        group.bench_function("bgra>i420", move |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::new(0, 0);
                for _i in 0..iters {
                    total += bgra_i420(&mut input_file, output_path)
                        .expect("Benchmark iteration failed");
                }

                total
            });
        });
    }

    {
        let output_path = &BGRA_I444_OUTPUT;
        if Path::new(output_path).exists() {
            remove_file(Path::new(output_path)).expect("Unable to delete benchmark output");
        }

        let mut input_file: Cursor<&[u8]> = Cursor::new(include_bytes!("input.bgra"));
        let (width, height) =
            { pnm_size(&mut input_file).expect("Malformed benchmark input file") };
        group.throughput(Throughput::Elements((width as u64) * (height as u64)));
        group.bench_function("bgra>i444", move |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::new(0, 0);
                for _i in 0..iters {
                    total += bgra_i444(&mut input_file, output_path)
                        .expect("Benchmark iteration failed");
                }

                total
            });
        });
    }

    {
        let output_path = &BGRA_OUTPUT;
        if Path::new(output_path).exists() {
            remove_file(Path::new(output_path)).expect("Unable to delete benchmark output");
        }

        let mut input_file: Cursor<&[u8]> = Cursor::new(include_bytes!("input.nv12"));
        let (width, height) =
            { pnm_size(&mut input_file).expect("Malformed benchmark input file") };
        group.throughput(Throughput::Elements((width as u64) * (height as u64)));
        group.bench_function("nv12>bgra", move |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::new(0, 0);
                for _i in 0..iters {
                    total += nv12_bgra(&mut input_file, output_path)
                        .expect("Benchmark iteration failed");
                }

                total
            });
        });
    }

    {
        let output_path = &RGB_BGRA_OUTPUT;
        if Path::new(output_path).exists() {
            remove_file(Path::new(output_path)).expect("Unable to delete benchmark output");
        }

        let mut input_file: Cursor<&[u8]> = Cursor::new(include_bytes!("input.rgb"));
        let (width, height) =
            { pnm_size(&mut input_file).expect("Malformed benchmark input file") };
        group.throughput(Throughput::Elements((width as u64) * (height as u64)));
        group.bench_function("rgb>bgra", move |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::new(0, 0);
                for _i in 0..iters {
                    total +=
                        rgb_bgra(&mut input_file, output_path).expect("Benchmark iteration failed");
                }

                total
            });
        });
    }

    {
        let output_path = &BGRA_RGB_OUTPUT;
        if Path::new(output_path).exists() {
            remove_file(Path::new(output_path)).expect("Unable to delete benchmark output");
        }

        let mut input_file: Cursor<&[u8]> = Cursor::new(include_bytes!("input.bgra"));
        let (width, height) =
            { pnm_size(&mut input_file).expect("Malformed benchmark input file") };
        group.throughput(Throughput::Elements((width as u64) * (height as u64)));
        group.bench_function("bgra>rgb", move |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::new(0, 0);
                for _i in 0..iters {
                    total +=
                        bgra_rgb(&mut input_file, output_path).expect("Benchmark iteration failed");
                }

                total
            });
        });
    }

    {
        let output_path = &I420_OUTPUT;
        if Path::new(output_path).exists() {
            remove_file(Path::new(output_path)).expect("Unable to delete benchmark output");
        }

        let mut input_file: Cursor<&[u8]> = Cursor::new(include_bytes!("input.i420"));
        let (width, height) =
            { pnm_size(&mut input_file).expect("Malformed benchmark input file") };
        group.throughput(Throughput::Elements((width as u64) * (height as u64)));
        group.bench_function("i420>bgra", move |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::new(0, 0);
                for _i in 0..iters {
                    total += i420_bgra(&mut input_file, output_path)
                        .expect("Benchmark iteration failed");
                }

                total
            });
        });
    }

    {
        let output_path = &I444_OUTPUT;
        if Path::new(output_path).exists() {
            remove_file(Path::new(output_path)).expect("Unable to delete benchmark output");
        }

        let mut input_file: Cursor<&[u8]> = Cursor::new(include_bytes!("input.i444"));
        let (width, height) =
            { pnm_size(&mut input_file).expect("Malformed benchmark input file") };
        group.throughput(Throughput::Elements((width as u64) * (height as u64)));
        group.bench_function("i444>bgra", move |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::new(0, 0);
                for _i in 0..iters {
                    total += i444_bgra(&mut input_file, output_path)
                        .expect("Benchmark iteration failed");
                }

                total
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
