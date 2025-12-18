use criterion::{
    BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
    measurement::{Measurement, ValueFormatter},
};
#[cfg(target_os = "linux")]
use std::env;
use std::time::{Duration, Instant};
use std::{
    alloc::{Layout, alloc, dealloc},
    arch::asm,
    ptr::write_bytes,
    slice::from_raw_parts_mut,
};

#[cfg(target_os = "linux")]
use perf_event::{Builder, Counter, events::Hardware};

use dcp::*;
use dcv_color_primitives as dcp;

const SRC_FORMATS: &[PixelFormat] = &[PixelFormat::Bgra, PixelFormat::Bgr];
const DST_FORMATS: &[PixelFormat] = &[
    PixelFormat::Nv12,
    PixelFormat::I420,
    PixelFormat::I444,
    PixelFormat::Rgb,
];

#[cfg(target_os = "linux")]
fn cycles_wanted() -> bool {
    if let Ok(value) = env::var("DCP_USE_CYCLES") {
        if let Ok(val) = value.parse::<i32>() {
            return val != 0;
        }
    }

    false
}

struct PerfEvent {
    #[cfg(target_os = "linux")]
    counter: Option<Counter>,
    started: Option<Instant>,
}

impl PerfEvent {
    fn clobber_memory() {
        // Set read/write memory barrier for the compiler
        unsafe {
            asm!("");
        }
    }

    #[cfg(target_os = "linux")]
    fn create_cycle_counter() -> Option<Counter> {
        Builder::new().kind(Hardware::CPU_CYCLES).build().ok()
    }

    fn new() -> PerfEvent {
        PerfEvent {
            #[cfg(target_os = "linux")]
            counter: if cycles_wanted() {
                Self::create_cycle_counter()
            } else {
                None
            },
            started: None,
        }
    }

    fn start(&mut self) {
        assert!(self.started.is_none());

        Self::clobber_memory();

        let instant = Instant::now();

        #[cfg(target_os = "linux")]
        if let Some(counter) = self.counter.as_mut() {
            if counter.reset().is_ok() && counter.enable().is_ok() {
                self.started = Some(instant);
            }

            return;
        }

        self.started = Some(instant);
    }

    fn end(&mut self) -> u64 {
        Self::clobber_memory();

        if let Some(instant) = self.started.take() {
            #[cfg(target_os = "linux")]
            if let Some(counter) = self.counter.as_mut() {
                if counter.disable().is_err() {
                    return 0;
                }

                return counter.read().unwrap_or_default();
            }

            return instant.elapsed().as_nanos() as u64;
        }

        0
    }
}

struct CycleFormatter;

impl CycleFormatter {
    fn cycles_per_element(&self, elems: f64, typical: f64, values: &mut [f64]) -> &'static str {
        let cycles_per_elem = typical / elems;
        let (denominator, unit) = if cycles_per_elem < 10f64.powi(3) {
            (1.0, " cy/px")
        } else if cycles_per_elem < 10f64.powi(6) {
            (10f64.powi(3), "Kcy/px")
        } else if cycles_per_elem < 10f64.powi(9) {
            (10f64.powi(6), "Mcy/px")
        } else {
            (10f64.powi(9), "Gcy/px")
        };

        for val in values {
            let cycles_per_elem = *val / elems;
            *val = cycles_per_elem / denominator;
        }

        unit
    }
}

impl ValueFormatter for CycleFormatter {
    fn scale_throughputs(
        &self,
        typical: f64,
        throughput: &Throughput,
        values: &mut [f64],
    ) -> &'static str {
        match *throughput {
            Throughput::Elements(elems) => self.cycles_per_element(elems as f64, typical, values),
            _ => unimplemented!(),
        }
    }

    fn scale_values(&self, ns: f64, values: &mut [f64]) -> &'static str {
        let (factor, unit) = if ns < 10f64.powi(3) {
            (10f64.powi(0), " cy   ")
        } else if ns < 10f64.powi(6) {
            (10f64.powi(-3), "Kcy   ")
        } else if ns < 10f64.powi(9) {
            (10f64.powi(-6), "Mcy   ")
        } else {
            (10f64.powi(-9), "Gcy   ")
        };

        for val in values {
            *val *= factor;
        }

        unit
    }

    fn scale_for_machines(&self, _values: &mut [f64]) -> &'static str {
        "cy"
    }
}

struct TimeFormatter;

impl TimeFormatter {
    fn elements_per_second(&self, elems: f64, typical: f64, values: &mut [f64]) -> &'static str {
        let elems_per_second = elems * (1e9 / typical);
        let (denominator, unit) = if elems_per_second < 10f64.powi(3) {
            (1.0, " px/s ")
        } else if elems_per_second < 10f64.powi(6) {
            (10f64.powi(3), "Kpx/s ")
        } else if elems_per_second < 10f64.powi(9) {
            (10f64.powi(6), "Mpx/s ")
        } else {
            (10f64.powi(9), "Gpx/s ")
        };

        for val in values {
            let elems_per_second = elems * (1e9 / *val);
            *val = elems_per_second / denominator;
        }

        unit
    }
}

impl ValueFormatter for TimeFormatter {
    fn scale_throughputs(
        &self,
        typical: f64,
        throughput: &Throughput,
        values: &mut [f64],
    ) -> &'static str {
        match *throughput {
            Throughput::Elements(elems) => self.elements_per_second(elems as f64, typical, values),
            _ => unimplemented!(),
        }
    }

    fn scale_values(&self, ns: f64, values: &mut [f64]) -> &'static str {
        let (factor, unit) = if ns < 10f64.powi(3) {
            (10f64.powi(0), "    ns")
        } else if ns < 10f64.powi(6) {
            (10f64.powi(-3), "    µs")
        } else if ns < 10f64.powi(9) {
            (10f64.powi(-6), "    ms")
        } else {
            (10f64.powi(-9), "    s")
        };

        for val in values {
            *val *= factor;
        }

        unit
    }

    fn scale_for_machines(&self, _values: &mut [f64]) -> &'static str {
        "ns"
    }
}

enum PerfUnit {
    #[cfg(target_os = "linux")]
    Cycle,
    Time,
}

struct Perf {
    unit: PerfUnit,
}

impl Perf {
    fn new() -> Self {
        #[cfg(target_os = "linux")]
        {
            Perf {
                unit: if cycles_wanted() && PerfEvent::create_cycle_counter().is_some() {
                    PerfUnit::Cycle
                } else {
                    PerfUnit::Time
                },
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            Perf {
                unit: PerfUnit::Time,
            }
        }
    }
}

impl Measurement for Perf {
    type Intermediate = u64;
    type Value = u64;

    fn start(&self) -> Self::Intermediate {
        unimplemented!();
    }

    fn end(&self, _i: Self::Intermediate) -> Self::Value {
        unimplemented!();
    }

    fn add(&self, v1: &Self::Value, v2: &Self::Value) -> Self::Value {
        *v1 + *v2
    }

    fn zero(&self) -> Self::Value {
        0
    }

    fn to_f64(&self, val: &Self::Value) -> f64 {
        *val as f64
    }

    fn formatter(&self) -> &dyn ValueFormatter {
        match self.unit {
            #[cfg(target_os = "linux")]
            PerfUnit::Cycle => &CycleFormatter,
            _ => &TimeFormatter,
        }
    }
}

type RawBuffer = (Layout, *mut u8, usize);

fn alloc_buffer(size: usize, seed: Option<u8>) -> RawBuffer {
    unsafe {
        let layout = Layout::from_size_align_unchecked(size, 32);
        let ptr = alloc(layout);
        if let Some(seed) = seed {
            write_bytes::<u8>(ptr, seed, size);
        }

        (layout, ptr, size)
    }
}

fn dealloc_buffer(buffer: RawBuffer) {
    unsafe { dealloc(buffer.1, buffer.0) }
}

fn slice_from_buffer(buffer: RawBuffer, seed: Option<u8>) -> &'static [u8] {
    unsafe {
        let slice: &mut [u8] = from_raw_parts_mut(buffer.1, buffer.2);
        if let Some(s) = seed {
            slice.fill(s);
        }

        slice
    }
}

fn slice_from_buffer_mut(buffer: RawBuffer, seed: Option<u8>) -> &'static mut [u8] {
    unsafe {
        let slice: &mut [u8] = from_raw_parts_mut(buffer.1, buffer.2);
        if let Some(s) = seed {
            slice.fill(s);
        }

        slice
    }
}

fn make_seed(n: u64) -> u8 {
    // Returns a value in between [64,191]
    64_u8 + (n & 0x7F) as u8
}

fn image_dimensions(i: u32, format: PixelFormat) -> (u32, u32) {
    let components = match format {
        PixelFormat::Bgr => 3,
        _ /* Bgra */ => 4,
    };

    let payload = 1 << i;
    let pixels: u32 = payload.div_ceil(components);

    let width = (16f32 * ((pixels as f32) / 144_f32).sqrt()).floor() as u32;
    let width = (width + 31) & !31;

    let height = pixels.div_ceil(width);
    let height = (height + 1) & !1;

    (width, height)
}

fn convert_from(
    width: u32,
    height: u32,
    from_format: PixelFormat,
    to_format: PixelFormat,
    seed: u8,
    perf: &mut PerfEvent,
) -> Option<u64> {
    let pixels = (width as usize) * (height as usize);

    let src_channels = match from_format {
        PixelFormat::Bgr => 3,
        _ /* Bgra */ => 4,
    };

    let src = alloc_buffer(src_channels * pixels, None);
    let input_buffer = slice_from_buffer(src, Some(seed));
    let input_data = &[input_buffer];

    let shift = match to_format {
        PixelFormat::Rgb | PixelFormat::I444 => 0,
        _ => 1,
    };
    let num_planes = match to_format {
        PixelFormat::Rgb | PixelFormat::Nv12 => 1,
        _ => 3,
    };
    let color_space = match to_format {
        PixelFormat::Rgb => ColorSpace::Rgb,
        _ => ColorSpace::Bt601FR,
    };

    let dst = alloc_buffer((3 * pixels) >> shift, Some(seed));
    let output_buffer = slice_from_buffer_mut(dst, None);

    let mut output_data = Vec::with_capacity(3);
    match to_format {
        PixelFormat::Rgb | PixelFormat::Nv12 => output_data.push(&mut output_buffer[..]),
        _ => {
            let (y_data, uv_data) = output_buffer.split_at_mut(pixels);
            let (u_data, v_data) = uv_data.split_at_mut(pixels >> (2 * shift));

            output_data.push(&mut *y_data);
            output_data.push(&mut *u_data);
            output_data.push(&mut *v_data);
        }
    }

    let src_format = ImageFormat {
        pixel_format: from_format,
        color_space: ColorSpace::Rgb,
        num_planes: 1,
    };
    let dst_format = ImageFormat {
        pixel_format: to_format,
        color_space,
        num_planes,
    };

    perf.start();
    let result = convert_image(
        width,
        height,
        &src_format,
        None,
        input_data,
        &dst_format,
        None,
        &mut output_data,
    );

    let elapsed = perf.end();
    if result.is_ok() {
        assert!(!output_buffer.iter().all(|&x| x == 0));
    }

    dealloc_buffer(src);
    dealloc_buffer(dst);

    result.ok().map(|_x| elapsed)
}

fn convert_to(
    width: u32,
    height: u32,
    from_format: PixelFormat,
    seed: u8,
    perf: &mut PerfEvent,
) -> Option<u64> {
    let pixels = (width as usize) * (height as usize);

    let shift = match from_format {
        PixelFormat::Rgb | PixelFormat::I444 => 0,
        _ => 1,
    };
    let num_planes = match from_format {
        PixelFormat::Rgb | PixelFormat::Nv12 => 1,
        _ => 3,
    };
    let color_space = match from_format {
        PixelFormat::Rgb => ColorSpace::Rgb,
        _ => ColorSpace::Bt601FR,
    };

    let src = alloc_buffer((3 * pixels) >> shift, None);
    let input_buffer = slice_from_buffer(src, Some(seed));

    let mut input_data = Vec::with_capacity(3);
    match from_format {
        PixelFormat::Rgb | PixelFormat::Nv12 => input_data.push(input_buffer),
        _ => {
            let (y_data, uv_data) = input_buffer.split_at(pixels);
            let (u_data, v_data) = uv_data.split_at(pixels >> (2 * shift));

            input_data.push(y_data);
            input_data.push(u_data);
            input_data.push(v_data);
        }
    }

    let dst = alloc_buffer(4 * pixels, Some(seed));
    let output_buffer = slice_from_buffer_mut(dst, None);
    let output_data: &mut [&mut [u8]] = &mut [&mut output_buffer[..]];

    let src_format = ImageFormat {
        pixel_format: from_format,
        color_space,
        num_planes,
    };
    let dst_format = ImageFormat {
        pixel_format: PixelFormat::Bgra,
        color_space: ColorSpace::Rgb,
        num_planes: 1,
    };

    perf.start();
    let result = convert_image(
        width,
        height,
        &src_format,
        None,
        &input_data,
        &dst_format,
        None,
        output_data,
    );

    let elapsed = perf.end();
    if result.is_ok() {
        assert!(!output_buffer.iter().all(|&x| x == 0));
    }

    dealloc_buffer(src);
    dealloc_buffer(dst);

    result.ok().map(|_x| elapsed)
}

fn bench(c: &mut Criterion<Perf>) {
    let mut perf = PerfEvent::new();

    for src_format in SRC_FORMATS {
        let src_format_name = format!("{src_format}");
        let mut group = c.benchmark_group(format!("{src_format_name:<4}"));
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(8));
        group.sampling_mode(SamplingMode::Flat);

        for i in 18..=26 {
            let (width, height) = image_dimensions(i, *src_format);

            group.throughput(Throughput::Elements((width as u64) * (height as u64)));
            for dst_format in DST_FORMATS {
                let dst_format_name = format!("{dst_format}");
                group.bench_with_input(
                    BenchmarkId::new(format!("{dst_format_name:<4} {width:>4}x{height:<4}"), i),
                    &i,
                    |b, _i| {
                        b.iter_custom(|iters| {
                            let mut total = 0;
                            for j in 0..iters {
                                total += convert_from(
                                    width,
                                    height,
                                    *src_format,
                                    *dst_format,
                                    make_seed(j),
                                    &mut perf,
                                )
                                .expect("Benchmark iteration failed");
                            }

                            total
                        });
                    },
                );
            }
        }

        group.finish();
    }

    for src_format in DST_FORMATS {
        let src_format_name = format!("{src_format}");
        let mut group = c.benchmark_group(format!("{src_format_name:<4}"));
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(8));
        group.sampling_mode(SamplingMode::Flat);

        for i in 18..=26 {
            let dst_format = PixelFormat::Bgra;
            let dst_format_name = format!("{dst_format}");
            let (width, height) = image_dimensions(i, dst_format);

            group.throughput(Throughput::Elements((width as u64) * (height as u64)));
            group.bench_with_input(
                BenchmarkId::new(format!("{dst_format_name:<4} {width:>4}x{height:<4}"), i),
                &i,
                |b, _i| {
                    b.iter_custom(|iters| {
                        let mut total = 0;
                        for j in 0..iters {
                            total +=
                                convert_to(width, height, *src_format, make_seed(j), &mut perf)
                                    .expect("Benchmark iteration failed");
                        }

                        total
                    });
                },
            );
        }

        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_measurement(Perf::new());
    targets = bench
}

criterion_main!(benches);
