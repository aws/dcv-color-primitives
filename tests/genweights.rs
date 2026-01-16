// Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify,
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::similar_names)]

mod common;

use common::{Coefficients, FIX18, FULL_RANGE, RGB_SRC, UV_SHIFT_18, compute_coefficients};
use std::env;
use std::process;

const FIX16: i32 = 16;
const FIX_8_14: i32 = 14;
const FIX_8_14_HALF: i32 = 1 << (FIX_8_14 - 1);

const FIX8_14_MULT: i32 = 1 << FIX_8_14;
const FIX8_14_MULT_F64: f64 = FIX8_14_MULT as f64;
const UV_SHIFT: i32 = (128 << FIX16) + (1 << (FIX16 - 1)) - 1;

fn print_direct_transformation(model: i32, suffix: &str, coefficients: &Coefficients) {
    let ((xr, xg, xb), (yr, yg), zg, _) = coefficients;

    println!("pub const XR_{model}{suffix}: i32 = {xr};");
    println!("pub const XG_{model}{suffix}: i32 = {xg};");
    println!("pub const XB_{model}{suffix}: i32 = {xb};");
    println!("pub const YR_{model}{suffix}: i32 = {yr};");
    println!("pub const YG_{model}{suffix}: i32 = {yg};");
    println!("pub const ZG_{model}{suffix}: i32 = {zg};");
}

fn print_inverse_transformation(
    model: i32,
    suffix: &str,
    (kr, kg, kb): (f64, f64, f64),
    (y_min, y_scale): (i32, f64),
    (c_half, c_scale): (i32, f64),
) {
    let ikb = 1.0 - kb;
    let ikr = 1.0 - kr;
    let y_scale_inv = 1.0 / y_scale;

    let rz = 2.0 * ikr / c_scale;
    let gy = (2.0 * ikb * kb) / (c_scale * kg);
    let gz = (2.0 * ikr * kr) / (c_scale * kg);
    let by = 2.0 * ikb / c_scale;

    let s = (FIX8_14_MULT_F64 * y_scale_inv + 0.5) as i32;
    let rz = (FIX8_14_MULT_F64 * rz + 0.5) as i32;
    let gy = (FIX8_14_MULT_F64 * gy + 0.5) as i32;
    let gz = (FIX8_14_MULT_F64 * gz + 0.5) as i32;
    let by = (FIX8_14_MULT_F64 * by + 0.5) as i32;

    let rw = rz * c_half + s * y_min - FIX_8_14_HALF;
    let gw = (gy * c_half) + (gz * c_half) - (s * y_min) + FIX_8_14_HALF;
    let bw = s * y_min + by * c_half - FIX_8_14_HALF;

    println!("pub const XXYM_{model}{suffix}: i32 = {s};");
    println!("pub const RCRM_{model}{suffix}: i32 = {rz};");
    println!("pub const GCRM_{model}{suffix}: i32 = {gz};");
    println!("pub const GCBM_{model}{suffix}: i32 = {gy};");
    println!("pub const BCBM_{model}{suffix}: i32 = {by};");
    println!("pub const RN_{model}{suffix}: i32 = {};", rw >> 8);
    println!("pub const GP_{model}{suffix}: i32 = {};", gw >> 8);
    println!("pub const BN_{model}{suffix}: i32 = {};", bw >> 8);
}

fn print_regression_data(model: i32, suffix: &str, coefficients: &Coefficients) {
    let ((xr, xg, xb), (yr, yg), zg, y_min) = coefficients;

    let yb = -(yr + yg);
    let zr = yb;
    let zb = -(zr + zg);
    let y_shift = (y_min << FIX16) + (1 << (FIX16 - 1));

    println!("const Y_BT{model}{suffix}_REF: FullPlane = [");
    for row in &RGB_SRC {
        let ys: Vec<i32> = row
            .iter()
            .map(|pixel| (xr * pixel[0] + xg * pixel[1] + xb * pixel[2] + y_shift) >> FIX16)
            .collect();
        println!("    {ys:?},");
    }
    println!("];");
    println!();

    println!("const CB_BT{model}{suffix}_REF: FullPlane = [");
    for row in &RGB_SRC {
        let us: Vec<i32> = row
            .iter()
            .map(|pixel| (yr * pixel[0] + yg * pixel[1] + yb * pixel[2] + UV_SHIFT) >> FIX16)
            .collect();
        println!("    {us:?},");
    }
    println!("];");
    println!();

    println!("const CR_BT{model}{suffix}_REF: FullPlane = [");
    for row in &RGB_SRC {
        let vs: Vec<i32> = row
            .iter()
            .map(|pixel| (zr * pixel[0] + zg * pixel[1] + zb * pixel[2] + UV_SHIFT) >> FIX16)
            .collect();
        println!("    {vs:?},");
    }
    println!("];");
    println!();

    // Generate subsampled regression data
    println!("const CB2_BT{model}{suffix}_REF: SubSampledPlane = [");
    RGB_SRC
        .chunks_exact(2)
        .map(|rows| {
            rows[0]
                .chunks_exact(2)
                .zip(rows[1].chunks_exact(2))
                .map(|(top, bottom)| {
                    let sum = top
                        .iter()
                        .chain(bottom.iter())
                        .fold((0i32, 0i32, 0i32), |sum, i| {
                            (sum.0 + i[0], sum.1 + i[1], sum.2 + i[2])
                        });
                    (yr * sum.0 + yg * sum.1 + yb * sum.2 + UV_SHIFT_18) >> FIX18
                })
                .collect::<Vec<_>>()
        })
        .for_each(|row| println!("    {row:?},"));
    println!("];");
    println!();

    println!("const CR2_BT{model}{suffix}_REF: SubSampledPlane = [");
    RGB_SRC
        .chunks_exact(2)
        .map(|rows| {
            rows[0]
                .chunks_exact(2)
                .zip(rows[1].chunks_exact(2))
                .map(|(top, bottom)| {
                    let sum = top
                        .iter()
                        .chain(bottom.iter())
                        .fold((0i32, 0i32, 0i32), |sum, i| {
                            (sum.0 + i[0], sum.1 + i[1], sum.2 + i[2])
                        });
                    (zr * sum.0 + zg * sum.1 + zb * sum.2 + UV_SHIFT_18) >> FIX18
                })
                .collect::<Vec<_>>()
        })
        .for_each(|row| println!("    {row:?},"));
    println!("];");
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("usage: {} [601|709] [0|1]", args[0]);
        process::exit(1);
    }

    let model: i32 = args[1].parse().unwrap_or_else(|_| {
        eprintln!("Invalid model {}", args[1]);
        process::exit(1);
    });

    let full_range = args[2].parse::<i32>().unwrap_or(0) == 1;

    let (kr, kg, kb) = match model {
        601 => (0.299, 0.587, 0.114),
        709 => (0.2126, 0.7152, 0.0722),
        _ => {
            eprintln!("Invalid model {model}");
            process::exit(1);
        }
    };

    println!("Model: {model}");
    println!("Full range: {}", i32::from(full_range));
    println!();

    let (y_min, y_max, c_min, c_max, suffix) = if full_range {
        (0, 255, 0, 255, "FR")
    } else {
        (16, 235, 16, 240, "")
    };

    let c_half = (c_max + c_min) >> 1;
    let y_scale = if full_range {
        1f64
    } else {
        f64::from(y_max - y_min) / FULL_RANGE
    };
    let c_scale = if full_range {
        1f64
    } else {
        f64::from(c_max - c_min) / FULL_RANGE
    };

    // Forward transformation
    let coefficients = compute_coefficients::<true>((kr, kg, kb), y_min, y_scale, c_scale);

    println!();
    println!(
        "// Coefficient table for {}{}",
        model,
        if full_range { " (full range)" } else { "" }
    );
    print_direct_transformation(model, suffix, &coefficients);
    println!();
    print_inverse_transformation(
        model,
        suffix,
        (kr, kg, kb),
        (y_min, y_scale),
        (c_half, c_scale),
    );
    println!();

    print_regression_data(model, suffix, &coefficients);
}
