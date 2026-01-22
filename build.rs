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
use itertools::iproduct;
use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

mod utils {
    include!("utils.rs");
}

use utils::{Coefficients, FULL_RANGE, compute_coefficients};

const FIX_8_14: i32 = 14;
const FIX_8_14_HALF: i32 = 1 << (FIX_8_14 - 1);
const FIX8_14_MULT: i32 = 1 << FIX_8_14;
const FIX8_14_MULT_F64: f64 = FIX8_14_MULT as f64;

fn generate_direct_transformation(
    f: &mut File,
    model: i32,
    suffix: &str,
    coefficients: &Coefficients,
) -> std::io::Result<()> {
    let ((xr, xg, xb), (yr, yg), zg, _) = coefficients;
    writeln!(f, "pub const XR_{model}{suffix}: i32 = {xr};")?;
    writeln!(f, "pub const XG_{model}{suffix}: i32 = {xg};")?;
    writeln!(f, "pub const XB_{model}{suffix}: i32 = {xb};")?;
    writeln!(f, "pub const YR_{model}{suffix}: i32 = {yr};")?;
    writeln!(f, "pub const YG_{model}{suffix}: i32 = {yg};")?;
    writeln!(f, "pub const ZG_{model}{suffix}: i32 = {zg};")?;
    Ok(())
}

fn generate_inverse_transformation(
    f: &mut File,
    model: i32,
    suffix: &str,
    (kr, kg, kb): (f64, f64, f64),
    (y_min, y_scale): (i32, f64),
    (c_half, c_scale): (i32, f64),
) -> std::io::Result<()> {
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

    writeln!(f, "pub const XXYM_{model}{suffix}: i32 = {s};")?;
    writeln!(f, "pub const RCRM_{model}{suffix}: i32 = {rz};")?;
    writeln!(f, "pub const GCRM_{model}{suffix}: i32 = {gz};")?;
    writeln!(f, "pub const GCBM_{model}{suffix}: i32 = {gy};")?;
    writeln!(f, "pub const BCBM_{model}{suffix}: i32 = {by};")?;
    writeln!(f, "pub const RN_{model}{suffix}: i32 = {};", rw >> 8)?;
    writeln!(f, "pub const GP_{model}{suffix}: i32 = {};", gw >> 8)?;
    writeln!(f, "pub const BN_{model}{suffix}: i32 = {};", bw >> 8)?;
    Ok(())
}

#[derive(Copy, Clone)]
enum ColorModel {
    Bt601,
    Bt709,
}

impl ColorModel {
    fn coefficients(self) -> (f64, f64, f64) {
        match self {
            ColorModel::Bt601 => (0.299, 0.587, 0.114),
            ColorModel::Bt709 => (0.2126, 0.7152, 0.0722),
        }
    }

    fn number(self) -> i32 {
        match self {
            ColorModel::Bt601 => 601,
            ColorModel::Bt709 => 709,
        }
    }
}

fn generate_coefficients(f: &mut File, model: ColorModel, full_range: bool) -> std::io::Result<()> {
    let (kr, kg, kb) = model.coefficients();
    let model_num = model.number();

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

    let coefficients = compute_coefficients::<false>((kr, kg, kb), y_min, y_scale, c_scale);

    writeln!(
        f,
        "// Coefficient table for {}{}",
        model_num,
        if full_range { " (full range)" } else { "" }
    )?;
    generate_direct_transformation(f, model_num, suffix, &coefficients)?;
    writeln!(f)?;
    generate_inverse_transformation(
        f,
        model_num,
        suffix,
        (kr, kg, kb),
        (y_min, y_scale),
        (c_half, c_scale),
    )?;

    Ok(())
}

fn main() -> std::io::Result<()> {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("weights.rs");
    let mut f = File::create(&dest_path)?;

    writeln!(f, "// Auto-generated color conversion weights")?;
    writeln!(f, "// Do not edit manually - generated by build.rs")?;
    writeln!(f)?;

    for (model, full_range) in iproduct!([ColorModel::Bt601, ColorModel::Bt709], [false, true]) {
        generate_coefficients(&mut f, model, full_range)?;
        writeln!(f)?;
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=utils.rs");

    Ok(())
}
