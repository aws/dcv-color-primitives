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

const COLOR_MODELS: [ColorModel; 2] = [ColorModel::Bt601, ColorModel::Bt709];
const COLOR_RANGES: [bool; 2] = [false, true];
const NUM_CONFIGS: usize = COLOR_MODELS.len() * COLOR_RANGES.len();

const FIX_8_14: i32 = 14;
const FIX_8_14_HALF: i32 = 1 << (FIX_8_14 - 1);
const FIX8_14_MULT: i32 = 1 << FIX_8_14;
const FIX8_14_MULT_F64: f64 = FIX8_14_MULT as f64;

fn compute_yuv_coefficients(
    kr: f64,
    kg: f64,
    kb: f64,
    y_scale: f64,
    c_scale: f64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let y_r = kr * y_scale;
    let y_g = kg * y_scale;
    let y_b = kb * y_scale;

    let cb_r = -0.5 * kr / (1.0 - kb) * c_scale;
    let cb_g = -0.5 * kg / (1.0 - kb) * c_scale;
    let cb_b = 0.5 * c_scale;

    let cr_r = 0.5 * c_scale;
    let cr_g = -0.5 * kg / (1.0 - kr) * c_scale;
    let cr_b = -0.5 * kb / (1.0 - kr) * c_scale;

    (y_r, y_g, y_b, cb_r, cb_g, cb_b, cr_r, cr_g, cr_b)
}

fn yuv_convert(
    kr: f64,
    kg: f64,
    kb: f64,
    y_min: f64,
    c_offset: f64,
    y_scale: f64,
    c_scale: f64,
    r: u8,
    g: u8,
    b: u8,
) -> (u8, u8, u8) {
    let (y_r, y_g, y_b, cb_r, cb_g, cb_b, cr_r, cr_g, cr_b) =
        compute_yuv_coefficients(kr, kg, kb, y_scale, c_scale);

    let r = r as f64;
    let g = g as f64;
    let b = b as f64;

    let y = y_r * r + y_g * g + y_b * b + y_min;
    let cb = cb_r * r + cb_g * g + cb_b * b + c_offset;
    let cr = cr_r * r + cr_g * g + cr_b * b + c_offset;

    (y as u8, cb as u8, cr as u8)
}

fn generate_yuv_tables(f: &mut File) -> std::io::Result<()> {
    let colors = [
        (0, 0, 0),       // black
        (255, 0, 0),     // red
        (0, 255, 0),     // green
        (255, 255, 0),   // yellow
        (0, 0, 255),     // blue
        (255, 0, 255),   // magenta
        (0, 255, 255),   // cyan
        (255, 255, 255), // white
    ];

    let yuv_data: Vec<Vec<(u8, u8, u8)>> = iproduct!(COLOR_RANGES, COLOR_MODELS)
        .map(|(full_range, model)| {
            let (kr, kg, kb) = model.coefficients();
            let (y_min, c_offset, y_scale, c_scale) = if full_range {
                (0.0, 128.0, 1.0, 1.0)
            } else {
                (16.0, 128.0, 219.0 / 255.0, 224.0 / 255.0)
            };
            colors
                .iter()
                .map(|(r, g, b)| {
                    yuv_convert(kr, kg, kb, y_min, c_offset, y_scale, c_scale, *r, *g, *b)
                })
                .collect()
        })
        .collect();

    for (component, name) in [(0, "Y_SRC"), (1, "U_SRC"), (2, "V_SRC")] {
        writeln!(f, "pub const {name}: [[u8; 8]; {NUM_CONFIGS}] = [")?;
        for row in &yuv_data {
            let values: Vec<String> = row
                .iter()
                .map(|yuv| {
                    match component {
                        0 => yuv.0,
                        1 => yuv.1,
                        _ => yuv.2,
                    }
                    .to_string()
                })
                .collect();
            writeln!(f, "    [{}],", values.join(", "))?;
        }
        writeln!(f, "];")?;
    }

    Ok(())
}

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
    let weights_path = Path::new(&out_dir).join("weights.rs");
    let mut weights_file = File::create(&weights_path)?;

    writeln!(weights_file, "// Auto-generated color conversion weights")?;
    writeln!(
        weights_file,
        "// Do not edit manually - generated by build.rs"
    )?;
    writeln!(weights_file)?;
    for (model, full_range) in iproduct!(COLOR_MODELS, COLOR_RANGES) {
        generate_coefficients(&mut weights_file, model, full_range)?;
        writeln!(weights_file)?;
    }

    let tables_path = Path::new(&out_dir).join("yuv_tables.rs");
    let mut tables_file = File::create(&tables_path)?;

    writeln!(tables_file, "// Auto-generated YUV test tables")?;
    writeln!(
        tables_file,
        "// Do not edit manually - generated by build.rs"
    )?;
    writeln!(tables_file)?;
    generate_yuv_tables(&mut tables_file)?;

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=utils.rs");

    Ok(())
}
