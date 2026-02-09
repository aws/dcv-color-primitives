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
use itertools::{Itertools, iproduct};
use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

mod utils {
    include!("utils.rs");
}

use utils::{
    Coefficients, FIX16, FIX18, FULL_RANGE, RGB_SRC, UV_SHIFT, UV_SHIFT_18, compute_coefficients,
};

const COLOR_MODELS: [ColorModel; 3] = [ColorModel::Bt601, ColorModel::Bt709, ColorModel::Bt2020];
const COLOR_RANGES: [bool; 2] = [false, true];
const NUM_CONFIGS: usize = COLOR_MODELS.len() * COLOR_RANGES.len();

const FIX_8_14: i32 = 14;
const FIX_8_14_HALF: i32 = 1 << (FIX_8_14 - 1);
const FIX8_14_MULT: i32 = 1 << FIX_8_14;
const FIX8_14_MULT_F64: f64 = FIX8_14_MULT as f64;

const COMPONENT_NAMES: [&str; 3] = ["Y_SRC", "U_SRC", "V_SRC"];
const TEST_COLORS: [(u8, u8, u8); 8] = [
    (0, 0, 0),       // black
    (255, 0, 0),     // red
    (0, 255, 0),     // green
    (255, 255, 0),   // yellow
    (0, 0, 255),     // blue
    (255, 0, 255),   // magenta
    (0, 255, 255),   // cyan
    (255, 255, 255), // white
];

#[derive(Copy, Clone)]
enum ColorModel {
    Bt601,
    Bt709,
    Bt2020,
}

impl ColorModel {
    fn coefficients(self) -> (f64, f64, f64) {
        match self {
            ColorModel::Bt601 => (0.299, 0.587, 0.114),
            ColorModel::Bt709 => (0.2126, 0.7152, 0.0722),
            ColorModel::Bt2020 => (0.2627, 0.6780, 0.0593),
        }
    }

    fn number(self) -> i32 {
        match self {
            ColorModel::Bt601 => 601,
            ColorModel::Bt709 => 709,
            ColorModel::Bt2020 => 2020,
        }
    }
}

struct RangeParams {
    yuv_min: i32,
    uv_max: i32,
    y_scale: f64,
    uv_scale: f64,
    suffix: &'static str,
}

fn get_range_params(full_range: bool) -> RangeParams {
    if full_range {
        RangeParams {
            yuv_min: 0,
            uv_max: 255,
            y_scale: 1.0,
            uv_scale: 1.0,
            suffix: "FR",
        }
    } else {
        RangeParams {
            yuv_min: 16,
            uv_max: 240,
            y_scale: 219.0 / FULL_RANGE,
            uv_scale: 224.0 / FULL_RANGE,
            suffix: "",
        }
    }
}

fn compute_yuv_coefficients(
    (kr, kg, kb): (f64, f64, f64),
    params: &RangeParams,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let y_r = kr * params.y_scale;
    let y_g = kg * params.y_scale;
    let y_b = kb * params.y_scale;

    let cb_r = -0.5 * kr / (1.0 - kb) * params.uv_scale;
    let cb_g = -0.5 * kg / (1.0 - kb) * params.uv_scale;
    let cb_b = 0.5 * params.uv_scale;

    let cr_r = 0.5 * params.uv_scale;
    let cr_g = -0.5 * kg / (1.0 - kr) * params.uv_scale;
    let cr_b = -0.5 * kb / (1.0 - kr) * params.uv_scale;

    (y_r, y_g, y_b, cb_r, cb_g, cb_b, cr_r, cr_g, cr_b)
}

fn yuv_convert(
    (kr, kg, kb): (f64, f64, f64),
    params: &RangeParams,
    r: u8,
    g: u8,
    b: u8,
) -> (u8, u8, u8) {
    let y_min = params.yuv_min as f64;
    let (y_r, y_g, y_b, cb_r, cb_g, cb_b, cr_r, cr_g, cr_b) =
        compute_yuv_coefficients((kr, kg, kb), params);

    let r = r as f64;
    let g = g as f64;
    let b = b as f64;

    let y = y_r * r + y_g * g + y_b * b + y_min;
    let cb = cb_r * r + cb_g * g + cb_b * b + 128.0;
    let cr = cr_r * r + cr_g * g + cr_b * b + 128.0;

    (y as u8, cb as u8, cr as u8)
}

fn generate_full_plane(
    f: &mut File,
    xr: i32,
    xg: i32,
    xb: i32,
    offset: i32,
) -> std::io::Result<()> {
    writeln!(
        f,
        "{}",
        RGB_SRC
            .iter()
            .map(|row| {
                format!(
                    "    [{}],",
                    row.iter()
                        .map(|p| ((xr * p[0] + xg * p[1] + xb * p[2] + offset) >> FIX16) as u8)
                        .join(", ")
                )
            })
            .join("\n")
    )?;
    writeln!(f, "];")
}

fn generate_subsampled_plane(f: &mut File, yr: i32, yg: i32, yb: i32) -> std::io::Result<()> {
    writeln!(
        f,
        "{}",
        RGB_SRC
            .chunks_exact(2)
            .map(|rows| {
                format!(
                    "    [{}],",
                    rows[0]
                        .chunks_exact(2)
                        .zip(rows[1].chunks_exact(2))
                        .map(|(chunk0, chunk1)| {
                            let [red, green, blue] = chunk0
                                .iter()
                                .chain(chunk1.iter())
                                .fold([0i32; 3], |acc, p| {
                                    [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]]
                                });
                            ((yr * red + yg * green + yb * blue + UV_SHIFT_18) >> FIX18) as u8
                        })
                        .join(", ")
                )
            })
            .join("\n")
    )?;
    writeln!(f, "];")
}

fn generate_yuv_tables(f: &mut File) -> std::io::Result<()> {
    let yuv_data: Vec<Vec<(u8, u8, u8)>> = iproduct!(COLOR_RANGES, COLOR_MODELS)
        .map(|(full_range, model)| {
            let (kr, kg, kb) = model.coefficients();
            let params = get_range_params(full_range);
            TEST_COLORS
                .iter()
                .map(|(r, g, b)| yuv_convert((kr, kg, kb), &params, *r, *g, *b))
                .collect()
        })
        .collect();

    for (component, name) in COMPONENT_NAMES.iter().enumerate() {
        writeln!(f, "pub const {name}: [[u8; 8]; {NUM_CONFIGS}] = [")?;
        for row in &yuv_data {
            writeln!(
                f,
                "    [{}],",
                row.iter()
                    .map(|yuv| {
                        match component {
                            0 => yuv.0,
                            1 => yuv.1,
                            _ => yuv.2,
                        }
                        .to_string()
                    })
                    .join(", ")
            )?;
        }
        writeln!(f, "];")?;
    }

    Ok(())
}

fn generate_rgb_tables(f: &mut File) -> std::io::Result<()> {
    for (full_range, model) in iproduct!(COLOR_RANGES, COLOR_MODELS) {
        let (kr, kg, kb) = model.coefficients();
        let params = get_range_params(full_range);
        let name = format!("BT{}{}", model.number(), params.suffix);
        let coefficients = compute_coefficients::<false>(
            (kr, kg, kb),
            params.yuv_min,
            params.y_scale,
            params.uv_scale,
        );
        let ((xr, xg, xb), (yr, yg), zg, _) = coefficients;
        let yb = -(yr + yg);
        let zr = yb;
        let zb = -(zr + zg);
        let y_offset = (params.yuv_min << FIX16) + (1 << (FIX16 - 1));

        writeln!(f, "pub const Y_{name}_REF: FullPlane = [")?;
        generate_full_plane(f, xr, xg, xb, y_offset)?;

        writeln!(f, "pub const CB_{name}_REF: FullPlane = [")?;
        generate_full_plane(f, yr, yg, yb, UV_SHIFT)?;

        writeln!(f, "pub const CR_{name}_REF: FullPlane = [")?;
        generate_full_plane(f, zr, zg, zb, UV_SHIFT)?;

        writeln!(f, "pub const CB2_{name}_REF: SubSampledPlane = [")?;
        generate_subsampled_plane(f, yr, yg, yb)?;

        writeln!(f, "pub const CR2_{name}_REF: SubSampledPlane = [")?;
        generate_subsampled_plane(f, zr, zg, zb)?;
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
    (kr, kg, kb): (f64, f64, f64),
    params: &RangeParams,
) -> std::io::Result<()> {
    let ikb = 1.0 - kb;
    let ikr = 1.0 - kr;
    let y_scale_inv = 1.0 / params.y_scale;
    let c_half = (params.uv_max + params.yuv_min) >> 1;

    let rz = 2.0 * ikr / params.uv_scale;
    let gy = (2.0 * ikb * kb) / (params.uv_scale * kg);
    let gz = (2.0 * ikr * kr) / (params.uv_scale * kg);
    let by = 2.0 * ikb / params.uv_scale;

    let s = (FIX8_14_MULT_F64 * y_scale_inv + 0.5) as i32;
    let rz = (FIX8_14_MULT_F64 * rz + 0.5) as i32;
    let gy = (FIX8_14_MULT_F64 * gy + 0.5) as i32;
    let gz = (FIX8_14_MULT_F64 * gz + 0.5) as i32;
    let by = (FIX8_14_MULT_F64 * by + 0.5) as i32;

    let rw = (rz * c_half + s * params.yuv_min - FIX_8_14_HALF) >> 8;
    let gw = ((gy * c_half) + (gz * c_half) - (s * params.yuv_min) + FIX_8_14_HALF) >> 8;
    let bw = (s * params.yuv_min + by * c_half - FIX_8_14_HALF) >> 8;

    writeln!(f, "pub const XXYM_{model}{}: i32 = {s};", params.suffix)?;
    writeln!(f, "pub const RCRM_{model}{}: i32 = {rz};", params.suffix)?;
    writeln!(f, "pub const GCRM_{model}{}: i32 = {gz};", params.suffix)?;
    writeln!(f, "pub const GCBM_{model}{}: i32 = {gy};", params.suffix)?;
    writeln!(f, "pub const BCBM_{model}{}: i32 = {by};", params.suffix)?;
    writeln!(f, "pub const RN_{model}{}: i32 = {};", params.suffix, rw)?;
    writeln!(f, "pub const GP_{model}{}: i32 = {};", params.suffix, gw)?;
    writeln!(f, "pub const BN_{model}{}: i32 = {};", params.suffix, bw)?;
    Ok(())
}

fn generate_coefficients(f: &mut File, model: ColorModel, full_range: bool) -> std::io::Result<()> {
    let (kr, kg, kb) = model.coefficients();
    let model_num = model.number();
    let params = get_range_params(full_range);
    let coefficients = compute_coefficients::<false>(
        (kr, kg, kb),
        params.yuv_min,
        params.y_scale,
        params.uv_scale,
    );

    writeln!(f, "// Coefficient table for {model_num}{}", params.suffix)?;
    generate_direct_transformation(f, model_num, params.suffix, &coefficients)?;
    writeln!(f)?;
    generate_inverse_transformation(f, model_num, (kr, kg, kb), &params)?;

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
    generate_rgb_tables(&mut tables_file)?;

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=utils.rs");

    Ok(())
}
