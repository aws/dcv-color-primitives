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
pub const FIX16: i32 = 16;

pub const FIX16_MULT: i32 = 1 << FIX16;
pub const FIX16_MULT_F64: f64 = FIX16_MULT as f64;
pub const UV_SHIFT: i32 = (128 << FIX16) + (1 << (FIX16 - 1)) - 1;
pub const FULL_RANGE: f64 = 255.0;

pub type Coefficients = ((i32, i32, i32), (i32, i32), i32, i32);

fn max_y_error((xr, xg, xb): (i32, i32, i32), (ar, ag, ab): (f64, f64, f64), y_min: i32) -> f64 {
    let shift = (y_min << FIX16) + (1 << (FIX16 - 1));

    (0..256)
        .flat_map(|red| {
            let y_tmp = xr * red + shift;
            let yf_tmp = ar * f64::from(red) + f64::from(y_min);

            (0..256).flat_map(move |green| {
                let y_tmp2 = xg * green + y_tmp;
                let yf_tmp2 = ag * f64::from(green) + yf_tmp;

                (0..256).map(move |blue| {
                    let y = (xb * blue + y_tmp2) >> FIX16;
                    let yf = ab * f64::from(blue) + yf_tmp2;
                    (yf - f64::from(y)).abs()
                })
            })
        })
        .fold(0.0, f64::max)
}

pub fn max_uv_error(
    (yr, yg): (i32, i32),
    zg: i32,
    (br, bg, bb): (f64, f64, f64),
    (cr, cg, cb): (f64, f64, f64),
) -> (f64, f64) {
    let yb = -(yr + yg);
    let zr = yb;
    let zb = -(zr + zg);

    (0..256)
        .flat_map(|red| {
            let u_tmp = yr * red + UV_SHIFT;
            let v_tmp = zr * red + UV_SHIFT;
            let uf_tmp = br * f64::from(red) + 128.0;
            let vf_tmp = cr * f64::from(red) + 128.0;

            (0..256).flat_map(move |green| {
                let u_tmp2 = yg * green + u_tmp;
                let v_tmp2 = zg * green + v_tmp;
                let uf_tmp2 = bg * f64::from(green) + uf_tmp;
                let vf_tmp2 = cg * f64::from(green) + vf_tmp;

                (0..256).map(move |blue| {
                    let u = (yb * blue + u_tmp2) >> FIX16;
                    let v = (zb * blue + v_tmp2) >> FIX16;
                    let uf = bb * f64::from(blue) + uf_tmp2;
                    let vf = cb * f64::from(blue) + vf_tmp2;
                    ((uf - f64::from(u)).abs(), (vf - f64::from(v)).abs())
                })
            })
        })
        .fold((0.0_f64, 0.0_f64), |(u_max, v_max), (u_err, v_err)| {
            (u_max.max(u_err), v_max.max(v_err))
        })
}

pub fn compute_coefficients<const LOG_ERROR: bool>(
    (kr, kg, kb): (f64, f64, f64),
    y_min: i32,
    y_scale: f64,
    c_scale: f64,
) -> Coefficients {
    let ar = y_scale * kr;
    let ag = y_scale * kg;
    let ab = y_scale * kb;
    let br = c_scale * (-kr / (2.0 * (1.0 - kb)));
    let bg = c_scale * (-kg / (2.0 * (1.0 - kb)));
    let bb = c_scale * (0.5);
    let cr = c_scale * (0.5);
    let cg = c_scale * (-kg / (2.0 * (1.0 - kr)));
    let cb = c_scale * (-kb / (2.0 * (1.0 - kr)));

    let xr = (FIX16_MULT_F64 * ar).round() as i32;
    let xg = (FIX16_MULT_F64 * ag).round() as i32;
    let xb = (FIX16_MULT_F64 * ab).round() as i32;

    let mut yr = (FIX16_MULT_F64 * br).round() as i32;
    let mut yg = (FIX16_MULT_F64 * bg).round() as i32;
    let zg = (FIX16_MULT_F64 * cg).round() as i32;
    let diff = -32767 - (yr + yg);

    let uv_err = if diff > 0 {
        let uv_err = max_uv_error((yr + diff, yg), zg, (br, bg, bb), (cr, cg, cb));
        let uv2_err = max_uv_error((yr, yg + diff), zg, (br, bg, bb), (cr, cg, cb));

        if uv_err.0.max(uv_err.1) <= uv2_err.0.max(uv2_err.1) {
            yr += diff;
            uv_err
        } else {
            yg += diff;
            uv2_err
        }
    } else if LOG_ERROR {
        max_uv_error((yr, yg), zg, (br, bg, bb), (cr, cg, cb))
    } else {
        (0f64, 0f64)
    };

    if LOG_ERROR {
        let y_err = max_y_error((xr, xg, xb), (ar, ag, ab), y_min);
        println!("Error: y={y_err:.5}, u={:.5}, v={:.5}", uv_err.0, uv_err.1);
    }

    ((xr, xg, xb), (yr, yg), zg, y_min)
}