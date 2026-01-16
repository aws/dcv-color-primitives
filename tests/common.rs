pub const FIX16: i32 = 16;
pub const FIX18: i32 = 18;

pub const FIX16_MULT: i32 = 1 << FIX16;
pub const FIX16_MULT_F64: f64 = FIX16_MULT as f64;
pub const UV_SHIFT: i32 = (128 << FIX16) + (1 << (FIX16 - 1)) - 1;
pub const UV_SHIFT_18: i32 = (128 << FIX18) + (1 << (FIX18 - 1)) - 1;
pub const FULL_RANGE: f64 = 255.0;

pub type Coefficients = ((i32, i32, i32), (i32, i32), i32, i32);

pub const RGB_SRC: [[[i32; 4]; 8]; 8] = [
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
