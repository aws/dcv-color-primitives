#!/usr/bin/env python3

# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import sys

FIX16 = 16
FIX18 = 18
FIX16_HALF = (1 << (FIX16 - 1))
FIX_8_14 = 14
FIX_8_14_HALF = (1 << (FIX_8_14 - 1))

FIX16_MULT = (1 << FIX16)
FIX8_14_MULT = (1 << FIX_8_14)
FULL_RANGE = 255

RGB_SRC = [
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
]


def max_y_error(xr, xg, xb, ar, ag, ab, y_min):
    err = 0.0
    shift = (y_min << FIX16) + (1 << (FIX16 - 1))

    for red in range(256):
        y_tmp = xr * red + shift
        yf_tmp = ar * red + y_min

        for green in range(256):
            y_tmp2 = xg * green + y_tmp
            yf_tmp2 = ag * green + yf_tmp

            for blue in range(256):
                y = (xb * blue + y_tmp2) >> FIX16
                yf = ab * blue + yf_tmp2
                err = max(err, abs(yf - y))

    return err


def max_uv_error(yr, yg, zg, br, bg, bb, cr, cg, cb):
    yb = -(yr + yg)
    zr = yb
    zb = -(zr + zg)

    shift = (128 << FIX16) + (1 << (FIX16 - 1)) - 1
    u_err = 0.0
    v_err = 0.0

    for red in range(256):
        u_tmp = yr * red + shift
        v_tmp = zr * red + shift
        uf_tmp = br * red + 128
        vf_tmp = cr * red + 128

        for green in range(256):
            u_tmp2 = yg * green + u_tmp
            v_tmp2 = zg * green + v_tmp
            uf_tmp2 = bg * green + uf_tmp
            vf_tmp2 = cg * green + vf_tmp

            for blue in range(256):
                u = (yb * blue + u_tmp2) >> FIX16
                v = (zb * blue + v_tmp2) >> FIX16
                uf = bb * blue + uf_tmp2
                vf = cb * blue + vf_tmp2
                u_err = max(u_err, abs(uf - u))
                v_err = max(v_err, abs(vf - v))

    return (u_err, v_err)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: python genweights.py [601|709] [0|1]')
        exit(0)

    model = int(sys.argv[1])
    full_range = int(sys.argv[2]) == 1

    if model == 601:
        kr = 0.299
        kg = 0.587
        kb = 0.114
    elif model == 709:
        kr = 0.2126
        kg = 0.7152
        kb = 0.0722
    else:
        print('Invalid model %s' % sys.argv[1])
        exit(0)

    print('Model: %d' % model)
    print('Full range: %d' % full_range)
    print('')

    if full_range:
        Y_MIN = 0
        Y_MAX = 255
        C_MIN = 0
        C_MAX = 255
        SUFFIX = "FR"
    else:
        Y_MIN = 16
        Y_MAX = 235
        C_MIN = 16
        C_MAX = 240
        SUFFIX = ""

    Y_RANGE = Y_MAX - Y_MIN
    C_HALF = (C_MAX + C_MIN) >> 1
    C_RANGE = C_MAX - C_MIN
    Y_SCALE = (Y_RANGE / FULL_RANGE)
    C_SCALE = (C_RANGE / FULL_RANGE)

    # Forward transformation
    ar = kr
    ag = kg
    ab = kb
    br = (-kr / (2.0 * (1.0 - kb)))
    bg = (-kg / (2.0 * (1.0 - kb)))
    bb = 0.5
    cr = 0.5
    cg = (-kg / (2.0 * (1.0 - kr)))
    cb = (-kb / (2.0 * (1.0 - kr)))

    if not full_range:
        ar *= Y_SCALE
        ag *= Y_SCALE
        ab *= Y_SCALE
        br *= C_SCALE
        bg *= C_SCALE
        bb *= C_SCALE
        cr *= C_SCALE
        cg *= C_SCALE
        cb *= C_SCALE

    xr = round(FIX16_MULT * ar)
    xg = round(FIX16_MULT * ag)
    xb = round(FIX16_MULT * ab)
    y_err = max_y_error(xr, xg, xb, ar, ag, ab, Y_MIN)

    yr = round(FIX16_MULT * br)
    yg = round(FIX16_MULT * bg)
    zg = round(FIX16_MULT * cg)
    diff = -32767 - (yr + yg)
    if diff > 0:
        # Prevent overflow
        # (decide which variable between yr and yg should be incremented)
        uv_err = max_uv_error(yr + diff, yg,
                              zg, br, bg, bb, cr, cg, cb)
        uv2_err = max_uv_error(yr, yg + diff, zg,
                               br, bg, bb, cr, cg, cb)

        if max(*uv_err) <= max(*uv2_err):
            yr += diff
        else:
            yg += diff
            uv_err = uv2_err
    else:
        uv_err = max_uv_error(yr, yg, zg, br, bg, bb, cr, cg, cb)

    print('Error: y=%.5f, u=%.5f, v=%.5f' % (y_err, *uv_err))
    print('')

    print('// Coefficient table for %s%s' %
          (model, " (full range)" if full_range else ""))
    print('pub const XR_%d%s: i32 = %d;' % (model, SUFFIX, xr))
    print('pub const XG_%d%s: i32 = %d;' % (model, SUFFIX, xg))
    print('pub const XB_%d%s: i32 = %d;' % (model, SUFFIX, xb))
    print('pub const YR_%d%s: i32 = %d;' % (model, SUFFIX, yr))
    print('pub const YG_%d%s: i32 = %d;' % (model, SUFFIX, yg))
    print('pub const ZG_%d%s: i32 = %d;' % (model, SUFFIX, zg))
    print('')

    # Inverse transformation
    ikb = 1.0 - kb
    ikr = 1.0 - kr
    y_scale = 1 / Y_SCALE

    rz = 2.0 * ikr / C_SCALE
    gy = (2.0 * ikb * kb) / (C_SCALE * kg)
    gz = (2.0 * ikr * kr) / (C_SCALE * kg)
    by = 2.0 * ikb / C_SCALE

    s = int(FIX8_14_MULT * y_scale + 0.5)
    rz = int(FIX8_14_MULT * rz + 0.5)
    gy = int(FIX8_14_MULT * gy + 0.5)
    gz = int(FIX8_14_MULT * gz + 0.5)
    by = int(FIX8_14_MULT * by + 0.5)

    rw = rz * C_HALF + s * Y_MIN - FIX_8_14_HALF
    gw = (gy * C_HALF) + (gz * C_HALF) - (s * Y_MIN) + FIX_8_14_HALF
    bw = s * Y_MIN + by * C_HALF - FIX_8_14_HALF

    print('pub const XXYM_%d%s: i32 = %d;' % (model, SUFFIX, s))
    print('pub const RCRM_%d%s: i32 = %d;' % (model, SUFFIX, rz))
    print('pub const GCRM_%d%s: i32 = %d;' % (model, SUFFIX, gz))
    print('pub const GCBM_%d%s: i32 = %d;' % (model, SUFFIX, gy))
    print('pub const BCBM_%d%s: i32 = %d;' % (model, SUFFIX, by))
    print('pub const RN_%d%s: i32 = %d;' % (model, SUFFIX, rw >> 8))
    print('pub const GP_%d%s: i32 = %d;' % (model, SUFFIX, gw >> 8))
    print('pub const BN_%d%s: i32 = %d;' % (model, SUFFIX, bw >> 8))

    # Reference data for regression tests
    yb = -(yr + yg)
    zr = yb
    zb = -(zr + zg)
    uv_shift = (128 << FIX16) + (1 << (FIX16 - 1)) - 1
    uv2_shift = (128 << FIX18) + (1 << (FIX18 - 1)) - 1
    y_shift = (Y_MIN << FIX16) + (1 << (FIX16 - 1))

    print('const Y_BT%d%s_REF: &[&[u8]] = &[' % (model, SUFFIX))
    for row in RGB_SRC:
        ys = [(xr * r + xg * g + xb * b + y_shift) >> FIX16
              for [r, g, b, a] in row]
        print('    &%s,' % ys)
    print('];\n')

    print('const CB_BT%d%s_REF: &[&[u8]] = &[' % (model, SUFFIX))
    for row in RGB_SRC:
        us = [(yr * r + yg * g + yb * b + uv_shift) >> FIX16
              for [r, g, b, a] in row]
        print('    &%s,' % us)
    print('];\n')

    print('const CR_BT%d%s_REF: &[&[u8]] = &[' % (model, SUFFIX))
    for row in RGB_SRC:
        vs = [(zr * r + zg * g + zb * b + uv_shift) >> FIX16
              for [r, g, b, a] in row]
        print('    &%s,' % vs)
    print('];\n')

    # The same but subsampled
    print('const CB2_BT%d%s_REF: &[&[u8]] = &[' % (model, SUFFIX))
    for j in range(4):
        us = []
        for i in range(4):
            ii = 2 * i
            ij = 2 * j

            r = RGB_SRC[ij][ii][0] + RGB_SRC[ij + 1][ii][0] + \
                RGB_SRC[ij][ii + 1][0] + RGB_SRC[ij + 1][ii + 1][0]
            g = RGB_SRC[ij][ii][1] + RGB_SRC[ij + 1][ii][1] + \
                RGB_SRC[ij][ii + 1][1] + RGB_SRC[ij + 1][ii + 1][1]
            b = RGB_SRC[ij][ii][2] + RGB_SRC[ij + 1][ii][2] + \
                RGB_SRC[ij][ii + 1][2] + RGB_SRC[ij + 1][ii + 1][2]

            us.append((yr * r + yg * g + yb * b + uv2_shift) >> FIX18)

        print('    &%s,' % us)
    print('];\n')

    print('const CR2_BT%d%s_REF: &[&[u8]] = &[' % (model, SUFFIX))
    for j in range(4):
        us = []
        for i in range(4):
            ii = 2 * i
            ij = 2 * j

            r = RGB_SRC[ij][ii][0] + RGB_SRC[ij + 1][ii][0] + \
                RGB_SRC[ij][ii + 1][0] + RGB_SRC[ij + 1][ii + 1][0]
            g = RGB_SRC[ij][ii][1] + RGB_SRC[ij + 1][ii][1] + \
                RGB_SRC[ij][ii + 1][1] + RGB_SRC[ij + 1][ii + 1][1]
            b = RGB_SRC[ij][ii][2] + RGB_SRC[ij + 1][ii][2] + \
                RGB_SRC[ij][ii + 1][2] + RGB_SRC[ij + 1][ii + 1][2]

            us.append((zr * r + zg * g + zb * b + uv2_shift) >> FIX18)

        print('    &%s,' % us)
    print('];')
