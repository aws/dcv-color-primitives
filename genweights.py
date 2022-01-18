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

from fractions import Fraction as frac
import sys

FIX16 = 16
FIX14 = (FIX16 + 2)
FIX16_HALF = (1 << (FIX16 - 1))
FIX14_HALF = (1 << (FIX16 + 1))
FIX_8_14 = 14
FIX_8_14_HALF = (1 << (FIX_8_14 - 1))

FIX16_MULT = (1 << FIX16)
FIX8_14_MULT = (1 << FIX_8_14)

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
    FULL_RANGE = 255
    Y_SCALE = (Y_RANGE / FULL_RANGE)
    C_SCALE = (C_RANGE / FULL_RANGE)

    if full_range:
        ar = kr
        ag = kg
        ab = kb
        br = (-kr / (2.0 * (1.0 - kb)))
        bg = (-kg / (2.0 * (1.0 - kb)))
        bb = 0.5
        cr = 0.5
        cg = (-kg / (2.0 * (1.0 - kr)))
        cb = (-kb / (2.0 * (1.0 - kr)))
    else:
        ar = Y_SCALE * kr
        ag = Y_SCALE * kg
        ab = Y_SCALE * kb
        br = C_SCALE * (-kr / (2.0 * (1.0 - kb)))
        bg = C_SCALE * (-kg / (2.0 * (1.0 - kb)))
        bb = C_SCALE * 0.5
        cr = C_SCALE * 0.5
        cg = C_SCALE * (-kg / (2.0 * (1.0 - kr)))
        cb = C_SCALE * (-kb / (2.0 * (1.0 - kr)))

    ikb = 1.0 - kb
    ikr = 1.0 - kr
    y_scale = 1 / Y_SCALE
    c_scale = 2 * FULL_RANGE

    inverse = (
        (y_scale,  0, c_scale * ikr / C_RANGE, ),
        (y_scale, -c_scale * ikb * kb / (C_RANGE * kg), -
         c_scale * ikr * kr / (C_RANGE * kg), ),
        (y_scale, c_scale * ikb / C_RANGE, 0, )
    )
    offset = (
        -((8.0 / 7.0) * ikr + (Y_MIN / Y_RANGE)),
        (
            (Y_MIN / Y_RANGE) * (kr - ikb) +
            (8.0 / 7.0) * (ikr * kr + ikb * kb)
        ) / kg,
        -((8.0 / 7.0) * ikb + (Y_MIN / Y_RANGE))
    )

    print('[ Y\']   [ %6.3f, %6.3f %6.3f] [ R ]   [%6d]' % (ar, ag, ab, Y_MIN))
    print('[Cb\'] = [ %6.3f, %6.3f %6.3f] [ G ] + [%6d]' %
          (br, bg, bb, C_HALF))
    print('[Cr\']   [ %6.3f, %6.3f %6.3f] [ B ]   [%6d]' %
          (cr, cg, cb, C_HALF))
    print('')

    print('[ R ]   [ %6.3f, %6.3f %6.3f] [ Y\']   [%6.3f]' %
          (inverse[0][0], inverse[0][1], inverse[0][2], offset[0]))
    print('[ G ] = [ %6.3f, %6.3f %6.3f] [Cb\'] + [%6.3f]' %
          (inverse[1][0], inverse[1][1], inverse[1][2], offset[1]))
    print('[ B ]   [ %6.3f, %6.3f %6.3f] [Cr\']   [%6.3f]' %
          (inverse[2][0], inverse[2][1], inverse[2][2], offset[2]))
    print('')

    xr = int(FIX16_MULT * ar + 0.5)
    xg = int(FIX16_MULT * ag + 0.5)
    xb = int(FIX16_MULT * ab + 0.5)
    yr = int(FIX16_MULT * br + 0.5)
    yg = int(FIX16_MULT * bg + 0.5)
    yb = int(FIX16_MULT * bb + 0.5)
    zr = int(FIX16_MULT * cr + 0.5)
    zg = int(FIX16_MULT * cg + 0.5)
    zb = int(FIX16_MULT * cb + 0.5)

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

    print('// Coefficient table for %s%s' %
          (model, " (full range)" if full_range else ""))
    print('pub const XR_%d%s: i32 = %d;' % (model, SUFFIX, xr))
    print('pub const XG_%d%s: i32 = %d;' % (model, SUFFIX, xg))
    print('pub const XB_%d%s: i32 = %d;' % (model, SUFFIX, xb))
    print('pub const YR_%d%s: i32 = %d;' % (model, SUFFIX, yr))
    print('pub const YG_%d%s: i32 = %d;' % (model, SUFFIX, yg))
    print('pub const YB_%d%s: i32 = %d;' % (model, SUFFIX, yb))
    print('pub const ZR_%d%s: i32 = %d;' % (model, SUFFIX, zr))
    print('pub const ZG_%d%s: i32 = %d;' % (model, SUFFIX, zg))
    print('pub const ZB_%d%s: i32 = %d;' % (model, SUFFIX, zb))
    print('')

    print('pub const XXYM_%d%s: i32 = %d;' % (model, SUFFIX, s))
    print('pub const RCRM_%d%s: i32 = %d;' % (model, SUFFIX, rz))
    print('pub const GCRM_%d%s: i32 = %d;' % (model, SUFFIX, gz))
    print('pub const GCBM_%d%s: i32 = %d;' % (model, SUFFIX, gy))
    print('pub const BCBM_%d%s: i32 = %d;' % (model, SUFFIX, by))
    print('pub const RN_%d%s: i32 = %d;' % (model, SUFFIX, rw >> 8))
    print('pub const GP_%d%s: i32 = %d;' % (model, SUFFIX, gw >> 8))
    print('pub const BN_%d%s: i32 = %d;' % (model, SUFFIX, bw >> 8))
    print('')
