#!/usr/bin/env python3
from array import array
from itertools import product as cartesian_product
from random import Random
from os.path import join, exists, dirname, realpath
from os import environ
from fractions import Fraction as frac
import sys

sys.path.append(dirname(realpath(__file__)))

kr = 0.299
kg = 0.587
kb = 0.114

Y_MIN = 16
Y_MAX = 235
Y_RANGE = Y_MAX - Y_MIN

C_MIN = 16
C_MAX = 240
C_HALF = (C_MAX + C_MIN) >> 1
C_RANGE = C_MAX - C_MIN

FULL_RANGE = 255
Y_SCALE = (Y_RANGE / FULL_RANGE)
C_SCALE = (C_RANGE / FULL_RANGE)

ATLAS_WIDTH = 8192

SHUFFLER = Random(0xDEADBEEF)

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

if __name__ == '__main__':
    root_dir = dirname(realpath(__file__))

    # write encode input test source
    bgra_input = join(root_dir, 'input.bgra')
    rgb_input = join(root_dir, 'input.rgb')

    if not exists(bgra_input) or not exists(rgb_input):
        sample_list = list(cartesian_product(range(256), repeat=3))
        rgb_gamut_size = len(sample_list)
        SHUFFLER.shuffle(sample_list)

    if not exists(bgra_input):
        height = ((4 * len(sample_list)) + (ATLAS_WIDTH - 1)) // ATLAS_WIDTH
        BGRA = array('B', [0] * (ATLAS_WIDTH * height))

        color = 0
        for r, g, b in sample_list:
            BGRA[4 * color] = b
            BGRA[4 * color + 1] = g
            BGRA[4 * color + 2] = r
            BGRA[4 * color + 3] = hash((r, g, b)) & 0xFF
            color += 1

        with open(bgra_input, 'wb') as fn:
            fn.write(('P5\n%d %d\n255\n' % (ATLAS_WIDTH, height)).encode('utf-8'))
            BGRA.tofile(fn)

    # the same in RGB format
    if not exists(rgb_input):
        atlas_width = 6144 # 3 * 2048 -> this way width evenly divisible by 3

        height = ((3 * len(sample_list)) + (atlas_width - 1)) // atlas_width
        RGB = array('B', [0] * (atlas_width * height))

        color = 0
        for r, g, b in sample_list:
            RGB[3 * color] = r
            RGB[3 * color + 1] = g
            RGB[3 * color + 2] = b
            color += 1

        with open(rgb_input, 'wb') as fn:
            fn.write(('P5\n%d %d\n255\n' % (atlas_width, height)).encode('utf-8'))
            RGB.tofile(fn)

    # write decode input test source
    decode_input_nv12 = join(root_dir, 'input.nv12')
    decode_input_i420 = join(root_dir, 'input.i420')
    decode_input_i444 = join(root_dir, 'input.i444')
    if not exists(decode_input_nv12) or not exists(decode_input_i420) or not exists(decode_input_i444):
        frac_kr = frac(int(kr * 10000), 10000)
        frac_kg = frac(int(kg * 10000), 10000)
        frac_kb = frac(int(kb * 10000), 10000)
        frac_S = frac(Y_RANGE, FULL_RANGE)
        frac_P = frac(C_RANGE, FULL_RANGE)
        frac_iS = 1 / frac_S
        frac_ikr = 1 - frac_kr
        frac_ikb = 1 - frac_kb

        r0 = frac_iS
        r1 = 2 * frac_ikr / frac_P
        g0 = (frac_ikr - frac_kb) / (frac_kg * frac_S)
        g1 = (-2 * frac_ikb * frac_kb) / (frac_kg * frac_P)
        g2 = (-2 * frac_ikr * frac_kr) / (frac_kg * frac_P)
        b0 = frac_iS
        b1 = 2 * frac_ikb / frac_P

        sample_list = []
        for y in range(1 + Y_RANGE):
            for cb in range(1 + C_RANGE):
                b = b0 * y + b1 * cb
                if -1 < b < 256:
                    computed_red = r0 * y
                    computed_green = g0 * y + g1 * cb
                    for cr in range(1 + C_RANGE):
                        r = computed_red + r1 * cr
                        g = computed_green + g2 * cr
                        if (-1 < r < 256) and (-1 < g < 256):
                            sample_list.append((y, cb, cr))

        ycbcr_gamut_size = len(sample_list)
        SHUFFLER.shuffle(sample_list)

        cols = 2 * len(sample_list)
        height = 2 * ((cols + (ATLAS_WIDTH - 1)) // ATLAS_WIDTH)
        Y = array('B', [Y_MIN] * (ATLAS_WIDTH * height))
        CbCr = array('B', [C_HALF] * (ATLAS_WIDTH * height // 2))
        Cb = array('B', [C_HALF] * (ATLAS_WIDTH * height // 4))
        Cr = array('B', [C_HALF] * (ATLAS_WIDTH * height // 4))
        i = 0
        j = 0

        # half because y has to be repeated
        for samples in chunks(sample_list, ATLAS_WIDTH // 2):
            for (y, cb, cr) in samples:
                y += Y_MIN
                Y[i] = y
                Y[i + 1] = y
                Y[ATLAS_WIDTH + i] = y
                Y[ATLAS_WIDTH + i + 1] = y
                CbCr[j] = cb + C_HALF
                CbCr[j + 1] = cr + C_HALF
                Cb[j // 2] = cb + C_HALF
                Cr[j // 2] = cb + C_HALF

                i += 2
                j += 2

            i += ATLAS_WIDTH

        with open(decode_input_nv12, 'wb') as fn:
            fn.write(('P5\n%d %d\n255\n' % (ATLAS_WIDTH, height + height // 2)).encode('utf-8'))
            Y.tofile(fn)
            CbCr.tofile(fn)

        # i420
        with open(decode_input_i420, 'wb') as fn:
            fn.write(('P5\n%d %d\n255\n' % (ATLAS_WIDTH, height + height // 2)).encode('utf-8'))
            Y.tofile(fn)
            Cb.tofile(fn)
            Cr.tofile(fn)

        # i444
        Y = array('B', [Y_MIN] * (ATLAS_WIDTH * height))
        Cb = array('B', [C_HALF] * (ATLAS_WIDTH * height))
        Cr = array('B', [C_HALF] * (ATLAS_WIDTH * height))
        i = 0

        # half because y has to be repeated
        for samples in chunks(sample_list, ATLAS_WIDTH // 2):
            for (y, cb, cr) in samples:
                y += Y_MIN
                cb += C_HALF
                cr += C_HALF

                Y[i] = y
                Y[i + 1] = y
                Y[ATLAS_WIDTH + i] = y
                Y[ATLAS_WIDTH + i + 1] = y

                Cb[i] = cb
                Cb[i + 1] = cb
                Cb[ATLAS_WIDTH + i] = cb
                Cb[ATLAS_WIDTH + i + 1] = cb

                Cr[i] = cr
                Cr[i + 1] = cr
                Cr[ATLAS_WIDTH + i] = cr
                Cr[ATLAS_WIDTH + i + 1] = cr

                i += 2
            i += ATLAS_WIDTH

        with open(decode_input_i444, 'wb') as fn:
            fn.write(('P5\n%d %d\n255\n' % (ATLAS_WIDTH, height * 3)).encode('utf-8'))
            Y.tofile(fn)
            Cb.tofile(fn)
            Cr.tofile(fn)

