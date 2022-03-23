#!/usr/bin/env python3
from array import array
from itertools import product as cartesian_product
from random import Random
from os.path import join, exists, dirname, realpath
import sys

sys.path.append(dirname(realpath(__file__)))

ATLAS_WIDTH = 8192
SHUFFLER = Random(0xDEADBEEF)


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
            fn.write(('P5\n%d %d\n255\n' % (ATLAS_WIDTH, height))
                     .encode('utf-8'))

            BGRA.tofile(fn)

    # the same in RGB format
    if not exists(rgb_input):
        atlas_width = 6144  # 3 * 2048 -> this way width evenly divisible by 3

        height = ((3 * len(sample_list)) + (atlas_width - 1)) // atlas_width
        RGB = array('B', [0] * (atlas_width * height))

        color = 0
        for r, g, b in sample_list:
            RGB[3 * color] = r
            RGB[3 * color + 1] = g
            RGB[3 * color + 2] = b
            color += 1

        with open(rgb_input, 'wb') as fn:
            fn.write(('P5\n%d %d\n255\n' % (atlas_width, height))
                     .encode('utf-8'))

            RGB.tofile(fn)

