/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: MIT-0
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the "Software"), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify,
 * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dcv_color_primitives.h>

static char domain[256];
static uint8_t domain_size = 0;
static char line_buffer[1024];

typedef struct {
    DcpResult result;
    DcpErrorKind error;
} DcpStatus;

DcpStatus dcp_status()
{
    DcpStatus result = {
       DCP_RESULT_OK,
       (DcpErrorKind) -1
    };

    return result;
}

#define EXIT_SKIP_CODE 77

#define TEST_BEGIN_GROUP(x, ...)                                                                 \
    sprintf(line_buffer, x, ##__VA_ARGS__);                                                      \
    fprintf(stdout, "%.*s%s\r", domain_size, domain, line_buffer);                               \
    domain[domain_size] = ' ';                                                                   \
    domain_size++;

#define TEST_END_GROUP() domain_size--;
#define TEST_BEGIN(x, ...)  { sprintf(line_buffer, x, ##__VA_ARGS__);
#define TEST_END() fprintf(stdout, "%.*s%s:OK\r", domain_size, domain, line_buffer); }

#define TEST_ASSERT(r, e)                                                                        \
    if (status.result != r) {                                                                    \
        fprintf(stderr, "%s: FAIL; Unexpected result: was %d, expected %d at line %d\n",         \
            line_buffer, status.result, r, __LINE__);                                            \
        exit(EXIT_FAILURE);                                                                      \
    } else if (status.result == DCP_RESULT_ERR && status.error != e) {                           \
        fprintf(stderr, "%s: FAIL; Unexpected error kind: was 0x%X, expected 0x%X at line %d\n", \
            line_buffer, status.error, e, __LINE__);                                             \
        exit(EXIT_FAILURE);                                                                      \
    }

#define TEST_ASSERT_EQ(val, x)                                                                   \
    if (val != x) {                                                                              \
        fprintf(stderr, "%s: FAIL; Unexpected status: was %lld, expected %lld at line %d\n",     \
            line_buffer, (int64_t)val, (int64_t)x, __LINE__);                                    \
        exit(EXIT_FAILURE);                                                                      \
    }

#define TEST_ASSERT_EQ_T(val, x, t)                                                              \
    if (abs((int32_t)val - (int32_t)x) > t) {                                                    \
        fprintf(stderr, "%s: FAIL; Unexpected status: was %lld, expected %lld at line %d\n",     \
            line_buffer, (int64_t)val, (int64_t)x, __LINE__);                                    \
        exit(EXIT_FAILURE);                                                                      \
    }                                                                                            \

#define SET_EXPECTED(pred, val)                                                                  \
    if (expected.result == DCP_RESULT_OK && (pred)) {                                            \
        expected.result = DCP_RESULT_ERR;                                                        \
        expected.error = val;                                                                    \
    }                                                                                            \

typedef struct {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
} rgba;

typedef struct {
    uint8_t r;
    uint8_t g;
    uint8_t b;
} rgb;

static const rgba rgb_to_yuv_input[8][8] = {
    {
        { 161, 24, 44, 58 },
        { 35, 95, 51, 205 },
        { 177, 30, 252, 158 },
        { 248, 94, 62, 28 },
        { 247, 51, 135, 38 },
        { 98, 147, 200, 127 },
        { 68, 103, 20, 124 },
        { 233, 227, 165, 0 }
    }, {
        { 251, 19, 32, 170 },
        { 235, 183, 25, 77 },
        { 146, 81, 218, 161 },
        { 25, 124, 96, 56 },
        { 22, 127, 167, 179 },
        { 247, 34, 40, 53 },
        { 164, 193, 159, 24 },
        { 96, 158, 17, 223 },
    }, {
        { 240, 123, 14, 108 },
        { 0, 105, 52, 116 },
        { 194, 219, 244, 47 },
        { 216, 254, 153, 84 },
        { 116, 77, 133, 68 },
        { 190, 96, 190, 133 },
        { 118, 4, 170, 115 },
        { 218, 145, 23, 50 }
    }, {
        { 202, 120, 126, 231 },
        { 42, 28, 137, 40 },
        { 136, 227, 210, 177 },
        { 254, 140, 238, 88 },
        { 90, 195, 170, 67 },
        { 125, 242, 148, 88 },
        { 1, 91, 190, 245 },
        { 31, 100, 190, 225 }
    }, {
        { 207, 49, 249, 131 },
        { 48, 120, 34, 82 },
        { 43, 145, 253, 141 },
        { 83, 205, 105, 44 },
        { 16, 9, 157, 22 },
        { 253, 131, 178, 148 },
        { 142, 236, 98, 6 },
        { 246, 190, 15, 213 }
    }, {
        { 72, 207, 6, 168 },
        { 220, 39, 6, 219 },
        { 244, 14, 252, 45 },
        { 159, 106, 17, 184 },
        { 222, 72, 230, 39 },
        { 6, 185, 30, 35 },
        { 101, 223, 30, 14 },
        { 40, 71, 16, 244 }
    },{
        { 124, 121, 46, 190 },
        { 244, 206, 61, 169 },
        { 43, 130, 87, 247 },
        { 170, 10, 238, 229 },
        { 12, 168, 14, 220 },
        { 96, 60, 226, 235 },
        { 206, 93, 122, 117 },
        { 126, 168, 203, 39 }
    }, {
        { 181, 88, 248, 45 },
        { 65, 24, 208, 166 },
        { 24, 21, 151, 85 },
        { 60, 86, 9, 153 },
        { 225, 80, 156, 159 },
        { 210, 181, 6, 214 },
        { 17, 142, 255, 163 },
        { 189, 137, 72, 87 }
    }
};

static const uint8_t rgb_to_yuv_bt601_output[12][8] = {
    { 74, 78, 101, 133, 118, 135, 87, 206 },
    { 93, 171, 116, 94, 102, 100, 171, 122 },
    { 141, 74, 200, 214, 98, 132, 65, 147 },
    { 141, 54, 186, 175, 154, 185, 81, 93 },
    { 118, 92, 125, 151, 40, 164, 181, 176 },
    { 139, 93, 110, 112, 132, 114, 157, 64 },
    { 113, 188, 101, 88, 105, 93, 128, 153 },
    { 131, 65, 48, 76, 129, 162, 117, 141 },

    { 96, 171, 151, 152, 139, 153, 97, 121 },
    { 119, 141, 130, 124, 135, 118, 153, 127 },
    { 110, 145, 143, 132, 146, 135, 73, 117 },
    { 135, 145, 152, 129, 116, 135, 140, 126 }
};

static const uint8_t rgb_to_yuv_bt709_output[12][8] = {
    { 63, 84, 82, 123, 101, 137, 93, 208 },
    { 75, 173, 106, 103, 108, 84, 174, 132 },
    { 136, 84, 201, 221, 93, 121, 51, 146 },
    { 134, 49, 193, 163, 163, 197, 84, 95 },
    { 99, 101, 129, 164, 34, 154, 193, 179 },
    { 157, 80, 85, 111, 115, 133, 173, 68 },
    { 116, 191, 109, 68, 122, 84, 118, 155 },
    { 118, 56, 43, 80, 116, 166, 122, 139 },

    { 100, 169, 154, 154, 142, 154, 96, 118 },
    { 120, 140, 130, 124, 134, 118, 153, 129 },
    { 112, 144, 144, 133, 147, 137, 71, 113 },
    { 137, 146, 153, 131, 117, 135, 140, 127 }
};

#define MAX_NUMBER_OF_PLANES 3

static const size_t num_log2_den[][2] = {
    { 4, 0, },
    { 4, 0, },
    { 3, 0, },
    { 4, 0, },
    { 3, 0, },
    { 3, 0, },
    { 2, 0, },
    { 3, 1, },
    { 3, 1, },
};

static const size_t num_log2_den_per_plane[][3 * MAX_NUMBER_OF_PLANES] = {
    { 4, 0,  0, 0,  0, 0, },
    { 4, 0,  0, 0,  0, 0, },
    { 3, 0,  0, 0,  0, 0, },
    { 4, 0,  0, 0,  0, 0, },
    { 3, 0,  0, 0,  0, 0, },
    { 1, 0,  1, 0,  1, 0, },
    { 1, 0,  1, 1,  1, 1, },
    { 1, 0,  1, 2,  1, 2, },
    { 1, 0,  1, 1,  0, 0, },
};

/*
 * Largest group that uses neither avx2 nor sse2 is 62x64.
 * We can arrange the image as blocks:
 * y0 y0 | y1 y1 | ...
 * y0 y0 | y1 y1 | ...
 * u0 v0 | u1 v1 | ...
 *
 * Min width to represent 8 colors below: 2 * 8 = 16x16
 *
 * Color table (bt601):
 *             r    g    b
 * black    (  0,   0,   0):    16, 128, 128
 * red      (255,   0,   0):    82,  90, 240
 * green    (  0, 255,   0):   145,  54,  34
 * yellow   (255, 255,   0):   210,  16, 146
 * blue     (  0,   0, 255):    41, 240, 110
 * magenta  (255,   0, 255):   107, 202, 222
 * cyan     (  0, 255, 255):   169, 166,  16
 * white    (255, 255, 255):   235, 128, 128
 *
 * Color table (bt709):
 *             r    g    b
 * black    (  0,   0,   0):    16, 128, 128
 * red      (255,   0,   0):    63, 102, 240
 * green    (  0, 255,   0):   173,  42,  26
 * yellow   (255, 255,   0):   219,  16, 138
 * blue     (  0,   0, 255):    32, 240, 118
 * magenta  (255,   0, 255):    78, 214, 230
 * cyan     (  0, 255, 255):   188, 154,  16
 * white    (255, 255, 255):   235, 128, 128
 */
static const uint8_t y_to_rgb_input[2][8] = {
    {  16,  82, 145, 210,  41, 107, 169, 235 },
    {  16,  63, 173, 219,  32,  78, 188, 235 }
};

static const uint8_t cb_to_rgb_input[2][8] = {
    { 128,  90,  54,  16, 240, 202, 166, 128 },
    { 128, 102,  42,  16, 240, 214, 154, 128 }
};

static const uint8_t cr_to_rgb_input[2][8] = {
    { 128, 240,  34, 146, 110, 222,  16, 128 },
    { 128, 240,  26, 138, 118, 230,  16, 128 }
};

static uint8_t stack[1024 * 1024];
static uint8_t *ptr = stack;

static void *
stack_alloc(size_t size)
{
    uint8_t *result = ptr;
    memset(result, 0, size);
    ptr += size;
    return result;
}

static void
stack_reset(void)
{
    ptr = stack;
}

static void
init(void)
{
    dcp_initialize();
}

static void
unit_init(void)
{
    char *desc;

    TEST_BEGIN_GROUP(__FUNCTION__);

    TEST_BEGIN("init");

    desc = dcp_describe_acceleration();
    TEST_ASSERT_EQ((desc == NULL), 1);

    init();

    desc = dcp_describe_acceleration();
    TEST_ASSERT_EQ((desc != NULL), 1);

    printf("%s\n", desc);
    dcp_unref_string(desc);

    TEST_END();

    TEST_END_GROUP();
}

static void
convert_image_rgb_to_yuv_size_mode_stride(uint32_t       num_planes,
                                          uint32_t       width,
                                          uint32_t       height,
                                          DcpColorSpace  color_space,
                                          DcpPixelFormat pixel_format,
                                          size_t         src_stride,
                                          size_t         luma_stride,
                                          size_t         chroma_stride)
{
    const uint32_t chroma_height = (height / 2);
    const size_t in_size = src_stride * height;
    const size_t out_size = (luma_stride * height) + (chroma_stride * chroma_height);
    const size_t min_src_stride = (size_t)width * ((pixel_format == DCP_PIXEL_FORMAT_BGR) ? 3 : 4);
    const size_t src_fill_bytes = src_stride - min_src_stride;
    const size_t luma_fill_bytes = luma_stride - width;
    const size_t chroma_fill_bytes = chroma_stride - width;

    uint8_t *test_input;
    uint8_t *test_output;
    uint8_t *input;
    const uint8_t *output;
    size_t dst_strides[2];
    uint8_t *dst_buffers[2];
    uint32_t y;
    size_t count;

    DcpStatus status = dcp_status();

    DcpImageFormat src_format = {
        pixel_format,
        DCP_COLOR_SPACE_LRGB,
        1
    };

    DcpImageFormat dst_format = {
        DCP_PIXEL_FORMAT_NV12,
        color_space,
        num_planes
    };

    /* Allocate and initialize input */
    test_input = (uint8_t *)stack_alloc(in_size);
    input = test_input;

    for (y = 0; y < height; y++, input += src_fill_bytes) {
        uint32_t x;

        for (x = 0; x < width; x++) {
            if (pixel_format == DCP_PIXEL_FORMAT_ARGB) {
                *input++ = rgb_to_yuv_input[y][x].a;
                *input++ = rgb_to_yuv_input[y][x].r;
                *input++ = rgb_to_yuv_input[y][x].g;
                *input++ = rgb_to_yuv_input[y][x].b;
            } else if (pixel_format == DCP_PIXEL_FORMAT_BGRA) {
                *input++ = rgb_to_yuv_input[y][x].b;
                *input++ = rgb_to_yuv_input[y][x].g;
                *input++ = rgb_to_yuv_input[y][x].r;
                *input++ = rgb_to_yuv_input[y][x].a;
            } else { /* BGR */
                *input++ = rgb_to_yuv_input[y][x].b;
                *input++ = rgb_to_yuv_input[y][x].g;
                *input++ = rgb_to_yuv_input[y][x].r;
            }
        }
    }

    /* Allocate and initialize output */
    test_output = (uint8_t *)stack_alloc(out_size);

    /* Compute strides */
    src_stride = src_fill_bytes == 0 ? DCP_STRIDE_AUTO : src_stride;
    if (num_planes == 1) {
        dst_strides[0] = luma_fill_bytes == 0 ? DCP_STRIDE_AUTO : luma_stride;
        dst_buffers[0] = test_output;
    } else { /* 2 */
        dst_strides[0] = luma_fill_bytes == 0 ? DCP_STRIDE_AUTO : luma_stride;
        dst_strides[1] = chroma_fill_bytes == 0 ? DCP_STRIDE_AUTO : chroma_stride;
        dst_buffers[0] = test_output;
        dst_buffers[1] = &test_output[luma_stride * height];
    }

    /* Perform conversion */
    status.result = dcp_convert_image(width, height,
                                      &src_format, &src_stride, &test_input,
                                      &dst_format, dst_strides, dst_buffers, &status.error);

    TEST_ASSERT(DCP_RESULT_OK, -1);

    /* Check output */
    count = 0;
    output = test_output;

    /* Check all luma samples are correct */
    for (y = 0; y < height; y++) {
        size_t x;

        for (x = 0; x < width; x++, count++, output++) {
            TEST_ASSERT_EQ(*output, (color_space == DCP_COLOR_SPACE_BT601 ? rgb_to_yuv_bt601_output : rgb_to_yuv_bt709_output)[y][x]);
        }

        for (x = 0; x < luma_fill_bytes; x++, count++, output++) {
            TEST_ASSERT_EQ(*output, 0);
        }
    }

    /* Check all chroma samples are correct */
    for (y = 0; y < chroma_height; y++) {
        size_t x;

        for (x = 0; x < width; x++, count++, output++) {
            TEST_ASSERT_EQ(*output, (color_space == DCP_COLOR_SPACE_BT601 ? rgb_to_yuv_bt601_output : rgb_to_yuv_bt709_output)[8 + y][x]);
        }

        for (x = 0; x < chroma_fill_bytes; x++, count++, output++) {
            TEST_ASSERT_EQ(*output, 0);
        }
    }

    /* Rest must be identically null */
    for (; count < out_size; count++, output++) {
        TEST_ASSERT_EQ(*output, 0);
    }

    stack_reset();
}

static void
convert_image_rgb_to_yuv_size_mode(uint32_t       num_planes,
                                   uint32_t       width,
                                   uint32_t       height,
                                   DcpColorSpace  color_space,
                                   DcpPixelFormat pixel_format)
{
    const size_t max_fill_bytes = 15;
    const size_t min_src_stride = (size_t)width * (pixel_format == DCP_PIXEL_FORMAT_BGR ? 3 : 4);
    const size_t max_src_stride = min_src_stride + max_fill_bytes;
    const size_t min_dst_stride = width;
    const size_t max_dst_stride = min_dst_stride + max_fill_bytes;

    TEST_BEGIN_GROUP("cs=%d,pf=%d", color_space, pixel_format);

    if (num_planes == 1) {
        size_t luma_stride;

        for (luma_stride = min_dst_stride; luma_stride <= max_dst_stride; luma_stride++) {
            size_t src_stride;

            TEST_BEGIN("D=%llu", luma_stride);

            for (src_stride = min_src_stride; src_stride <= max_src_stride; src_stride++) {
                convert_image_rgb_to_yuv_size_mode_stride(num_planes, width, height, color_space, pixel_format, src_stride, luma_stride, luma_stride);
            }

            TEST_END();
        }
    } else { /* 2 planes */
        size_t luma_stride;

        for (luma_stride = min_dst_stride; luma_stride <= max_dst_stride; luma_stride++) {
            size_t chroma_stride;

            for (chroma_stride = min_dst_stride; chroma_stride <= max_dst_stride; chroma_stride++) {
                size_t src_stride;

                TEST_BEGIN("Y=%llu,C=%llu", luma_stride, chroma_stride);

                for (src_stride = min_src_stride; src_stride <= max_src_stride; src_stride++) {
                    convert_image_rgb_to_yuv_size_mode_stride(num_planes, width, height, color_space, pixel_format, src_stride, luma_stride, chroma_stride);
                }

                TEST_END();
            }
        }
    }

    TEST_END_GROUP();
}

static void
convert_image_rgb_to_yuv_size(uint32_t num_planes,
                              uint32_t width,
                              uint32_t height)
{
    static const DcpPixelFormat supported_pixel_formats[] = {
        DCP_PIXEL_FORMAT_ARGB,
        DCP_PIXEL_FORMAT_BGRA,
        DCP_PIXEL_FORMAT_BGR
    };

    static const size_t supported_pixel_formats_count = sizeof(supported_pixel_formats) / sizeof(supported_pixel_formats[0]);

    DcpColorSpace color_space;

    TEST_BEGIN_GROUP("%ux%u", width, height);

    for (color_space = DCP_COLOR_SPACE_BT601; color_space <= DCP_COLOR_SPACE_BT709; color_space++) {
        size_t i;

        for (i = 0; i < supported_pixel_formats_count; i++) {
            convert_image_rgb_to_yuv_size_mode(num_planes, width, height, color_space, supported_pixel_formats[i]);
        }
    }

    TEST_END_GROUP();
}

static void
unit_convert_image_rgb_to_yuv(uint32_t num_planes)
{
    const uint32_t max_width = 8;
    const uint32_t max_height = 8;
    uint32_t width;

    TEST_BEGIN_GROUP(__FUNCTION__);
    init();

    for (width = 2; width <= max_width; width += 2) {
        uint32_t height;

        for (height = 2; height <= max_height; height += 2) {
            convert_image_rgb_to_yuv_size(num_planes, width, height);
        }
    }

    TEST_END_GROUP();
}

static void
convert_image_yuv_to_rgb_size_mode_stride(uint32_t       num_planes,
                                          uint32_t       width,
                                          uint32_t       height,
                                          DcpColorSpace  color_space,
                                          size_t         luma_fill_bytes,
                                          size_t         u_chroma_fill_bytes,
                                          size_t         v_chroma_fill_bytes,
                                          size_t         dst_fill_bytes,
                                          DcpPixelFormat format)
{

    const uint32_t chroma_height = 
        format == DCP_PIXEL_FORMAT_NV12 || format == DCP_PIXEL_FORMAT_I420 ?
            (height / 2)
        : format == DCP_PIXEL_FORMAT_I444 ?
            height
        : 0;

    const size_t luma_stride = width + luma_fill_bytes;

    const size_t u_chroma_stride =
        format == DCP_PIXEL_FORMAT_NV12 || format == DCP_PIXEL_FORMAT_I444 ?
            width + u_chroma_fill_bytes
        : format == DCP_PIXEL_FORMAT_I420 ?
            (width / 2) + u_chroma_fill_bytes
        : 0;

    const size_t v_chroma_stride =
        format == DCP_PIXEL_FORMAT_NV12 ?
            u_chroma_stride
        : format == DCP_PIXEL_FORMAT_I420 ?
            (width / 2) + v_chroma_fill_bytes
        : format == DCP_PIXEL_FORMAT_I444 ?
            width + v_chroma_fill_bytes
        : 0;

    const size_t in_size =
        format == DCP_PIXEL_FORMAT_NV12 ?
            (luma_stride * height) + (u_chroma_stride * chroma_height)
        : format == DCP_PIXEL_FORMAT_I420 || format == DCP_PIXEL_FORMAT_I444 ?
            (luma_stride * height) + (u_chroma_stride * chroma_height) + (v_chroma_stride * chroma_height)
        : 0;

    const size_t dst_stride =
        dst_fill_bytes == 0 ?
            DCP_STRIDE_AUTO
        : (width * 4) + dst_fill_bytes;

    const size_t out_size = ((width * 4) + dst_fill_bytes) * height;
    uint8_t *test_input;
    uint8_t *test_output;
    uint8_t *input;
    const uint8_t *output;
    size_t src_strides[3];
    uint8_t *src_buffers[3];
    uint32_t y;
    size_t count;
    int32_t color_space_index = color_space - DCP_COLOR_SPACE_BT601;

    DcpImageFormat src_format = {
        format,
        color_space,
        num_planes
    };

    DcpImageFormat dst_format = {
        DCP_PIXEL_FORMAT_BGRA,
        DCP_COLOR_SPACE_LRGB,
        1
    };

    DcpStatus status = dcp_status();

    /* Allocate and initialize input */
    test_input = (uint8_t *)stack_alloc(in_size);
    input = test_input;

    for (y = 0; y < height; y++, input += luma_fill_bytes) {
        uint32_t x;

        for (x = 0; x < width; x += 2, input += 2) {
            uint32_t index = (x >> 1) & 0x7;
            uint8_t luma = y_to_rgb_input[color_space_index][index];
            input[0] = luma;
            input[1] = luma;
        }
    }

    if (format == DCP_PIXEL_FORMAT_NV12) {
        for (y = 0; y < chroma_height; y++, input += u_chroma_fill_bytes) {
            uint32_t x;

            for (x = 0; x < width; x += 2, input += 2) {
                uint32_t index = (x >> 1) & 0x7;
                input[0] = cb_to_rgb_input[color_space_index][index];
                input[1] = cr_to_rgb_input[color_space_index][index];
            }
        }
    } else if (format == DCP_PIXEL_FORMAT_I420) {
        for (y = 0; y < chroma_height; y++, input += u_chroma_fill_bytes) {
            uint32_t x;

            for (x = 0; x < width; x += 2, input++) {
                uint32_t index = (x >> 1) & 0x7;
                input[0] = cb_to_rgb_input[color_space_index][index];
            }
        }

        for (y = 0; y < chroma_height; y++, input += v_chroma_fill_bytes) {
            uint32_t x;

            for (x = 0; x < width; x += 2, input++) {
                uint32_t index = (x >> 1) & 0x7;
                input[0] = cr_to_rgb_input[color_space_index][index];
            }
        }
    } else if (format == DCP_PIXEL_FORMAT_I444) {
        for (y = 0; y < chroma_height; y++, input += u_chroma_fill_bytes) {
            uint32_t x;

            for (x = 0; x < width; x++, input++) {
                uint32_t index = (x >> 1) & 0x7;
                input[0] = cb_to_rgb_input[color_space_index][index];
            }
        }

        for (y = 0; y < chroma_height; y++, input += v_chroma_fill_bytes) {
            uint32_t x;

            for (x = 0; x < width; x++, input++) {
                uint32_t index = (x >> 1) & 0x7;
                input[0] = cr_to_rgb_input[color_space_index][index];
            }
        }
    } else {
        abort();
    }

    /* Allocate and initialize output */
    test_output = (uint8_t *)stack_alloc(out_size);

    /* Compute strides */
    src_strides[0] = luma_fill_bytes == 0 ? DCP_STRIDE_AUTO : luma_stride;

    if (format == DCP_PIXEL_FORMAT_NV12) {
        if (num_planes == 1) {
            src_buffers[0] = test_input;
        } else if (num_planes == 2) { /* 2 */
            src_strides[1] = u_chroma_fill_bytes == 0 ? DCP_STRIDE_AUTO : u_chroma_stride;
            src_buffers[0] = test_input;
            src_buffers[1] = &test_input[luma_stride * height];
        }
    } else if (format == DCP_PIXEL_FORMAT_I420 || format == DCP_PIXEL_FORMAT_I444) {
        if (num_planes == 3) {
            src_strides[1] = u_chroma_fill_bytes == 0 ? DCP_STRIDE_AUTO : u_chroma_stride;
            src_strides[2] = v_chroma_fill_bytes == 0 ? DCP_STRIDE_AUTO : v_chroma_stride;

            src_buffers[0] = test_input;
            src_buffers[1] = &test_input[luma_stride * height];
            src_buffers[2] = &test_input[(luma_stride * height) + (u_chroma_stride * chroma_height)];
        }
    } else {
        abort();
    }

    /* Perform conversion */
    status.result = dcp_convert_image(width, height,
                               &src_format, src_strides, src_buffers,
                               &dst_format, &dst_stride, &test_output, &status.error);

    TEST_ASSERT(DCP_RESULT_OK, -1);

    /* Check output */
    count = 0;
    output = test_output;

    /* Check all samples are correct */
    for (y = 0; y < height; y++) {
        size_t x;

        for (x = 0; x < width; x++, count += 4, output += 4) {
            uint32_t index = (x >> 1) & 0x7;
            int32_t exp_b = ((index >> 2) & 1) == 0 ? 0: 255;
            int32_t exp_g = ((index >> 1) & 1) == 0 ? 0: 255;
            int32_t exp_r = (index & 1) == 0 ? 0: 255;

            TEST_ASSERT_EQ_T(output[0], exp_b, 2);
            TEST_ASSERT_EQ_T(output[1], exp_g, 2);
            TEST_ASSERT_EQ_T(output[2], exp_r, 2);
            TEST_ASSERT_EQ(output[3], 255);
        }

        for (x = 0; x < dst_fill_bytes; x++, count++, output++) {
            TEST_ASSERT_EQ(*output, 0);
        }
    }

    /* Rest must be identically null */
    for (; count < out_size; count++, output++) {
        TEST_ASSERT_EQ(*output, 0);
    }

    stack_reset();
}

static void
convert_image_yuv_to_rgb_size_mode(uint32_t       num_planes,
                                   uint32_t       width,
                                   uint32_t       height,
                                   DcpColorSpace  color_space,
                                   DcpPixelFormat format)
{
    const size_t max_fill_bytes = 3;

    TEST_BEGIN_GROUP("format=%d", format);

    if (num_planes == 1) {
        size_t luma_fill_bytes;

        for (luma_fill_bytes = 0; luma_fill_bytes <= max_fill_bytes; luma_fill_bytes++) {
            size_t dst_fill_bytes;

            TEST_BEGIN("S=%llu", luma_fill_bytes);

            for (dst_fill_bytes = 0; dst_fill_bytes <= max_fill_bytes; dst_fill_bytes++) {
                convert_image_yuv_to_rgb_size_mode_stride(num_planes,
                                                          width, height,
                                                          color_space,
                                                          luma_fill_bytes,
                                                          luma_fill_bytes,
                                                          luma_fill_bytes,
                                                          dst_fill_bytes,
                                                          format);
            }

            TEST_END();
        }
    } else if (num_planes == 2) {
        size_t luma_fill_bytes;

        for (luma_fill_bytes = 0; luma_fill_bytes <= max_fill_bytes; luma_fill_bytes++) {
            size_t chroma_fill_bytes;

            for (chroma_fill_bytes = 0; chroma_fill_bytes <= max_fill_bytes; chroma_fill_bytes++) {
                size_t dst_fill_bytes;

                TEST_BEGIN("Y=%llu,C=%llu", luma_fill_bytes, chroma_fill_bytes);

                for (dst_fill_bytes = 0; dst_fill_bytes <= max_fill_bytes; dst_fill_bytes++) {
                    convert_image_yuv_to_rgb_size_mode_stride(num_planes,
                                                              width,
                                                              height,
                                                              color_space,
                                                              luma_fill_bytes,
                                                              chroma_fill_bytes,
                                                              chroma_fill_bytes,
                                                              dst_fill_bytes,
                                                              format);
                }

                TEST_END();
            }
        }
    } else if (num_planes == 3) {
        size_t luma_fill_bytes;

        for (luma_fill_bytes = 0; luma_fill_bytes <= max_fill_bytes; luma_fill_bytes++) {
            size_t u_chroma_fill_bytes;

            for (u_chroma_fill_bytes = 0; u_chroma_fill_bytes <= max_fill_bytes; u_chroma_fill_bytes++) {
                size_t v_chroma_fill_bytes;

                for (v_chroma_fill_bytes = 0; v_chroma_fill_bytes <= max_fill_bytes; v_chroma_fill_bytes++) {
                    size_t dst_fill_bytes;

                    TEST_BEGIN("Y=%llu,U=%llu, V=%llu", luma_fill_bytes, u_chroma_fill_bytes, v_chroma_fill_bytes);

                    for (dst_fill_bytes = 0; dst_fill_bytes <= max_fill_bytes; dst_fill_bytes++) {
                        convert_image_yuv_to_rgb_size_mode_stride(num_planes,
                                                                  width,
                                                                  height,
                                                                  color_space,
                                                                  luma_fill_bytes,
                                                                  u_chroma_fill_bytes,
                                                                  v_chroma_fill_bytes,
                                                                  dst_fill_bytes,
                                                                  format);
                    }

                    TEST_END();
                }
            }
        }
    }

    TEST_END_GROUP();
}

static void
convert_image_yuv_to_rgb_size_format(DcpPixelFormat format,
                                     uint32_t width,
                                     uint32_t height,
                                     DcpColorSpace color_space)
{
    TEST_BEGIN_GROUP("cs=%d", color_space);

    if (format == DCP_PIXEL_FORMAT_NV12) {
        convert_image_yuv_to_rgb_size_mode(1, width, height, color_space, format);
        convert_image_yuv_to_rgb_size_mode(2, width, height, color_space, format);
    } else if (format == DCP_PIXEL_FORMAT_I420 || format == DCP_PIXEL_FORMAT_I444) {
        convert_image_yuv_to_rgb_size_mode(3, width, height, color_space, format);
    }

    TEST_END_GROUP();
}

static void
convert_image_yuv_to_rgb_size(DcpPixelFormat format,
                              uint32_t width,
                              uint32_t height)
{
    DcpColorSpace color_space;

    TEST_BEGIN_GROUP("%ux%u", width, height);

    for (color_space = DCP_COLOR_SPACE_BT601; color_space <= DCP_COLOR_SPACE_BT709; color_space++) {
        convert_image_yuv_to_rgb_size_format(format, width, height, color_space);
    }

    TEST_END_GROUP();
}

static void
unit_convert_image_yuv_to_rgb(DcpPixelFormat format)
{
    /*
     * Those are the tested sizes:
     * 2x2 up to 64x64
     */
    const uint32_t max_width = 64;
    const uint32_t max_height = 64;
    uint32_t width;

    TEST_BEGIN_GROUP(__FUNCTION__);
    init();

    for (width = 2; width <= max_width; width += 2) {
        uint32_t height;

        for (height = 2; height <= max_height; height += 2) {
            convert_image_yuv_to_rgb_size(format, width, height);
        }
    }

    TEST_END_GROUP();
}

static void
unit_convert_image_rgb_to_yuv_errors(void)
{
    const uint32_t width = 2;
    const uint32_t height = 2;
    const uint32_t chroma_height = (height / 2);
    const size_t src_stride = (size_t)width * 4;
    const size_t in_size = src_stride * height;
    const size_t out_size = (size_t)width * ((size_t)height + chroma_height);
    uint8_t *test_input;
    uint8_t *test_output;

    TEST_BEGIN_GROUP(__FUNCTION__);
    init();

    /* Allocate and initialize input */
    test_input = (uint8_t *)stack_alloc(in_size);

    /* Allocate and initialize output */
    test_output = (uint8_t *)stack_alloc(out_size);

    uint32_t num_planes;
    for (num_planes = 0; num_planes <= 3; num_planes++) { /* Only 1 and 2 are valid values */
        int32_t src_pixel_format;

        for (src_pixel_format = 0; src_pixel_format <= DCP_PIXEL_FORMAT_NV12 + 1; src_pixel_format++) {
            int32_t src_color_space;

            for (src_color_space = 0; src_color_space <= DCP_COLOR_SPACE_BT709 + 1; src_color_space++) {
                int32_t dst_color_space;

                for (dst_color_space = 0; dst_color_space <= DCP_COLOR_SPACE_BT709 + 1; dst_color_space++) {
                    int32_t corrupt;

                    for (corrupt = 0; corrupt < 4; corrupt++) {
                        uint8_t *src_buffer;
                        size_t dst_strides[2];
                        uint8_t *dst_buffers[2];

                        DcpImageFormat src_format = {
                            src_pixel_format,
                            src_color_space,
                            1
                        };

                        DcpImageFormat dst_format = {
                            DCP_PIXEL_FORMAT_NV12,
                            dst_color_space,
                            num_planes
                        };

                        DcpStatus status = dcp_status();
                        DcpStatus expected = dcp_status();

                        TEST_BEGIN("num_planes=%d,src_pf=%d,src_cs=%d,dst_cs=%d,corrupt=%d", num_planes, src_pixel_format, src_color_space, dst_color_space, corrupt);

                        /* Compute strides */
                        src_buffer = (corrupt & 2) ? NULL : test_input;
                        if (num_planes == 1) {
                            dst_strides[0] = width;
                            dst_buffers[0] = (corrupt & 1) ? NULL : test_output;
                        } else { /* 2 or error value, do not care */
                            dst_strides[0] = width;
                            dst_strides[1] = width;
                            dst_buffers[0] = test_output;
                            dst_buffers[1] = (corrupt & 1) ? NULL : &test_output[width * height];
                        }

                        /* Test image convert */
                        status.result = dcp_convert_image(width, height,
                                                   NULL, &src_stride, &src_buffer,
                                                   &dst_format, dst_strides, dst_buffers, &status.error);

                        TEST_ASSERT(DCP_RESULT_ERR, DCP_ERROR_KIND_INVALID_VALUE);

                        status.result = dcp_convert_image(width, height,
                                                   &src_format, &src_stride, &src_buffer,
                                                   NULL, dst_strides, dst_buffers, &status.error);

                        TEST_ASSERT(DCP_RESULT_ERR, DCP_ERROR_KIND_INVALID_VALUE);

                        status.result = dcp_convert_image(width, height,
                                                   NULL, &src_stride, &src_buffer,
                                                   NULL, dst_strides, dst_buffers, &status.error);

                        TEST_ASSERT(DCP_RESULT_ERR, DCP_ERROR_KIND_INVALID_VALUE);

                        expected = dcp_status();

                        SET_EXPECTED(src_pixel_format > DCP_PIXEL_FORMAT_NV12, DCP_ERROR_KIND_INVALID_VALUE);
                        SET_EXPECTED(src_color_space > DCP_COLOR_SPACE_BT709, DCP_ERROR_KIND_INVALID_VALUE);
                        SET_EXPECTED(dst_color_space > DCP_COLOR_SPACE_BT709, DCP_ERROR_KIND_INVALID_VALUE);

                        SET_EXPECTED((width & 1) != 0, DCP_ERROR_KIND_INVALID_VALUE);
                        SET_EXPECTED((height & 1) != 0, DCP_ERROR_KIND_INVALID_VALUE);

                        SET_EXPECTED((src_pixel_format >= DCP_PIXEL_FORMAT_I444) && (src_color_space <= DCP_COLOR_SPACE_LRGB), DCP_ERROR_KIND_INVALID_VALUE);
                        SET_EXPECTED((src_pixel_format < DCP_PIXEL_FORMAT_I444) && (src_color_space > DCP_COLOR_SPACE_LRGB), DCP_ERROR_KIND_INVALID_VALUE);

                        SET_EXPECTED(dst_color_space <= DCP_COLOR_SPACE_LRGB, DCP_ERROR_KIND_INVALID_VALUE);

                        SET_EXPECTED(num_planes < 1, DCP_ERROR_KIND_INVALID_VALUE);
                        SET_EXPECTED(num_planes > 2, DCP_ERROR_KIND_INVALID_VALUE);

                        SET_EXPECTED(corrupt != 0, DCP_ERROR_KIND_INVALID_VALUE);

                        SET_EXPECTED((src_pixel_format != DCP_PIXEL_FORMAT_ARGB &&
                                      src_pixel_format != DCP_PIXEL_FORMAT_BGRA &&
                                      src_pixel_format != DCP_PIXEL_FORMAT_BGR), DCP_ERROR_KIND_INVALID_OPERATION);

                        SET_EXPECTED(src_color_space != DCP_COLOR_SPACE_LRGB, DCP_ERROR_KIND_INVALID_OPERATION);

                        status.result = dcp_convert_image(width, height,
                                                   &src_format, &src_stride, &src_buffer,
                                                   &dst_format, dst_strides, dst_buffers, &status.error);

                        TEST_ASSERT(expected.result, expected.error);

                        TEST_END();
                    }
                }
            }
        }
    }

    stack_reset();

    TEST_END_GROUP();
}

static void
unit_convert_image_yuv_to_rgb_errors(void)
{
    const uint32_t width = 2;
    const uint32_t height = 2;
    const uint32_t chroma_height = (height / 2);
    const size_t dst_stride = (size_t)width * 4;
    const size_t in_size = (size_t)width * ((size_t)height + chroma_height);
    const size_t out_size = dst_stride * height;
    uint8_t *test_input;
    uint8_t *test_output;

    TEST_BEGIN_GROUP(__FUNCTION__);
    init();

    /* Allocate and initialize input */
    test_input = (uint8_t *)stack_alloc(in_size);

    /* Allocate and initialize output */
    test_output = (uint8_t *)stack_alloc(out_size);

    uint32_t num_planes;
    for (num_planes = 0; num_planes <= 3; num_planes++) { /* Only 1 and 2 are valid values */
        int32_t dst_pixel_format;

        for (dst_pixel_format = 0; dst_pixel_format <= DCP_PIXEL_FORMAT_NV12 + 1; dst_pixel_format++) {
            int32_t dst_color_space;

            for (dst_color_space = 0; dst_color_space <= DCP_COLOR_SPACE_BT709 + 1; dst_color_space++) {
                int32_t src_color_space;

                for (src_color_space = 0; src_color_space <= DCP_COLOR_SPACE_BT709 + 1; src_color_space++) {
                    int32_t corrupt;

                    for (corrupt = 0; corrupt < 4; corrupt++) {
                        size_t src_strides[2];
                        uint8_t *src_buffers[2];
                        uint8_t *dst_buffer;

                        DcpImageFormat src_format = {
                            DCP_PIXEL_FORMAT_NV12,
                            src_color_space,
                            num_planes
                        };

                        DcpImageFormat dst_format = {
                            dst_pixel_format,
                            dst_color_space,
                            1
                        };

                        DcpStatus status;
                        DcpStatus expected;

                        TEST_BEGIN("num_planes=%d,src_cs=%d,dst_pf=%d,dst_cs=%d,corrupt=%d", num_planes, src_color_space, dst_pixel_format, dst_color_space, corrupt);

                        /* Compute strides */
                        dst_buffer = (corrupt & 2) ? NULL : test_output;
                        if (num_planes == 1) {
                            src_strides[0] = width;
                            src_buffers[0] = (corrupt & 1) ? NULL : test_input;
                        } else { /* 2 or error value, do not care */
                            src_strides[0] = width;
                            src_strides[1] = width;
                            src_buffers[0] = test_input;
                            src_buffers[1] = (corrupt & 1) ? NULL : &test_input[width * height];
                        }

                        /* Test image convert */
                        status.result = dcp_convert_image(width, height,
                                                   NULL, src_strides, src_buffers,
                                                   &dst_format, &dst_stride, &dst_buffer, &status.error);

                        TEST_ASSERT(DCP_RESULT_ERR, DCP_ERROR_KIND_INVALID_VALUE);

                        status.result = dcp_convert_image(width, height,
                                                   &src_format, src_strides, src_buffers,
                                                   NULL, &dst_stride, &dst_buffer, &status.error);

                        TEST_ASSERT(DCP_RESULT_ERR, DCP_ERROR_KIND_INVALID_VALUE);

                        status.result = dcp_convert_image(width, height,
                                                   NULL, src_strides, src_buffers,
                                                   NULL, &dst_stride, &dst_buffer, &status.error);

                        TEST_ASSERT(DCP_RESULT_ERR, DCP_ERROR_KIND_INVALID_VALUE);

                        expected = dcp_status();

                        SET_EXPECTED(src_color_space > DCP_COLOR_SPACE_BT709, DCP_ERROR_KIND_INVALID_VALUE);
                        SET_EXPECTED(dst_pixel_format > DCP_PIXEL_FORMAT_NV12, DCP_ERROR_KIND_INVALID_VALUE);
                        SET_EXPECTED(dst_color_space > DCP_COLOR_SPACE_BT709, DCP_ERROR_KIND_INVALID_VALUE);

                        SET_EXPECTED((width & 1) != 0, DCP_ERROR_KIND_INVALID_VALUE);
                        SET_EXPECTED((height & 1) != 0, DCP_ERROR_KIND_INVALID_VALUE);

                        SET_EXPECTED(src_color_space <= DCP_COLOR_SPACE_LRGB, DCP_ERROR_KIND_INVALID_VALUE);

                        SET_EXPECTED((dst_pixel_format >= DCP_PIXEL_FORMAT_I444) && (dst_color_space <= DCP_COLOR_SPACE_LRGB), DCP_ERROR_KIND_INVALID_VALUE);
                        SET_EXPECTED((dst_pixel_format < DCP_PIXEL_FORMAT_I444) && (dst_color_space > DCP_COLOR_SPACE_LRGB), DCP_ERROR_KIND_INVALID_VALUE);

                        SET_EXPECTED(num_planes < 1, DCP_ERROR_KIND_INVALID_VALUE);
                        SET_EXPECTED(num_planes > 2, DCP_ERROR_KIND_INVALID_VALUE);

                        SET_EXPECTED(corrupt != 0, DCP_ERROR_KIND_INVALID_VALUE);

                        SET_EXPECTED(dst_pixel_format != DCP_PIXEL_FORMAT_BGRA, DCP_ERROR_KIND_INVALID_OPERATION);
                        SET_EXPECTED(dst_color_space != DCP_COLOR_SPACE_LRGB, DCP_ERROR_KIND_INVALID_OPERATION);

                        status.result = dcp_convert_image(width, height,
                                                   &src_format, src_strides, src_buffers,
                                                   &dst_format, &dst_stride, &dst_buffer, &status.error);

                        TEST_ASSERT(expected.result, expected.error);

                        TEST_END();
                    }
                }
            }
        }
    }

    stack_reset();

    TEST_END_GROUP();
}

static void
unit_get_buffers_size(void)
{
    static const uint32_t valid_width = 4098;
    static const uint32_t valid_height = 258;

    DcpStatus status = dcp_status();
    int32_t num_planes;

    TEST_BEGIN_GROUP(__FUNCTION__);
    init();

    TEST_BEGIN("no_format");
    status.result = dcp_get_buffers_size(1, 1, NULL, NULL, NULL, &status.error);
    TEST_ASSERT(DCP_RESULT_ERR, DCP_ERROR_KIND_INVALID_VALUE);
    TEST_END();

    TEST_BEGIN("no_buffer");
    DcpImageFormat format = {
        DCP_PIXEL_FORMAT_ARGB,
        0xDEADBEEF,
        1
    };

    status.result = dcp_get_buffers_size(1, 1, &format, NULL, NULL, &status.error);
    TEST_ASSERT(DCP_RESULT_ERR, DCP_ERROR_KIND_INVALID_VALUE);
    TEST_END();

    for (num_planes = -1; num_planes <= MAX_NUMBER_OF_PLANES + 1; num_planes++) {
        int32_t pf;

        TEST_BEGIN_GROUP("num_planes=%d", num_planes);

        for (pf = DCP_PIXEL_FORMAT_ARGB - 1; pf <= DCP_PIXEL_FORMAT_NV12 + 1; pf++) {
            size_t buffers_size[MAX_NUMBER_OF_PLANES];
            uint32_t max_number_of_planes;
            DcpStatus expected;
            DcpStatus status;
            uint8_t is_pf_valid = (pf >= DCP_PIXEL_FORMAT_ARGB && pf <= DCP_PIXEL_FORMAT_NV12);

            DcpImageFormat format = {
                pf,
                0xDEADBEEF,
                num_planes
            };

            TEST_BEGIN("pixel_format=%d", pf);

            /* Compute valid number of planes. */
            if (is_pf_valid) {
                for (max_number_of_planes = 0; max_number_of_planes < MAX_NUMBER_OF_PLANES; max_number_of_planes++) {
                    if (num_log2_den_per_plane[pf][2 * max_number_of_planes] == 0) {
                        break;
                    }
                }
            }

            /* Invalid width */
            expected = dcp_status();
            SET_EXPECTED(!is_pf_valid, DCP_ERROR_KIND_INVALID_VALUE);
            SET_EXPECTED(pf >= DCP_PIXEL_FORMAT_I422, DCP_ERROR_KIND_INVALID_VALUE);
            SET_EXPECTED(num_planes != 1 && num_planes != max_number_of_planes, DCP_ERROR_KIND_INVALID_VALUE);
            status.result = dcp_get_buffers_size(1, valid_height, &format, NULL, buffers_size, &status.error);
            TEST_ASSERT(expected.result, expected.error);

            /* Invalid height */
            expected = dcp_status();
            SET_EXPECTED(!is_pf_valid, DCP_ERROR_KIND_INVALID_VALUE);
            SET_EXPECTED(pf >= DCP_PIXEL_FORMAT_I420, DCP_ERROR_KIND_INVALID_VALUE);
            SET_EXPECTED(num_planes != 1 && num_planes != max_number_of_planes, DCP_ERROR_KIND_INVALID_VALUE);
            status.result = dcp_get_buffers_size(valid_width, 1, &format, NULL, buffers_size, &status.error);
            TEST_ASSERT(expected.result, expected.error);

            /* Test size is valid */
            expected = dcp_status();
            SET_EXPECTED(!is_pf_valid, DCP_ERROR_KIND_INVALID_VALUE);
            SET_EXPECTED(num_planes != 1 && num_planes != max_number_of_planes, DCP_ERROR_KIND_INVALID_VALUE);
            status.result = dcp_get_buffers_size(valid_width, valid_height, &format, NULL, buffers_size, &status.error);
            TEST_ASSERT(expected.result, expected.error);
            if (expected.result == DCP_RESULT_OK) {
                if (num_planes == 1) {
                    TEST_ASSERT_EQ(buffers_size[0], (((size_t)valid_width * (size_t)valid_height * num_log2_den[pf][0]) >> num_log2_den[pf][1]));
                } else {
                    uint32_t i;

                    for (i = 0; i < num_planes; i++) {
                        TEST_ASSERT_EQ(buffers_size[i], (((size_t)valid_width * (size_t)valid_height * num_log2_den_per_plane[pf][2 * i]) >> num_log2_den_per_plane[pf][2 * i + 1]));
                    }
                }
            }

            TEST_END();
        }

        TEST_END_GROUP();
    }

    TEST_END_GROUP();
}


static void
unit_convert_image_over_4gb_limit(void)
{
    const uint32_t width = 0x40000000;
    const uint32_t height = 2;

    DcpStatus status = dcp_status();
    size_t src_buffer_size;
    const size_t expected_src_buffer_size = (size_t)width * height * 4;

    size_t dst_buffer_size;
    const size_t expected_dst_buffer_size = ((size_t)width * height * 3) / 2;

    uint8_t *src_image;
    uint8_t *dst_image;
    uint8_t *output;

    size_t luma_size = (size_t)width * height;
    size_t chroma_size = luma_size / 2;
    size_t i;

    DcpImageFormat src_format = {
        DCP_PIXEL_FORMAT_ARGB,
        DCP_COLOR_SPACE_LRGB,
        1
    };

    DcpImageFormat dst_format = {
        DCP_PIXEL_FORMAT_NV12,
        DCP_COLOR_SPACE_BT601,
        1
    };

    TEST_BEGIN_GROUP(__FUNCTION__);
    init();

    if (sizeof(void *) <= 4) {
        exit(EXIT_SKIP_CODE);
    }

    status.result = dcp_get_buffers_size(width, height, &src_format, NULL, &src_buffer_size, &status.error);
    TEST_ASSERT(DCP_RESULT_OK, -1);
    TEST_ASSERT_EQ(src_buffer_size, expected_src_buffer_size);

    status.result = dcp_get_buffers_size(width, height, &dst_format, NULL, &dst_buffer_size, &status.error);
    TEST_ASSERT(DCP_RESULT_OK, -1);
    TEST_ASSERT_EQ(dst_buffer_size, expected_dst_buffer_size);

    src_image = (uint8_t *)calloc(src_buffer_size, 1);
    if (src_image == NULL) {
        exit(EXIT_SKIP_CODE);
    }

    dst_image = (uint8_t *)calloc(dst_buffer_size, 1);
    if (dst_image == NULL) {
        exit(EXIT_SKIP_CODE);
    }

    status.result = dcp_convert_image(width, height,
                               &src_format, NULL, &src_image,
                               &dst_format, NULL, &dst_image, &status.error);

    TEST_ASSERT(DCP_RESULT_OK, -1);

    /* Check all luma samples are correct */
    output = dst_image;
    for (i = 0; i < luma_size; i++, output++) {
        TEST_ASSERT_EQ(*output, 16);
    }

    /* Check all chroma samples are correct */
    for (i = 0; i < chroma_size; i++, output++) {
        TEST_ASSERT_EQ(*output, 128);
    }

    TEST_END_GROUP();
}

static void
unit_image_convert_rgb_to_bgra_ok(void)
{
    const uint32_t MAX_WIDTH = 32;
    const uint32_t MAX_HEIGHT = 32;
    const uint32_t MAX_FILL_BYTES = 7;
    const uint32_t auto_stride = DCP_STRIDE_AUTO;
    const uint8_t src_bpp = 3;
    const uint8_t dst_bpp = 4;

    uint8_t *src_buffers[1];
    uint8_t *dst_buffers[1];

    uint32_t width;
    uint32_t height;
    uint32_t src_stride_bytes;
    uint32_t dst_stride_bytes;
    uint32_t src_stride_diff;
    uint32_t dst_stride_diff;
    size_t src_stride_param;
    size_t dst_stride_param;

    DcpImageFormat src_format = {
        DCP_PIXEL_FORMAT_RGB,
        DCP_COLOR_SPACE_LRGB,
        1
    };

    DcpImageFormat dst_format = {
        DCP_PIXEL_FORMAT_BGRA,
        DCP_COLOR_SPACE_LRGB,
        1
    };

    DcpStatus status = dcp_status();
    TEST_BEGIN_GROUP(__FUNCTION__);
    init();
    for (width = 0; width < MAX_WIDTH; width++) {
        for (height = 0; height < MAX_HEIGHT; height++) {
            for (src_stride_diff = 0; src_stride_diff < MAX_FILL_BYTES; src_stride_diff++) {
                for (dst_stride_diff = 0; dst_stride_diff < MAX_FILL_BYTES; dst_stride_diff++) {
                    uint32_t h;
                    uint32_t w;
                    TEST_BEGIN("Width=%d, Height=%d, Strides: src=%d, dst=%d", width, height, src_stride_diff, dst_stride_diff);

                    src_stride_bytes = (src_bpp * width) + src_stride_diff;
                    dst_stride_bytes = (dst_bpp * width) + dst_stride_diff;
                    src_stride_param = src_stride_diff == 0 ? auto_stride : src_stride_bytes;
                    dst_stride_param = dst_stride_diff == 0 ? auto_stride : dst_stride_bytes;

                    src_buffers[0] = stack_alloc(src_stride_bytes * height);
                    dst_buffers[0] = stack_alloc(dst_stride_bytes * height);

                    for (h = 0; h < height; h++) {
                        for (w = 0; w < width; w++) {
                            src_buffers[0][(h * src_stride_bytes) + (w * src_bpp) + 0] = rand() & 0xFF;
                            src_buffers[0][(h * src_stride_bytes) + (w * src_bpp) + 1] = rand() & 0xFF;
                            src_buffers[0][(h * src_stride_bytes) + (w * src_bpp) + 2] = rand() & 0xFF;
                        }
                    }

                    status.result = dcp_convert_image(width, height, &src_format, &src_stride_param, src_buffers, &dst_format, &dst_stride_param, dst_buffers, &status.error);
                    TEST_ASSERT(DCP_RESULT_OK, -1);

                    for (h = 0; h < height; h++) {
                        for (w = 0; w < width; w++) {
                            TEST_ASSERT_EQ(dst_buffers[0][(h * dst_stride_bytes) + (w * dst_bpp) + 0], src_buffers[0][(h * src_stride_bytes) + (w * src_bpp) + 2]);
                            TEST_ASSERT_EQ(dst_buffers[0][(h * dst_stride_bytes) + (w * dst_bpp) + 1], src_buffers[0][(h * src_stride_bytes) + (w * src_bpp) + 1]);
                            TEST_ASSERT_EQ(dst_buffers[0][(h * dst_stride_bytes) + (w * dst_bpp) + 2], src_buffers[0][(h * src_stride_bytes) + (w * src_bpp) + 0]);
                            TEST_ASSERT_EQ(dst_buffers[0][(h * dst_stride_bytes) + (w * dst_bpp) + 3], 255);
                        }
                    }
                    stack_reset();

                    TEST_END();
                }
            }
        }
    }


    TEST_END_GROUP();
}

int
main(int   argc,
     char *argv[])
{
    char *test_name = argv[1];

    if (strcmp(test_name, "unit_init") == 0) {
        unit_init();
    } else if (strcmp(test_name, "unit_get_buffers_size") == 0) {
        unit_get_buffers_size();
    } else if (strcmp(test_name, "unit_convert_image_rgb_to_yuv_ok") == 0) {
        int32_t planes = *(argv[2]) - '0';
        unit_convert_image_rgb_to_yuv(planes);
    } else if (strcmp(test_name, "unit_convert_image_nv12_to_rgb_ok") == 0) {
        unit_convert_image_yuv_to_rgb(DCP_PIXEL_FORMAT_NV12);
    } else if (strcmp(test_name, "unit_convert_image_i420_to_rgb_ok") == 0) {
        unit_convert_image_yuv_to_rgb(DCP_PIXEL_FORMAT_I420);
    } else if (strcmp(test_name, "unit_convert_image_i444_to_rgb_ok") == 0) {
        unit_convert_image_yuv_to_rgb(DCP_PIXEL_FORMAT_I444);
    } else if (strcmp(test_name, "unit_convert_image_rgb_to_yuv_errors") == 0) {
        unit_convert_image_rgb_to_yuv_errors();
    } else if (strcmp(test_name, "unit_convert_image_yuv_to_rgb_errors") == 0) {
        unit_convert_image_yuv_to_rgb_errors();
    } else if (strcmp(test_name, "unit_convert_image_over_4gb_limit") == 0) {
        unit_convert_image_over_4gb_limit();
    } else if (strcmp(test_name, "unit_image_convert_rgb_to_bgra_ok") == 0) {
        unit_image_convert_rgb_to_bgra_ok();
    } else {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/* ex:set ts=4 et: */