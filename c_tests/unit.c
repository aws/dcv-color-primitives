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
#include <inttypes.h>
#include <dcv_color_primitives.h>

#define MAX_ALLOCATIONS 16

static char domain[256];
static uint8_t domain_size = 0;
static char line_buffer[1024];

typedef struct {
    DcpResult result;
    DcpErrorKind error;
} DcpStatus;

typedef struct _Allocator {
    uint8_t *data[MAX_ALLOCATIONS];
    uint32_t count;
} Allocator;

static DcpStatus dcp_status(void)
{
    DcpStatus result = {
       DCP_RESULT_OK,
       (DcpErrorKind) -1
    };

    return result;
}

static uint8_t *
alloc_new(Allocator *alloc,
          size_t     size)
{
    uint8_t *data;

    if (alloc == NULL) {
        return NULL;
    }

    if (alloc->count == MAX_ALLOCATIONS) {
        return NULL;
    }

    data = malloc(size);
    if (data == NULL) {
        return NULL;
    }

    memset(data, 0, size);
    alloc->data[alloc->count++] = data;

    return data;
}

static void
alloc_free(Allocator *alloc)
{
    uint32_t i;

    if (alloc == NULL) {
        return;
    }

    for (i = 0; i < alloc->count; i++) {
        free(alloc->data[i]);
        alloc->data[i] = NULL;
    }

    alloc->count = 0;
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
        alloc_free(&alloc);                                                                      \
        exit(EXIT_FAILURE);                                                                      \
    } else if (status.result == DCP_RESULT_ERR && (int32_t)status.error != (int32_t)e) {         \
        fprintf(stderr, "%s: FAIL; Unexpected error kind: was 0x%X, expected 0x%X at line %d\n", \
            line_buffer, status.error, e, __LINE__);                                             \
        alloc_free(&alloc);                                                                      \
        exit(EXIT_FAILURE);                                                                      \
    }

#define TEST_ASSERT_EQ(val, x)                                                                   \
    if (val != x) {                                                                              \
        fprintf(stderr, "%s: FAIL; Unexpected status: was %"PRId64", expected %"PRId64" at line %d\n", \
            line_buffer, (int64_t)val, (int64_t)x, __LINE__);                                    \
        alloc_free(&alloc);                                                                      \
        exit(EXIT_FAILURE);                                                                      \
    }

#define TEST_ASSERT_EQ_T(val, x, t)                                                              \
    if (abs((int32_t)val - (int32_t)x) > t) {                                                    \
        fprintf(stderr, "%s: FAIL; Unexpected status: was %"PRId64", expected %"PRId64" at line %d\n", \
            line_buffer, (int64_t)val, (int64_t)x, __LINE__);                                    \
        alloc_free(&alloc);                                                                      \
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

typedef uint8_t uint8_mat8_t[8][8];
typedef uint8_t uint8_mat4_t[4][4];

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

static const uint8_mat8_t rgb_to_yuv_y_bt601_output = {
    { 74, 78, 101, 133, 118, 135, 87, 206 },
    { 93, 171, 116, 94, 102, 100, 171, 122 },
    { 141, 74, 200, 214, 98, 132, 65, 147 },
    { 141, 54, 186, 175, 154, 185, 81, 93 },
    { 118, 92, 125, 151, 40, 164, 181, 176 },
    { 139, 93, 110, 112, 132, 114, 157, 64 },
    { 113, 188, 101, 88, 105, 93, 128, 153 },
    { 131, 65, 48, 76, 129, 162, 117, 141 },
};

static const uint8_mat8_t rgb_to_yuv_cb_bt601_output = {
    { 116, 118, 204, 91, 136, 159, 97, 100 },
    { 99, 51, 179, 130, 161, 99, 117, 75 },
    { 63, 120, 143, 89, 147, 155, 184, 64 },
    { 118, 174, 134, 154, 133, 104, 185, 178 },
    { 192, 101, 191, 102, 192, 131, 81, 43 },
    { 60, 87, 198, 81, 175, 86, 61, 108 },
    { 95, 59, 122, 204, 83, 196, 124, 150 },
    { 184, 203, 185, 98, 140, 47, 196, 92 },
};

static const uint8_mat8_t rgb_to_yuv_cr_bt601_output = {
    { 187, 105, 177, 198, 208, 103, 119, 135 },
    { 229, 162, 147, 87, 79, 221, 118, 111 },
    { 187, 86, 115, 119, 141, 163, 166, 169 },
    { 164, 126, 89, 171, 84, 83, 81, 91 },
    { 183, 103, 75, 82, 121, 178, 97, 165 },
    { 83, 210, 212, 158, 183, 60, 88, 118 },
    { 135, 155, 93, 182, 70, 132, 176, 107 },
    { 157, 133, 120, 122, 186, 153, 65, 155 },
};

static const uint8_mat4_t rgb_to_yuv_cb2_bt601_output = {
    { 96, 151, 139, 97 },
    { 119, 130, 135, 153 },
    { 110, 143, 146, 73 },
    { 135, 152, 116, 140 },
};

static const uint8_mat4_t rgb_to_yuv_cr2_bt601_output = {
    { 171, 152, 153, 121 },
    { 141, 124, 118, 127 },
    { 145, 132, 135, 117 },
    { 145, 129, 135, 126 },
};

static const uint8_mat8_t rgb_to_yuv_y_bt709_output = {
    { 63, 84, 82, 123, 101, 137, 93, 208 },
    { 75, 173, 106, 103, 108, 84, 174, 132 },
    { 136, 84, 201, 221, 93, 121, 51, 146 },
    { 134, 49, 193, 163, 163, 197, 84, 95 },
    { 99, 101, 129, 164, 34, 154, 193, 179 },
    { 157, 80, 85, 111, 115, 133, 173, 68 },
    { 116, 191, 109, 68, 122, 84, 118, 155 },
    { 118, 56, 43, 80, 116, 166, 122, 139 },
};

static const uint8_mat8_t rgb_to_yuv_cb_bt709_output = {
    { 123, 115, 211, 98, 145, 156, 95, 100 },
    { 110, 53, 182, 126, 156, 109, 116, 72 },
    { 68, 115, 141, 87, 149, 160, 189, 67 },
    { 122, 174, 130, 160, 128, 98, 181, 174 },
    { 200, 97, 186, 96, 192, 136, 77, 45 },
    { 53, 95, 209, 84, 182, 78, 56, 107 },
    { 95, 60, 118, 212, 76, 197, 129, 148 },
    { 189, 205, 185, 97, 147, 48, 190, 94 },
};

static const uint8_mat8_t rgb_to_yuv_cr_bt709_output = {
    { 187, 103, 184, 197, 211, 104, 116, 133 },
    { 229, 157, 151, 86, 80, 221, 117, 106 },
    { 184, 84, 116, 115, 143, 166, 171, 165 },
    { 164, 130, 89, 174, 83, 80, 84, 94 },
    { 189, 100, 79, 78, 125, 180, 92, 160 },
    { 77, 209, 219, 155, 188, 56, 82, 117 },
    { 132, 151, 92, 189, 66, 137, 176, 108 },
    { 162, 139, 124, 120, 189, 148, 69, 153 },
};

static const uint8_mat4_t rgb_to_yuv_cb2_bt709_output = {
    { 100, 154, 142, 96 },
    { 120, 130, 134, 153 },
    { 112, 144, 147, 71 },
    { 137, 153, 117, 140 },
};

static const uint8_mat4_t rgb_to_yuv_cr2_bt709_output = {
    { 169, 154, 154, 118 },
    { 140, 124, 118, 129 },
    { 144, 133, 137, 113 },
    { 146, 131, 135, 127 },
};

static const uint8_mat8_t rgb_to_yuv_y_bt601fr_output = {
    { 67, 72, 99, 136, 119, 138, 83, 222 },
    { 90, 181, 116, 91, 100, 98, 180, 123 },
    { 146, 68, 214, 231, 95, 135, 57, 153 },
    { 145, 45, 198, 185, 161, 196, 75, 90 },
    { 119, 89, 127, 157, 28, 173, 192, 187 },
    { 144, 89, 110, 112, 135, 114, 165, 55 },
    { 113, 201, 99, 84, 104, 90, 130, 159 },
    { 134, 57, 37, 69, 132, 170, 118, 145 },
};

static const uint8_mat8_t rgb_to_yuv_cb_bt601fr_output = {
    { 115, 116, 214, 86, 137, 163, 92, 96 },
    { 95, 40, 186, 131, 166, 95, 116, 68 },
    { 54, 119, 145, 84, 149, 159, 192, 55 },
    { 117, 180, 135, 158, 133, 101, 193, 185 },
    { 201, 97, 199, 99, 201, 131, 75, 31 },
    { 50, 81, 208, 75, 182, 81, 52, 106 },
    { 90, 49, 121, 215, 77, 205, 123, 153 },
    { 192, 213, 192, 94, 142, 36, 206, 87 },
};

static const uint8_mat8_t rgb_to_yuv_cr_bt601fr_output = {
    { 195, 102, 183, 208, 219, 99, 117, 136 },
    { 243, 167, 149, 81, 72, 234, 116, 108 },
    { 195, 80, 113, 117, 143, 167, 172, 174 },
    { 169, 126, 84, 177, 78, 77, 75, 86 },
    { 191, 99, 68, 75, 119, 185, 92, 170 },
    { 77, 221, 224, 162, 190, 51, 83, 117 },
    { 136, 159, 88, 189, 63, 133, 182, 104 },
    { 161, 134, 119, 121, 194, 157, 56, 159 },
};

static const uint8_mat4_t rgb_to_yuv_cb2_bt601fr_output = {
    { 92, 154, 140, 93 },
    { 118, 130, 136, 156 },
    { 107, 145, 149, 66 },
    { 136, 156, 115, 142 },
};

static const uint8_mat4_t rgb_to_yuv_cr2_bt601fr_output = {
    { 177, 155, 156, 120 },
    { 142, 123, 116, 127 },
    { 147, 132, 136, 116 },
    { 147, 129, 137, 125 },
};

static const uint8_mat8_t rgb_to_yuv_y_bt709fr_output = {
    { 55, 79, 77, 124, 99, 140, 90, 224 },
    { 69, 183, 105, 101, 108, 80, 184, 135 },
    { 140, 79, 215, 239, 89, 123, 40, 152 },
    { 138, 39, 206, 171, 171, 210, 79, 92 },
    { 97, 98, 131, 172, 21, 160, 206, 189 },
    { 164, 75, 80, 111, 115, 136, 183, 60 },
    { 116, 204, 108, 60, 124, 80, 119, 162 },
    { 119, 46, 31, 75, 116, 175, 124, 143 },
};

static const uint8_mat8_t rgb_to_yuv_cb_bt709fr_output = {
    { 122, 113, 222, 94, 148, 160, 91, 96 },
    { 108, 43, 189, 125, 160, 107, 114, 65 },
    { 60, 114, 143, 82, 152, 164, 198, 59 },
    { 122, 181, 130, 164, 128, 94, 188, 181 },
    { 210, 93, 194, 92, 201, 138, 70, 34 },
    { 43, 91, 221, 77, 190, 71, 45, 104 },
    { 90, 51, 116, 224, 69, 207, 130, 150 },
    { 197, 215, 193, 92, 149, 37, 199, 90 },
};

static const uint8_mat8_t rgb_to_yuv_cr_bt709fr_output = {
    { 196, 100, 191, 206, 222, 101, 114, 134 },
    { 243, 161, 154, 80, 74, 234, 115, 103 },
    { 191, 78, 114, 114, 145, 171, 177, 170 },
    { 169, 130, 83, 181, 77, 74, 78, 89 },
    { 198, 96, 72, 72, 125, 187, 87, 164 },
    { 70, 220, 232, 159, 196, 46, 76, 115 },
    { 133, 154, 86, 198, 57, 138, 183, 105 },
    { 167, 140, 124, 119, 197, 151, 60, 157 },
};

static const uint8_mat4_t rgb_to_yuv_cb2_bt709fr_output = {
    { 97, 158, 144, 91 },
    { 119, 130, 134, 156 },
    { 109, 146, 150, 63 },
    { 138, 156, 116, 142 },
};

static const uint8_mat4_t rgb_to_yuv_cr2_bt709fr_output = {
    { 175, 158, 158, 117 },
    { 142, 123, 117, 129 },
    { 146, 134, 138, 111 },
    { 148, 132, 136, 126 },
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
 *
 * Color table (bt601 full range):
 *             r    g    b
 * black    (  0,   0,   0):     0, 128, 128
 * red      (255,   0,   0):    76,  84, 255
 * green    (  0, 255,   0):   149,  43,  21
 * yellow   (255, 255,   0):   225,   0, 148
 * blue     (  0,   0, 255):    29, 255, 107
 * magenta  (255,   0, 255):   105, 212, 234
 * cyan     (  0, 255, 255):   178, 171,   0
 * white    (255, 255, 255):   255, 128, 128
 *
 * Color table (bt709 full range):
 *             r    g    b
 * black    (  0,   0,   0):     0, 128, 128
 * red      (255,   0,   0):    54,  98, 255
 * green    (  0, 255,   0):   182,  29,  12
 * yellow   (255, 255,   0):   237,   0, 139
 * blue     (  0,   0, 255):    18, 255, 116
 * magenta  (255,   0, 255):    73, 226, 243
 * cyan     (  0, 255, 255):   201, 157,   0
 * white    (255, 255, 255):   255, 128, 128
 */
static const uint8_t y_to_rgb_input[4][8] = {
    {  16,  82, 145, 210,  41, 107, 169, 235 },
    {  16,  63, 173, 219,  32,  78, 188, 235 },
    {   0,  76, 149, 225,  29, 105, 178, 255 },
    {   0,  54, 182, 237,  18,  73, 201, 255 },
};

static const uint8_t cb_to_rgb_input[4][8] = {
    { 128,  90,  54,  16, 240, 202, 166, 128 },
    { 128, 102,  42,  16, 240, 214, 154, 128 },
    { 128,  84,  43,   0, 255, 212, 171, 128 },
    { 128,  98,  29,   0, 255, 226, 157, 128 },
};

static const uint8_t cr_to_rgb_input[4][8] = {
    { 128, 240,  34, 146, 110, 222,  16, 128 },
    { 128, 240,  26, 138, 118, 230,  16, 128 },
    { 128, 255,  21, 148, 107, 234,   0, 128 },
    { 128, 255,  12, 139, 116, 243,   0, 128 },
};

static int32_t
is_valid_format(const DcpImageFormat *format,
                uint32_t              width,
                uint32_t              height)
{
    if (format->pixel_format == DCP_PIXEL_FORMAT_I444) {
        return format->num_planes == 3;
    } else if (format->pixel_format == DCP_PIXEL_FORMAT_I422 || format->pixel_format == DCP_PIXEL_FORMAT_I420) {
        return format->num_planes == 3 && (width & 1) == 0 && (height & 1) == 0;
    } else if (format->pixel_format == DCP_PIXEL_FORMAT_NV12) {
        return (format->num_planes >= 1 && format->num_planes <= 2) && (width & 1) == 0 && (height & 1) == 0;
    } else {
        return format->num_planes == 1;
    }
}

static void
unit_init(void)
{
    const char *desc;
    Allocator alloc = { 0, };

    TEST_BEGIN_GROUP(__FUNCTION__);

    TEST_BEGIN("init");

    desc = dcp_describe_acceleration();
    TEST_ASSERT_EQ((desc != NULL), 1);

    printf("%s\n", desc);

    TEST_END();

    TEST_END_GROUP();
}

static void
convert_image_rgb_to_yuv_size_mode_stride(uint32_t       num_planes,
                                          uint32_t       width,
                                          uint32_t       height,
                                          DcpColorSpace  color_space,
                                          DcpPixelFormat src_pixel_format,
                                          DcpPixelFormat dst_pixel_format,
                                          size_t         src_fill_bytes,
                                          size_t         luma_fill_bytes,
                                          size_t         u_chroma_fill_bytes,
                                          size_t         v_chroma_fill_bytes)
{
    const size_t depth = src_pixel_format == DCP_PIXEL_FORMAT_BGR ? 3 : 4;
    size_t src_stride = (width * depth) + src_fill_bytes;
    const size_t in_size = src_stride * height;
    DcpStatus status = dcp_status();
    DcpImageFormat src_format = {
        src_pixel_format,
        DCP_COLOR_SPACE_RGB,
        1
    };
    DcpImageFormat dst_format = {
        dst_pixel_format,
        color_space,
        num_planes
    };

    size_t luma_stride;
    size_t u_chroma_stride;
    size_t v_chroma_stride;
    uint32_t chroma_height;
    size_t out_size;
    uint8_t *test_input;
    uint8_t *input;
    uint32_t y;
    uint8_t *test_output;
    const uint8_t *output;
    size_t dst_strides[3];
    uint8_t *dst_buffers[3];
    size_t count;
    Allocator alloc = { 0, };

    luma_stride = width + luma_fill_bytes;
    u_chroma_stride = (dst_pixel_format == DCP_PIXEL_FORMAT_I444 || dst_pixel_format == DCP_PIXEL_FORMAT_NV12) ?
                       width + u_chroma_fill_bytes: (width / 2) + u_chroma_fill_bytes;
    if (dst_pixel_format == DCP_PIXEL_FORMAT_I444) {
        v_chroma_stride = width + v_chroma_fill_bytes;
    } else if (dst_pixel_format == DCP_PIXEL_FORMAT_NV12) {
        v_chroma_stride = u_chroma_stride;
    } else {
        v_chroma_stride = (width / 2) + v_chroma_fill_bytes;
    }

    chroma_height = height / 2;

    if (dst_pixel_format == DCP_PIXEL_FORMAT_I444) {
        out_size = (luma_stride * height) + (u_chroma_stride * height) + (v_chroma_stride * height);
    } else if (dst_pixel_format == DCP_PIXEL_FORMAT_NV12) {
        out_size = (luma_stride * height) + (u_chroma_stride * chroma_height);
    } else {
        out_size = (luma_stride * height) +
                   (u_chroma_stride * chroma_height) +
                   (v_chroma_stride * chroma_height);
    }

    /* Allocate and initialize input */
    test_input = alloc_new(&alloc, in_size);
    input = test_input;

    for (y = 0; y < height; y++, input += src_fill_bytes) {
        uint32_t x;

        for (x = 0; x < width; x++) {
            if (src_pixel_format == DCP_PIXEL_FORMAT_ARGB) {
                *input++ = rgb_to_yuv_input[y][x].a;
                *input++ = rgb_to_yuv_input[y][x].r;
                *input++ = rgb_to_yuv_input[y][x].g;
                *input++ = rgb_to_yuv_input[y][x].b;
            } else if (src_pixel_format == DCP_PIXEL_FORMAT_BGRA) {
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

    test_output = alloc_new(&alloc, out_size);

    /* Compute strides and buffers */
    src_stride = src_fill_bytes == 0 ? DCP_STRIDE_AUTO : src_stride;
    dst_strides[0] = luma_fill_bytes == 0 ? DCP_STRIDE_AUTO : luma_stride;
    dst_buffers[0] = test_output;

    if (dst_pixel_format == DCP_PIXEL_FORMAT_I444) {
        dst_strides[1] = u_chroma_fill_bytes == 0 ? DCP_STRIDE_AUTO : u_chroma_stride;
        dst_strides[2] = v_chroma_fill_bytes == 0 ? DCP_STRIDE_AUTO : v_chroma_stride;
        dst_buffers[1] = test_output + (luma_stride * height);
        dst_buffers[2] = test_output + (luma_stride * height) + (u_chroma_stride * height);
    } else if (dst_pixel_format == DCP_PIXEL_FORMAT_NV12) {
        if (num_planes == 2) {
            dst_strides[1] = u_chroma_fill_bytes == 0 ? DCP_STRIDE_AUTO : u_chroma_stride;
            dst_buffers[1] = test_output + (luma_stride * height);
        }
    } else {
        dst_strides[1] = u_chroma_fill_bytes == 0 ? DCP_STRIDE_AUTO : u_chroma_stride;
        dst_strides[2] = v_chroma_fill_bytes == 0 ? DCP_STRIDE_AUTO : v_chroma_stride;
        dst_buffers[1] = test_output + (luma_stride * height);
        dst_buffers[2] = test_output + (luma_stride * height) + (u_chroma_stride * chroma_height);
    }

    /* Perform conversion */
    status.result = dcp_convert_image(width, height,
                                      &src_format, &src_stride, (const uint8_t * const *)&test_input,
                                      &dst_format, dst_strides, dst_buffers, &status.error);

    TEST_ASSERT(DCP_RESULT_OK, -1);

    /* Check output */
    count = 0;
    output = test_output;

    /* Check all luma samples are correct */
    const uint8_mat8_t *expected_y = (color_space == DCP_COLOR_SPACE_BT601 ? &rgb_to_yuv_y_bt601_output:
                                      color_space == DCP_COLOR_SPACE_BT709 ? &rgb_to_yuv_y_bt709_output:
                                      color_space == DCP_COLOR_SPACE_BT601FR ? &rgb_to_yuv_y_bt601fr_output:
                                                                               &rgb_to_yuv_y_bt709fr_output);

    for (y = 0; y < height; y++) {
        size_t x;

        for (x = 0; x < width; x++, count++, output++) {
            TEST_ASSERT_EQ(*output, (*expected_y)[y][x]);
        }

        for (x = 0; x < luma_fill_bytes; x++, count++, output++) {
            TEST_ASSERT_EQ(*output, 0);
        }
    }

    if (dst_pixel_format == DCP_PIXEL_FORMAT_I444) {
        const uint8_mat8_t *expected_cb = (color_space == DCP_COLOR_SPACE_BT601 ? &rgb_to_yuv_cb_bt601_output:
                                           color_space == DCP_COLOR_SPACE_BT709 ? &rgb_to_yuv_cb_bt709_output:
                                           color_space == DCP_COLOR_SPACE_BT601FR ? &rgb_to_yuv_cb_bt601fr_output:
                                                                                    &rgb_to_yuv_cb_bt709fr_output);
        const uint8_mat8_t *expected_cr = (color_space == DCP_COLOR_SPACE_BT601 ? &rgb_to_yuv_cr_bt601_output:
                                           color_space == DCP_COLOR_SPACE_BT709 ? &rgb_to_yuv_cr_bt709_output:
                                           color_space == DCP_COLOR_SPACE_BT601FR ? &rgb_to_yuv_cr_bt601fr_output:
                                                                                    &rgb_to_yuv_cr_bt709fr_output);

        for (y = 0; y < height; y++) {
            size_t x;

            for (x = 0; x < width; x++, count++, output++) {
                TEST_ASSERT_EQ(*output, (*expected_cb)[y][x]);
            }

            for (x = 0; x < u_chroma_fill_bytes; x++, count++, output++) {
                TEST_ASSERT_EQ(*output, 0);
            }
        }

        for (y = 0; y < height; y++) {
            size_t x;

            for (x = 0; x < width; x++, count++, output++) {
                TEST_ASSERT_EQ(*output, (*expected_cr)[y][x]);
            }

            for (x = 0; x < v_chroma_fill_bytes; x++, count++, output++) {
                TEST_ASSERT_EQ(*output, 0);
            }
        }
    } else if (dst_pixel_format == DCP_PIXEL_FORMAT_NV12) {
        const uint8_mat4_t *expected_cb = (color_space == DCP_COLOR_SPACE_BT601 ? &rgb_to_yuv_cb2_bt601_output:
                                           color_space == DCP_COLOR_SPACE_BT709 ? &rgb_to_yuv_cb2_bt709_output:
                                           color_space == DCP_COLOR_SPACE_BT601FR ? &rgb_to_yuv_cb2_bt601fr_output:
                                                                                    &rgb_to_yuv_cb2_bt709fr_output);
        const uint8_mat4_t *expected_cr = (color_space == DCP_COLOR_SPACE_BT601 ? &rgb_to_yuv_cr2_bt601_output:
                                           color_space == DCP_COLOR_SPACE_BT709 ? &rgb_to_yuv_cr2_bt709_output:
                                           color_space == DCP_COLOR_SPACE_BT601FR ? &rgb_to_yuv_cr2_bt601fr_output:
                                                                                    &rgb_to_yuv_cr2_bt709fr_output);

        for (y = 0; y < chroma_height; y++) {
            size_t x;

            for (x = 0; x < width / 2; x++, count += 2, output += 2) {
                TEST_ASSERT_EQ(*output, (*expected_cb)[y][x]);
                TEST_ASSERT_EQ(*(output + 1), (*expected_cr)[y][x]);
            }

            for (x = 0; x < u_chroma_fill_bytes; x++, count++, output++) {
                TEST_ASSERT_EQ(*output, 0);
            }
        }
    } else {
        const uint8_mat4_t *expected_cb = (color_space == DCP_COLOR_SPACE_BT601 ? &rgb_to_yuv_cb2_bt601_output:
                                           color_space == DCP_COLOR_SPACE_BT709 ? &rgb_to_yuv_cb2_bt709_output:
                                           color_space == DCP_COLOR_SPACE_BT601FR ? &rgb_to_yuv_cb2_bt601fr_output:
                                                                                    &rgb_to_yuv_cb2_bt709fr_output);
        const uint8_mat4_t *expected_cr = (color_space == DCP_COLOR_SPACE_BT601 ? &rgb_to_yuv_cr2_bt601_output:
                                           color_space == DCP_COLOR_SPACE_BT709 ? &rgb_to_yuv_cr2_bt709_output:
                                           color_space == DCP_COLOR_SPACE_BT601FR ? &rgb_to_yuv_cr2_bt601fr_output:
                                                                                    &rgb_to_yuv_cr2_bt709fr_output);

        for (y = 0; y < chroma_height; y++) {
            size_t x;

            for (x = 0; x < width / 2; x++, count++, output++) {
                TEST_ASSERT_EQ(*output, (*expected_cb)[y][x]);
            }

            for (x = 0; x < u_chroma_fill_bytes; x++, count++, output++) {
                TEST_ASSERT_EQ(*output, 0);
            }
        }

        for (y = 0; y < chroma_height; y++) {
            size_t x;

            for (x = 0; x < width / 2; x++, count++, output++) {
                TEST_ASSERT_EQ(*output, (*expected_cr)[y][x]);
            }

            for (x = 0; x < v_chroma_fill_bytes; x++, count++, output++) {
                TEST_ASSERT_EQ(*output, 0);
            }
        }
    }

    /* Rest must be identically null */
    for (; count < out_size; count++, output++) {
        TEST_ASSERT_EQ(*output, 0);
    }

    alloc_free(&alloc);
}

static void
convert_image_rgb_to_yuv_size_mode(uint32_t       num_planes,
                                   uint32_t       width,
                                   uint32_t       height,
                                   DcpColorSpace  color_space,
                                   DcpPixelFormat src_pixel_format,
                                   DcpPixelFormat dst_pixel_format)
{
    const size_t max_fill_bytes = 1;

    TEST_BEGIN_GROUP("cs=%d,spf=%d,dpf=%d", color_space, src_pixel_format, dst_pixel_format);

    if (num_planes == 1) {
        size_t luma_stride;

        for (luma_stride = 0; luma_stride <= max_fill_bytes; luma_stride++) {
            size_t src_stride;

            TEST_BEGIN("D=%zu", luma_stride);

            for (src_stride = 0; src_stride <= max_fill_bytes; src_stride++) {
                convert_image_rgb_to_yuv_size_mode_stride(num_planes, width, height, color_space, src_pixel_format, dst_pixel_format, src_stride, luma_stride, luma_stride, luma_stride);
            }

            TEST_END();
        }
    } else if (num_planes == 2) {
        size_t luma_stride;

        for (luma_stride = 0; luma_stride <= max_fill_bytes; luma_stride++) {
            size_t chroma_stride;

            for (chroma_stride = 0; chroma_stride <= max_fill_bytes; chroma_stride++) {
                size_t src_stride;

                TEST_BEGIN("Y=%zu,C=%zu", luma_stride, chroma_stride);

                for (src_stride = 0; src_stride <= max_fill_bytes; src_stride++) {
                    convert_image_rgb_to_yuv_size_mode_stride(num_planes, width, height, color_space, src_pixel_format, dst_pixel_format, src_stride, luma_stride, chroma_stride, chroma_stride);
                }

                TEST_END();
            }
        }
    } else { /* 3 planes */
        size_t luma_stride;

        for (luma_stride = 0; luma_stride <= max_fill_bytes; luma_stride++) {
            size_t u_stride;

            for (u_stride = 0; u_stride <= max_fill_bytes; u_stride++) {
                size_t v_stride;

                for (v_stride = 0; v_stride <= max_fill_bytes; v_stride++) {
                    size_t src_stride;

                    TEST_BEGIN("Y=%zu,U=%zu,V=%zu", luma_stride, u_stride, v_stride);

                    for (src_stride = 0; src_stride <= max_fill_bytes; src_stride++) {
                        convert_image_rgb_to_yuv_size_mode_stride(num_planes, width, height, color_space, src_pixel_format, dst_pixel_format, src_stride, luma_stride, u_stride, v_stride);
                    }

                    TEST_END();
                }
            }
        }
    }

    TEST_END_GROUP();
}

static void
convert_image_rgb_to_yuv_size(uint32_t       num_planes,
                              uint32_t       width,
                              uint32_t       height,
                              DcpPixelFormat dst_pixel_format)
{
    static const DcpPixelFormat supported_pixel_formats[] = {
        DCP_PIXEL_FORMAT_ARGB,
        DCP_PIXEL_FORMAT_BGRA,
        DCP_PIXEL_FORMAT_BGR
    };

    static const size_t supported_pixel_formats_count = sizeof(supported_pixel_formats) / sizeof(supported_pixel_formats[0]);

    DcpColorSpace color_space;

    TEST_BEGIN_GROUP("%ux%u", width, height);

    for (color_space = DCP_COLOR_SPACE_BT601; color_space <= DCP_COLOR_SPACE_BT709FR; color_space++) {
        size_t i;

        for (i = 0; i < supported_pixel_formats_count; i++) {
            convert_image_rgb_to_yuv_size_mode(num_planes, width, height, color_space, supported_pixel_formats[i], dst_pixel_format);
        }
    }

    TEST_END_GROUP();
}

static void
unit_convert_image_rgb_to_yuv(DcpPixelFormat dst_pixel_format,
                              uint32_t       num_planes)
{
    const uint32_t max_width = 8;
    const uint32_t max_height = 8;
    const uint32_t step = dst_pixel_format == DCP_PIXEL_FORMAT_I444 ? 1 : 2;
    uint32_t width;

    TEST_BEGIN_GROUP(__FUNCTION__);

    for (width = 0; width <= max_width; width += step) {
        uint32_t height;

        for (height = 0; height <= max_height; height += step) {
            convert_image_rgb_to_yuv_size(num_planes, width, height, dst_pixel_format);
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
                                          DcpPixelFormat format,
                                          DcpPixelFormat dst_pixel_format)
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
    uint32_t red_offset;
    uint32_t blue_offset;
    Allocator alloc = { 0, };

    DcpImageFormat src_format = {
        format,
        color_space,
        num_planes
    };

    DcpImageFormat dst_format = {
        dst_pixel_format,
        DCP_COLOR_SPACE_RGB,
        1
    };

    DcpStatus status = dcp_status();

    /* Allocate and initialize input */
    test_input = alloc_new(&alloc, in_size);
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
        alloc_free(&alloc);
        exit(EXIT_FAILURE);
    }

    /* Allocate and initialize output */
    test_output = alloc_new(&alloc, out_size);

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
        alloc_free(&alloc);
        exit(EXIT_FAILURE);
    }

    /* Perform conversion */
    status.result = dcp_convert_image(width, height,
                                      &src_format, src_strides, (const uint8_t * const *)src_buffers,
                                      &dst_format, &dst_stride, &test_output, &status.error);

    TEST_ASSERT(DCP_RESULT_OK, -1);

    /* Check output */
    count = 0;
    output = test_output;

    /* Check all samples are correct */
    red_offset = dst_pixel_format == DCP_PIXEL_FORMAT_BGRA ? 0 : 2;
    blue_offset = dst_pixel_format == DCP_PIXEL_FORMAT_BGRA ? 2 : 0;
    for (y = 0; y < height; y++) {
        size_t x;

        for (x = 0; x < width; x++, count += 4, output += 4) {
            uint32_t index = (x >> 1) & 0x7;
            int32_t exp_b = ((index >> blue_offset) & 1) == 0 ? 0: 255;
            int32_t exp_g = ((index >> 1) & 1) == 0 ? 0: 255;
            int32_t exp_r = ((index >> red_offset) & 1) == 0 ? 0: 255;

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

    alloc_free(&alloc);
}

static void
convert_image_yuv_to_rgb_size_mode(uint32_t       num_planes,
                                   uint32_t       width,
                                   uint32_t       height,
                                   DcpColorSpace  color_space,
                                   DcpPixelFormat format,
                                   DcpPixelFormat dst_pixel_format)
{
    const size_t max_fill_bytes = 4;

    TEST_BEGIN_GROUP("format=%d", format);

    if (num_planes == 1) {
        size_t luma_fill_bytes;

        for (luma_fill_bytes = 0; luma_fill_bytes <= max_fill_bytes; luma_fill_bytes++) {
            size_t dst_fill_bytes;

            TEST_BEGIN("S=%zu", luma_fill_bytes);

            for (dst_fill_bytes = 0; dst_fill_bytes <= max_fill_bytes; dst_fill_bytes++) {
                convert_image_yuv_to_rgb_size_mode_stride(num_planes,
                                                          width, height,
                                                          color_space,
                                                          luma_fill_bytes,
                                                          luma_fill_bytes,
                                                          luma_fill_bytes,
                                                          dst_fill_bytes,
                                                          format,
                                                          dst_pixel_format);
            }

            TEST_END();
        }
    } else if (num_planes == 2) {
        size_t luma_fill_bytes;

        for (luma_fill_bytes = 0; luma_fill_bytes <= max_fill_bytes; luma_fill_bytes++) {
            size_t chroma_fill_bytes;

            for (chroma_fill_bytes = 0; chroma_fill_bytes <= max_fill_bytes; chroma_fill_bytes++) {
                size_t dst_fill_bytes;

                TEST_BEGIN("Y=%zu,C=%zu", luma_fill_bytes, chroma_fill_bytes);

                for (dst_fill_bytes = 0; dst_fill_bytes <= max_fill_bytes; dst_fill_bytes++) {
                    convert_image_yuv_to_rgb_size_mode_stride(num_planes,
                                                              width,
                                                              height,
                                                              color_space,
                                                              luma_fill_bytes,
                                                              chroma_fill_bytes,
                                                              chroma_fill_bytes,
                                                              dst_fill_bytes,
                                                              format,
                                                              dst_pixel_format);
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

                    TEST_BEGIN("Y=%zu,U=%zu,V=%zu", luma_fill_bytes, u_chroma_fill_bytes, v_chroma_fill_bytes);

                    for (dst_fill_bytes = 0; dst_fill_bytes <= max_fill_bytes; dst_fill_bytes++) {
                        convert_image_yuv_to_rgb_size_mode_stride(num_planes,
                                                                  width,
                                                                  height,
                                                                  color_space,
                                                                  luma_fill_bytes,
                                                                  u_chroma_fill_bytes,
                                                                  v_chroma_fill_bytes,
                                                                  dst_fill_bytes,
                                                                  format,
                                                                  dst_pixel_format);
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
                                     uint32_t       width,
                                     uint32_t       height,
                                     DcpColorSpace  color_space,
                                     DcpPixelFormat dst_pixel_format)
{
    TEST_BEGIN_GROUP("cs=%d", color_space);

    if (format == DCP_PIXEL_FORMAT_NV12) {
        convert_image_yuv_to_rgb_size_mode(1, width, height, color_space, format, dst_pixel_format);
        convert_image_yuv_to_rgb_size_mode(2, width, height, color_space, format, dst_pixel_format);
    } else if (format == DCP_PIXEL_FORMAT_I420 || format == DCP_PIXEL_FORMAT_I444) {
        convert_image_yuv_to_rgb_size_mode(3, width, height, color_space, format, dst_pixel_format);
    }

    TEST_END_GROUP();
}

static void
convert_image_yuv_to_rgb_size(DcpPixelFormat format,
                              uint32_t       width,
                              uint32_t       height)
{
    static const DcpPixelFormat dst_pixel_formats[] = { DCP_PIXEL_FORMAT_BGRA, DCP_PIXEL_FORMAT_RGBA };
    static const size_t n_pixel_formats = sizeof(dst_pixel_formats) / sizeof(dst_pixel_formats[0]);
    size_t i;

    TEST_BEGIN_GROUP("%ux%u", width, height);

    for (i = 0; i < n_pixel_formats; i++) {
        DcpColorSpace color_space;

        for (color_space = DCP_COLOR_SPACE_BT601; color_space <= DCP_COLOR_SPACE_BT709FR; color_space++) {
            convert_image_yuv_to_rgb_size_format(format, width, height, color_space, dst_pixel_formats[i]);
        }
    }

    TEST_END_GROUP();
}

static void
unit_convert_image_yuv_to_rgb(DcpPixelFormat format)
{
    const uint32_t max_width = 34;
    const uint32_t max_height = 4;
    const uint32_t step = format == DCP_PIXEL_FORMAT_I444 ? 1 : 2;
    uint32_t width;

    TEST_BEGIN_GROUP(__FUNCTION__);

    for (width = 0; width <= max_width; width += step) {
        uint32_t height;

        for (height = 0; height <= max_height; height += step) {
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
    Allocator alloc = { 0, };

    TEST_BEGIN_GROUP(__FUNCTION__);

    /* Allocate and initialize input */
    test_input = alloc_new(&alloc, in_size);

    /* Allocate and initialize output */
    test_output = alloc_new(&alloc, out_size);

    uint32_t num_planes;
    for (num_planes = 0; num_planes <= 3; num_planes++) { /* Only 1 and 2 are valid values */
        int32_t src_pixel_format;

        for (src_pixel_format = 0; src_pixel_format <= DCP_PIXEL_FORMAT_NV12 + 1; src_pixel_format++) {
            int32_t src_color_space;

            for (src_color_space = 0; src_color_space <= DCP_COLOR_SPACE_BT709FR + 1; src_color_space++) {
                int32_t dst_color_space;

                for (dst_color_space = 0; dst_color_space <= DCP_COLOR_SPACE_BT709FR + 1; dst_color_space++) {
                    int32_t corrupt;

                    for (corrupt = 0; corrupt < 4; corrupt++) {
                        uint8_t *src_buffer;
                        size_t dst_strides[2];
                        uint8_t *dst_buffers[2];
                        int32_t src_pf_rgb;
                        int32_t src_cs_rgb;

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
                                                          NULL, &src_stride, (const uint8_t * const *)&src_buffer,
                                                          &dst_format, dst_strides, dst_buffers, &status.error);

                        TEST_ASSERT(DCP_RESULT_ERR, DCP_ERROR_KIND_INVALID_VALUE);

                        status.result = dcp_convert_image(width, height,
                                                          &src_format, &src_stride, (const uint8_t * const *)&src_buffer,
                                                          NULL, dst_strides, dst_buffers, &status.error);

                        TEST_ASSERT(DCP_RESULT_ERR, DCP_ERROR_KIND_INVALID_VALUE);

                        status.result = dcp_convert_image(width, height,
                                                          NULL, &src_stride, (const uint8_t * const *)&src_buffer,
                                                          NULL, dst_strides, dst_buffers, &status.error);

                        TEST_ASSERT(DCP_RESULT_ERR, DCP_ERROR_KIND_INVALID_VALUE);

                        src_pf_rgb = src_pixel_format < DCP_PIXEL_FORMAT_I444;
                        src_cs_rgb = src_color_space == DCP_COLOR_SPACE_RGB;
                        expected = dcp_status();

                        SET_EXPECTED(src_pixel_format > DCP_PIXEL_FORMAT_NV12, DCP_ERROR_KIND_INVALID_VALUE);
                        SET_EXPECTED(src_color_space > DCP_COLOR_SPACE_BT709FR, DCP_ERROR_KIND_INVALID_VALUE);
                        SET_EXPECTED(dst_color_space > DCP_COLOR_SPACE_BT709FR, DCP_ERROR_KIND_INVALID_VALUE);

                        SET_EXPECTED(!src_pf_rgb && src_cs_rgb, DCP_ERROR_KIND_INVALID_VALUE);
                        SET_EXPECTED(src_pf_rgb && !src_cs_rgb, DCP_ERROR_KIND_INVALID_VALUE);

                        SET_EXPECTED(dst_color_space <= DCP_COLOR_SPACE_RGB, DCP_ERROR_KIND_INVALID_VALUE);

                        SET_EXPECTED(is_valid_format(&src_format, width, height) == 0, DCP_ERROR_KIND_INVALID_VALUE);
                        SET_EXPECTED(is_valid_format(&dst_format, width, height) == 0, DCP_ERROR_KIND_INVALID_VALUE);

                        SET_EXPECTED(corrupt != 0, DCP_ERROR_KIND_INVALID_VALUE);

                        SET_EXPECTED((src_pixel_format != DCP_PIXEL_FORMAT_ARGB &&
                                      src_pixel_format != DCP_PIXEL_FORMAT_BGRA &&
                                      src_pixel_format != DCP_PIXEL_FORMAT_BGR), DCP_ERROR_KIND_INVALID_OPERATION);

                        SET_EXPECTED(src_color_space != DCP_COLOR_SPACE_RGB, DCP_ERROR_KIND_INVALID_OPERATION);

                        status.result = dcp_convert_image(width, height,
                                                          &src_format, &src_stride, (const uint8_t * const *)&src_buffer,
                                                          &dst_format, dst_strides, dst_buffers, &status.error);

                        TEST_ASSERT(expected.result, expected.error);

                        TEST_END();
                    }
                }
            }
        }
    }

    alloc_free(&alloc);

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
    Allocator alloc = { 0, };

    TEST_BEGIN_GROUP(__FUNCTION__);

    /* Allocate and initialize input */
    test_input = alloc_new(&alloc, in_size);

    /* Allocate and initialize output */
    test_output = alloc_new(&alloc, out_size);

    uint32_t num_planes;
    for (num_planes = 0; num_planes <= 3; num_planes++) { /* Only 1 and 2 are valid values */
        int32_t dst_pixel_format;

        for (dst_pixel_format = 0; dst_pixel_format <= DCP_PIXEL_FORMAT_NV12 + 1; dst_pixel_format++) {
            int32_t dst_color_space;

            for (dst_color_space = 0; dst_color_space <= DCP_COLOR_SPACE_BT709FR + 1; dst_color_space++) {
                int32_t src_color_space;

                for (src_color_space = 0; src_color_space <= DCP_COLOR_SPACE_BT709FR + 1; src_color_space++) {
                    int32_t corrupt;

                    for (corrupt = 0; corrupt < 4; corrupt++) {
                        size_t src_strides[2];
                        uint8_t *src_buffers[2];
                        uint8_t *dst_buffer;
                        int32_t dst_pf_rgb;
                        int32_t dst_cs_rgb;

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
                                                          NULL, src_strides, (const uint8_t * const *)src_buffers,
                                                          &dst_format, &dst_stride, &dst_buffer, &status.error);

                        TEST_ASSERT(DCP_RESULT_ERR, DCP_ERROR_KIND_INVALID_VALUE);

                        status.result = dcp_convert_image(width, height,
                                                          &src_format, src_strides, (const uint8_t * const *)src_buffers,
                                                          NULL, &dst_stride, &dst_buffer, &status.error);

                        TEST_ASSERT(DCP_RESULT_ERR, DCP_ERROR_KIND_INVALID_VALUE);

                        status.result = dcp_convert_image(width, height,
                                                          NULL, src_strides, (const uint8_t * const *)src_buffers,
                                                          NULL, &dst_stride, &dst_buffer, &status.error);

                        TEST_ASSERT(DCP_RESULT_ERR, DCP_ERROR_KIND_INVALID_VALUE);

                        dst_pf_rgb = dst_pixel_format < DCP_PIXEL_FORMAT_I444;
                        dst_cs_rgb = dst_color_space == DCP_COLOR_SPACE_RGB;
                        expected = dcp_status();

                        SET_EXPECTED(src_color_space > DCP_COLOR_SPACE_BT709FR, DCP_ERROR_KIND_INVALID_VALUE);
                        SET_EXPECTED(dst_pixel_format > DCP_PIXEL_FORMAT_NV12, DCP_ERROR_KIND_INVALID_VALUE);
                        SET_EXPECTED(dst_color_space > DCP_COLOR_SPACE_BT709FR, DCP_ERROR_KIND_INVALID_VALUE);

                        SET_EXPECTED(src_color_space <= DCP_COLOR_SPACE_RGB, DCP_ERROR_KIND_INVALID_VALUE);

                        SET_EXPECTED(is_valid_format(&src_format, width, height) == 0, DCP_ERROR_KIND_INVALID_VALUE);
                        SET_EXPECTED(is_valid_format(&dst_format, width, height) == 0, DCP_ERROR_KIND_INVALID_VALUE);

                        SET_EXPECTED(!dst_pf_rgb && dst_cs_rgb, DCP_ERROR_KIND_INVALID_VALUE);
                        SET_EXPECTED(dst_pf_rgb && !dst_cs_rgb, DCP_ERROR_KIND_INVALID_VALUE);

                        SET_EXPECTED(corrupt != 0, DCP_ERROR_KIND_INVALID_VALUE);

                        SET_EXPECTED(!(dst_pixel_format == DCP_PIXEL_FORMAT_BGRA ||
                                       dst_pixel_format == DCP_PIXEL_FORMAT_RGBA ||
                                       dst_pixel_format == DCP_PIXEL_FORMAT_RGB), DCP_ERROR_KIND_INVALID_OPERATION);
                        SET_EXPECTED(dst_color_space != DCP_COLOR_SPACE_RGB, DCP_ERROR_KIND_INVALID_OPERATION);

                        status.result = dcp_convert_image(width, height,
                                                          &src_format, src_strides, (const uint8_t * const *)src_buffers,
                                                          &dst_format, &dst_stride, &dst_buffer, &status.error);

                        TEST_ASSERT(expected.result, expected.error);

                        TEST_END();
                    }
                }
            }
        }
    }

    alloc_free(&alloc);

    TEST_END_GROUP();
}

static void
unit_get_buffers_plane(int32_t num_planes)
{
    static const uint32_t valid_width = 4098;
    static const uint32_t valid_height = 258;
    Allocator alloc = { 0, };
    int32_t pf;

    for (pf = DCP_PIXEL_FORMAT_ARGB - 1; pf <= DCP_PIXEL_FORMAT_NV12 + 1; pf++) {
        size_t buffers_size[MAX_NUMBER_OF_PLANES];
        int32_t max_number_of_planes;
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
        SET_EXPECTED(is_valid_format(&format, valid_width, valid_height) == 0, DCP_ERROR_KIND_INVALID_VALUE);
        status.result = dcp_get_buffers_size(1, valid_height, &format, NULL, buffers_size, &status.error);
        TEST_ASSERT(expected.result, expected.error);

        /* Invalid height */
        expected = dcp_status();
        SET_EXPECTED(!is_pf_valid, DCP_ERROR_KIND_INVALID_VALUE);
        SET_EXPECTED(pf >= DCP_PIXEL_FORMAT_I420, DCP_ERROR_KIND_INVALID_VALUE);
        SET_EXPECTED(is_valid_format(&format, valid_width, valid_height) == 0, DCP_ERROR_KIND_INVALID_VALUE);
        status.result = dcp_get_buffers_size(valid_width, 1, &format, NULL, buffers_size, &status.error);
        TEST_ASSERT(expected.result, expected.error);

        /* Test size is valid */
        expected = dcp_status();
        SET_EXPECTED(!is_pf_valid, DCP_ERROR_KIND_INVALID_VALUE);
        SET_EXPECTED(is_valid_format(&format, valid_width, valid_height) == 0, DCP_ERROR_KIND_INVALID_VALUE);
        status.result = dcp_get_buffers_size(valid_width, valid_height, &format, NULL, buffers_size, &status.error);
        TEST_ASSERT(expected.result, expected.error);
        if (expected.result == DCP_RESULT_OK) {
            if (num_planes == 1) {
                TEST_ASSERT_EQ(buffers_size[0], (((size_t)valid_width * (size_t)valid_height * num_log2_den[pf][0]) >> num_log2_den[pf][1]));
            } else {
                int32_t i;

                for (i = 0; i < num_planes; i++) {
                    TEST_ASSERT_EQ(buffers_size[i], (((size_t)valid_width * (size_t)valid_height * num_log2_den_per_plane[pf][2 * i]) >> num_log2_den_per_plane[pf][2 * i + 1]));
                }
            }
        }

        TEST_END();
    }
}

static void
unit_get_buffers_size(void)
{
    DcpStatus status = dcp_status();
    int32_t num_planes;
    Allocator alloc = { 0, };

    TEST_BEGIN_GROUP(__FUNCTION__);

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
        TEST_BEGIN_GROUP("num_planes=%d", num_planes);
        unit_get_buffers_plane(num_planes);
        TEST_END_GROUP();
    }

    TEST_END_GROUP();
}


static void
unit_convert_image_over_4gb_limit(void)
{
    const uint32_t width = 0x20000020;
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
        DCP_COLOR_SPACE_RGB,
        1
    };

    DcpImageFormat dst_format = {
        DCP_PIXEL_FORMAT_NV12,
        DCP_COLOR_SPACE_BT601,
        1
    };

    Allocator alloc = { 0, };

    TEST_BEGIN_GROUP(__FUNCTION__);

    if (sizeof(void *) <= 4) {
        exit(EXIT_SKIP_CODE);
    }

    status.result = dcp_get_buffers_size(width, height, &src_format, NULL, &src_buffer_size, &status.error);
    TEST_ASSERT(DCP_RESULT_OK, -1);
    TEST_ASSERT_EQ(src_buffer_size, expected_src_buffer_size);

    status.result = dcp_get_buffers_size(width, height, &dst_format, NULL, &dst_buffer_size, &status.error);
    TEST_ASSERT(DCP_RESULT_OK, -1);
    TEST_ASSERT_EQ(dst_buffer_size, expected_dst_buffer_size);

    src_image = alloc_new(&alloc, src_buffer_size);
    if (src_image == NULL) {
        alloc_free(&alloc);
        exit(EXIT_SKIP_CODE);
    }

    dst_image = alloc_new(&alloc, dst_buffer_size);
    if (dst_image == NULL) {
        alloc_free(&alloc);
        exit(EXIT_SKIP_CODE);
    }

    status.result = dcp_convert_image(width, height,
                                      &src_format, NULL, (const uint8_t * const *)&src_image,
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

    alloc_free(&alloc);

    TEST_END_GROUP();
}

static uint8_t
get_format_bpp_rgb(DcpPixelFormat format)
{
    if (format == DCP_PIXEL_FORMAT_ARGB ||
        format == DCP_PIXEL_FORMAT_BGRA ||
        format == DCP_PIXEL_FORMAT_RGBA) {
        return 4;
    }

    return 3;
}

static void
unit_image_convert_rgb_ok(DcpPixelFormat src_pixel_format,
                           DcpPixelFormat dst_pixel_format)
{
    const uint32_t MAX_WIDTH = 49;
    const uint32_t MAX_HEIGHT = 8;
    const uint32_t MAX_FILL_BYTES = 2;
    const uint32_t auto_stride = DCP_STRIDE_AUTO;
    const uint8_t src_bpp = get_format_bpp_rgb(src_pixel_format);
    const uint8_t dst_bpp = get_format_bpp_rgb(dst_pixel_format);;

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
    Allocator alloc = { 0, };

    DcpImageFormat src_format = {
        src_pixel_format,
        DCP_COLOR_SPACE_RGB,
        1
    };

    DcpImageFormat dst_format = {
        dst_pixel_format,
        DCP_COLOR_SPACE_RGB,
        1
    };

    DcpStatus status = dcp_status();
    TEST_BEGIN_GROUP(__FUNCTION__);
    for (width = 0; width <= MAX_WIDTH; width++) {
        for (height = 0; height <= MAX_HEIGHT; height++) {
            for (src_stride_diff = 0; src_stride_diff <= MAX_FILL_BYTES; src_stride_diff++) {
                for (dst_stride_diff = 0; dst_stride_diff <= MAX_FILL_BYTES; dst_stride_diff++) {
                    uint32_t h;
                    uint32_t w;
                    TEST_BEGIN("Width=%d, Height=%d, Strides: src=%d, dst=%d", width, height, src_stride_diff, dst_stride_diff);

                    src_stride_bytes = (src_bpp * width) + src_stride_diff;
                    dst_stride_bytes = (dst_bpp * width) + dst_stride_diff;
                    src_stride_param = src_stride_diff == 0 ? auto_stride : src_stride_bytes;
                    dst_stride_param = dst_stride_diff == 0 ? auto_stride : dst_stride_bytes;

                    src_buffers[0] = alloc_new(&alloc, src_stride_bytes * height);
                    dst_buffers[0] = alloc_new(&alloc, dst_stride_bytes * height);

                    for (h = 0; h < height; h++) {
                        for (w = 0; w < width; w++) {
                            src_buffers[0][(h * src_stride_bytes) + (w * src_bpp) + 0] = rand() & 0xFF;
                            src_buffers[0][(h * src_stride_bytes) + (w * src_bpp) + 1] = rand() & 0xFF;
                            src_buffers[0][(h * src_stride_bytes) + (w * src_bpp) + 2] = rand() & 0xFF;
                            if (src_bpp == 4) {
                                src_buffers[0][(h * src_stride_bytes) + (w * src_bpp) + 3] = 0xFF;
                            }
                        }
                    }

                    status.result = dcp_convert_image(width, height,
                                                      &src_format, &src_stride_param, (const uint8_t * const *)src_buffers,
                                                      &dst_format, &dst_stride_param, dst_buffers, &status.error);
                    TEST_ASSERT(DCP_RESULT_OK, -1);

                    for (h = 0; h < height; h++) {
                        for (w = 0; w < width; w++) {
                            TEST_ASSERT_EQ(dst_buffers[0][(h * dst_stride_bytes) + (w * dst_bpp) + 0], src_buffers[0][(h * src_stride_bytes) + (w * src_bpp) + 2]);
                            TEST_ASSERT_EQ(dst_buffers[0][(h * dst_stride_bytes) + (w * dst_bpp) + 1], src_buffers[0][(h * src_stride_bytes) + (w * src_bpp) + 1]);
                            TEST_ASSERT_EQ(dst_buffers[0][(h * dst_stride_bytes) + (w * dst_bpp) + 2], src_buffers[0][(h * src_stride_bytes) + (w * src_bpp) + 0]);
                            if (dst_bpp == 4) {
                                TEST_ASSERT_EQ(dst_buffers[0][(h * dst_stride_bytes) + (w * dst_bpp) + 3], 0xFF);
                            }
                        }
                    }

                    alloc_free(&alloc);

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
    } else if (strcmp(test_name, "unit_convert_image_rgb_to_nv12_ok") == 0) {
        int32_t planes = *(argv[2]) - '0';
        unit_convert_image_rgb_to_yuv(DCP_PIXEL_FORMAT_NV12, planes);
    } else if (strcmp(test_name, "unit_convert_image_rgb_to_i420_ok") == 0) {
        unit_convert_image_rgb_to_yuv(DCP_PIXEL_FORMAT_I420, 3);
    } else if (strcmp(test_name, "unit_convert_image_rgb_to_i444_ok") == 0) {
        unit_convert_image_rgb_to_yuv(DCP_PIXEL_FORMAT_I444, 3);
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
        unit_image_convert_rgb_ok(DCP_PIXEL_FORMAT_RGB, DCP_PIXEL_FORMAT_BGRA);
    } else if (strcmp(test_name, "unit_image_convert_bgra_to_rgb_ok") == 0) {
        unit_image_convert_rgb_ok(DCP_PIXEL_FORMAT_BGRA, DCP_PIXEL_FORMAT_RGB);
    }  else if (strcmp(test_name, "unit_image_convert_bgr_to_rgb_ok") == 0) {
        unit_image_convert_rgb_ok(DCP_PIXEL_FORMAT_BGR, DCP_PIXEL_FORMAT_RGB);
    } else {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/* ex:set ts=4 et: */