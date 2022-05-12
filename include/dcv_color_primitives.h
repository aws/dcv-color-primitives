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

/**
 * dcv_color_primitives:
 * DCV color primitives is a library to perform image color model conversion.
 *
 * It is able to convert the following pixel formats:
 *
 * | Source pixel format  | Destination pixel formats  |
 * | -------------------- | -------------------------- |
 * | ARGB                 | I420, I444, NV12           |
 * | BGR                  | I420, I444, NV12, RGB      |
 * | BGRA                 | I420, I444, NV12, RGB      |
 * | I420                 | BGRA                       |
 * | I444                 | BGRA                       |
 * | NV12                 | BGRA                       |
 * | RGB                  | BGRA                       |
 *
 * The supported color models are:
 * - YCbCr, ITU-R Recommendation BT.601 (standard video system)
 * - YCbCr, ITU-R Recommendation BT.709 (CSC systems)
 *
 * Both standard range (0-235) and full range (0-255) are supported.
 *
 * # Examples
 *
 * Initialize the library:
 *
 * |[<!-- language="C" -->
 * dcp_initialize();
 * ]|
 *
 * Convert an image from bgra to nv12 (single plane) format, with Bt601 color space:
 *
 * |[<!-- language="C" -->
 * static const uint32_t WIDTH = 640;
 * static const uint32_t HEIGHT = 480;
 *
 * static uint8_t src_image[4 * (size_t)WIDTH * (size_t)HEIGHT] = { 0 };
 * static uint8_t dst_image[3 * (size_t)WIDTH * (size_t)HEIGHT / 2] = { 0 };
 *
 * static uint8_t *src_buffers[] = { src_image };
 * static uint8_t *dst_buffers[] = { dst_image };
 *
 * DcpImageFormat src_format = {
 *     DCP_PIXEL_FORMAT_BGRA,
 *     DCP_COLOR_SPACE_RGB,
 *     1,
 * };
 *
 * DcpImageFormat dst_format = {
 *     DCP_PIXEL_FORMAT_NV12,
 *     DCP_COLOR_SPACE_BT601,
 *     1,
 * };
 *
 * dcp_initialize();
 * dcp_convert_image(WIDTH, HEIGHT,
 *                   &src_format, NULL, src_buffers,
 *                   &dst_format, NULL, dst_buffers,
 *                   NULL);
 * ]|
 *
 * Handle conversion errors:
 *
 * |[<!-- language="C" -->
 * static const uint32_t WIDTH = 640;
 * static const uint32_t HEIGHT = 480;
 *
 * static uint8_t src_image[4 * (size_t)WIDTH * (size_t)HEIGHT] = { 0 };
 * static uint8_t dst_image[3 * (size_t)WIDTH * (size_t)HEIGHT / 2] = { 0 };
 *
 * static uint8_t *src_buffers[] = { src_image };
 * static uint8_t *dst_buffers[] = { dst_image };
 *
 * DcpImageFormat src_format = {
 *     DCP_PIXEL_FORMAT_BGRA,
 *     DCP_COLOR_SPACE_RGB,
 *     1,
 * };
 *
 * DcpImageFormat dst_format = {
 *     DCP_PIXEL_FORMAT_NV12,
 *     DCP_COLOR_SPACE_BT601,
 *     1,
 * };
 *
 * DcpResult result;
 * DcpErrorKind error;
 *
 * dcp_initialize();
 * result = dcp_convert_image(WIDTH, HEIGHT,
 *                            &src_format, NULL, src_buffers,
 *                            &dst_format, NULL, dst_buffers,
 *                            &error);
 *
 * if (result == DCP_RESULT_OK) {
 *     printf("Conversion ok\n");
 * } else {
 *     printf("Conversion failed with error 0x%X\n", error);
 * }
 * ]|
 *
 * Compute how many bytes are needed to store and image of a given format and size:
 *
 * |[<!-- language="C" -->
 * static const uint32_t WIDTH = 640;
 * static const uint32_t HEIGHT = 480;
 * static const uint32_t NUM_PLANES = 1;
 *
 * DcpImageFormat format = {
 *     DCP_PIXEL_FORMAT_BGRA,
 *     DCP_COLOR_SPACE_RGB,
 *     NUM_PLANES,
 * };
 *
 * static size_t sizes[NUM_PLANES] = { 0 };
 *
 * DcpResult result;
 * DcpErrorKind error;
 * uint8_t *buffer = NULL;
 *
 * dcp_initialize();
 * result = dcp_get_buffers_size(WIDTH, HEIGHT, &format, NULL, sizes, &error);
 * if (result == DCP_RESULT_OK) {
 *     buffer = (uint8_t *)malloc(sizes[0]);
 * } else {
 *     printf("Cannot compute buffer size, error 0x%X\n", error);
 * }
 * ]|
 *
 * Provide image planes to hangle data scattered in multiple buffers that are not
 * necessarily contiguous:
 *
 * |[<!-- language="C" -->
 * static const uint32_t WIDTH = 640;
 * static const uint32_t HEIGHT = 480;
 * static const uint32_t NUM_SRC_PLANES = 2;
 * static const uint32_t NUM_DST_PLANES = 1;
 *
 * DcpImageFormat src_format = {
 *     DCP_PIXEL_FORMAT_NV12,
 *     DCP_COLOR_SPACE_BT709,
 *     NUM_SRC_PLANES,
 * };
 *
 * DcpImageFormat dst_format = {
 *     DCP_PIXEL_FORMAT_BGRA,
 *     DCP_COLOR_SPACE_RGB,
 *     NUM_DST_PLANES,
 * };
 *
 * static size_t src_sizes[NUM_SRC_PLANES] = { 0 };
 * static size_t dst_sizes[NUM_DST_PLANES] = { 0 };
 *
 * static uint8_t * src_buffers[NUM_SRC_PLANES] = { 0 };
 * static uint8_t * dst_buffers[NUM_DST_PLANES] = { 0 };
 *
 * DcpErrorKind error;
 *
 * dcp_initialize();
 *
 * if (dcp_get_buffers_size(WIDTH, HEIGHT, &src_format, NULL, src_sizes, &error) != DCP_RESULT_OK) {
 *     printf("Cannot compute source buffer size, error 0x%X\n", error);
 *     return;
 * }
 *
 * if (dcp_get_buffers_size(WIDTH, HEIGHT, &dst_format, NULL, dst_sizes, &error) != DCP_RESULT_OK) {
 *     printf("Cannot compute destination buffer size, error 0x%X\n", error);
 *     return;
 * }
 *
 * src_buffers[0] = (uint8_t *)malloc(src_sizes[0]);
 * src_buffers[1] = (uint8_t *)malloc(src_sizes[1]);
 * dst_buffers[0] = (uint8_t *)malloc(dst_sizes[0]);
 *
 * if (dcp_convert_image(WIDTH, HEIGHT,
 *                       &src_format, NULL, src_buffers,
 *                       &dst_format, NULL, dst_buffers,
 *                       &error) != DCP_RESULT_OK) {
 *     printf("Conversion failed with error 0x%X\n", error);
 * }
 * ]|
 *
 * Provide image strides to convert data which is not tightly packed:
 *
 * |[<!-- language="C" -->
 * static const uint32_t WIDTH = 640;
 * static const uint32_t HEIGHT = 480;
 * static const uint32_t NUM_SRC_PLANES = 1;
 * static const uint32_t NUM_DST_PLANES = 2;
 * static const size_t RGB_STRIDE = 4 * (((3 * (size_t)WIDTH) + 3) / 4);
 *
 * DcpImageFormat src_format = {
 *     DCP_PIXEL_FORMAT_BGR,
 *     DCP_COLOR_SPACE_RGB,
 *     NUM_SRC_PLANES,
 * };
 *
 * DcpImageFormat dst_format = {
 *     DCP_PIXEL_FORMAT_NV12,
 *     DCP_COLOR_SPACE_BT709,
 *     NUM_DST_PLANES,
 * };
 *
 * static size_t src_strides[NUM_SRC_PLANES] = { RGB_STRIDE };
 *
 * static size_t src_sizes[NUM_SRC_PLANES] = { 0 };
 * static size_t dst_sizes[NUM_DST_PLANES] = { 0 };
 *
 * static uint8_t * src_buffers[NUM_SRC_PLANES] = { 0 };
 * static uint8_t * dst_buffers[NUM_DST_PLANES] = { 0 };
 *
 * DcpErrorKind error;
 *
 * dcp_initialize();
 *
 * if (dcp_get_buffers_size(WIDTH, HEIGHT, &src_format, src_strides, src_sizes, &error) != DCP_RESULT_OK) {
 *     printf("Cannot compute source buffer size, error 0x%X\n", error);
 *     return 0;
 * }
 *
 * if (dcp_get_buffers_size(WIDTH, HEIGHT, &dst_format, NULL, dst_sizes, &error) != DCP_RESULT_OK) {
 *     printf("Cannot compute destination buffer size, error 0x%X\n", error);
 *     return 0;
 * }
 *
 * src_buffers[0] = (uint8_t *)malloc(src_sizes[0]);
 * dst_buffers[0] = (uint8_t *)malloc(dst_sizes[0]);
 * dst_buffers[1] = (uint8_t *)malloc(dst_sizes[1]);
 *
 * if (dcp_convert_image(WIDTH, HEIGHT,
 *                       &src_format, src_strides, src_buffers,
 *                       &dst_format, NULL, dst_buffers,
 *                       &error) != DCP_RESULT_OK) {
 *     printf("Conversion failed with error 0x%X\n", error);
 * }
 * ]|
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/**
 * DcpResult:
 * @DCP_RESULT_OK: The operation completed successfully
 * @DCP_RESULT_ERR: An error occured during the operation execution
 *
 * An enumeration of results.
 */
typedef enum {
    DCP_RESULT_OK,
    DCP_RESULT_ERR
} DcpResult;

/**
 * DcpErrorKind:
 * @DCP_ERROR_KIND_NOT_INITIALIZED: function(dcp_initialize) was never called
 * @DCP_ERROR_KIND_INVALID_VALUE: One or more parameters have invalid values for the called function
 * @DCP_ERROR_KIND_INVALID_OPERATION: The combination of parameters is unsupported for the called function
 * @DCP_ERROR_KIND_NOT_ENOUGH_DATA: Not enough data was provided to the called function.
 *                                  Typically, provided arrays are not correctly sized
 *
 * An enumeration of errors.
 */
typedef enum {
    DCP_ERROR_KIND_NOT_INITIALIZED,
    DCP_ERROR_KIND_INVALID_VALUE,
    DCP_ERROR_KIND_INVALID_OPERATION,
    DCP_ERROR_KIND_NOT_ENOUGH_DATA,
} DcpErrorKind;

/**
 * DcpPixelFormat:
 * @DCP_PIXEL_FORMAT_ARGB: RGB with alpha channel first. 32 bits per pixel
 * @DCP_PIXEL_FORMAT_BGRA: Reverse RGB with alpha channel last. 32 bits per pixel
 * @DCP_PIXEL_FORMAT_BGR: Reverse RGB packed into 24 bits without padding. 24 bits per pixel
 * @DCP_PIXEL_FORMAT_RGBA: RGB with alpha channel last. 32 bits per pixel
 * @DCP_PIXEL_FORMAT_RGB: RGB packed into 24 bits without padding. 24 bits per pixel
 * @DCP_PIXEL_FORMAT_I444: YUV with one luma plane Y then 2 chroma planes U and V.
 *                         Chroma planes are not sub-sampled. 24 bits per pixel
 * @DCP_PIXEL_FORMAT_I422: YUV with one luma plane Y then 2 chroma planes U, V.
 *                         Chroma planes are sub-sampled in the horizontal dimension, by a factor of 2.
 *                         16 bits per pixel
 * @DCP_PIXEL_FORMAT_I420: YUV with one luma plane Y then U chroma plane and last the V chroma plane.
 *                         The two chroma planes are sub-sampled in both the horizontal and vertical
 *                         dimensions by a factor of 2.
 *                         12 bits per pixel
 * @DCP_PIXEL_FORMAT_NV12: YUV with one luma plane Y then one plane with U and V values interleaved.
 *                         Chroma planes are subsampled in both the horizontal and vertical dimensions
 *                         by a factor of 2.
 *                         12 bits per pixel
 *
 * An enumeration of supported pixel formats.
 */
typedef enum {
    DCP_PIXEL_FORMAT_ARGB,
    DCP_PIXEL_FORMAT_BGRA,
    DCP_PIXEL_FORMAT_BGR,
    DCP_PIXEL_FORMAT_RGBA,
    DCP_PIXEL_FORMAT_RGB,
    DCP_PIXEL_FORMAT_I444,
    DCP_PIXEL_FORMAT_I422,
    DCP_PIXEL_FORMAT_I420,
    DCP_PIXEL_FORMAT_NV12,
} DcpPixelFormat;

/**
 * DcpColorSpace:
 * @DCP_COLOR_SPACE_RGB: Gamma-corrected R'G'B'.
 *                       The gamma is the one defined in ITU-R Recommendation BT.709-6 page 3, item 1.2
 *                       The relationship between gamma-corrected component (C') and its linear value (C)
 *                       is the following one:
 *                       C' = 4.5 * C                      if C < 0.018,
 *                            1.099 * pow(C, 0.45) - 0.099 otherwise.
 * @DCP_COLOR_SPACE_BT601: YCbCr, ITU-R Recommendation BT.601 (standard video system)
 * @DCP_COLOR_SPACE_BT709: YCbCr, ITU-R Recommendation BT.709 (CSC systems)
 * @DCP_COLOR_SPACE_BT601FR: YCbCr, BT.601 (full range)
 * @DCP_COLOR_SPACE_BT709FR: YCbCr, BT.709 (full range)
 * @DCP_COLOR_SPACE_LRGB: Deprecated: 0.5.0: Use @DCP_COLOR_SPACE_RGB instead
 *
 * An enumeration of supported color models.
 *
 * It includes:
 *
 * - Colorimetry
 * - Gamma
 * - Range (headroom / footroom)
 * - Primaries
 */
typedef enum {
    DCP_COLOR_SPACE_RGB,
    DCP_COLOR_SPACE_BT601,
    DCP_COLOR_SPACE_BT709,
    DCP_COLOR_SPACE_BT601FR,
    DCP_COLOR_SPACE_BT709FR,
    DCP_COLOR_SPACE_LRGB = DCP_COLOR_SPACE_RGB, /* Preserve backwards compatibility */
} DcpColorSpace;

/**
 * DcpImageFormat:
 * @pixel_format: Pixel format
 * @color_space: Color space
 * @num_planes: Number of planes
 *
 * Describes how the image data is laid out in memory and its color space.
 *
 * # Note # {#size-constraint}
 *
 * Not all combinations of pixel format, color space and number of planes
 * describe a valid image format.
 *
 * Each pixel format has one or more compatible color spaces:
 *
 * pixel_format          | color_space
 * ----------------------|---------------------------------------------
 * DCP_PIXEL_FORMAT_ARGB | DCP_COLOR_SPACE_RGB
 * DCP_PIXEL_FORMAT_BGRA | DCP_COLOR_SPACE_RGB
 * DCP_PIXEL_FORMAT_BGR  | DCP_COLOR_SPACE_RGB
 * DCP_PIXEL_FORMAT_RGBA | DCP_COLOR_SPACE_RGB
 * DCP_PIXEL_FORMAT_RGB  | DCP_COLOR_SPACE_RGB
 * DCP_PIXEL_FORMAT_I444 | DCP_COLOR_SPACE_BT601(FR), DCP_COLOR_SPACE_BT709(FR)
 * DCP_PIXEL_FORMAT_I422 | DCP_COLOR_SPACE_BT601(FR), DCP_COLOR_SPACE_BT709(FR)
 * DCP_PIXEL_FORMAT_I420 | DCP_COLOR_SPACE_BT601(FR), DCP_COLOR_SPACE_BT709(FR)
 * DCP_PIXEL_FORMAT_NV12 | DCP_COLOR_SPACE_BT601(FR), DCP_COLOR_SPACE_BT709(FR)
 *
 * Some pixel formats might impose additional restrictions on the accepted number of
 * planes and the image size:
 *
 * pixel_format          | subsampling | w   | h   | #planes | #1     | #2     | #3
 * ----------------------|:-----------:|:---:|:---:|:-------:|:------:|:------:|:-------:
 * DCP_PIXEL_FORMAT_ARGB | 4:4:4       |     |     | 1       | argb:4 |        |
 * DCP_PIXEL_FORMAT_BGRA | 4:4:4       |     |     | 1       | bgra:4 |        |
 * DCP_PIXEL_FORMAT_BGR  | 4:4:4       |     |     | 1       | bgr:3  |        |
 * DCP_PIXEL_FORMAT_RGBA | 4:4:4       |     |     | 1       | rgba:4 |        |
 * DCP_PIXEL_FORMAT_RGB  | 4:4:4       |     |     | 1       | rgb:3  |        |
 * DCP_PIXEL_FORMAT_I444 | 4:4:4       |     |     | 3       | y:1    | u:1    | v:1
 * DCP_PIXEL_FORMAT_I422 | 4:2:2       |  2  |     | 1, 3    | y:1    | u:1/2  | v:1/2
 * DCP_PIXEL_FORMAT_I420 | 4:2:0       |  2  |  2  | 3       | y:1    | u:1/4  | v:1/4
 * DCP_PIXEL_FORMAT_NV12 | 4:2:0       |  2  |  2  | 1, 2    | y:1    | uv:1/2 |
 *
 * The values reported in columns `w` and `h`, when specified, indicate that the described
 * image should have width and height that are multiples of the specified values
 */
typedef struct {
    DcpPixelFormat pixel_format;
    DcpColorSpace color_space;
    uint32_t num_planes;
} DcpImageFormat;

/**
 * DCP_STRIDE_AUTO:
 * If a plane stride is assigned to this constant, the plane will be assumed to contain packed data
 */
static const size_t DCP_STRIDE_AUTO = 0;

/**
 * dcp_initialize:
 * Automatically initializes the library functions that are most appropriate for
 * the current processor type.
 *
 * You should call this function before calling any other library function
 *
 * # Safety
 *
 * You can not use any other library function (also in other threads) while the initialization
 * is in progress. Failure to do so result in undefined behaviour
 *
 * # Examples
 *
 * |[<!-- language="C" -->
 * dcp_initialize();
 * ]|
 */
void                dcp_initialize              (void);

/**
 * dcp_describe_acceleration:
 * Returns a description of the algorithms that are best for the running cpu and
 * available instruction sets
 *
 * Returns: a null-terminated string that contains the description, or %NULL if the library was not
 *          initialized before. String has to freed using function(dcp_unref_string)
 *
 * # Examples
 *
 * |[<!-- language="C" -->
 * char *description = NULL;
 *
 * dcp_initialize();
 * description = dcp_describe_acceleration();
 * if (description != NULL) {
 *     printf("%s\n", description);
 * } else {
 *     printf("Unable to describe the acceleration\n");
 * }
 * // => {cpu-manufacturer:Intel,instruction-set:Avx2}
 * ]|
 *
 * When function(dcp_initialize) is not called:
 *
 * |[<!-- language="C" -->
 * char *description = dcp_describe_acceleration();
 * if (description != NULL) {
 *     printf("%s\n", description);
 * } else {
 *     printf("Unable to describe the acceleration\n");
 * }
 * // => Unable to describe the acceleration
 * ]|
 */
char *              dcp_describe_acceleration   (void);

/**
 * dcp_unref_string:
 * @string: A null-terminated string generated by the library. If the provided argument is %NULL,
 *          the function does nothing
 *
 * Frees strings generated by DCV Color Primitives library
 *
 * # Examples
 *
 * |[<!-- language="C" -->
 * char *description = NULL;
 *
 * dcp_initialize();
 * description = dcp_describe_acceleration();
 * dcp_unref_string(description);
 * ]|
 */
void                dcp_unref_string            (char *string);

/**
 * dcp_get_buffers_size:
 * @width: Width of the image in pixels
 * @height: Height of the image in pixels
 * @format: (not nullable) (in): Image format
 * @strides: (in): An array of distances in bytes between starts of consecutive lines in each image planes
 * @buffers_size: (not nullable) (out caller-allocates): An array describing the minimum number of
 *                                                       bytes required in each image planes
 * @error: An optional reference to an #DcpErrorKind to receive the error description if the operation
 *         does not complete successfully
 *
 * Compute number of bytes required to store an image given its format, dimensions
 * and optionally its strides
 *
 * Returns: %DCP_RESULT_OK if the operation succeeded, else %DCP_RESULT_ERR
 *
 * # Errors
 *
 * - %DCP_ERROR_KIND_INVALID_VALUE if @format or @buffers_size is %NULL
 * - %DCP_ERROR_KIND_INVALID_VALUE if the image pixel format is not a #DcpPixelFormat
 * - %DCP_ERROR_KIND_INVALID_VALUE if @width or @height violate the [size constraints][size-constraint]
 *   that might by imposed by the image pixel format
 * - %DCP_ERROR_KIND_INVALID_VALUE if the image format has a number of planes which is not compatible
 *   with its pixel format

 * # Undefined behaviour
 *
 * - @strides is not %NULL and its length is less than the image format number of planes
 * - @buffers_sizes is not %NULL and its length is less than the image format number of planes
 *
 * # Examples
 *
 * Compute how many bytes are needed to store and image of a given format and size
 * assuming all planes contain data which is tightly packed:
 *
 * |[<!-- language="C" -->
 * static const uint32_t WIDTH = 640;
 * static const uint32_t HEIGHT = 480;
 * static const uint32_t NUM_PLANES = 2;
 *
 * DcpImageFormat format = {
 *     DCP_PIXEL_FORMAT_NV12,
 *     DCP_COLOR_SPACE_BT601,
 *     NUM_PLANES,
 * };
 *
 * static size_t sizes[NUM_PLANES] = { 0 };
 *
 * DcpResult result;
 * DcpErrorKind error;
 *
 * dcp_initialize();
 * result = dcp_get_buffers_size(WIDTH, HEIGHT, &format, NULL, sizes, &error);
 * ]|
 *
 * Compute how many bytes are needed to store and image of a given format and size
 * in which all planes have custom strides:
 *
 * |[<!-- language="C" -->
 * static const uint32_t WIDTH = 640;
 * static const uint32_t HEIGHT = 480;
 * static const uint32_t NUM_PLANES = 2;
 * static const size_t Y_STRIDE = (size_t)WIDTH + 1;
 * static const size_t UV_STRIDE = (size_t)WIDTH + 3;
 *
 * DcpImageFormat format = {
 *     DCP_PIXEL_FORMAT_NV12,
 *     DCP_COLOR_SPACE_BT601,
 *     NUM_PLANES,
 * };
 *
 * static const size_t STRIDES[NUM_PLANES] = { Y_STRIDE, UV_STRIDE, };
 * static size_t sizes[NUM_PLANES] = { 0 };
 *
 * DcpResult result;
 * DcpErrorKind error;
 *
 * dcp_initialize();
 * result = dcp_get_buffers_size(WIDTH, HEIGHT, &format, STRIDES, sizes, &error);
 * ]|
 *
 * Compute how many bytes are needed to store and image of a given format and size
 * in which some planes have custom strides, while some other are assumed to
 * contain data which is tightly packed:
 *
 * |[<!-- language="C" -->
 * static const uint32_t WIDTH = 640;
 * static const uint32_t HEIGHT = 480;
 * static const uint32_t NUM_PLANES = 2;
 * static const size_t Y_STRIDE = (size_t)WIDTH + 1;
 *
 * DcpImageFormat format = {
 *     DCP_PIXEL_FORMAT_NV12,
 *     DCP_COLOR_SPACE_BT601,
 *     NUM_PLANES,
 * };
 *
 * static const size_t STRIDES[NUM_PLANES] = { Y_STRIDE, DCP_STRIDE_AUTO, };
 * static size_t sizes[NUM_PLANES] = { 0 };
 *
 * DcpResult result;
 * DcpErrorKind error;
 *
 * dcp_initialize();
 * result = dcp_get_buffers_size(WIDTH, HEIGHT, &format, STRIDES, sizes, &error);
 * ]|
 *
 * Default strides (e.g. the one you would set for tightly packed data) can be set
 * using the constant %DCP_STRIDE_AUTO
 */
DcpResult           dcp_get_buffers_size        (uint32_t              width,
                                                 uint32_t              height,
                                                 const DcpImageFormat *format,
                                                 const size_t         *strides,
                                                 size_t               *buffers_size,
                                                 DcpErrorKind         *error);

/**
 * dcp_convert_image:
 * @width: Width of the image to convert in pixels
 * @height: Height of the image to convert in pixels
 * @src_format: (not nullable) (in): Source image format
 * @src_strides: (in): An array of distances in bytes between starts of consecutive lines in each
 *                     source image planes
 * @src_buffers: (not nullable) (in): An array of image buffers in each source color plane
 * @dst_format: (not nullable) (in): Destination image format
 * @dst_strides: (in): An array of distances in bytes between starts of consecutive lines in each
 *                     destination image planes
 * @dst_buffers: (not nullable) (out caller-allocates): An array of image buffers in each destination
 *                                                      color plane
 * @error: An optional reference to an #DcpErrorKind to receive the error description if the operation
 *         does not complete successfully
 *
 * Converts from a color space to another one, applying downsampling/upsampling
 * to match destination image format
 *
 * Returns: %DCP_RESULT_OK if the operation succeeded, else %DCP_RESULT_ERR
 *
 * # Errors
 *
 * - %DCP_ERROR_KIND_INVALID_VALUE if @src_format or @dst_format is %NULL
 * - %DCP_ERROR_KIND_INVALID_VALUE if @src_buffers or @dst_buffers is %NULL
 * - %DCP_ERROR_KIND_INVALID_VALUE if the source or destination image pixel format is not a #DcpPixelFormat
 * - %DCP_ERROR_KIND_INVALID_VALUE if the source or destination image color space is not a #DcpColorSpace
 * - %DCP_ERROR_KIND_NOT_INITIALIZED if the library was not initialized before
 * - %DCP_ERROR_KIND_INVALID_VALUE if @width or @height violate the [size constraints][size-constraint]
 *   that might by imposed by the source and destination image pixel formats
 * - %DCP_ERROR_KIND_INVALID_VALUE if source or destination image formats have a number of planes
 *   which is not compatible with their pixel formats
 * - %DCP_ERROR_KIND_INVALID_OPERATION if there is no available method to convert the image with the
 *   source pixel format to the image with the destination pixel format.
 *
 *   The list of available conversions is specified here:
 *
 *   Source image pixel format         | Supported destination image pixel formats
 *   ----------------------------------|------------------------------------------
 *   DCP_PIXEL_FORMAT_ARGB             | DCP_PIXEL_FORMAT_I420 [1][algo-1]
 *   DCP_PIXEL_FORMAT_ARGB             | DCP_PIXEL_FORMAT_I444 [1][algo-1]
 *   DCP_PIXEL_FORMAT_ARGB             | DCP_PIXEL_FORMAT_NV12 [1][algo-1]
 *   DCP_PIXEL_FORMAT_BGRA             | DCP_PIXEL_FORMAT_I420 [1][algo-1]
 *   DCP_PIXEL_FORMAT_BGRA             | DCP_PIXEL_FORMAT_I444 [1][algo-1]
 *   DCP_PIXEL_FORMAT_BGRA             | DCP_PIXEL_FORMAT_NV12 [1][algo-1]
 *   DCP_PIXEL_FORMAT_BGRA             | DCP_PIXEL_FORMAT_RGB  [4][algo-4]
 *   DCP_PIXEL_FORMAT_BGR              | DCP_PIXEL_FORMAT_I420 [1][algo-1]
 *   DCP_PIXEL_FORMAT_BGR              | DCP_PIXEL_FORMAT_I444 [1][algo-1]
 *   DCP_PIXEL_FORMAT_BGR              | DCP_PIXEL_FORMAT_NV12 [1][algo-1]
 *   DCP_PIXEL_FORMAT_BGR              | DCP_PIXEL_FORMAT_RGB  [5][algo-5]
 *   DCP_PIXEL_FORMAT_I420             | DCP_PIXEL_FORMAT_BGRA [2][algo-2]
 *   DCP_PIXEL_FORMAT_I444             | DCP_PIXEL_FORMAT_BGRA [2][algo-2]
 *   DCP_PIXEL_FORMAT_NV12             | DCP_PIXEL_FORMAT_BGRA [2][algo-2]
 *   DCP_PIXEL_FORMAT_RGB              | DCP_PIXEL_FORMAT_BGRA [3][algo-3]
 *
 * # Undefined behaviour
 *
 * - @src_stride is not %NULL and its length is less than the source image format number of planes
 * - @dst_stride is not %NULL and its length is less than the destination image format number of planes
 * - @src_buffers length is less than the source image format number of planes
 * - @dst_buffers length is less than the destination image format number of planes
 * - One or more source/destination buffers does not provide enough data.
 *   The minimum number of bytes to provide for each buffer depends from the image format, dimensions,
 *   and strides (if they are not %NULL).
 *   You can compute the buffers' size using function(dcp_get_buffers_size)
 *
 * # Algorithm 1 # {#algo-1}
 *
 * Conversion from linear RGB model to YCbCr color model, with 4:2:0 downsampling
 *
 * If the destination image color space is Bt601, the following formula is applied:
 *
 * |[
 * y  =  0.257 * r + 0.504 * g + 0.098 * b + 16
 * cb = -0.148 * r - 0.291 * g + 0.439 * b + 128
 * cr =  0.439 * r - 0.368 * g - 0.071 * b + 128
 * ]|
 *
 * If the destination image color space is Bt709, the following formula is applied:
 *
 * |[
 * y  =  0.213 * r + 0.715 * g + 0.072 * b + 16
 * cb = -0.117 * r - 0.394 * g + 0.511 * b + 128
 * cr =  0.511 * r - 0.464 * g - 0.047 * b + 128
 * ]|
 *
 * If the destination image color space is Bt601FR, the following formula is applied:
 *
 * |[
 * y  =  0.299 * r + 0.587 * g + 0.114 * b
 * cb = -0.169 * r - 0.331 * g + 0.500 * b + 128
 * cr =  0.500 * r - 0.419 * g - 0.081 * b + 128
 * ]|
 *
 * If the destination image color space is Bt709FR, the following formula is applied:
 *
 * |[
 * y  =  0.213 * r + 0.715 * g + 0.072 * b
 * cb = -0.115 * r - 0.385 * g + 0.500 * b + 128
 * cr =  0.500 * r - 0.454 * g - 0.046 * b + 128
 * ]|
 *
 * # Algorithm 2 # {#algo-2}
 *
 * Conversion from YCbCr model to linear RGB model, with 4:4:4 upsampling
 *
 * If the destination image contains an alpha channel, each component will be set to 255
 *
 * If the source image color space is Bt601, the following formula is applied:
 *
 * |[
 * r = 1.164 * (y - 16) + 1.596 * (cr - 128)
 * g = 1.164 * (y - 16) - 0.813 * (cr - 128) - 0.392 * (cb - 128)
 * b = 1.164 * (y - 16) + 2.017 * (cb - 128)
 * ]|
 *
 * If the source image color space is Bt709, the following formula is applied:
 *
 * |[
 * r = 1.164 * (y - 16) + 1.793 * (cr - 128)
 * g = 1.164 * (y - 16) - 0.534 * (cr - 128) - 0.213 * (cb - 128)
 * b = 1.164 * (y - 16) + 2.115 * (cb - 128)
 * ]|
 *
 * If the source image color space is Bt601FR, the following formula is applied:
 *
 * |[
 * r = y + 1.402 * (cr - 128)
 * g = y - 0.714 * (cr - 128) - 0.344 * (cb - 128)
 * b = y + 1.772 * (cb - 128)
 * ]|
 *
 * If the source image color space is Bt709FR, the following formula is applied:
 *
 * |[
 * r = y + 1.575 * (cr - 128)
 * g = y - 0.468 * (cr - 128) - 0.187 * (cb - 128)
 * b = y + 1.856 * (cb - 128)
 * ]|
 *
 * # Algorithm 3 # {#algo-3}
 *
 * Conversion from RGB to BGRA
 *
 * # Algorithm 4 # {#algo-4}
 *
 * Conversion from BGRA to RGB
 *
 * # Algorithm 5 # {#algo-5}
 *
 * Conversion from BGR to RGB
 */
DcpResult           dcp_convert_image           (uint32_t               width,
                                                 uint32_t               height,
                                                 const DcpImageFormat  *src_format,
                                                 const size_t          *src_strides,
                                                 const uint8_t * const *src_buffers,
                                                 const DcpImageFormat  *dst_format,
                                                 const size_t          *dst_strides,
                                                 uint8_t * const       *dst_buffers,
                                                 DcpErrorKind          *error);

#ifdef __cplusplus
}
#endif

/* ex:set ts=4 et: */