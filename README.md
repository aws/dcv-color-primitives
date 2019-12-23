# DCV Color Primitives - dcp

[![Build Status](https://travis-ci.org/aws/dcv-color-primitives.png?branch=master)](https://travis-ci.org/aws/dcv-color-primitives)

DCV Color Primitives is a library to perform image color model conversion.

## Design guidelines

* Aware of the underlying hardware and supplemental cpu extension sets (up to avx2)
* Support data coming from a single buffer or coming from multiple image planes
* Support non-tightly packed data
* Support images greater than 4GB (64 bit)

## Color space conversion

The library is currenty able to convert the following pixel formats:

* BGRA8
* RGBA8
* BGR8

To the following destination pixel formats:

* NV12 (single plane or biplanar 4:2:0)

Additionally, it is possible to convert the following pixel formats:

* NV12 (single plane or biplanar 4:2:0)

To the following destination pixel formats:

* BGRA8 (alpha is set to 255)

### Color models

The supported color models are:

* YCbCr, ITU-R Recommendation BT.601 (standard video system)
* YCbCr, ITU-R Recommendation BT.709 (CSC systems)

## Requirements

* Rust stable (at least 1.39)
* Python3 and Fractions module (needed only if you want to execute the benchmark)

### Windows

* Install rustup: https://www.rust-lang.org/tools/install
* If you want to execute the benchmark:
    * Install Python 3: https://www.python.org/downloads/
    * Install fraction module: `pip install Fraction`

### Linux

* Install rustup (see https://forge.rust-lang.org/infra/other-installation-methods.html)
    ```
    curl https://sh.rustup.rs -sSf | sh
    ```
* If you want to execute the benchmark:
    * Install Python 3 (example for Ubuntu): `apt install python3`
    * Install fraction module: `pip install Fraction`

You may require administrative privileges.

## Building

Open a terminal inside the library root directory.

To build for debug experience:
```
cargo build
```

To build an optimized library:
```
cargo build --release
```

Run unit tests:
```
cargo test
```

Run benchmark:
```
python benches/geninput.py
cargo bench
```

## Usage

### Initialize the library

This function has to be called before any other library function call:

```rust
use dcv_color_primitives as dcp;

fn main() {
    dcp::initialize();
}
```

### Image conversion

Convert an image from bgra to nv12 (single plane) format containing yuv in BT601:

```rust
use dcv_color_primitives as dcp;
use dcp::{convert_image, ColorSpace, ImageFormat, PixelFormat};

fn main() {
    dcp::initialize();

    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 480;

    let src_buffers: &[&[u8]] = &[&[0u8; 4 * (WIDTH as usize) * (HEIGHT as usize)]];
    let dst_buffers: &mut [&mut [u8]] =
        &mut [&mut [0u8; 3 * (WIDTH as usize) * (HEIGHT as usize) / 2]];

    let src_format = ImageFormat {
        pixel_format: PixelFormat::Bgra,
        color_space: ColorSpace::Lrgb,
        num_planes: 1,
    };

    let dst_format = ImageFormat {
        pixel_format: PixelFormat::Nv12,
        color_space: ColorSpace::Bt601,
        num_planes: 1,
    };

    convert_image(
        WIDTH,
        HEIGHT,
        &src_format,
        None,
        src_buffers,
        &dst_format,
        None,
        dst_buffers,
    );
}
```

### Error Handling

The library functions return a `Result` describing the operation outcome:

| Result                             | Description                                                           |
| ---------------------------------- | --------------------------------------------------------------------- |
| `Ok(())`                           | The operation succeeded                                               |
| `Err(ErrorKind::NotInitialized)`   | The library is not initialized                                        |
| `Err(ErrorKind::InvalidValue)`     | One or more parameters have invalid values for the called function    |
| `Err(ErrorKind::InvalidOperation)` | The combination of parameters is unsupported for the called function  |
| `Err(ErrorKind::NotEnoughData)`    | One or more buffers are not correctly sized                           |

In the following example, `result` will match `Err(ErrorKind::InvalidValue)`, because `ColorSpace::Bt709`
color space is not compatible with `PixelFormat::Bgra`:

```rust
use dcv_color_primitives as dcp;
use dcp::{convert_image, ColorSpace, ErrorKind, ImageFormat, PixelFormat};

fn main() {
    dcp::initialize();

    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 480;

    let src_buffers: &[&[u8]] = &[&[0u8; 4 * (WIDTH as usize) * (HEIGHT as usize)]];
    let dst_buffers: &mut [&mut [u8]] =
        &mut [&mut [0u8; 3 * (WIDTH as usize) * (HEIGHT as usize) / 2]];

    let src_format = ImageFormat {
        pixel_format: PixelFormat::Bgra,
        color_space: ColorSpace::Bt709,
        num_planes: 1,
    };

    let dst_format = ImageFormat {
        pixel_format: PixelFormat::Nv12,
        color_space: ColorSpace::Bt601,
        num_planes: 1,
    };

    let status = convert_image(
        WIDTH,
        HEIGHT,
        &src_format,
        None,
        src_buffers,
        &dst_format,
        None,
        dst_buffers,
    );

    match status {
        Err(ErrorKind::InvalidValue) => (),
        _ => panic!("Expected ErrorKind::InvalidValue"),
    }
}
```

Even better, you might want to propagate errors to the caller function or mix with some other error types:
```rust
use dcv_color_primitives as dcp;
use dcp::{convert_image, ColorSpace, ImageFormat, PixelFormat};
use std::error;

fn main() -> Result<(), Box<dyn error::Error>> {
    dcp::initialize();

    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 480;

    let src_buffers: &[&[u8]] = &[&[0u8; 4 * (WIDTH as usize) * (HEIGHT as usize)]];
    let dst_buffers: &mut [&mut [u8]] =
        &mut [&mut [0u8; 3 * (WIDTH as usize) * (HEIGHT as usize) / 2]];

    let src_format = ImageFormat {
        pixel_format: PixelFormat::Bgra,
        color_space: ColorSpace::Bt709,
        num_planes: 1,
    };

    let dst_format = ImageFormat {
        pixel_format: PixelFormat::Nv12,
        color_space: ColorSpace::Bt601,
        num_planes: 1,
    };

    convert_image(
        WIDTH,
        HEIGHT,
        &src_format,
        None,
        src_buffers,
        &dst_format,
        None,
        dst_buffers,
    )?;

    Ok(())
}
```

### Buffer size computation

So far, buffers were sized taking into account the image pixel format and dimensions; However,
you can use a function to compute how many bytes are needed to store an image of a given format
and size:

```rust
use dcv_color_primitives as dcp;
use dcp::{get_buffers_size, ColorSpace, ImageFormat, PixelFormat};
use std::error;

fn main() -> Result<(), Box<dyn error::Error>> {
    dcp::initialize();

    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 480;
    const NUM_PLANES: u32 = 1;

    let format = ImageFormat {
        pixel_format: PixelFormat::Bgra,
        color_space: ColorSpace::Lrgb,
        num_planes: NUM_PLANES,
    };

    let sizes: &mut [usize] = &mut [0usize; NUM_PLANES as usize];
    get_buffers_size(WIDTH, HEIGHT, &format, None, sizes)?;

    let buffer: Vec<_> = vec![0u8; sizes[0]];

    // Do something with buffer
    // --snip--

    Ok(())
}
```

### Image planes

If your data is scattered in multiple buffers that are not necessarily contiguous, you can provide image planes:

```rust
use dcv_color_primitives as dcp;
use dcp::{convert_image, get_buffers_size, ColorSpace, ImageFormat, PixelFormat};
use std::error;

fn main() -> Result<(), Box<dyn error::Error>> {
    dcp::initialize();

    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 480;
    const NUM_SRC_PLANES: u32 = 2;
    const NUM_DST_PLANES: u32 = 1;

    let src_format = ImageFormat {
        pixel_format: PixelFormat::Nv12,
        color_space: ColorSpace::Bt709,
        num_planes: NUM_SRC_PLANES,
    };

    let src_sizes: &mut [usize] = &mut [0usize; NUM_SRC_PLANES as usize];
    get_buffers_size(WIDTH, HEIGHT, &src_format, None, src_sizes)?;

    let src_y: Vec<_> = vec![0u8; src_sizes[0]];
    let src_uv: Vec<_> = vec![0u8; src_sizes[1]];
    let src_buffers: &[&[u8]] = &[&src_y[..], &src_uv[..]];

    let dst_format = ImageFormat {
        pixel_format: PixelFormat::Bgra,
        color_space: ColorSpace::Lrgb,
        num_planes: NUM_DST_PLANES,
    };

    let dst_sizes: &mut [usize] = &mut [0usize; NUM_DST_PLANES as usize];
    get_buffers_size(WIDTH, HEIGHT, &dst_format, None, dst_sizes)?;

    let mut dst_rgba: Vec<_> = vec![0u8; dst_sizes[0]];
    let dst_buffers: &mut [&mut [u8]] = &mut [&mut dst_rgba[..]];

    convert_image(
        WIDTH,
        HEIGHT,
        &src_format,
        None,
        src_buffers,
        &dst_format,
        None,
        dst_buffers,
    )?;

    Ok(())
}
```

### Stride support

To take into account data which is not tightly packed, you can provide image strides:

```rust
use dcv_color_primitives as dcp;
use dcp::{convert_image, get_buffers_size, ColorSpace, ImageFormat, PixelFormat};
use std::error;

fn main() -> Result<(), Box<dyn error::Error>> {
    dcp::initialize();

    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 480;
    const NUM_SRC_PLANES: u32 = 1;
    const NUM_DST_PLANES: u32 = 2;
    const RGB_STRIDE: usize = 4 * (((3 * (WIDTH as usize)) + 3) / 4);

    let src_format = ImageFormat {
        pixel_format: PixelFormat::Bgr,
        color_space: ColorSpace::Lrgb,
        num_planes: NUM_SRC_PLANES,
    };

    let src_strides: &[usize] = &[RGB_STRIDE];

    let src_sizes: &mut [usize] = &mut [0usize; NUM_SRC_PLANES as usize];
    get_buffers_size(WIDTH, HEIGHT, &src_format, Some(src_strides), src_sizes)?;

    let src_rgba: Vec<_> = vec![0u8; src_sizes[0]];
    let src_buffers: &[&[u8]] = &[&src_rgba[..]];

    let dst_format = ImageFormat {
        pixel_format: PixelFormat::Nv12,
        color_space: ColorSpace::Bt709,
        num_planes: NUM_DST_PLANES,
    };

    let dst_sizes: &mut [usize] = &mut [0usize; NUM_DST_PLANES as usize];
    get_buffers_size(WIDTH, HEIGHT, &dst_format, None, dst_sizes)?;

    let mut dst_y: Vec<_> = vec![0u8; dst_sizes[0]];
    let mut dst_uv: Vec<_> = vec![0u8; dst_sizes[1]];
    let dst_buffers: &mut [&mut [u8]] = &mut [&mut dst_y[..], &mut dst_uv[..]];

    convert_image(
        WIDTH,
        HEIGHT,
        &src_format,
        Some(src_strides),
        src_buffers,
        &dst_format,
        None,
        dst_buffers,
    )?;

    Ok(())
}
```

See documentation for further information.

## C bindings

DCV Color Primitives provides C bindings. A static library will be automatically generated for the 
default build.

API is slighter different than rust one. Check dcv_color_primitives.h for examples and further information.

There are also some unit tests written in C, to add some coverage also for the bindings. If you
want to build those, you need meson build system.

* **Windows**  
  Visual Studio is required. At least the following packages are required:
  * MSBuild
  * MSVC - C++ build tools
  * Windows 10 SDK

  Install meson, you can choose one of the following methods:
  1. Using meson msi installer
   * Download from https://github.com/mesonbuild/meson/releases
   * Install both Meson and Ninja
  2. Install meson through pip
   * Download and install python3: https://www.python.org/downloads/
   * Install meson and ninja:
     ```
     pip install meson ninja
     ```

  All build commands have to be issued from Native Tools Command Prompt for VS (x86 or x64 depending on what platform you want to build)
* **Linux**  
  The following example is for Ubuntu:

  ```
  #install python3
  apt install python3

  #install meson. See https://mesonbuild.com/Getting-meson.html for details or if you want to install through pip.
  apt install meson

  #install ninja
  apt install ninja-build
  ```

  You may require administrative privileges.
* **Build and run**  
  Move inside the library root directory:
  ```
  cd `dcv_color_primitives_root_dir`
  ```

  Then:
  ```
  meson --buildtype release c_bindings_test
  cd c_bindings_test
  meson test -t 10
  ```

  A timeout scale factor of 10 is required because some tests take longer than default
  30 seconds to complete.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
  