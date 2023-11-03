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

/// An enumeration of supported color models.
///
/// It includes:
/// * Colorimetry
/// * Gamma
/// * Range (headroom / footroom)
/// * Primaries
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub enum ColorSpace {
    /// Gamma-corrected R'G'B'.
    /// The gamma is the one defined in ITU-R Recommendation BT.709-6 page 3, item 1.2
    /// The relationship between gamma-corrected component (C') and its linear value (C)
    /// is the following one:
    /// C' = 4.5 * C                      if C < 0.018,
    ///      1.099 * pow(C, 0.45) - 0.099 otherwise.
    Rgb,
    /// YCbCr, ITU-R Recommendation BT.601 (standard video system)
    Bt601,
    /// YCbCr, ITU-R Recommendation BT.709 (CSC systems)
    Bt709,
    /// YCbCr, BT.601 (full range)
    Bt601FR,
    /// YCbCr, BT.709 (full range)
    Bt709FR,
}

#[cfg(not(tarpaulin_include))]
impl std::fmt::Display for ColorSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ColorSpace::Rgb => write!(f, "rgb"),
            ColorSpace::Bt601 => write!(f, "bt-601"),
            ColorSpace::Bt709 => write!(f, "bt-709"),
            ColorSpace::Bt601FR => write!(f, "bt-601-fr"),
            ColorSpace::Bt709FR => write!(f, "bt-709-fr"),
        }
    }
}
