// Copyright (C) 2020 Intel Corporation
// Modifications Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// SPDX-License-Identifier: Apache-2.0
//
// Upstream Intel HEXL generates this header from defines.hpp.in at CMake time.
// rvv-hexl builds with plain make, so the compiler is detected here instead.
// HEXL_DEBUG is NOT baked in: the library gets it from `make BUILD=debug`
// (-DHEXL_DEBUG), which turns every HEXL_CHECK into a throwing argument check.

#pragma once

#if defined(_MSC_VER) && !defined(__clang__)
#error "rvv-hexl: MSVC is not supported (the port targets GCC/Clang on Linux)"
#elif defined(__clang__)
#define HEXL_USE_CLANG
#elif defined(__GNUC__)
#define HEXL_USE_GNU
#else
#error "rvv-hexl: unsupported compiler"
#endif

#define HEXL_UNUSED(x) (void)(x)

// ---- rvv-hexl additions (not present upstream) ---------------------------
// Lets consumers (benches, OpenFHE patches) tell the port apart from Intel HEXL.
#define HEXL_RVV_PORT 1
#define HEXL_RVV_PORT_VERSION_MAJOR 0
#define HEXL_RVV_PORT_VERSION_MINOR 1
// The upstream API version this port is source-compatible with. OpenFHE's
// openfhe-hexl v1.5.1.0 asks for find_package(HEXL 1.2.6).
#define HEXL_API_VERSION_STRING "1.2.6"
// uint32_t overloads of the NTT and eltwise functions, for OpenFHE built with
// NATIVE_SIZE=32 (see third_party/patches/openfhe-hexl-wordsize.py).
#define HEXL_RVV_HAS_32BIT_API 1
// EltwiseMultAddMod: (a * b + c) mod q on three vectors, both storage words
// (include/hexl/eltwise/eltwise-mult-add-mod.hpp).
#define HEXL_RVV_HAS_MULT_ADD_MOD 1
// SignedDigitDecompose: OpenFHE binfhe's signed gadget decomposition, both storage
// words (include/hexl/eltwise/eltwise-signed-digit-decompose.hpp).
#define HEXL_RVV_HAS_SIGNED_DIGIT_DECOMPOSE 1
// BaseConvert: the approximate RNS base conversion of OpenFHE's key switching
// (include/hexl/eltwise/eltwise-base-convert.hpp), 64-bit storage.
#define HEXL_RVV_HAS_BASE_CONVERT 1
// EltwiseExternalProductMod: the external product of OpenFHE binfhe's blind rotation
// in one call, both storage words (include/hexl/eltwise/eltwise-external-product.hpp).
#define HEXL_RVV_HAS_EXTERNAL_PRODUCT 1
