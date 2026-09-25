// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
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
