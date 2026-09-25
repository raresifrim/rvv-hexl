// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "hexl/util/defines.hpp"

// 128-bit integers. RV64 GCC/Clang lower a 64x64->128 multiply to mul + mulhu,
// so these are the scalar building block of every modular multiplication.
#if defined(HEXL_USE_GNU) || defined(HEXL_USE_CLANG)

// The library's own names. Unqualified lookup inside intel::hexl finds these
// before any global typedef, so HEXL code always gets a real 128-bit type.
namespace intel {
namespace hexl {
__extension__ typedef __int128 int128_t;
__extension__ typedef unsigned __int128 uint128_t;
}  // namespace hexl
}  // namespace intel

// Upstream also declares both names at global scope. OpenFHE declares the same
// global names in math/hal/basicint.h: identical types at NATIVE_SIZE=64/128,
// but uint64_t/int64_t at NATIVE_SIZE=32, which would be a redefinition error.
// OpenFHE's config_core.h (NATIVEINT) is always included before its HEXL
// headers, so skip ours in exactly that case.
#if !(defined(NATIVEINT) && NATIVEINT == 32)
__extension__ typedef __int128 int128_t;
__extension__ typedef unsigned __int128 uint128_t;
#endif

#endif
