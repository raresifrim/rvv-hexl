// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "hexl/util/defines.hpp"

// 128-bit integers. RV64 GCC/Clang lower a 64x64->128 multiply to mul + mulhu,
// so these are the scalar building block of every modular multiplication.
// (Declared at global scope exactly like upstream; OpenFHE relies on that.)
#if defined(HEXL_USE_GNU) || defined(HEXL_USE_CLANG)
__extension__ typedef __int128 int128_t;
__extension__ typedef unsigned __int128 uint128_t;
#endif
