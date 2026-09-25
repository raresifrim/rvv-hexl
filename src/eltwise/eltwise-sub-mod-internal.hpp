// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Internal kernels behind EltwiseAddMod/EltwiseSubMod-style entry points.
// Native = portable scalar C++ (also the "no RVV" reference path).
// RVV    = hand-written RVV 1.0 kernels (src/eltwise/eltwise-sub-mod-rvv.cpp).

#pragma once

#include <stdint.h>

#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

/// @brief Native vector-vector: result[i] = (operand1[i]  operand2[i]) mod modulus
void EltwiseSubModNative(uint64_t* result, const uint64_t* operand1,
                          const uint64_t* operand2, uint64_t n,
                          uint64_t modulus);

/// @brief Native vector-scalar: result[i] = (operand1[i]  operand2) mod modulus
void EltwiseSubModNative(uint64_t* result, const uint64_t* operand1,
                          uint64_t operand2, uint64_t n, uint64_t modulus);

#ifdef HEXL_HAS_RVV
/// @brief RVV vector-vector variant (same contract as the native one)
void EltwiseSubModRVV(uint64_t* result, const uint64_t* operand1,
                       const uint64_t* operand2, uint64_t n, uint64_t modulus);

/// @brief RVV vector-scalar variant (same contract as the native one)
void EltwiseSubModRVV(uint64_t* result, const uint64_t* operand1,
                       uint64_t operand2, uint64_t n, uint64_t modulus);
#endif

}  // namespace hexl
}  // namespace intel
