// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

/// @brief Native: result[i] = (operand1[i] * operand2[i]) mod modulus, with
/// inputs in [0, InputModFactor * modulus) and output in [0, modulus).
/// InputModFactor is 1, 2 or 4.
template <int InputModFactor>
void EltwiseMultModNative(uint64_t* result, const uint64_t* operand1,
                          const uint64_t* operand2, uint64_t n,
                          uint64_t modulus);

#ifdef HEXL_HAS_RVV
/// @brief RVV, 32-bit lanes. Selected when modulus < kMaxModulusRVV32
/// (so every input fits in 32 bits for InputModFactor <= 4).
template <int InputModFactor>
void EltwiseMultModRVV32(uint64_t* result, const uint64_t* operand1,
                         const uint64_t* operand2, uint64_t n,
                         uint64_t modulus);

/// @brief RVV, 64-bit lanes. Every other modulus (BFV's 60-bit primes).
template <int InputModFactor>
void EltwiseMultModRVV64(uint64_t* result, const uint64_t* operand1,
                         const uint64_t* operand2, uint64_t n,
                         uint64_t modulus);
#endif

}  // namespace hexl
}  // namespace intel
