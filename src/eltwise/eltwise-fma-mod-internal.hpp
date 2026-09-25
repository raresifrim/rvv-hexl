// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

/// @brief Native: result[i] = (arg1[i] * arg2 + arg3[i]) mod modulus
/// (arg3 == nullptr: no addition). All inputs in [0, InputModFactor * q),
/// output in [0, q). InputModFactor is 1, 2, 4 or 8.
template <int InputModFactor>
void EltwiseFMAModNative(uint64_t* result, const uint64_t* arg1, uint64_t arg2,
                         const uint64_t* arg3, uint64_t n, uint64_t modulus);

#ifdef HEXL_HAS_RVV
/// @brief RVV, 32-bit lanes (InputModFactor * modulus < 2^32 and
/// modulus < kMaxModulusRVV32).
template <int InputModFactor>
void EltwiseFMAModRVV32(uint64_t* result, const uint64_t* arg1, uint64_t arg2,
                        const uint64_t* arg3, uint64_t n, uint64_t modulus);

/// @brief RVV, 64-bit lanes.
template <int InputModFactor>
void EltwiseFMAModRVV64(uint64_t* result, const uint64_t* arg1, uint64_t arg2,
                        const uint64_t* arg3, uint64_t n, uint64_t modulus);
#endif

}  // namespace hexl
}  // namespace intel
