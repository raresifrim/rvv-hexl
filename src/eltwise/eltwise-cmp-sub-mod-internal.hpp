// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "hexl/util/util.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

/// @brief Native: r = operand1[i] mod modulus;
///   result[i] = cmp(operand1[i], bound) ? (r - diff) mod modulus : r
///   (the comparison uses the ORIGINAL value, before reduction)
void EltwiseCmpSubModNative(uint64_t* result, const uint64_t* operand1,
                        uint64_t n, uint64_t modulus, CMPINT cmp,
                        uint64_t bound, uint64_t diff);

#ifdef HEXL_HAS_RVV
/// @brief RVV variant, same contract.
void EltwiseCmpSubModRVV(uint64_t* result, const uint64_t* operand1,
                        uint64_t n, uint64_t modulus, CMPINT cmp,
                        uint64_t bound, uint64_t diff);
#endif

}  // namespace hexl
}  // namespace intel
