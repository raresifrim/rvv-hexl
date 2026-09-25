// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "hexl/util/util.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

/// @brief Native: result[i] = cmp(operand1[i], bound) ? operand1[i] + diff : operand1[i]
///   (plain wrapping 64-bit add, no reduction)
void EltwiseCmpAddNative(uint64_t* result, const uint64_t* operand1,
                        uint64_t n, CMPINT cmp, uint64_t bound,
                        uint64_t diff);

#ifdef HEXL_HAS_RVV
/// @brief RVV variant, same contract.
void EltwiseCmpAddRVV(uint64_t* result, const uint64_t* operand1,
                        uint64_t n, CMPINT cmp, uint64_t bound,
                        uint64_t diff);
#endif

}  // namespace hexl
}  // namespace intel
