// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Public API: signature identical to Intel HEXL v1.2.6.
// Implementation: src/eltwise/eltwise-cmp-add.cpp (dispatch + native), eltwise-cmp-add-rvv.cpp (RVV).
// OpenFHE uses this (with EltwiseCmpSubMod) to switch a vector to a new
// modulus while keeping the centred representative: SwitchModulus().

#pragma once

#include <stdint.h>

#include "hexl/util/util.hpp"

namespace intel {
namespace hexl {

/// @brief Computes element-wise conditional addition.
/// @param[out] result Stores the result. May alias operand1.
/// @param[in] operand1 Vector of elements to compare
/// @param[in] n Number of elements in \p operand1
/// @param[in] cmp Comparison operation
/// @param[in] bound Scalar to compare against
/// @param[in] diff Scalar to conditionally add. Must be non-zero.
/// @details Computes result[i] = cmp(operand1[i], bound) ? operand1[i] +
/// diff : operand1[i] for all \f$i=0, ..., n-1\f$. The addition is a plain
/// (wrapping) 64-bit add, no modular reduction.
void EltwiseCmpAdd(uint64_t* result, const uint64_t* operand1, uint64_t n,
                   CMPINT cmp, uint64_t bound, uint64_t diff);

}  // namespace hexl
}  // namespace intel
