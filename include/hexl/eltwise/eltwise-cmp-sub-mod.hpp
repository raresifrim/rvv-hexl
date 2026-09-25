// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Public API: signature identical to Intel HEXL v1.2.6.
// Implementation: src/eltwise/eltwise-cmp-sub-mod.cpp (dispatch + native), eltwise-cmp-sub-mod-rvv.cpp.
// OpenFHE uses this in SwitchModulus() when the new modulus is smaller.

#pragma once

#include <stdint.h>

#include "hexl/util/util.hpp"

namespace intel {
namespace hexl {

/// @brief Computes element-wise conditional modular subtraction.
/// @param[out] result Stores the result. May alias operand1.
/// @param[in] operand1 Vector of elements to compare (any 64-bit value)
/// @param[in] n Number of elements in \p operand1
/// @param[in] modulus Modulus to reduce by
/// @param[in] cmp Comparison function
/// @param[in] bound Scalar to compare against
/// @param[in] diff Scalar to subtract by. Must be in (0, modulus).
/// @details Computes, for all i:
///   r = operand1[i] mod modulus;
///   result[i] = cmp(operand1[i], bound) ? (r - diff) mod modulus : r
/// Note the comparison is made on the ORIGINAL (unreduced) value.
void EltwiseCmpSubMod(uint64_t* result, const uint64_t* operand1, uint64_t n,
                      uint64_t modulus, CMPINT cmp, uint64_t bound,
                      uint64_t diff);

}  // namespace hexl
}  // namespace intel
