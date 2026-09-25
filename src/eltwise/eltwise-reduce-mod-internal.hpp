// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

/// @brief Native: result[i] = operand[i] reduced into [0, output_mod_factor*q).
/// input_mod_factor is modulus (arbitrary 64-bit input), 2 or 4;
/// output_mod_factor is 1 or 2; the two are never equal here (the public
/// entry point handles the "copy" case).
void EltwiseReduceModNative(uint64_t* result, const uint64_t* operand,
                            uint64_t n, uint64_t modulus,
                            uint64_t input_mod_factor,
                            uint64_t output_mod_factor);

#ifdef HEXL_HAS_RVV
/// @brief RVV variant, same contract.
void EltwiseReduceModRVV(uint64_t* result, const uint64_t* operand, uint64_t n,
                         uint64_t modulus, uint64_t input_mod_factor,
                         uint64_t output_mod_factor);
#endif

}  // namespace hexl
}  // namespace intel
