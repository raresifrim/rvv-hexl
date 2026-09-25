// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Public API: signature identical to Intel HEXL v1.2.6.
// Implementation: src/eltwise/eltwise-reduce-mod.cpp (dispatch + native), eltwise-reduce-mod-rvv.cpp (RVV).
// OpenFHE uses this in NativeVector::SwitchModulus / Mod (mubintvecnathexl.cpp).

#pragma once

#include <stdint.h>

namespace intel {
namespace hexl {

/// @brief Performs elementwise modular reduction
/// @param[out] result Stores the result. May alias operand.
/// @param[in] operand Data on which to compute the elementwise modular
/// reduction
/// @param[in] n Number of elements in operand
/// @param[in] modulus Modulus with which to perform modular reduction
/// @param[in] input_mod_factor Assumes input elements are in [0,
/// input_mod_factor * p). Must be modulus, 2 or 4. input_mod_factor == modulus
/// is the special "arbitrary 64-bit input" case, reduced with Barrett.
/// If input_mod_factor == output_mod_factor the data is only copied.
/// @param[in] output_mod_factor output elements will be in [0,
/// output_mod_factor * modulus). Must be 1 or 2. Any representative in that
/// range congruent to the input mod p is correct.
void EltwiseReduceMod(uint64_t* result, const uint64_t* operand, uint64_t n,
                      uint64_t modulus, uint64_t input_mod_factor,
                      uint64_t output_mod_factor);


// ---- rvv-hexl extension: 32-bit storage (OpenFHE NATIVE_SIZE=32) -----------
// Same contract as above with uint32_t data; scalars stay uint64_t. All values
// (inputs and outputs, including the "lazy" ranges) must fit in 32 bits.
// q < 2^30 runs the RVV e32 kernels without any 64<->32 conversion; larger q
// falls back to the native kernels. Not part of upstream Intel HEXL: guard uses
// with #ifdef HEXL_RVV_HAS_32BIT_API.
/// input_mod_factor == modulus means "any 32-bit value".
void EltwiseReduceMod(uint32_t* result, const uint32_t* operand, uint64_t n,
                      uint64_t modulus, uint64_t input_mod_factor,
                      uint64_t output_mod_factor);

}  // namespace hexl
}  // namespace intel
