// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Public API: signature identical to Intel HEXL v1.2.6.
// Implementation: src/eltwise/eltwise-mult-mod.cpp (dispatch + native), eltwise-mult-mod-rvv.cpp (RVV).
// OpenFHE uses this for NativeVector::ModMul{,Eq} and DCRTPoly '*' (the
// Hadamard product between NTT-domain polynomials): the #2 hotspot after the NTT.

#pragma once

#include <stdint.h>

namespace intel {
namespace hexl {

/// @brief Multiplies two vectors elementwise with modular reduction
/// @param[out] result Result of element-wise multiplication. May alias operand1.
/// @param[in] operand1 Vector of elements to multiply. Each element must be
/// less than input_mod_factor * modulus.
/// @param[in] operand2 Vector of elements to multiply. Each element must be
/// less than input_mod_factor * modulus.
/// @param[in] n Number of elements in each vector
/// @param[in] modulus Modulus with which to perform modular reduction.
/// Requires input_mod_factor * modulus < 2^63.
/// @param[in] input_mod_factor Assumes input elements are in [0,
/// input_mod_factor * p) Must be 1, 2 or 4.
/// @details Computes \p result[i] = (\p operand1[i] * \p operand2[i]) mod \p
/// modulus for i=0, ..., \p n - 1. Output is fully reduced, in [0, modulus).
void EltwiseMultMod(uint64_t* result, const uint64_t* operand1,
		const uint64_t* operand2, uint64_t n, uint64_t modulus,
		uint64_t input_mod_factor);


// ---- rvv-hexl extension: 32-bit storage (OpenFHE NATIVE_SIZE=32) -----------
// Same contract as above with uint32_t data; scalars stay uint64_t. All values
// (inputs and outputs, including the "lazy" ranges) must fit in 32 bits.
// q < 2^30 runs the RVV e32 kernels without any 64<->32 conversion; larger q
// falls back to the native kernels. Not part of upstream Intel HEXL: guard uses
// with #ifdef HEXL_RVV_HAS_32BIT_API.
void EltwiseMultMod(uint32_t* result, const uint32_t* operand1,
		const uint32_t* operand2, uint64_t n, uint64_t modulus,
		uint64_t input_mod_factor);

}  // namespace hexl
}  // namespace intel
