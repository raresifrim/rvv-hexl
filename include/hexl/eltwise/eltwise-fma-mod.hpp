// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Public API: signature identical to Intel HEXL v1.2.6.
// Implementation: src/eltwise/eltwise-fma-mod.cpp (dispatch + native), eltwise-fma-mod-rvv.cpp (RVV).
// OpenFHE uses this for scalar ModMul (arg3 == nullptr) and for the CRT/RNS
// basis-switching loops in DCRTPoly (hexldcrtpoly-impl.h).

#pragma once

#include <stdint.h>

namespace intel {
namespace hexl {

/// @brief Computes fused multiply-add (\p arg1 * \p arg2 + \p arg3) mod \p
/// modulus element-wise, broadcasting scalars to vectors.
/// @param[out] result Stores the result. May alias arg1 and/or arg3.
/// @param[in] arg1 Vector to multiply
/// @param[in] arg2 Scalar to multiply
/// @param[in] arg3 Vector to add. Will not add if \p arg3 == nullptr
/// @param[in] n Number of elements in each vector
/// @param[in] modulus Modulus with which to perform modular reduction. Must be
/// in the range \f$ [2, 2^{61} - 1]\f$
/// @param[in] input_mod_factor Assumes input elements (arg1, arg2 and arg3) are
/// in [0, input_mod_factor * modulus). Must be 1, 2, 4, or 8.
/// @details Output is fully reduced, in [0, modulus).
void EltwiseFMAMod(uint64_t* result, const uint64_t* arg1, uint64_t arg2,
		const uint64_t* arg3, uint64_t n, uint64_t modulus,
		uint64_t input_mod_factor);


// ---- rvv-hexl extension: 32-bit storage (OpenFHE NATIVE_SIZE=32) -----------
// Same contract as above with uint32_t data; scalars stay uint64_t. All values
// (inputs and outputs, including the "lazy" ranges) must fit in 32 bits.
// q < 2^30 runs the RVV e32 kernels without any 64<->32 conversion; larger q
// falls back to the native kernels. Not part of upstream Intel HEXL: guard uses
// with #ifdef HEXL_RVV_HAS_32BIT_API.
void EltwiseFMAMod(uint32_t* result, const uint32_t* arg1, uint64_t arg2,
		const uint32_t* arg3, uint64_t n, uint64_t modulus,
		uint64_t input_mod_factor);

}  // namespace hexl
}  // namespace intel
