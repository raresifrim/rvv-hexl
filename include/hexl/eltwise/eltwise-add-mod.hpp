// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Public API: signature identical to Intel HEXL v1.2.6.
// Implementation: src/eltwise/eltwise-add-mod.cpp (dispatch + native), eltwise-add-mod-rvv.cpp (RVV).
// OpenFHE uses this for NativeVector::ModAdd{,Eq} and for DCRTPoly '+'.

#pragma once

#include <stdint.h>

namespace intel {
namespace hexl {

/// @brief Adds two vectors elementwise with modular reduction
/// @param[out] result Stores result. May alias operand1 (OpenFHE does this).
/// @param[in] operand1 Vector of elements to add. Each element must be less
/// than the modulus
/// @param[in] operand2 Vector of elements to add. Each element must be less
/// than the modulus
/// @param[in] n Number of elements in each vector
/// @param[in] modulus Modulus with which to perform modular reduction. Must be
/// in the range \f$[2, 2^{63} - 1]\f$
/// @details Computes \f$ result[i] = (operand1[i] + operand2[i]) \mod modulus
/// \f$ for \f$ i=0, ..., n-1\f$.
void EltwiseAddMod(uint64_t* result, const uint64_t* operand1,
                   const uint64_t* operand2, uint64_t n, uint64_t modulus);

/// @brief Adds a vector and scalar elementwise with modular reduction
/// @param[out] result Stores result. May alias operand1.
/// @param[in] operand1 Vector of elements to add. Each element must be less
/// than the modulus
/// @param[in] operand2 Scalar to add. Must be less than the modulus
/// @param[in] n Number of elements in each vector
/// @param[in] modulus Modulus with which to perform modular reduction. Must be
/// in the range \f$[2, 2^{63} - 1]\f$
/// @details Computes \f$ result[i] = (operand1[i] + operand2) \mod modulus
/// \f$ for \f$ i=0, ..., n-1\f$.
void EltwiseAddMod(uint64_t* result, const uint64_t* operand1,
                   uint64_t operand2, uint64_t n, uint64_t modulus);


// ---- rvv-hexl extension: 32-bit storage (OpenFHE NATIVE_SIZE=32) -----------
// Same contract as above with uint32_t data; scalars stay uint64_t. All values
// (inputs and outputs, including the "lazy" ranges) must fit in 32 bits.
// q < 2^30 runs the RVV e32 kernels without any 64<->32 conversion; larger q
// falls back to the native kernels. Not part of upstream Intel HEXL: guard uses
// with #ifdef HEXL_RVV_HAS_32BIT_API.
void EltwiseAddMod(uint32_t* result, const uint32_t* operand1,
                   const uint32_t* operand2, uint64_t n, uint64_t modulus);
void EltwiseAddMod(uint32_t* result, const uint32_t* operand1,
                   uint64_t operand2, uint64_t n, uint64_t modulus);

}  // namespace hexl
}  // namespace intel
