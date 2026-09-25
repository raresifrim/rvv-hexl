// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Free-function NTT kernels behind NTT::ComputeForward / ComputeInverse.
//
// Conventions shared by every kernel below (they are what OpenFHE expects):
//   * negacyclic: Z_q[X]/(X^N + 1), w = primitive 2N-th root of unity
//   * forward:  natural order in  -> bit-reversed order out
//               result[i] = sum_j operand[j] * w^((2*rev(i)+1)*j)
//   * inverse:  bit-reversed in   -> natural order out, scaled by N^{-1}
//   * result may equal operand (in place); if not, operand must be untouched
//   * input_mod_factor / output_mod_factor: see NTT::ComputeForward/Inverse.
//     Harvey's lazy butterflies keep values in [0, 4q) between stages, which
//     is what makes the 1/2/4 factors cheap.
//   * no state: these run concurrently from many OpenMP threads.
//   * Word is the storage type: uint64_t (upstream API, OpenFHE NATIVE_SIZE=64)
//     or uint32_t (rvv-hexl extension, NATIVE_SIZE=32). The twiddle tables
//     are the same for both. Specialise per type inside one body with
//       if constexpr (std::is_same_v<Word, uint32_t>) { ... }

#pragma once

#include <stdint.h>

#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

// ---------------------------------------------------------------------------
// Native (portable scalar) path: src/ntt/ntt-radix-2.cpp        TODO(port)
// ---------------------------------------------------------------------------

/// @brief Radix-2 Cooley-Tukey forward NTT (Harvey butterflies).
/// @param root_of_unity_powers  NTT::GetRootOfUnityPowers() (bit-reversed)
/// @param precon_root_of_unity_powers NTT::GetPrecon64RootOfUnityPowers()
template <typename Word>
void ForwardTransformToBitReverseRadix2(
    Word* result, const Word* operand, uint64_t n, uint64_t modulus,
    const uint64_t* root_of_unity_powers,
    const uint64_t* precon_root_of_unity_powers, uint64_t input_mod_factor = 1,
    uint64_t output_mod_factor = 1);

/// @brief Radix-2 Gentleman-Sande inverse NTT (Harvey butterflies), including
/// the final multiplication by N^{-1}.
/// @param inv_root_of_unity_powers NTT::GetInvRootOfUnityPowers()
/// @param precon_inv_root_of_unity_powers NTT::GetPrecon64InvRootOfUnityPowers()
template <typename Word>
void InverseTransformFromBitReverseRadix2(
    Word* result, const Word* operand, uint64_t n, uint64_t modulus,
    const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers,
    uint64_t input_mod_factor = 1, uint64_t output_mod_factor = 1);

/// @brief Textbook in-place forward NTT (plain MultiplyMod, fully reduced after
/// every butterfly). Slow on purpose: a debugging aid to diff the fast kernels
/// against, stage by stage. Same output order as the forward kernels.
void ReferenceForwardTransformToBitReverse(uint64_t* operand, uint64_t n,
                                           uint64_t modulus,
                                           const uint64_t* root_of_unity_powers);

/// @brief Textbook in-place inverse NTT, including the N^{-1} scaling.
void ReferenceInverseTransformFromBitReverse(
    uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* inv_root_of_unity_powers);

// ---------------------------------------------------------------------------
// RVV path: src/ntt/ntt-rvv.cpp                               TODO(port-rvv)
// ---------------------------------------------------------------------------
#ifdef HEXL_HAS_RVV

/// @brief RVV forward NTT, 32-bit lanes. Used when modulus <
/// NTT::s_max_fwd_32_modulus (2^30): the binfhe/TFHE moduli (~2^27-2^28),
/// for either storage type (rvv::Load32 / rvv::Store32 pick the conversion).
/// @param w, w_precon NTT::GetRVV32RootOfUnityPowers() /
/// GetRVV32PreconRootOfUnityPowers(); layout decided by YOU in
/// NTT::ComputeRootOfUnityPowers.
template <typename Word>
void ForwardTransformToBitReverseRVV32(Word* result,
                                       const Word* operand, uint64_t n,
                                       uint64_t modulus, const uint32_t* w,
                                       const uint32_t* w_precon,
                                       uint64_t input_mod_factor,
                                       uint64_t output_mod_factor);

/// @brief RVV inverse NTT, 32-bit lanes (incl. N^{-1} scaling).
template <typename Word>
void InverseTransformFromBitReverseRVV32(Word* result,
                                         const Word* operand, uint64_t n,
                                         uint64_t modulus, const uint32_t* w_inv,
                                         const uint32_t* w_inv_precon,
                                         uint64_t input_mod_factor,
                                         uint64_t output_mod_factor);

/// @brief RVV forward NTT, 64-bit lanes. Every modulus >= 2^30 (the 49/60-bit
/// BFV/CKKS primes). Uses the same tables as the native path. 64-bit storage
/// only (32-bit storage with q >= 2^30 goes to the native path).
void ForwardTransformToBitReverseRVV64(
    uint64_t* result, const uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* root_of_unity_powers,
    const uint64_t* precon_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor);

/// @brief RVV inverse NTT, 64-bit lanes (incl. N^{-1} scaling).
void InverseTransformFromBitReverseRVV64(
    uint64_t* result, const uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor);

#endif  // HEXL_HAS_RVV

}  // namespace hexl
}  // namespace intel
