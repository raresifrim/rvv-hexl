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
#include <algorithm>
#include "util/cpu-features.hpp"
#include "util/rvv-config.hpp"  // lane types (empty without RVV)

namespace intel {
namespace hexl {

// ---------------------------------------------------------------------------
// Native (portable scalar) path: src/ntt/ntt-radix-2.cpp
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
// RVV path: src/ntt/ntt-rvv.cpp
// ---------------------------------------------------------------------------
#ifdef HEXL_HAS_RVV

// Small-span twiddle tables (src/ntt/ntt-rvv-small-span.hpp).
template <class Strategy, typename E>
struct NttSmallSpanTables;

// Every RVV kernel takes two template parameters on top of the storage word,
// both defaulted from src/util/rvv-config.hpp:
//   V          lane type (SEW + LMUL)
//   SmallSpan  rvv::ntt::PerBlock / FusedSlides / Transposed: how the stages
//              with span t < VLMAX of V run
// and, last, the SmallSpan tables built for this (N, q) and VLMAX of V
// (NTT keeps them for the configured lanes and strategy). Without them, or
// with tables for another VL, every stage runs per block.

/// @brief RVV forward NTT, 32-bit lanes. Used when modulus <
/// NTT::s_max_fwd_32_modulus (2^30): the binfhe/TFHE moduli (~2^27-2^28),
/// for either storage type (rvv::Load32 / rvv::Store32 pick the conversion).
/// @param w, w_precon NTT::GetRVV32RootOfUnityPowers() /
/// GetRVV32PreconRootOfUnityPowers() (forward table, bit-reversed).
/// Lane type V: rvv::cfg::Ntt32 (uint32_t storage) or Ntt32U64 (uint64_t
/// storage, at most u32m4 because of the narrowing load).
template <typename Word,
	 class V = rvv::LaneFor<Word, rvv::cfg::Ntt32U64, rvv::cfg::Ntt32>,
	 class SmallSpan = rvv::cfg::NttSmallSpan32>
		 void ForwardTransformToBitReverseRVV32(Word* result,
				 const Word* operand, uint64_t n,
				 uint64_t modulus, const uint32_t* w,
				 const uint32_t* w_precon,
				 uint64_t input_mod_factor,
				 uint64_t output_mod_factor,
				 const NttSmallSpanTables<SmallSpan, uint32_t>* small_span = nullptr);

/// @brief RVV inverse NTT, 32-bit lanes (incl. N^{-1} scaling).
/// @param w_inv, w_inv_precon NTT::GetRVV32InvRootOfUnityPowers() /
/// GetRVV32PreconInvRootOfUnityPowers() (inverse table, stage order).
template <typename Word,
	 class V = rvv::LaneFor<Word, rvv::cfg::Ntt32U64, rvv::cfg::Ntt32>,
	 class SmallSpan = rvv::cfg::NttSmallSpan32>
		 void InverseTransformFromBitReverseRVV32(Word* result,
				 const Word* operand, uint64_t n,
				 uint64_t modulus, const uint32_t* w_inv,
				 const uint32_t* w_inv_precon,
				 uint64_t input_mod_factor,
				 uint64_t output_mod_factor,
				 const NttSmallSpanTables<SmallSpan, uint32_t>* small_span = nullptr);

/// @brief RVV forward NTT, 64-bit lanes. Every modulus >= 2^30 (the 49/60-bit
/// BFV/CKKS primes). Uses the same tables as the native path. 64-bit storage
/// only (32-bit storage with q >= 2^30 goes to the native path).
/// Lane type V: rvv::cfg::Ntt64.
template <class V = rvv::cfg::Ntt64, class SmallSpan = rvv::cfg::NttSmallSpan64>
void ForwardTransformToBitReverseRVV64(
		uint64_t* result, const uint64_t* operand, uint64_t n, uint64_t modulus,
		const uint64_t* root_of_unity_powers,
		const uint64_t* precon_root_of_unity_powers, uint64_t input_mod_factor,
		uint64_t output_mod_factor,
		const NttSmallSpanTables<SmallSpan, uint64_t>* small_span = nullptr);

/// @brief RVV inverse NTT, 64-bit lanes (incl. N^{-1} scaling).
template <class V = rvv::cfg::Ntt64, class SmallSpan = rvv::cfg::NttSmallSpan64>
void InverseTransformFromBitReverseRVV64(
		uint64_t* result, const uint64_t* operand, uint64_t n, uint64_t modulus,
		const uint64_t* inv_root_of_unity_powers,
		const uint64_t* precon_inv_root_of_unity_powers, uint64_t input_mod_factor,
		uint64_t output_mod_factor,
		const NttSmallSpanTables<SmallSpan, uint64_t>* small_span = nullptr);

#endif  // HEXL_HAS_RVV

}  // namespace hexl
}  // namespace intel
