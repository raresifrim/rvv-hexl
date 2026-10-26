// Copyright (C) 2020 Intel Corporation
// Modifications Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// SPDX-License-Identifier: Apache-2.0
//
// Kernels behind EltwiseMultMod.
// Kernels are templates on the storage word:
//   Word = uint64_t  OpenFHE NATIVE_SIZE=64 (upstream HEXL API)
//   Word = uint32_t  OpenFHE NATIVE_SIZE=32 (rvv-hexl extension)
// Write one body and specialise where it pays with
//   if constexpr (std::is_same_v<Word, uint32_t>) { ... } else { ... }

#pragma once

#include <stdint.h>

#include "util/cpu-features.hpp"
#include "util/rvv-config.hpp"  // lane types (empty without RVV)

namespace intel {
namespace hexl {

/// @brief Native: result[i] = (operand1[i] * operand2[i]) mod modulus, with
/// inputs in [0, InputModFactor * modulus) and output in [0, modulus).
/// InputModFactor is 1, 2 or 4.
template <typename Word, int InputModFactor>
void EltwiseMultModNative(Word* result, const Word* operand1, const Word* operand2, uint64_t n, uint64_t modulus);

#ifdef HEXL_HAS_RVV
/// @brief RVV, 32-bit lanes, modulus < kMaxModulusRVV32 (2^30), so every
/// input fits in 32 bits. Word = uint64_t narrows on load / widens on store
/// (rvv::Load32 / rvv::Store32); Word = uint32_t loads and stores directly.
/// Default lane: rvv::cfg::MultMod32 (uint32_t) or MultMod32U64 (uint64_t,
/// at most u32m4 because of the narrowing load).
template <typename Word, int InputModFactor,
	 class V = rvv::LaneFor<Word, rvv::cfg::MultMod32U64, rvv::cfg::MultMod32>>
		 void EltwiseMultModRVV32(Word* result, const Word* operand1, const Word* operand2, uint64_t n, uint64_t modulus);

/// @brief RVV, 64-bit lanes: every other modulus (BFV's 60-bit primes).
/// 64-bit storage only.
template <int InputModFactor, class V = rvv::cfg::MultMod64>
void EltwiseMultModRVV64(uint64_t* result, const uint64_t* operand1,
		const uint64_t* operand2, uint64_t n,
		uint64_t modulus);

/// @brief RVV, FP64 lanes (rvv-hexl extension, eltwise-fp64-rvv.cpp): the
/// modular product on the FP64 FMA, for
/// ModMulFp64Applicable moduli (util/fp64-mod.hpp). 64-bit storage; same
/// contract as EltwiseMultModRVV64. VF: vfloat64m1_t .. vfloat64m8_t.
template <int InputModFactor, class VF = rvv::cfg::MultModF64>
void EltwiseMultModRVVF64(uint64_t* result, const uint64_t* operand1,
		const uint64_t* operand2, uint64_t n, uint64_t modulus);
#endif

}  // namespace hexl
}  // namespace intel
