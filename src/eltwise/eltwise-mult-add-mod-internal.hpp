// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
//
// Kernels behind EltwiseMultAddMod (rvv-hexl extension):
//   result[i] = (arg1[i] * arg2[i] + arg3[i]) mod modulus
// All inputs in [0, InputModFactor * q), output in [0, q); result may alias
// any input. Templates on the storage word, like the other eltwise kernels:
//   Word = uint64_t  OpenFHE NATIVE_SIZE=64
//   Word = uint32_t  OpenFHE NATIVE_SIZE=32
//
// One pass: the pre-shift Barrett product of EltwiseMultMod plus arg3, then
// two conditional subtractions (2q, q). RVV32 uses rvv::MulAddModBarrett32.
// Against EltwiseMultMod into a temporary + EltwiseAddMod (2026-10-14, N=4096):
// 1.05-1.33x faster with RVV, 1.09-1.22x scalar (no-V build).

#pragma once

#include <stdint.h>

#include <algorithm>

#include "util/cpu-features.hpp"
#include "util/rvv-config.hpp"  // lane types (empty without RVV)

namespace intel {
namespace hexl {

/// @brief Native: result[i] = (arg1[i] * arg2[i] + arg3[i]) mod modulus.
/// InputModFactor is 1, 2 or 4.
template <typename Word, int InputModFactor>
void EltwiseMultAddModNative(Word* result, const Word* arg1, const Word* arg2,
		const Word* arg3, uint64_t n, uint64_t modulus);

#ifdef HEXL_HAS_RVV
/// @brief RVV, 32-bit lanes: modulus < kMaxModulusRVV32 (2^30), every input
/// fits in 32 bits. Word = uint64_t narrows on load / widens on store
/// (rvv::Load32 / Store32, so at most u32m4); uint32_t loads directly.
template <typename Word, int InputModFactor,
	 class V = rvv::LaneFor<Word, rvv::cfg::MultAddMod32U64, rvv::cfg::MultAddMod32>>
		 void EltwiseMultAddModRVV32(Word* result, const Word* arg1, const Word* arg2,
				 const Word* arg3, uint64_t n, uint64_t modulus);

/// @brief RVV, 64-bit lanes: every other modulus. 64-bit storage only.
template <int InputModFactor, class V = rvv::cfg::MultAddMod64>
void EltwiseMultAddModRVV64(uint64_t* result, const uint64_t* arg1,
		const uint64_t* arg2, const uint64_t* arg3, uint64_t n,
		uint64_t modulus);

/// @brief RVV, FP64 lanes (rvv-hexl extension, STUB: calls
/// EltwiseMultAddModRVV64): for ModMulFp64Applicable moduli
/// (util/fp64-mod.hpp), the addend joining the FP64 remainder exactly. Same
/// contract as EltwiseMultAddModRVV64 (result may alias arg3).
/// VF: vfloat64m1_t .. vfloat64m8_t.
template <int InputModFactor, class VF = rvv::cfg::MultAddModF64>
void EltwiseMultAddModRVVF64(uint64_t* result, const uint64_t* arg1,
		const uint64_t* arg2, const uint64_t* arg3, uint64_t n,
		uint64_t modulus);
#endif

}  // namespace hexl
}  // namespace intel
