// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
//
// Kernels behind EltwiseDotProductMod (rvv-hexl extension): a multi-output
// modular dot product (see the public header). Templates on the storage word:
//   Word = uint64_t  SEAL, OpenFHE NATIVE_SIZE=64
//   Word = uint32_t  OpenFHE NATIVE_SIZE=32
// Three RVV kernels, like EltwiseMultAddMod:
//   RVV32   q < kMaxModulusRVV32 (2^30), e32 lanes, either word (binfhe, small
//           RNS primes);
//   RVVF64  2^30 <= q < 2^50 with the FP64 eltwise path on (UseEltwiseFp64):
//           SEAL's ~49-bit and the CKKS primes;
//   RVV64   every other modulus, e64 integer lanes (OpenFHE's 60-bit BFV limbs).
// The sums are kept lazy across the k terms, with a correction every few terms
// so they stay exact: at e32 a product in [0, 2q) with q < 2^28 can be added
// about eight times; in FP64 the sum of k products in (-q, q) is exact while
// k q < 2^53; at e64 lazy sums near 2^60 leave room for a few terms only.

#pragma once

#include <stdint.h>

#include "hexl/number-theory/number-theory.hpp"
#include "util/cpu-features.hpp"
#include "util/fp64-mod.hpp"
#include "util/rvv-config.hpp"  // lane types (empty without RVV)

namespace intel {
namespace hexl {

/// @brief Reference: the defining sums, one exact product per term (MultiplyMod
/// on a reduced first), element by element. The lane tests compare the kernels
/// against it; it is itself checked against SEAL's and OpenFHE's call sequences
/// (EltwiseMultMod / EltwiseMultAddMod) in test-eltwise.cpp.
template <typename Word>
inline void DotProductModReference(Word* const* result, uint64_t num_results,
		const Word* const* a, uint64_t num_terms, const Word* const* b,
		const Word* post, uint64_t n, uint64_t modulus, bool accumulate) {
	for (uint64_t e = 0; e < n; ++e) {
		for (uint64_t m = 0; m < num_results; ++m) {
			uint64_t s = 0;
			for (uint64_t i = 0; i < num_terms; ++i) {
				const uint64_t x = a[i][e] % modulus;
				s = AddUIntMod(s, MultiplyMod(x, b[i * num_results + m][e], modulus), modulus);
			}
			if (post) s = MultiplyMod(s, post[e], modulus);
			if (accumulate) s = AddUIntMod(s, result[m][e], modulus);
			result[m][e] = static_cast<Word>(s);
		}
	}
}

/// @brief Native (scalar) kernel.
template <typename Word>
void DotProductModNative(Word* const* result, uint64_t num_results,
		const Word* const* a, uint64_t num_terms, const Word* const* b,
		const Word* post, uint64_t n, uint64_t modulus, uint64_t input_mod_factor,
		bool accumulate);

#ifdef HEXL_HAS_RVV
/// @brief RVV, 32-bit lanes: modulus < kMaxModulusRVV32 (2^30), so every input
/// (a < 4q) fits 32 bits. Word = uint64_t narrows on load / widens on store
/// (rvv::Load32 / Store32, so at most u32m4); uint32_t loads directly.
template <typename Word,
	 class V = rvv::LaneFor<Word, rvv::cfg::DotProduct32U64, rvv::cfg::DotProduct32>>
	 void DotProductModRVV32(Word* const* result, uint64_t num_results,
			 const Word* const* a, uint64_t num_terms, const Word* const* b,
			 const Word* post, uint64_t n, uint64_t modulus,
			 uint64_t input_mod_factor, bool accumulate);

/// @brief RVV, FP64 lanes: 2^30 <= modulus < 2^50 (UseEltwiseFp64). 64-bit storage.
template <class VF = rvv::cfg::DotProductF64>
void DotProductModRVVF64(uint64_t* const* result, uint64_t num_results,
		const uint64_t* const* a, uint64_t num_terms, const uint64_t* const* b,
		const uint64_t* post, uint64_t n, uint64_t modulus,
		uint64_t input_mod_factor, bool accumulate);

/// @brief RVV, 64-bit integer lanes: every other modulus. 64-bit storage only.
template <class V = rvv::cfg::DotProduct64>
void DotProductModRVV64(uint64_t* const* result, uint64_t num_results,
		const uint64_t* const* a, uint64_t num_terms, const uint64_t* const* b,
		const uint64_t* post, uint64_t n, uint64_t modulus,
		uint64_t input_mod_factor, bool accumulate);
#endif

}  // namespace hexl
}  // namespace intel
