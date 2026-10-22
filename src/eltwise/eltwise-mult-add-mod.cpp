// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.

#include "hexl/eltwise/eltwise-mult-add-mod.hpp"

#include <algorithm>
#include <type_traits>
#include "eltwise/eltwise-mult-add-mod-internal.hpp"
#include "hexl/eltwise/eltwise-mult-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "util/cpu-features.hpp"
#include "util/fp64-mod.hpp"

namespace intel {
namespace hexl {

void EltwiseMultAddMod(uint64_t* result, const uint64_t* arg1,
		const uint64_t* arg2, const uint64_t* arg3, uint64_t n,
		uint64_t modulus, uint64_t input_mod_factor) {
	HEXL_CHECK(result != nullptr, "Require result != nullptr");
	HEXL_CHECK(arg1 != nullptr, "Require arg1 != nullptr");
	HEXL_CHECK(arg2 != nullptr, "Require arg2 != nullptr");
	HEXL_CHECK(arg3 != nullptr, "Require arg3 != nullptr (no addend: EltwiseMultMod)");
	HEXL_CHECK(n != 0, "Require n != 0");
	HEXL_CHECK(modulus > 1, "Require modulus > 1");
	HEXL_CHECK(modulus < (1ULL << 61), "Require modulus < (1ULL << 61)");
	HEXL_CHECK(
			input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4,
			"Require input_mod_factor = 1, 2, or 4");
	HEXL_CHECK_BOUNDS(arg1, n, input_mod_factor * modulus,
			"arg1 exceeds bound " << (input_mod_factor * modulus));
	HEXL_CHECK_BOUNDS(arg2, n, input_mod_factor * modulus,
			"arg2 exceeds bound " << (input_mod_factor * modulus));
	HEXL_CHECK_BOUNDS(arg3, n, input_mod_factor * modulus,
			"arg3 exceeds bound " << (input_mod_factor * modulus));

#ifdef HEXL_HAS_RVV
	if (has_rvv) {
		if (UseEltwiseFp64(modulus)) {  // switched on and 2^30 <= q < 2^50
			HEXL_VLOG(3, "Calling EltwiseMultAddModRVVF64");
			switch (input_mod_factor) {
				case 1: EltwiseMultAddModRVVF64<1>(result, arg1, arg2, arg3, n, modulus); break;
				case 2: EltwiseMultAddModRVVF64<2>(result, arg1, arg2, arg3, n, modulus); break;
				case 4: EltwiseMultAddModRVVF64<4>(result, arg1, arg2, arg3, n, modulus); break;
			}
		} else if (modulus < kMaxModulusRVV32 && input_mod_factor * modulus < (1ULL << 32)) {
			HEXL_VLOG(3, "Calling EltwiseMultAddModRVV32<uint64_t>");
			switch (input_mod_factor) {
				case 1: EltwiseMultAddModRVV32<uint64_t, 1>(result, arg1, arg2, arg3, n, modulus); break;
				case 2: EltwiseMultAddModRVV32<uint64_t, 2>(result, arg1, arg2, arg3, n, modulus); break;
				case 4: EltwiseMultAddModRVV32<uint64_t, 4>(result, arg1, arg2, arg3, n, modulus); break;
			}
		} else {
			HEXL_VLOG(3, "Calling EltwiseMultAddModRVV64");
			switch (input_mod_factor) {
				case 1: EltwiseMultAddModRVV64<1>(result, arg1, arg2, arg3, n, modulus); break;
				case 2: EltwiseMultAddModRVV64<2>(result, arg1, arg2, arg3, n, modulus); break;
				case 4: EltwiseMultAddModRVV64<4>(result, arg1, arg2, arg3, n, modulus); break;
			}
		}
		return;
	}
#endif

	HEXL_VLOG(3, "Calling EltwiseMultAddModNative<uint64_t>");
	switch (input_mod_factor) {
		case 1: EltwiseMultAddModNative<uint64_t, 1>(result, arg1, arg2, arg3, n, modulus); break;
		case 2: EltwiseMultAddModNative<uint64_t, 2>(result, arg1, arg2, arg3, n, modulus); break;
		case 4: EltwiseMultAddModNative<uint64_t, 4>(result, arg1, arg2, arg3, n, modulus); break;
	}
}

void EltwiseMultAddMod(uint32_t* result, const uint32_t* arg1,
		const uint32_t* arg2, const uint32_t* arg3, uint64_t n,
		uint64_t modulus, uint64_t input_mod_factor) {
	HEXL_CHECK(result != nullptr, "Require result != nullptr");
	HEXL_CHECK(arg1 != nullptr, "Require arg1 != nullptr");
	HEXL_CHECK(arg2 != nullptr, "Require arg2 != nullptr");
	HEXL_CHECK(arg3 != nullptr, "Require arg3 != nullptr (no addend: EltwiseMultMod)");
	HEXL_CHECK(n != 0, "Require n != 0");
	HEXL_CHECK(modulus > 1, "Require modulus > 1");
	HEXL_CHECK(
			input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4,
			"Require input_mod_factor = 1, 2, or 4");
	HEXL_CHECK(input_mod_factor * modulus <= (1ULL << 32),
			"Require input_mod_factor * modulus <= 2**32");
	HEXL_CHECK_BOUNDS(arg1, n, input_mod_factor * modulus,
			"arg1 exceeds bound " << (input_mod_factor * modulus));
	HEXL_CHECK_BOUNDS(arg2, n, input_mod_factor * modulus,
			"arg2 exceeds bound " << (input_mod_factor * modulus));
	HEXL_CHECK_BOUNDS(arg3, n, input_mod_factor * modulus,
			"arg3 exceeds bound " << (input_mod_factor * modulus));

#ifdef HEXL_HAS_RVV
	if (has_rvv && modulus < kMaxModulusRVV32) {
		HEXL_VLOG(3, "Calling EltwiseMultAddModRVV32<uint32_t>");
		switch (input_mod_factor) {
			case 1: EltwiseMultAddModRVV32<uint32_t, 1>(result, arg1, arg2, arg3, n, modulus); break;
			case 2: EltwiseMultAddModRVV32<uint32_t, 2>(result, arg1, arg2, arg3, n, modulus); break;
			case 4: EltwiseMultAddModRVV32<uint32_t, 4>(result, arg1, arg2, arg3, n, modulus); break;
		}
		return;
	}
#endif

	HEXL_VLOG(3, "Calling EltwiseMultAddModNative<uint32_t>");
	switch (input_mod_factor) {
		case 1: EltwiseMultAddModNative<uint32_t, 1>(result, arg1, arg2, arg3, n, modulus); break;
		case 2: EltwiseMultAddModNative<uint32_t, 2>(result, arg1, arg2, arg3, n, modulus); break;
		case 4: EltwiseMultAddModNative<uint32_t, 4>(result, arg1, arg2, arg3, n, modulus); break;
	}
}

// ---------------------------------------------------------------------------
// Native (scalar) kernel.
// ---------------------------------------------------------------------------

template <typename Word, int InputModFactor>
void EltwiseMultAddModNative(Word* result, const Word* arg1, const Word* arg2,
		const Word* arg3, uint64_t n, uint64_t modulus) {

	constexpr bool kWord32 = std::is_same_v<Word, uint32_t>;
	const uint64_t shift = kWord32 ? 0 : MSB(modulus) - 1;
	const uint64_t mu = kWord32 ? 0 : MultiplyFactor(uint64_t{1} << shift, 64, modulus).BarrettFactor();
	const uint64_t q_barr = kWord32 ? MultiplyFactor(1, 64, modulus).BarrettFactor() : 0;
#pragma GCC novector
	for(uint64_t i=0; i<n; ++i){
		uint64_t a = arg1[i];
		uint64_t b = arg2[i];
		uint64_t c = arg3[i];
		if constexpr (InputModFactor >= 4) { 
			a = std::min(a, a - 2*modulus);
			b = std::min(b, b - 2*modulus);
			c = std::min(c, c - 2*modulus);
		}
		if constexpr (InputModFactor >= 2) {
			a = std::min(a, a - modulus);
			b = std::min(b, b - modulus);
			c = std::min(c, c - modulus);
		}

		if constexpr (kWord32) {
			result[i] = static_cast<Word>(BarrettReduce64(a * b + c, modulus, q_barr));  // a*b+c < 2^64
		} else {
			uint64_t hi, lo;
			asm("mulhu %0, %1, %2" : "=r"(hi) : "r"(a), "r"(b));
			asm("mul %0, %1, %2" : "=r"(lo) : "r"(a), "r"(b));

			uint64_t qt = lo >> shift | hi << (64-shift);
			uint64_t q;
			asm("mulhu %0, %1, %2" : "=r"(q) : "r"(qt), "r"(mu));

			uint64_t r = lo - (q * modulus);
			r = std::min(r, r - modulus);
			r += c;
			result[i] = std::min(r, r - modulus);
		}
	}}

template void EltwiseMultAddModNative<uint64_t, 1>(uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultAddModNative<uint64_t, 2>(uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultAddModNative<uint64_t, 4>(uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultAddModNative<uint32_t, 1>(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void EltwiseMultAddModNative<uint32_t, 2>(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void EltwiseMultAddModNative<uint32_t, 4>(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);

}  // namespace hexl
}  // namespace intel
