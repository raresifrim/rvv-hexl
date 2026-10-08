// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#include "hexl/eltwise/eltwise-mult-mod.hpp"

#include <algorithm>
#include <type_traits>

#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

void EltwiseMultMod(uint64_t* result, const uint64_t* operand1,
		const uint64_t* operand2, uint64_t n, uint64_t modulus,
		uint64_t input_mod_factor) {
	HEXL_CHECK(result != nullptr, "Require result != nullptr");
	HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
	HEXL_CHECK(operand2 != nullptr, "Require operand2 != nullptr");
	HEXL_CHECK(n != 0, "Require n != 0");
	HEXL_CHECK(modulus > 1, "Require modulus > 1");
	// The pre-shift Barrett estimate is at most one short only for moduli of up
	// to 61 bits (OpenFHE <= 60, SEAL <= 61); same bound as EltwiseFMAMod.
	HEXL_CHECK(modulus < (1ULL << 61), "Require modulus < (1ULL << 61)");
	HEXL_CHECK(input_mod_factor * modulus < (1ULL << 63),
			"Require input_mod_factor * modulus < (1ULL << 63)");
	HEXL_CHECK(
			input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4,
			"Require input_mod_factor = 1, 2, or 4")
		HEXL_CHECK_BOUNDS(operand1, n, input_mod_factor * modulus,
				"operand1 exceeds bound " << (input_mod_factor * modulus))
		HEXL_CHECK_BOUNDS(operand2, n, input_mod_factor * modulus,
				"operand2 exceeds bound " << (input_mod_factor * modulus))

#ifdef HEXL_HAS_RVV
		if (has_rvv) {
			if (modulus < kMaxModulusRVV32) {
				HEXL_VLOG(3, "Calling EltwiseMultModRVV32<uint64_t>");
				switch (input_mod_factor) {
					case 1: EltwiseMultModRVV32<uint64_t, 1>(result, operand1, operand2, n, modulus); break;
					case 2: EltwiseMultModRVV32<uint64_t, 2>(result, operand1, operand2, n, modulus); break;
					case 4: EltwiseMultModRVV32<uint64_t, 4>(result, operand1, operand2, n, modulus); break;
				}
			} else {
				HEXL_VLOG(3, "Calling EltwiseMultModRVV64");
				switch (input_mod_factor) {
					case 1: EltwiseMultModRVV64<1>(result, operand1, operand2, n, modulus); break;
					case 2: EltwiseMultModRVV64<2>(result, operand1, operand2, n, modulus); break;
					case 4: EltwiseMultModRVV64<4>(result, operand1, operand2, n, modulus); break;
				}
			}
			return;
		}
#endif

	HEXL_VLOG(3, "Calling EltwiseMultModNative<uint64_t>");
	switch (input_mod_factor) {
		case 1: EltwiseMultModNative<uint64_t, 1>(result, operand1, operand2, n, modulus); break;
		case 2: EltwiseMultModNative<uint64_t, 2>(result, operand1, operand2, n, modulus); break;
		case 4: EltwiseMultModNative<uint64_t, 4>(result, operand1, operand2, n, modulus); break;
	}
}

// ---- rvv-hexl extension: 32-bit storage ----------------------------------

void EltwiseMultMod(uint32_t* result, const uint32_t* operand1,
		const uint32_t* operand2, uint64_t n, uint64_t modulus,
		uint64_t input_mod_factor) {
	HEXL_CHECK(result != nullptr, "Require result != nullptr");
	HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
	HEXL_CHECK(operand2 != nullptr, "Require operand2 != nullptr");
	HEXL_CHECK(n != 0, "Require n != 0");
	HEXL_CHECK(modulus > 1, "Require modulus > 1");
	HEXL_CHECK(
			input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4,
			"Require input_mod_factor = 1, 2, or 4")
		HEXL_CHECK(input_mod_factor * modulus <= (1ULL << 32),
				"Require input_mod_factor * modulus <= 2**32");
	HEXL_CHECK_BOUNDS(operand1, n, input_mod_factor * modulus,
			"operand1 exceeds bound " << (input_mod_factor * modulus))
		HEXL_CHECK_BOUNDS(operand2, n, input_mod_factor * modulus,
				"operand2 exceeds bound " << (input_mod_factor * modulus))

#ifdef HEXL_HAS_RVV
		if (has_rvv && modulus < kMaxModulusRVV32) {
			HEXL_VLOG(3, "Calling EltwiseMultModRVV32<uint32_t>");
			switch (input_mod_factor) {
				case 1: EltwiseMultModRVV32<uint32_t, 1>(result, operand1, operand2, n, modulus); break;
				case 2: EltwiseMultModRVV32<uint32_t, 2>(result, operand1, operand2, n, modulus); break;
				case 4: EltwiseMultModRVV32<uint32_t, 4>(result, operand1, operand2, n, modulus); break;
			}
			return;
		}
#endif

	HEXL_VLOG(3, "Calling EltwiseMultModNative<uint32_t>");
	switch (input_mod_factor) {
		case 1: EltwiseMultModNative<uint32_t, 1>(result, operand1, operand2, n, modulus); break;
		case 2: EltwiseMultModNative<uint32_t, 2>(result, operand1, operand2, n, modulus); break;
		case 4: EltwiseMultModNative<uint32_t, 4>(result, operand1, operand2, n, modulus); break;
	}
}

// ---------------------------------------------------------------------------
// Native (scalar) kernel.
// ---------------------------------------------------------------------------

template <typename Word, int InputModFactor>
void EltwiseMultModNative(Word* result, const Word* operand1, const Word* operand2, uint64_t n, uint64_t modulus) {
	// result[i] = (operand1[i] * operand2[i]) mod modulus, in [0, q).
	// Inputs are NOT reduced: first bring each into [0, q) with branch-free
	constexpr bool kWord32 = std::is_same_v<Word, uint32_t>;
	const uint64_t shift = kWord32 ? 0 : MSB(modulus) - 1;
	const uint64_t mu = kWord32 ? 0 : MultiplyFactor(uint64_t{1} << shift, 64, modulus).BarrettFactor();
	const uint64_t q_barr = kWord32 ? MultiplyFactor(1, 64, modulus).BarrettFactor() : 0;
#pragma GCC novector
	for(uint64_t i=0; i<n; ++i){
		uint64_t a = operand1[i];
		uint64_t b = operand2[i];
		if constexpr (InputModFactor >= 4) { a = std::min(a, a - 2*modulus); b = std::min(b, b - 2*modulus); }
		if constexpr (InputModFactor >= 2) { a = std::min(a, a - modulus);  b = std::min(b, b - modulus);  }

		if constexpr (kWord32) {
			result[i] = static_cast<Word>(BarrettReduce64(a * b, modulus, q_barr));  // a*b < 2^64
		} else {
			uint64_t hi, lo;
			asm("mulhu %0, %1, %2" : "=r"(hi) : "r"(a), "r"(b));
			asm("mul %0, %1, %2" : "=r"(lo) : "r"(a), "r"(b));

			uint64_t c = lo >> shift | hi << (64-shift);
			uint64_t q;
			asm("mulhu %0, %1, %2" : "=r"(q) : "r"(c), "r"(mu));

			uint64_t r = lo - (q * modulus);
			result[i] = std::min(r, r - modulus);
		}
	}
}

}  // namespace hexl
}  // namespace intel
