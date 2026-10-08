// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#include "hexl/eltwise/eltwise-fma-mod.hpp"

#include "eltwise/eltwise-fma-mod-internal.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

void EltwiseFMAMod(uint64_t* result, const uint64_t* arg1, uint64_t arg2,
		const uint64_t* arg3, uint64_t n, uint64_t modulus,
		uint64_t input_mod_factor) {
	HEXL_CHECK(result != nullptr, "Require result != nullptr");
	HEXL_CHECK(arg1 != nullptr, "Require arg1 != nullptr");
	HEXL_CHECK(n != 0, "Require n != 0");
	HEXL_CHECK(modulus > 1, "Require modulus > 1");
	HEXL_CHECK(modulus < (1ULL << 61), "Require modulus < (1ULL << 61)");
	HEXL_CHECK(
			input_mod_factor == 1 || input_mod_factor == 2 ||
			input_mod_factor == 4 || input_mod_factor == 8,
			"input_mod_factor must be 1, 2, 4, or 8. Got " << input_mod_factor);
	HEXL_CHECK(arg2 < input_mod_factor * modulus,
			"arg2 " << arg2 << " exceeds bound "
			<< (input_mod_factor * modulus));
	HEXL_CHECK_BOUNDS(arg1, n, input_mod_factor * modulus,
			"arg1 value exceeds bound " << (input_mod_factor * modulus));
	if (arg3 != nullptr) {
		HEXL_CHECK_BOUNDS(arg3, n, input_mod_factor * modulus,
				"arg3 value exceeds bound " << (input_mod_factor * modulus));
	}

#ifdef HEXL_HAS_RVV
	if (has_rvv) {
		if (modulus < kMaxModulusRVV32 &&
				input_mod_factor * modulus < (1ULL << 32)) {
			HEXL_VLOG(3, "Calling EltwiseFMAModRVV32<uint64_t>");
			switch (input_mod_factor) {
				case 1: EltwiseFMAModRVV32<uint64_t, 1>(result, arg1, arg2, arg3, n, modulus); break;
				case 2: EltwiseFMAModRVV32<uint64_t, 2>(result, arg1, arg2, arg3, n, modulus); break;
				case 4: EltwiseFMAModRVV32<uint64_t, 4>(result, arg1, arg2, arg3, n, modulus); break;
				case 8: EltwiseFMAModRVV32<uint64_t, 8>(result, arg1, arg2, arg3, n, modulus); break;
			}
		} else {
			HEXL_VLOG(3, "Calling EltwiseFMAModRVV64");
			switch (input_mod_factor) {
				case 1: EltwiseFMAModRVV64<1>(result, arg1, arg2, arg3, n, modulus); break;
				case 2: EltwiseFMAModRVV64<2>(result, arg1, arg2, arg3, n, modulus); break;
				case 4: EltwiseFMAModRVV64<4>(result, arg1, arg2, arg3, n, modulus); break;
				case 8: EltwiseFMAModRVV64<8>(result, arg1, arg2, arg3, n, modulus); break;
			}
		}
		return;
	}
#endif

	HEXL_VLOG(3, "Calling EltwiseFMAModNative<uint64_t>");
	switch (input_mod_factor) {
		case 1: EltwiseFMAModNative<uint64_t, 1>(result, arg1, arg2, arg3, n, modulus); break;
		case 2: EltwiseFMAModNative<uint64_t, 2>(result, arg1, arg2, arg3, n, modulus); break;
		case 4: EltwiseFMAModNative<uint64_t, 4>(result, arg1, arg2, arg3, n, modulus); break;
		case 8: EltwiseFMAModNative<uint64_t, 8>(result, arg1, arg2, arg3, n, modulus); break;
	}
}

// ---- rvv-hexl extension: 32-bit storage ----------------------------------

void EltwiseFMAMod(uint32_t* result, const uint32_t* arg1, uint64_t arg2,
		const uint32_t* arg3, uint64_t n, uint64_t modulus,
		uint64_t input_mod_factor) {
	HEXL_CHECK(result != nullptr, "Require result != nullptr");
	HEXL_CHECK(arg1 != nullptr, "Require arg1 != nullptr");
	HEXL_CHECK(n != 0, "Require n != 0");
	HEXL_CHECK(modulus > 1, "Require modulus > 1");
	HEXL_CHECK(
			input_mod_factor == 1 || input_mod_factor == 2 ||
			input_mod_factor == 4 || input_mod_factor == 8,
			"input_mod_factor must be 1, 2, 4, or 8. Got " << input_mod_factor);
	HEXL_CHECK(input_mod_factor * modulus <= (1ULL << 32),
			"Require input_mod_factor * modulus <= 2**32");
	HEXL_CHECK(arg2 < input_mod_factor * modulus,
			"arg2 " << arg2 << " exceeds bound "
			<< (input_mod_factor * modulus));
	HEXL_CHECK_BOUNDS(arg1, n, input_mod_factor * modulus,
			"arg1 value exceeds bound " << (input_mod_factor * modulus));
	if (arg3 != nullptr) {
		HEXL_CHECK_BOUNDS(arg3, n, input_mod_factor * modulus,
				"arg3 value exceeds bound " << (input_mod_factor * modulus));
	}

#ifdef HEXL_HAS_RVV
	if (has_rvv && modulus < kMaxModulusRVV32) {
		HEXL_VLOG(3, "Calling EltwiseFMAModRVV32<uint32_t>");
		switch (input_mod_factor) {
			case 1: EltwiseFMAModRVV32<uint32_t, 1>(result, arg1, arg2, arg3, n, modulus); break;
			case 2: EltwiseFMAModRVV32<uint32_t, 2>(result, arg1, arg2, arg3, n, modulus); break;
			case 4: EltwiseFMAModRVV32<uint32_t, 4>(result, arg1, arg2, arg3, n, modulus); break;
			case 8: EltwiseFMAModRVV32<uint32_t, 8>(result, arg1, arg2, arg3, n, modulus); break;
		}
		return;
	}
#endif

	HEXL_VLOG(3, "Calling EltwiseFMAModNative<uint32_t>");
	switch (input_mod_factor) {
		case 1: EltwiseFMAModNative<uint32_t, 1>(result, arg1, arg2, arg3, n, modulus); break;
		case 2: EltwiseFMAModNative<uint32_t, 2>(result, arg1, arg2, arg3, n, modulus); break;
		case 4: EltwiseFMAModNative<uint32_t, 4>(result, arg1, arg2, arg3, n, modulus); break;
		case 8: EltwiseFMAModNative<uint32_t, 8>(result, arg1, arg2, arg3, n, modulus); break;
	}
}

// ---------------------------------------------------------------------------
// Native (scalar) kernel.
// ---------------------------------------------------------------------------

template <typename Word, int InputModFactor>
void EltwiseFMAModNative(Word* result, const Word* arg1, uint64_t arg2, const Word* arg3, uint64_t n, uint64_t modulus) {
	uint64_t w = arg2;
	if constexpr (InputModFactor == 8) w = std::min(w, w - 4 * modulus);
	if constexpr (InputModFactor >= 4) w = std::min(w, w - 2 * modulus);
	if constexpr (InputModFactor >= 2) w = std::min(w, w - modulus);
	const uint64_t w_precon = MultiplyFactor(w, 64, modulus).BarrettFactor();
	if(arg3 != nullptr){
#pragma GCC novector
		for(uint64_t i=0; i<n; ++i){
			result[i] = MultiplyAddMod<InputModFactor>(arg1[i], w, w_precon, arg3[i], modulus);
		}
	}
	else{
#pragma GCC novector
		for(uint64_t i=0; i<n; ++i){
			result[i] = MultiplyMod(arg1[i], w, w_precon, modulus);
		}
	}
}

}  // namespace hexl
}  // namespace intel
