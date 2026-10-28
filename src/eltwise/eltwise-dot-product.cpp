// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.

#include "hexl/eltwise/eltwise-dot-product.hpp"

#include "eltwise/eltwise-dot-product-internal.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/util/check.hpp"
#include "util/cpu-features.hpp"
#include "util/fp64-mod.hpp"

namespace intel {
namespace hexl {

namespace {

template <typename Word>
void CheckArgs(Word* const* result, uint64_t num_results, const Word* const* a, uint64_t num_terms,
		const Word* const* b, const Word* post, uint64_t n, uint64_t modulus,
		uint64_t input_mod_factor, bool accumulate) {
	HEXL_CHECK(result != nullptr && a != nullptr && b != nullptr, "Require result, a, b != nullptr");
	HEXL_CHECK(num_results >= 1, "Require num_results >= 1");
	HEXL_CHECK(num_terms >= 1, "Require num_terms >= 1");
	HEXL_CHECK(n != 0, "Require n != 0");
	HEXL_CHECK(modulus > 1, "Require modulus > 1");
	HEXL_CHECK(input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4,
			"Require input_mod_factor = 1, 2, or 4");
	for (uint64_t m = 0; m < num_results; ++m) {
		HEXL_CHECK(result[m] != nullptr, "Require result[" << m << "] != nullptr");
		if (accumulate) HEXL_CHECK_BOUNDS(result[m], n, modulus, "result[" << m << "] exceeds bound " << modulus);
		for (uint64_t r = 0; r < m; ++r) HEXL_CHECK(result[r] != result[m], "Require distinct results");
	}
	for (uint64_t i = 0; i < num_terms; ++i) {
		HEXL_CHECK(a[i] != nullptr, "Require a[" << i << "] != nullptr");
		HEXL_CHECK_BOUNDS(a[i], n, input_mod_factor * modulus,
				"a[" << i << "] exceeds bound " << input_mod_factor * modulus);
		for (uint64_t m = 0; m < num_results; ++m) {
			HEXL_CHECK(b[i * num_results + m] != nullptr, "Require b[" << i * num_results + m << "] != nullptr");
			HEXL_CHECK_BOUNDS(b[i * num_results + m], n, modulus, "b[" << i * num_results + m << "] exceeds bound " << modulus);
			HEXL_CHECK(a[i] != result[m] && b[i * num_results + m] != result[m], "Require result not to alias a or b");
		}
	}
	if (post) HEXL_CHECK_BOUNDS(post, n, modulus, "post exceeds bound " << modulus);
	(void)result; (void)num_results; (void)a; (void)num_terms; (void)b; (void)post;
	(void)n; (void)modulus; (void)input_mod_factor; (void)accumulate;
}

}  // namespace

void EltwiseDotProductMod(uint64_t* const* result, uint64_t num_results,
		const uint64_t* const* a, uint64_t num_terms, const uint64_t* const* b,
		const uint64_t* post, uint64_t n, uint64_t modulus,
		uint64_t input_mod_factor, bool accumulate) {
	CheckArgs(result, num_results, a, num_terms, b, post, n, modulus, input_mod_factor, accumulate);
	HEXL_CHECK(modulus < (1ULL << 61), "Require modulus < 2**61");
	HEXL_CHECK(input_mod_factor * modulus < (1ULL << 63), "Require input_mod_factor * modulus < 2**63");

#ifdef HEXL_HAS_RVV
	if (has_rvv) {
		if (UseEltwiseFp64(modulus)) {  // switched on and 2^30 <= q < 2^50
			HEXL_VLOG(3, "Calling DotProductModRVVF64");
			DotProductModRVVF64(result, num_results, a, num_terms, b, post, n, modulus, input_mod_factor, accumulate);
		} else if (modulus < kMaxModulusRVV32) {
			HEXL_VLOG(3, "Calling DotProductModRVV32<uint64_t>");
			DotProductModRVV32<uint64_t>(result, num_results, a, num_terms, b, post, n, modulus, input_mod_factor,
					accumulate);
		} else {
			HEXL_VLOG(3, "Calling DotProductModRVV64");
			DotProductModRVV64(result, num_results, a, num_terms, b, post, n, modulus, input_mod_factor, accumulate);
		}
		return;
	}
#endif

	HEXL_VLOG(3, "Calling DotProductModNative<uint64_t>");
	DotProductModNative<uint64_t>(result, num_results, a, num_terms, b, post, n, modulus, input_mod_factor,
			accumulate);
}

void EltwiseDotProductMod(uint32_t* const* result, uint64_t num_results,
		const uint32_t* const* a, uint64_t num_terms, const uint32_t* const* b,
		const uint32_t* post, uint64_t n, uint64_t modulus,
		uint64_t input_mod_factor, bool accumulate) {
	CheckArgs(result, num_results, a, num_terms, b, post, n, modulus, input_mod_factor, accumulate);
	HEXL_CHECK(input_mod_factor * modulus <= (1ULL << 32), "Require input_mod_factor * modulus <= 2**32");

#ifdef HEXL_HAS_RVV
	if (has_rvv && modulus < kMaxModulusRVV32) {
		HEXL_VLOG(3, "Calling DotProductModRVV32<uint32_t>");
		DotProductModRVV32<uint32_t>(result, num_results, a, num_terms, b, post, n, modulus, input_mod_factor,
				accumulate);
		return;
	}
#endif

	HEXL_VLOG(3, "Calling DotProductModNative<uint32_t>");
	DotProductModNative<uint32_t>(result, num_results, a, num_terms, b, post, n, modulus, input_mod_factor,
			accumulate);
}

// ---------------------------------------------------------------------------
// Native (scalar) kernel.
// ---------------------------------------------------------------------------

// STUB: the defining sums, one exact product per term (the reference). To be
// replaced by the lazy loop (SEAL keeps 128-bit sums and reduces every few
// digits: SEAL_MULTIPLY_ACCUMULATE_USER_MOD_MAX).
template <typename Word>
void DotProductModNative(Word* const* result, uint64_t num_results,
		const Word* const* a, uint64_t num_terms, const Word* const* b,
		const Word* post, uint64_t n, uint64_t modulus, uint64_t input_mod_factor,
		bool accumulate) {
	(void)input_mod_factor;
	DotProductModReference<Word>(result, num_results, a, num_terms, b, post, n, modulus, accumulate);
}

template void DotProductModNative<uint64_t>(uint64_t* const*, uint64_t, const uint64_t* const*, uint64_t,
		const uint64_t* const*, const uint64_t*, uint64_t, uint64_t, uint64_t, bool);
template void DotProductModNative<uint32_t>(uint32_t* const*, uint64_t, const uint32_t* const*, uint64_t,
		const uint32_t* const*, const uint32_t*, uint64_t, uint64_t, uint64_t, bool);

}  // namespace hexl
}  // namespace intel
