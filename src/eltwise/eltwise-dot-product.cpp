// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.

#include <algorithm>

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

namespace {

// The loop of DotProductModNative. kLazy (modulus < 2^32): a product of reduced operands is
// below 2^64, so the sums add raw products and BarrettReduce64 runs only every kmax terms,
// before a sum could overflow, and once per sum at the end; otherwise every term is a full
// MulAddModBarrett (the 128-bit pre-shift Barrett) and the sums stay in [0, q).
// #pragma GCC novector: GCC -O3 would vectorize the lazy loop into RVV.
template <bool kLazy, typename Word>
void DotProductNative(Word* const* result, uint64_t num_results,
		const Word* const* a, uint64_t num_terms, const Word* const* b,
		const Word* post, uint64_t n, uint64_t modulus, uint64_t input_mod_factor,
		bool accumulate) {

	const uint8_t shift = kLazy ? 0 : MSB(modulus) - 1;
	const uint64_t mu = kLazy ? 0 : MultiplyFactor(uint64_t(1U) << shift, 64, modulus).BarrettFactor();
	const uint64_t q_barr = kLazy ? MultiplyFactor(1, 64, modulus).BarrettFactor() : 0;
	// terms a sum below q can take before it could overflow: s + kmax (q - 1)^2 < 2^64
	const uint64_t kmax = kLazy ? (~uint64_t{0} - (modulus - 1)) / ((modulus - 1) * (modulus - 1)) : 0;
	const bool reduce4 = input_mod_factor >= 4, reduce2 = input_mod_factor >= 2;

	const auto reduce = [&](uint64_t x) { return kLazy ? BarrettReduce64(x, modulus, q_barr) : x; };
	const auto mul_add = [&](uint64_t x, uint64_t y, uint64_t c) {  // x, y, c < q
		return kLazy ? BarrettReduce64(x * y + c, modulus, q_barr) : MulAddModBarrett(x, y, c, modulus, mu, shift);
	};

	// Blocks of kBlock elements, term by term, two results at a time: a[i], the two b and the
	// sums streamed at a time. Element by element, every element reads all (M + 1) k + M
	// polynomials (26 for SEAL BFV), and once they leave L1 nearly every line misses (as in
	// ExternalProductModNative, rvv-hexl results/extprod-native-2026-10-29).
	constexpr uint64_t kBlock = 256;
	uint64_t s[2][kBlock];  // the sums of results m, m + 1 over one block (4 KB, in L1)

	for (uint64_t e0 = 0; e0 < n; e0 += kBlock) {
		const uint64_t len = std::min(kBlock, n - e0);
		for (uint64_t m = 0; m < num_results; m += 2) {  // results m and m + 1, sharing a[i]
			const bool two = m + 1 < num_results;
			const int pair = two ? 2 : 1;
			for (int r = 0; r < pair; ++r) {
#pragma GCC novector
				for (uint64_t e = 0; e < len; ++e) s[r][e] = accumulate && !post ? result[m + r][e0 + e] : 0;
			}

			uint64_t terms = 0;  // raw products in the sums since their last reduction (kLazy)
			for (uint64_t i = 0; i < num_terms; ++i) {
				if (kLazy && terms == kmax) {
					for (int r = 0; r < pair; ++r) {
#pragma GCC novector
						for (uint64_t e = 0; e < len; ++e) s[r][e] = reduce(s[r][e]);
					}
					terms = 0;
				}
				++terms;

				const Word* x = a[i] + e0;
				const Word* y0 = b[i*num_results + m] + e0;
				const Word* y1 = two ? b[i*num_results + m + 1] + e0 : y0;
#pragma GCC novector
				for (uint64_t e = 0; e < len; ++e) {
					uint64_t xe = x[e];  // reduced once per term
					if (reduce4) xe = ReduceFromTwice(xe, 2*modulus);
					if (reduce2) xe = ReduceFromTwice(xe, modulus);
					if constexpr (kLazy) {
						s[0][e] += xe * y0[e];
						if (two) s[1][e] += xe * y1[e];
					} else {
						s[0][e] = MulAddModBarrett(xe, y0[e], s[0][e], modulus, mu, shift);
						if (two) s[1][e] = MulAddModBarrett(xe, y1[e], s[1][e], modulus, mu, shift);
					}
				}
			}

			for (int r = 0; r < pair; ++r) {
				Word* out = result[m + r] + e0;
#pragma GCC novector
				for (uint64_t e = 0; e < len; ++e) {
					uint64_t v = reduce(s[r][e]);
					if (post) v = mul_add(v, post[e0 + e], accumulate ? out[e] : 0);
					out[e] = static_cast<Word>(v);
				}
			}
		}
	}
}

}  // namespace

template <typename Word>
void DotProductModNative(Word* const* result, uint64_t num_results,
		const Word* const* a, uint64_t num_terms, const Word* const* b,
		const Word* post, uint64_t n, uint64_t modulus, uint64_t input_mod_factor,
		bool accumulate) {
	if (modulus < (1ULL << 32)) {  // binfhe's Q, small RNS primes: lazy 64-bit sums
		DotProductNative<true, Word>(result, num_results, a, num_terms, b, post, n, modulus, input_mod_factor,
				accumulate);
	} else {
		DotProductNative<false, Word>(result, num_results, a, num_terms, b, post, n, modulus, input_mod_factor,
				accumulate);
	}
}

template void DotProductModNative<uint64_t>(uint64_t* const*, uint64_t, const uint64_t* const*, uint64_t,
		const uint64_t* const*, const uint64_t*, uint64_t, uint64_t, uint64_t, bool);
template void DotProductModNative<uint32_t>(uint32_t* const*, uint64_t, const uint32_t* const*, uint64_t,
		const uint32_t* const*, const uint32_t*, uint64_t, uint64_t, uint64_t, bool);

}  // namespace hexl
}  // namespace intel
