// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.

#include <algorithm>

#include "hexl/eltwise/eltwise-external-product.hpp"

#include "eltwise/eltwise-external-product-internal.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/util/check.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

namespace {

template <typename Word>
void CheckArgs(Word* acc0, Word* acc1, const Word* const* digits, uint64_t num_digits,
		const Word* const* key1, const Word* mono1, const Word* const* key2, const Word* mono2,
		uint64_t n, uint64_t modulus) {
	HEXL_CHECK(acc0 != nullptr && acc1 != nullptr, "Require acc0, acc1 != nullptr");
	HEXL_CHECK(acc0 != acc1, "Require acc0 != acc1");
	HEXL_CHECK(digits != nullptr, "Require digits != nullptr");
	HEXL_CHECK(key1 != nullptr, "Require key1 != nullptr");
	HEXL_CHECK(num_digits >= 1, "Require num_digits >= 1");
	HEXL_CHECK(n != 0, "Require n != 0");
	HEXL_CHECK(modulus > 1, "Require modulus > 1");
	for (uint64_t i = 0; i < num_digits; ++i) {
		HEXL_CHECK(digits[i] != nullptr, "Require digits[" << i << "] != nullptr");
		HEXL_CHECK(digits[i] != acc0 && digits[i] != acc1, "Require digits[" << i << "] not to alias acc");
		HEXL_CHECK_BOUNDS(digits[i], n, modulus, "digits[" << i << "] exceeds bound " << modulus);
		for (uint64_t j = 0; j < 2; ++j) {
			HEXL_CHECK(key1[2 * i + j] != nullptr, "Require key1[" << 2 * i + j << "] != nullptr");
			HEXL_CHECK_BOUNDS(key1[2 * i + j], n, modulus, "key1[" << 2 * i + j << "] exceeds bound " << modulus);
			if (key2) {
				HEXL_CHECK(key2[2 * i + j] != nullptr, "Require key2[" << 2 * i + j << "] != nullptr");
				HEXL_CHECK_BOUNDS(key2[2 * i + j], n, modulus, "key2[" << 2 * i + j << "] exceeds bound " << modulus);
			}
		}
	}
	if (mono1) HEXL_CHECK_BOUNDS(mono1, n, modulus, "mono1 exceeds bound " << modulus);
	if (key2 && mono2) HEXL_CHECK_BOUNDS(mono2, n, modulus, "mono2 exceeds bound " << modulus);
	HEXL_CHECK_BOUNDS(acc0, n, modulus, "acc0 exceeds bound " << modulus);
	HEXL_CHECK_BOUNDS(acc1, n, modulus, "acc1 exceeds bound " << modulus);
	(void)acc0; (void)acc1; (void)digits; (void)num_digits; (void)key1; (void)mono1;
	(void)key2; (void)mono2; (void)n; (void)modulus;
}

}  // namespace

void EltwiseExternalProductMod(uint64_t* acc0, uint64_t* acc1,
		const uint64_t* const* digits, uint64_t num_digits,
		const uint64_t* const* key1, const uint64_t* mono1,
		const uint64_t* const* key2, const uint64_t* mono2,
		uint64_t n, uint64_t modulus, bool accumulate) {
	CheckArgs(acc0, acc1, digits, num_digits, key1, mono1, key2, mono2, n, modulus);
	HEXL_CHECK(modulus < (1ULL << 62), "Require modulus < 2**62");

#ifdef HEXL_HAS_RVV
	if (has_rvv) {
		if (modulus < kMaxModulusRVV32) {
			HEXL_VLOG(3, "Calling ExternalProductModRVV32<uint64_t>");
			ExternalProductModRVV32<uint64_t>(acc0, acc1, digits, num_digits, key1, mono1, key2, mono2,
					n, modulus, accumulate);
		} else {
			HEXL_VLOG(3, "Calling ExternalProductModRVV64");
			ExternalProductModRVV64(acc0, acc1, digits, num_digits, key1, mono1, key2, mono2,
					n, modulus, accumulate);
		}
		return;
	}
#endif

	HEXL_VLOG(3, "Calling ExternalProductModNative<uint64_t>");
	ExternalProductModNative<uint64_t>(acc0, acc1, digits, num_digits, key1, mono1, key2, mono2,
			n, modulus, accumulate);
}

void EltwiseExternalProductMod(uint32_t* acc0, uint32_t* acc1,
		const uint32_t* const* digits, uint64_t num_digits,
		const uint32_t* const* key1, const uint32_t* mono1,
		const uint32_t* const* key2, const uint32_t* mono2,
		uint64_t n, uint64_t modulus, bool accumulate) {
	CheckArgs(acc0, acc1, digits, num_digits, key1, mono1, key2, mono2, n, modulus);
	HEXL_CHECK(modulus < (1ULL << 32), "Require modulus < 2**32");

#ifdef HEXL_HAS_RVV
	if (has_rvv && modulus < kMaxModulusRVV32) {
		HEXL_VLOG(3, "Calling ExternalProductModRVV32<uint32_t>");
		ExternalProductModRVV32<uint32_t>(acc0, acc1, digits, num_digits, key1, mono1, key2, mono2,
				n, modulus, accumulate);
		return;
	}
#endif

	HEXL_VLOG(3, "Calling ExternalProductModNative<uint32_t>");
	ExternalProductModNative<uint32_t>(acc0, acc1, digits, num_digits, key1, mono1, key2, mono2,
			n, modulus, accumulate);
}

// ---------------------------------------------------------------------------
// Native (scalar) kernel.
// ---------------------------------------------------------------------------

namespace {

// The blocked loop of ExternalProductModNative. kLazy (modulus < 2^32): a product is below
// 2^64, so the sums add raw products (s += d y) and BarrettReduce64 runs only every kmax terms,
// before a sum could overflow, and once per sum at the end; otherwise every term is a full
// MulAddModBarrett (the 128-bit pre-shift Barrett) and the sums stay in [0, q).\n// The element loops carry #pragma GCC novector: GCC -O3 would vectorize the lazy ones into RVV.
template <bool kLazy, typename Word>
void ExternalProductBlocked(Word* acc0, Word* acc1,
		const Word* const* digits, uint64_t num_digits,
		const Word* const* key1, const Word* mono1,
		const Word* const* key2, const Word* mono2,
		uint64_t n, uint64_t modulus, bool accumulate) {

	Word* acc[2] = {acc0, acc1};
	const uint8_t shift = kLazy ? 0 : MSB(modulus) - 1;
	const uint64_t mu = kLazy ? 0 : MultiplyFactor(uint64_t(1U) << shift, 64, modulus).BarrettFactor();
	const uint64_t q_barr = kLazy ? MultiplyFactor(1, 64, modulus).BarrettFactor() : 0;
	// terms a sum below q can take before it could overflow: s + kmax (q - 1)^2 < 2^64
	const uint64_t kmax = kLazy ? (~uint64_t{0} - (modulus - 1)) / ((modulus - 1) * (modulus - 1)) : 0;

	const auto reduce = [&](uint64_t x) { return kLazy ? BarrettReduce64(x, modulus, q_barr) : x; };
	const auto mul_add = [&](uint64_t x, uint64_t y, uint64_t c) {  // x, y, c < q
		return kLazy ? BarrettReduce64(x * y + c, modulus, q_barr) : MulAddModBarrett(x, y, c, modulus, mu, shift);
	};

	// Blocks of kBlock elements, term by term: a digit, a key and the sums streamed at a time.
	// Element by element, every element reads all 5k + 4 polynomials (34 in CGGI at k = 6),
	// and once they leave L1 nearly every line misses: 0.94 misses per line and 1.25x the
	// time of blocking at N = 1024 on the X100 (rvv-hexl results/extprod-native-2026-10-29).
	constexpr uint64_t kBlock = 256;
	uint64_t s1[2][kBlock], s2[2][kBlock];  // the sums of one block (8 KB, in L1)

	for (uint64_t e0 = 0; e0 < n; e0 += kBlock) {
		const uint64_t len = std::min(kBlock, n - e0);
		for (int j = 0; j < 2; ++j) {
#pragma GCC novector
			for (uint64_t e = 0; e < len; ++e) {
				s1[j][e] = accumulate && !mono1 ? acc[j][e0 + e] : 0;
				s2[j][e] = 0;
			}
		}

		uint64_t terms = 0;  // raw products in the sums since their last reduction (kLazy)
		for (uint64_t i = 0; i < num_digits; ++i) {
			if (kLazy && terms == kmax) {
				for (int j = 0; j < 2; ++j) {
#pragma GCC novector
					for (uint64_t e = 0; e < len; ++e) {
						s1[j][e] = reduce(s1[j][e]);
						s2[j][e] = reduce(s2[j][e]);
					}
				}
				terms = 0;
			}
			++terms;

			const Word* d = digits[i] + e0;
			for (int j = 0; j < 2; ++j) {
				const Word* y = key1[2 * i + j] + e0;
#pragma GCC novector
				for (uint64_t e = 0; e < len; ++e) {
					s1[j][e] = kLazy ? s1[j][e] + uint64_t{d[e]} * y[e] : MulAddModBarrett(d[e], y[e], s1[j][e], modulus, mu, shift);
				}
			}
			if (key2) {
				for (int j = 0; j < 2; ++j) {
					const Word* y = key2[2 * i + j] + e0;
#pragma GCC novector
					for (uint64_t e = 0; e < len; ++e) {
						s2[j][e] = kLazy ? s2[j][e] + uint64_t{d[e]} * y[e] : MulAddModBarrett(d[e], y[e], s2[j][e], modulus, mu, shift);
					}
				}
			}
		}

		for (int j = 0; j < 2; ++j) {
#pragma GCC novector
			for (uint64_t e = 0; e < len; ++e) {
				uint64_t r = reduce(s1[j][e]);
				if (mono1) r = mul_add(r, mono1[e0 + e], accumulate ? acc[j][e0 + e] : 0);
				if (key2) {
					const uint64_t t = reduce(s2[j][e]);
					r = mono2 ? mul_add(t, mono2[e0 + e], r) : AddUIntMod(r, t, modulus);
				}
				acc[j][e0 + e] = static_cast<Word>(r);
			}
		}
	}
}

}  // namespace

template <typename Word>
void ExternalProductModNative(Word* acc0, Word* acc1,
		const Word* const* digits, uint64_t num_digits,
		const Word* const* key1, const Word* mono1,
		const Word* const* key2, const Word* mono2,
		uint64_t n, uint64_t modulus, bool accumulate) {
	if (modulus < (1ULL << 32)) {  // binfhe's Q: lazy 64-bit sums
		ExternalProductBlocked<true, Word>(acc0, acc1, digits, num_digits, key1, mono1, key2, mono2,
				n, modulus, accumulate);
	} else {
		ExternalProductBlocked<false, Word>(acc0, acc1, digits, num_digits, key1, mono1, key2, mono2,
				n, modulus, accumulate);
	}
}

template void ExternalProductModNative<uint64_t>(uint64_t*, uint64_t*, const uint64_t* const*, uint64_t,
		const uint64_t* const*, const uint64_t*, const uint64_t* const*, const uint64_t*, uint64_t, uint64_t, bool);
template void ExternalProductModNative<uint32_t>(uint32_t*, uint32_t*, const uint32_t* const*, uint64_t,
		const uint32_t* const*, const uint32_t*, const uint32_t* const*, const uint32_t*, uint64_t, uint64_t, bool);

}  // namespace hexl
}  // namespace intel
