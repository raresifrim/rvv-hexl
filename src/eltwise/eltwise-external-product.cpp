// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.

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

// STUB: the defining sums, one exact product per term (the reference). To be
// replaced by the fused loop (lazy products, one correction per sum).
template <typename Word>
void ExternalProductModNative(Word* acc0, Word* acc1,
		const Word* const* digits, uint64_t num_digits,
		const Word* const* key1, const Word* mono1,
		const Word* const* key2, const Word* mono2,
		uint64_t n, uint64_t modulus, bool accumulate) {
	ExternalProductModReference<Word>(acc0, acc1, digits, num_digits, key1, mono1, key2, mono2,
			n, modulus, accumulate);
}

template void ExternalProductModNative<uint64_t>(uint64_t*, uint64_t*, const uint64_t* const*, uint64_t,
		const uint64_t* const*, const uint64_t*, const uint64_t* const*, const uint64_t*, uint64_t, uint64_t, bool);
template void ExternalProductModNative<uint32_t>(uint32_t*, uint32_t*, const uint32_t* const*, uint64_t,
		const uint32_t* const*, const uint32_t*, const uint32_t* const*, const uint32_t*, uint64_t, uint64_t, bool);

}  // namespace hexl
}  // namespace intel
