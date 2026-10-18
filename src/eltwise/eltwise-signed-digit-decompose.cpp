// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// SPDX-License-Identifier: Apache-2.0

#include "hexl/eltwise/eltwise-signed-digit-decompose.hpp"

#include "eltwise/eltwise-signed-digit-decompose-internal.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/util/check.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

namespace {

template <typename Word>
void CheckArgs(Word* const* result, const Word* operand, uint64_t n,
		uint64_t modulus, uint64_t log_base, uint64_t num_digits) {
	HEXL_CHECK(result != nullptr, "Require result != nullptr");
	HEXL_CHECK(operand != nullptr, "Require operand != nullptr");
	HEXL_CHECK(n != 0, "Require n != 0");
	HEXL_CHECK(modulus > 1, "Require modulus > 1");
	HEXL_CHECK(num_digits >= 1, "Require num_digits >= 1");
	for (uint64_t d = 0; d < num_digits; ++d) {
		HEXL_CHECK(result[d] != nullptr, "Require result[" << d << "] != nullptr");
		HEXL_CHECK(result[d] != operand, "Require result[" << d << "] != operand");
	}
	HEXL_CHECK_BOUNDS(operand, n, modulus, "operand exceeds bound " << modulus);
	(void)result; (void)operand; (void)n; (void)modulus; (void)log_base; (void)num_digits;
}

}  // namespace

void SignedDigitDecompose(uint64_t* const* result, const uint64_t* operand,
		uint64_t n, uint64_t modulus, uint64_t log_base, uint64_t num_digits) {
	CheckArgs(result, operand, n, modulus, log_base, num_digits);
	HEXL_CHECK(modulus < (1ULL << 62), "Require modulus < 2**62");
	HEXL_CHECK(log_base >= 1 && log_base <= 62, "Require log_base in [1, 62]");

#ifdef HEXL_HAS_RVV
	if (has_rvv) {
		if (modulus < (1ULL << 31) && log_base <= 30) {
			HEXL_VLOG(3, "Calling SignedDigitDecomposeRVV32<uint64_t>");
			SignedDigitDecomposeRVV32<uint64_t>(result, operand, n, modulus, log_base, num_digits);
		} else {
			HEXL_VLOG(3, "Calling SignedDigitDecomposeRVV64");
			SignedDigitDecomposeRVV64(result, operand, n, modulus, log_base, num_digits);
		}
		return;
	}
#endif

	HEXL_VLOG(3, "Calling SignedDigitDecomposeNative<uint64_t>");
	SignedDigitDecomposeNative<uint64_t>(result, operand, n, modulus, log_base, num_digits);
}

void SignedDigitDecompose(uint32_t* const* result, const uint32_t* operand,
		uint64_t n, uint64_t modulus, uint64_t log_base, uint64_t num_digits) {
	CheckArgs(result, operand, n, modulus, log_base, num_digits);
	HEXL_CHECK(modulus < (1ULL << 31), "Require modulus < 2**31");
	HEXL_CHECK(log_base >= 1 && log_base <= 30, "Require log_base in [1, 30]");

#ifdef HEXL_HAS_RVV
	if (has_rvv) {
		HEXL_VLOG(3, "Calling SignedDigitDecomposeRVV32<uint32_t>");
		SignedDigitDecomposeRVV32<uint32_t>(result, operand, n, modulus, log_base, num_digits);
		return;
	}
#endif

	HEXL_VLOG(3, "Calling SignedDigitDecomposeNative<uint32_t>");
	SignedDigitDecomposeNative<uint32_t>(result, operand, n, modulus, log_base, num_digits);
}

// ---------------------------------------------------------------------------
// Native (scalar) kernel.
// ---------------------------------------------------------------------------

template <typename Word>
void SignedDigitDecomposeNative(Word* const* result, const Word* operand,
		uint64_t n, uint64_t modulus, uint64_t log_base, uint64_t num_digits) {
	// STUB: replace with the native kernel
	SignedDigitDecomposeStub<Word>(result, operand, n, modulus, log_base, num_digits);
}

template void SignedDigitDecomposeNative<uint64_t>(uint64_t* const*, const uint64_t*, uint64_t, uint64_t, uint64_t, uint64_t);
template void SignedDigitDecomposeNative<uint32_t>(uint32_t* const*, const uint32_t*, uint64_t, uint64_t, uint64_t, uint64_t);

}  // namespace hexl
}  // namespace intel
