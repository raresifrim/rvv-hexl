// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#include "hexl/eltwise/eltwise-cmp-sub-mod.hpp"

#include <algorithm>

#include "eltwise/eltwise-cmp-sub-mod-internal.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "util/cpu-features.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

void EltwiseCmpSubMod(uint64_t* result, const uint64_t* operand1, uint64_t n,
		uint64_t modulus, CMPINT cmp, uint64_t bound,
		uint64_t diff) {
	HEXL_CHECK(result != nullptr, "Require result != nullptr");
	HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
	HEXL_CHECK(n != 0, "Require n != 0");
	HEXL_CHECK(modulus > 1, "Require modulus > 1");
	// The branch-free conditional subtract, min(d, d + q), needs d + q < 2^64
	// for every d < q. OpenFHE (<= 60 bits) and SEAL (<= 61 bits) are far below.
	HEXL_CHECK(modulus <= (1ULL << 63), "Require modulus <= 2**63");
	HEXL_CHECK(diff != 0, "Require diff != 0");
	HEXL_CHECK(diff < modulus, "Diff " << diff << " >= modulus " << modulus);

#ifdef HEXL_HAS_RVV
	if (has_rvv) {
		HEXL_VLOG(3, "Calling EltwiseCmpSubModRVV<uint64_t>");
		EltwiseCmpSubModRVV<uint64_t>(result, operand1, n, modulus, cmp, bound, diff);
		return;
	}
#endif

	HEXL_VLOG(3, "Calling EltwiseCmpSubModNative<uint64_t>");
	EltwiseCmpSubModNative<uint64_t>(result, operand1, n, modulus, cmp, bound, diff);
}

// ---- rvv-hexl extension: 32-bit storage ----------------------------------

void EltwiseCmpSubMod(uint32_t* result, const uint32_t* operand1, uint64_t n,
		uint64_t modulus, CMPINT cmp, uint64_t bound,
		uint64_t diff) {
	HEXL_CHECK(result != nullptr, "Require result != nullptr");
	HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
	HEXL_CHECK(n != 0, "Require n != 0");
	HEXL_CHECK(modulus > 1, "Require modulus > 1");
	HEXL_CHECK(diff != 0, "Require diff != 0");
	HEXL_CHECK(diff < modulus, "Diff " << diff << " >= modulus " << modulus);
	HEXL_CHECK(modulus <= (1ULL << 32), "Require modulus <= 2**32");
	HEXL_CHECK(bound < (1ULL << 32), "Require bound < 2**32");

#ifdef HEXL_HAS_RVV
	if (has_rvv && modulus < kMaxModulusRVV32) {
		HEXL_VLOG(3, "Calling EltwiseCmpSubModRVV<uint32_t>");
		EltwiseCmpSubModRVV<uint32_t>(result, operand1, n, modulus, cmp, bound, diff);
		return;
	}
#endif

	HEXL_VLOG(3, "Calling EltwiseCmpSubModNative<uint32_t>");
	EltwiseCmpSubModNative<uint32_t>(result, operand1, n, modulus, cmp, bound, diff);
}

// ---------------------------------------------------------------------------
// Native (scalar) kernel. 
// ---------------------------------------------------------------------------

// Per element: r = x mod q (Barrett), then subtract diff where hit, branch-free:
// select the subtrahend (sltu + czero), then the min(d, d + q) correction
// (minu). Requires q <= 2^63 (checked at the entry point) so that d + q cannot
// wrap. Measured on the K3 at n = 4096 (cycles/elem, X100 / A100): 5.8 / 21.6
// for NLE; selecting between two RESULTS instead made GCC emit a mispredicting
// branch (15.0 / 27.3), and calling the out-of-line SubUIntMod cost a call per
// element.
template <typename Word>
static inline Word CmpSubModOne(uint64_t x, bool hit, uint64_t modulus, uint64_t q_barr,
		uint64_t diff) {
	const uint64_t r = BarrettReduce64(x, modulus, q_barr);   // any value -> [0, q)
	const uint64_t d = r - (hit ? diff : 0U);                 // r - diff (may wrap) or r -> this uses czero instruction to avoid branching
	return (Word)std::min(d, d + modulus);                    // back into [0, q) -> this uses the minu instruction if present
}

// One loop per comparison, so the switch runs once per call. Every loop has
// "#pragma GCC novector": the library builds at -O3 with V enabled, and GCC's
// auto-vectorized Barrett loop is 3-4x slower than the scalar one on the K3.
template <typename Word>
void EltwiseCmpSubModNative(Word* result, const Word* operand1, uint64_t n, uint64_t modulus, CMPINT cmp, uint64_t bound, uint64_t diff) {

	const uint64_t q_barr = MultiplyFactor(1, 64, modulus).BarrettFactor();  // once per call

	switch (cmp) {
		case CMPINT::EQ:
#pragma GCC novector
			for (uint64_t i = 0; i < n; ++i) {
				result[i] = CmpSubModOne<Word>(operand1[i], operand1[i] == bound, modulus, q_barr, diff);
			}
			break;
		case CMPINT::LT:
#pragma GCC novector
			for (uint64_t i = 0; i < n; ++i) {
				result[i] = CmpSubModOne<Word>(operand1[i], operand1[i] < bound, modulus, q_barr, diff);
			}
			break;
		case CMPINT::LE:
#pragma GCC novector
			for (uint64_t i = 0; i < n; ++i) {
				result[i] = CmpSubModOne<Word>(operand1[i], operand1[i] <= bound, modulus, q_barr, diff);
			}
			break;
		case CMPINT::NE:
#pragma GCC novector
			for (uint64_t i = 0; i < n; ++i) {
				result[i] = CmpSubModOne<Word>(operand1[i], operand1[i] != bound, modulus, q_barr, diff);
			}
			break;
		case CMPINT::NLT:
#pragma GCC novector
			for (uint64_t i = 0; i < n; ++i) {
				result[i] = CmpSubModOne<Word>(operand1[i], operand1[i] >= bound, modulus, q_barr, diff);
			}
			break;
		case CMPINT::NLE:
#pragma GCC novector
			for (uint64_t i = 0; i < n; ++i) {
				result[i] = CmpSubModOne<Word>(operand1[i], operand1[i] > bound, modulus, q_barr, diff);
			}
			break;
		case CMPINT::TRUE:
#pragma GCC novector
			for (uint64_t i = 0; i < n; ++i) {
				result[i] = CmpSubModOne<Word>(operand1[i], true, modulus, q_barr, diff);
			}
			break;
		case CMPINT::FALSE:
			// no element is shifted, but every element is still reduced
#pragma GCC novector
			for (uint64_t i = 0; i < n; ++i) {
				result[i] = (Word)BarrettReduce64(operand1[i], modulus, q_barr);
			}
			break;
		default:
			// not a valid CMPINT; the scalar Compare() treats unknown values as TRUE
#pragma GCC novector
			for (uint64_t i = 0; i < n; ++i) {
				result[i] = CmpSubModOne<Word>(operand1[i], true, modulus, q_barr, diff);
			}
			break;
	}
}

}  // namespace hexl
}  // namespace intel
