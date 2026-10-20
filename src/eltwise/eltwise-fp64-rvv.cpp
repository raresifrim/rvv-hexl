// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// SPDX-License-Identifier: Apache-2.0
//
// FP64 eltwise kernels (rvv-hexl extension): EltwiseMultMod, EltwiseFMAMod and
// EltwiseMultAddMod on the FP64 FMA, for 2^30 <= q < 2^50 (method and
// conditions: util/fp64-mod.hpp). STUBS until written: each calls its e64
// integer kernel, so the dispatch, the tests and the benchmarks work from the
// start. Each strip: inputs reduced to [0, q) as the integer kernels do, then
// converted (vfcvt.f.xu), multiplied as in fp64-mod.hpp, the remainder
// converted back (vfcvt.x.f) and corrected in integer.

#include <stdint.h>

#include "eltwise/eltwise-fma-mod-internal.hpp"
#include "eltwise/eltwise-mult-add-mod-internal.hpp"
#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "util/fp64-mod.hpp"

#ifdef HEXL_HAS_RVV
#include <riscv_vector.h>
#endif

namespace intel {
namespace hexl {

// The bounds the FP64 eltwise kernels rely on, checked at compile time.
static_assert(!ModMulFp64Applicable(kFp64MinModulus - 1) && ModMulFp64Applicable(kFp64MinModulus));
static_assert(ModMulFp64Applicable(kFp64MaxModulus - 1) && !ModMulFp64Applicable(kFp64MaxModulus));
static_assert(8 * (kFp64MaxModulus - 1) < (1ULL << 53), "EltwiseFMAMod's inputs (< 8q) must be exact doubles");

#ifdef HEXL_HAS_RVV

template <int InputModFactor, class VF>
void EltwiseMultModRVVF64(uint64_t* result, const uint64_t* operand1, const uint64_t* operand2, uint64_t n,
		uint64_t modulus) {
	// STUB: replace with the FP64 kernel (util/fp64-mod.hpp)
	EltwiseMultModRVV64<InputModFactor>(result, operand1, operand2, n, modulus);
}

template <int InputModFactor, class VF>
void EltwiseFMAModRVVF64(uint64_t* result, const uint64_t* arg1, uint64_t arg2, const uint64_t* arg3, uint64_t n,
		uint64_t modulus) {
	// STUB: replace with the FP64 kernel (util/fp64-mod.hpp)
	EltwiseFMAModRVV64<InputModFactor>(result, arg1, arg2, arg3, n, modulus);
}

template <int InputModFactor, class VF>
void EltwiseMultAddModRVVF64(uint64_t* result, const uint64_t* arg1, const uint64_t* arg2, const uint64_t* arg3,
		uint64_t n, uint64_t modulus) {
	// STUB: replace with the FP64 kernel (util/fp64-mod.hpp)
	EltwiseMultAddModRVV64<InputModFactor>(result, arg1, arg2, arg3, n, modulus);
}

// Every input mod factor at every FP64 lane type, so tests and benchmarks can
// compare them; the dispatch uses rvv::cfg::MultModF64 / FMAModF64 / MultAddModF64.
#define HEXL_ELTWISE_RVVF64(F, VF)                                                                                   \
	template void EltwiseMultModRVVF64<F, VF>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);      \
	template void EltwiseFMAModRVVF64<F, VF>(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, uint64_t,        \
			uint64_t);                                                                                                   \
	template void EltwiseMultAddModRVVF64<F, VF>(uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*,       \
			uint64_t, uint64_t)
#define HEXL_ELTWISE_RVVF64_LANES(F)       \
	HEXL_ELTWISE_RVVF64(F, vfloat64m1_t); \
	HEXL_ELTWISE_RVVF64(F, vfloat64m2_t); \
	HEXL_ELTWISE_RVVF64(F, vfloat64m4_t); \
	HEXL_ELTWISE_RVVF64(F, vfloat64m8_t)
HEXL_ELTWISE_RVVF64_LANES(1);
HEXL_ELTWISE_RVVF64_LANES(2);
HEXL_ELTWISE_RVVF64_LANES(4);
#undef HEXL_ELTWISE_RVVF64_LANES
#undef HEXL_ELTWISE_RVVF64

// EltwiseFMAMod alone also takes input_mod_factor 8
template void EltwiseFMAModRVVF64<8, vfloat64m1_t>(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, uint64_t,
		uint64_t);
template void EltwiseFMAModRVVF64<8, vfloat64m2_t>(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, uint64_t,
		uint64_t);
template void EltwiseFMAModRVVF64<8, vfloat64m4_t>(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, uint64_t,
		uint64_t);
template void EltwiseFMAModRVVF64<8, vfloat64m8_t>(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, uint64_t,
		uint64_t);

#endif  // HEXL_HAS_RVV

}  // namespace hexl
}  // namespace intel
