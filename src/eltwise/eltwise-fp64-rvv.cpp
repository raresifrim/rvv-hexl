// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// SPDX-License-Identifier: Apache-2.0
//
// FP64 eltwise kernels (rvv-hexl extension): EltwiseMultMod, EltwiseFMAMod and
// EltwiseMultAddMod on the FP64 FMA, for 2^30 <= q < 2^50 (method and
// conditions: util/fp64-mod.hpp). Each strip: inputs reduced to [0, q) in
// integer lanes as the integer kernels do, converted (vfcvt.f.xu), multiplied
// as in fp64-mod.hpp (the addend c of FMAMod / MultAddMod folded into the
// quotient estimate, so the remainder stays in (-q, q)), converted back
// (vfcvt.x.f, signed) and corrected in integer (+ q when negative).

#include <stdint.h>

#include <algorithm>

#include "eltwise/eltwise-fma-mod-internal.hpp"
#include "eltwise/eltwise-mult-add-mod-internal.hpp"
#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "util/fp64-mod.hpp"

#ifdef HEXL_HAS_RVV
#include <riscv_vector.h>
#include "util/rvv-util.hpp"
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
	using VU = rvv::UintOf<VF>;
	const double qd = static_cast<double>(modulus);
	const double qi = 1.0 / qd;
	const double C = 0x1.8p52;
	for (size_t vl; n > 0; n -= vl, result += vl, operand1 += vl, operand2 += vl) {
		vl = rvv::SetVl<VU>(n);
		VU v1 = rvv::Load<VU>(operand1, vl);
		VU v2 = rvv::Load<VU>(operand2, vl);
		if constexpr (InputModFactor >= 4) { v1 = rvv::ReduceFromTwice(v1, 2 * modulus, vl); v2 = rvv::ReduceFromTwice(v2, 2 * modulus, vl); }
		if constexpr (InputModFactor >= 2) { v1 = rvv::ReduceFromTwice(v1, modulus, vl); v2 = rvv::ReduceFromTwice(v2, modulus, vl); }
		VF a = __riscv_vfcvt_f(v1, vl);
		VF b = __riscv_vfcvt_f(v2, vl);
		VF h = __riscv_vfmul(a, b, vl);                                                  // a b rounded
		VF l = __riscv_vfmsub(a, b, h, vl);                                              // a b - h, exact
		VF t = __riscv_vfsub(__riscv_vfadd(__riscv_vfmul(h, qi, vl), C, vl), C, vl);    // round(h / q)
		VF r = __riscv_vfadd(__riscv_vfnmsac(h, qd, t, vl), l, vl);                      // a b - t q, in (-q, q)
		VU ru = rvv::AsUnsigned(__riscv_vfcvt_x(r, vl));
		rvv::Store(result, __riscv_vminu(ru, __riscv_vadd(ru, modulus, vl), vl), vl);         // + q when negative
	}
}


template <int InputModFactor, class VF>
void EltwiseFMAModRVVF64(uint64_t* result, const uint64_t* arg1, uint64_t arg2, const uint64_t* arg3, uint64_t n,
		uint64_t modulus) {
	using VU = rvv::UintOf<VF>;
	const double qd = static_cast<double>(modulus);
	const double qi = 1.0 / qd;
	const double C = 0x1.8p52;
	uint64_t w = arg2;  // the scalar, reduced once to [0, q)
	if constexpr (InputModFactor == 8) w = std::min(w, w - 4 * modulus);
	if constexpr (InputModFactor >= 4) w = std::min(w, w - 2 * modulus);
	if constexpr (InputModFactor >= 2) w = std::min(w, w - modulus);
	const double wd = static_cast<double>(w);
	const auto reduce = [modulus](VU v, size_t vl) {  // [0, F q) -> [0, q), F up to 8
		if constexpr (InputModFactor == 8) v = rvv::ReduceFromTwice(v, 4 * modulus, vl);
		if constexpr (InputModFactor >= 4) v = rvv::ReduceFromTwice(v, 2 * modulus, vl);
		if constexpr (InputModFactor >= 2) v = rvv::ReduceFromTwice(v, modulus, vl);
		return v;
	};
	if (arg3 != nullptr) {
		for (size_t vl; n > 0; n -= vl, result += vl, arg1 += vl, arg3 += vl) {
			vl = rvv::SetVl<VU>(n);
			VF a = __riscv_vfcvt_f(reduce(rvv::Load<VU>(arg1, vl), vl), vl);
			VF c = __riscv_vfcvt_f(reduce(rvv::Load<VU>(arg3, vl), vl), vl);
			VF h = __riscv_vfmul(a, wd, vl);                                                 // a w rounded
			VF l = __riscv_vfmsub(a, wd, h, vl);                                             // a w - h, exact
			VF s = __riscv_vfadd(h, c, vl);                                                  // ~ a w + c, for the estimate
			VF t = __riscv_vfsub(__riscv_vfadd(__riscv_vfmul(s, qi, vl), C, vl), C, vl);    // round((a w + c) / q)
			VF r = __riscv_vfadd(__riscv_vfadd(__riscv_vfnmsac(h, qd, t, vl), l, vl), c, vl);  // a w + c - t q, in (-q, q)
			VU ru = rvv::AsUnsigned(__riscv_vfcvt_x(r, vl));
			rvv::Store(result, __riscv_vminu(ru, __riscv_vadd(ru, modulus, vl), vl), vl);         // + q when negative
		}
	} else {
		for (size_t vl; n > 0; n -= vl, result += vl, arg1 += vl) {
			vl = rvv::SetVl<VU>(n);
			VF a = __riscv_vfcvt_f(reduce(rvv::Load<VU>(arg1, vl), vl), vl);
			VF h = __riscv_vfmul(a, wd, vl);                                                 // a w rounded
			VF l = __riscv_vfmsub(a, wd, h, vl);                                             // a w - h, exact
			VF t = __riscv_vfsub(__riscv_vfadd(__riscv_vfmul(h, qi, vl), C, vl), C, vl);    // round(h / q)
			VF r = __riscv_vfadd(__riscv_vfnmsac(h, qd, t, vl), l, vl);                      // a w - t q, in (-q, q)
			VU ru = rvv::AsUnsigned(__riscv_vfcvt_x(r, vl));
			rvv::Store(result, __riscv_vminu(ru, __riscv_vadd(ru, modulus, vl), vl), vl);         // + q when negative
		}
	}
}

template <int InputModFactor, class VF>
void EltwiseMultAddModRVVF64(uint64_t* result, const uint64_t* arg1, const uint64_t* arg2, const uint64_t* arg3,
		uint64_t n, uint64_t modulus) {
	using VU = rvv::UintOf<VF>;
	const double qd = static_cast<double>(modulus);
	const double qi = 1.0 / qd;
	const double C = 0x1.8p52;
	// result may be arg3 (acc += a b): each strip loads arg3 before it stores result
	for (size_t vl; n > 0; n -= vl, result += vl, arg1 += vl, arg2 += vl, arg3 += vl) {
		vl = rvv::SetVl<VU>(n);
		VU v1 = rvv::Load<VU>(arg1, vl);
		VU v2 = rvv::Load<VU>(arg2, vl);
		VU v3 = rvv::Load<VU>(arg3, vl);
		if constexpr (InputModFactor >= 4) { v1 = rvv::ReduceFromTwice(v1, 2 * modulus, vl); v2 = rvv::ReduceFromTwice(v2, 2 * modulus, vl); v3 = rvv::ReduceFromTwice(v3, 2 * modulus, vl); }
		if constexpr (InputModFactor >= 2) { v1 = rvv::ReduceFromTwice(v1, modulus, vl); v2 = rvv::ReduceFromTwice(v2, modulus, vl); v3 = rvv::ReduceFromTwice(v3, modulus, vl); }
		VF a = __riscv_vfcvt_f(v1, vl);
		VF b = __riscv_vfcvt_f(v2, vl);
		VF c = __riscv_vfcvt_f(v3, vl);
		VF h = __riscv_vfmul(a, b, vl);                                                  // a b rounded
		VF l = __riscv_vfmsub(a, b, h, vl);                                              // a b - h, exact
		VF s = __riscv_vfadd(h, c, vl);                                                  // ~ a b + c, for the estimate
		VF t = __riscv_vfsub(__riscv_vfadd(__riscv_vfmul(s, qi, vl), C, vl), C, vl);    // round((a b + c) / q)
		VF r = __riscv_vfadd(__riscv_vfadd(__riscv_vfnmsac(h, qd, t, vl), l, vl), c, vl);  // a b + c - t q, in (-q, q)
		VU ru = rvv::AsUnsigned(__riscv_vfcvt_x(r, vl));
		rvv::Store(result, __riscv_vminu(ru, __riscv_vadd(ru, modulus, vl), vl), vl);         // + q when negative
	}
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
