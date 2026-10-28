// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
//
// RVV kernel for BaseConvert (rvv-hexl extension): OpenFHE's approximate RNS
// base conversion, BaseConvertNative's arithmetic (eltwise-base-convert-
// internal.hpp) on strips of vl coefficients, e64 lanes:
//   y_i   = x_i * q_hat_inv_i mod q_i: lazy Shoup + ReduceFromTwice, exact
//   acc_j = sum_i lazy Shoup (y_i * q_hat_mod_p[i][j] mod p_j), each add
//           followed by ReduceFromTwice(., 2 p_j): [0, 2 p_j)
//   out_j = ReduceFromTwice(acc_j, p_j)
// Every constant is a scalar (vmulhu.vx / vmul.vx / vnmsac.vx). Vector values
// cannot live in arrays and L, K are run-time values, so:
//   L = 1 (OpenFHE CKKS ModUp, one limb per digit, and ModDown, one P limb):
//         y stays in a register, each output is one product, one reduction,
//         one store (no accumulation);
//   L > 1: per strip the y_i go to a per-thread scratch (L * VLMAX words, in
//         L1), then each output accumulates in one register over i.
// Inputs are read once and outputs written once, unit-stride.

#include <stdint.h>

#include "eltwise/eltwise-base-convert-internal.hpp"

#ifdef HEXL_HAS_RVV
#include <riscv_vector.h>

#include "hexl/util/aligned-allocator.hpp"
#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

namespace {
// The y_i of one strip, L * VLMAX words. One buffer per thread (OpenFHE may
// convert from OpenMP threads), grown to the largest size seen, never shrunk.
uint64_t* BaseConvertScratch(size_t words) {
	thread_local AlignedVector64<uint64_t> scratch;
	if (scratch.size() < words) scratch.resize(words);
	return scratch.data();
}
}  // namespace

template <class V>
void BaseConvertRVV64(uint64_t* const* result, const uint64_t* const* operand,
		uint64_t n, const BaseConvertArgs& a) {
	const uint64_t L = a.in_count, K = a.out_count;
	if (L == 1) {  // y in a register
		const uint64_t q = a.in_moduli[0], w = a.q_hat_inv[0], wp = a.q_hat_inv_precon[0];
		for (size_t k = 0, vl; k < n; k += vl){
			vl = rvv::SetVl<V>(n - k);
			const V y = rvv::ReduceFromTwice(rvv::MulModShoupLazy(rvv::Load<V>(operand[0] + k, vl), w, wp, q, vl), q, vl);  // [0, q)
			for (uint64_t j = 0; j < K; ++j){
				const uint64_t p = a.out_moduli[j];
				const V t = rvv::MulModShoupLazy(y, a.q_hat_mod_p[j], a.q_hat_mod_p_precon[j], p, vl);  // [0, 2p)
				rvv::Store(result[j] + k, rvv::ReduceFromTwice(t, p, vl), vl);                          // [0, p)
			}
		}
		return;
	}
	const size_t vlmax = rvv::VlMax<V>();
	uint64_t* ys = BaseConvertScratch(L * vlmax);  // y_i at ys + i * vlmax
	for (size_t k = 0, vl; k < n; k += vl){
		vl = rvv::SetVl<V>(n - k);
		for (uint64_t i = 0; i < L; ++i){                                      // y_i, exact
			const uint64_t q = a.in_moduli[i];
			const V x = rvv::Load<V>(operand[i] + k, vl);
			rvv::Store(ys + i * vlmax, rvv::ReduceFromTwice(rvv::MulModShoupLazy(x, a.q_hat_inv[i], a.q_hat_inv_precon[i], q, vl), q, vl), vl);
		}
		for (uint64_t j = 0; j < K; ++j){                                      // out_j
			const uint64_t p = a.out_moduli[j], two_p = 2 * p;
			V s = rvv::MulModShoupLazy(rvv::Load<V>(ys, vl), a.q_hat_mod_p[j], a.q_hat_mod_p_precon[j], p, vl);  // [0, 2p)
			for (uint64_t i = 1; i < L; ++i){
				const V t = rvv::MulModShoupLazy(rvv::Load<V>(ys + i * vlmax, vl), a.q_hat_mod_p[i * K + j],
						a.q_hat_mod_p_precon[i * K + j], p, vl);                    // [0, 2p)
				s = rvv::ReduceFromTwice(__riscv_vadd(s, t, vl), two_p, vl);   // < 4p before: [0, 2p)
			}
			rvv::Store(result[j] + k, rvv::ReduceFromTwice(s, p, vl), vl);     // [0, p)
		}
	}
}

template void BaseConvertRVV64<vuint64m1_t>(uint64_t* const*, const uint64_t* const*, uint64_t, const BaseConvertArgs&);
template void BaseConvertRVV64<vuint64m2_t>(uint64_t* const*, const uint64_t* const*, uint64_t, const BaseConvertArgs&);
template void BaseConvertRVV64<vuint64m4_t>(uint64_t* const*, const uint64_t* const*, uint64_t, const BaseConvertArgs&);
template void BaseConvertRVV64<vuint64m8_t>(uint64_t* const*, const uint64_t* const*, uint64_t, const BaseConvertArgs&);

// STUB: calls the native kernel until the RVV body is written (BaseConvertRVV64's
// strips plus the FP64 nu accumulator and the alpha row per coefficient).
template <class V>
void BaseConvertExactRVV64(uint64_t* const* result, const uint64_t* const* operand,
		uint64_t n, const BaseConvertArgs& a) {
	static_assert(rvv::IsE64<V>, "BaseConvertExactRVV64 takes e64 lanes");
	BaseConvertExactNative(result, operand, n, a);
}

template void BaseConvertExactRVV64<vuint64m1_t>(uint64_t* const*, const uint64_t* const*, uint64_t, const BaseConvertArgs&);
template void BaseConvertExactRVV64<vuint64m2_t>(uint64_t* const*, const uint64_t* const*, uint64_t, const BaseConvertArgs&);
template void BaseConvertExactRVV64<vuint64m4_t>(uint64_t* const*, const uint64_t* const*, uint64_t, const BaseConvertArgs&);
template void BaseConvertExactRVV64<vuint64m8_t>(uint64_t* const*, const uint64_t* const*, uint64_t, const BaseConvertArgs&);

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
