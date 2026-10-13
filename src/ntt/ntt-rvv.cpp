// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// RVV NTT kernels: the RISC-V counterpart of upstream's fwd-ntt-avx512.cpp /
// inv-ntt-avx512.cpp, and the actual deliverable of "Etapa 1".
//
// Suggested order of work, straight from the D01 plan (each step measured on
// both K3 clusters before the next, against HEXL_DISABLE_RVV=1 and against
// stock OpenFHE):
//   1. Flatten the stage loop nest so the compiler / you get ONE inner loop
//      per stage over n/2 butterflies (est. ~1.24x end-to-end on its own).
//   2. Vectorise the final stages (t < VL) ACROSS blocks instead of running
//      them scalar: strided loads (vlse32 with stride 2t), segment loads
//      (vlseg2e32/vlseg4e32 for t = 1, 2) or replicated twiddle tables. The
//      microbenchmark already showed ~2x on X100 for this alone; on A100
//      avoid strided access (in-register transposes instead).
//   3. Block in the vector register file: radix-4 / radix-8 butterflies do
//      2-3 stages per pass over memory, raising arithmetic intensity toward
//      the roofline balance point (A100's 4x larger VRF gains the most).
//   4. Only then hand-tune instruction selection.
// Parameters: SEW=e32 whenever q < 2^30; LMUL m4 where a stage's butterfly
// span allows it (the Shoup multiply alone measured 1.5-2.4x over m1 on the
// X100), never below mf2. m8 leaves 4 register groups: a butterfly keeps more
// values live, so expect spills there. Lane types: rvv::cfg::Ntt64 / Ntt32 in
// util/rvv-config.hpp, with rvv::SetVl<V> / rvv::Load<V> / rvv::Store.

#include "ntt/ntt-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include <type_traits>

#include "hexl/number-theory/number-theory.hpp"
#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

template <typename Word, class V>
void ForwardTransformToBitReverseRVV32(Word* result,
		const Word* operand, uint64_t n,
		uint64_t modulus, const uint32_t* w,
		const uint32_t* w_precon,
		uint64_t input_mod_factor,
		uint64_t output_mod_factor) {

	uint32_t modulus32 = static_cast<uint32_t>(modulus);
	if (result != operand){ //copy operand into result, then work in place.
		Word *temp = result;
		for (size_t vl, m=n; m>0; m-=vl, operand+=vl, temp+=vl){
			vl = rvv::SetVl<V>(m);
			V v = rvv::Load32<V>(operand, vl);   // plain vle32 or narrowing load, by Word
			rvv::Store32(temp, v, vl);
		}
	}

	for (uint64_t t = n/2, m = 1; m < n; m <<= 1, t >>= 1){      // log2(n) stages
		for (uint64_t i = 0; i < m; i++){                          // m blocks
			uint32_t W = w[m + i];
			uint32_t Wp = w_precon[m + i];
			Word* x = result + 2*i*t;
			for (size_t j = 0, vl; j < t; j += vl){                   // t butterflies
				vl = rvv::SetVl<V>(t - j);
				V X = rvv::Load32<V>(x + j, vl);
				V Y = rvv::Load32<V>(x + j + t, vl);
				X = rvv::ReduceFromTwice32(X, 2*modulus32, vl);      // [0, 2q)
				V T  = rvv::MulModShoupLazy32(Y, W, Wp, modulus32, vl);         // in [0, 2q)
				V Xr = __riscv_vadd(X, T, vl);                                // [0, 4q)
				V Yr = __riscv_vsub(X, T, vl);
				Yr = __riscv_vadd(Yr, 2*modulus32, vl);                         // [0, 4q)
				rvv::Store32(x + j, Xr, vl);
				rvv::Store32(x + j + t, Yr, vl);
			}
		}
	}

	if (output_mod_factor == 1) 
		for (size_t vl; n>0; n-=vl, result+=vl){
			vl = rvv::SetVl<V>(n);
			V v = rvv::Load32<V>(result, vl);
			v = rvv::ReduceFromTwice32(v, 2*modulus32, vl);
			v = rvv::ReduceFromTwice32(v, modulus32, vl);
			rvv::Store32(result, v, vl);
		}

}

template <typename Word, class V>
void InverseTransformFromBitReverseRVV32(Word* result,
		const Word* operand, uint64_t n,
		uint64_t modulus, const uint32_t* w_inv,
		const uint32_t* w_inv_precon,
		uint64_t input_mod_factor,
		uint64_t output_mod_factor) {
	// Mirror of the RVV32 forward: Gentleman-Sande in 32-bit lanes, every load
	// and store through rvv::Load32 / Store32 (both storage words), 32-bit
	// twiddle tables in stage order, Harvey lazy butterflies in [0, 2q), and
	// the N^{-1} scaling folded into the last stage.
	const uint32_t modulus32 = static_cast<uint32_t>(modulus);
	const uint32_t twice_mod32 = 2*modulus32;
	if (result != operand){ //copy operand into result, then work in place.
		Word *temp = result;
		for (size_t vl, m=n; m>0; m-=vl, operand+=vl, temp+=vl){
			vl = rvv::SetVl<V>(m);
			V v = rvv::Load32<V>(operand, vl);   // plain vle32 or narrowing load, by Word
			rvv::Store32(temp, v, vl);
		}
	}

	uint64_t k = 1;                                                    // next inverse twiddle
	for (uint64_t t = 1, m = n/2; m > 1; m >>= 1, t <<= 1){            // all stages but the last
		for (uint64_t i = 0; i < m; i++, k++){                          // m blocks
			uint32_t W = w_inv[k];
			uint32_t Wp = w_inv_precon[k];
			Word* x = result + 2*i*t;
			for (size_t j = 0, vl; j < t; j += vl){                       // t butterflies
				vl = rvv::SetVl<V>(t - j);
				V X = rvv::Load32<V>(x + j, vl);                           // [0, 2q)
				V Y = rvv::Load32<V>(x + j + t, vl);                       // [0, 2q)
				V S = rvv::ReduceFromTwice32(__riscv_vadd(X, Y, vl), twice_mod32, vl);   // [0, 2q)
				V D = __riscv_vadd(__riscv_vsub(X, Y, vl), twice_mod32, vl);             // [0, 4q)
				rvv::Store32(x + j, S, vl);
				rvv::Store32(x + j + t, rvv::MulModShoupLazy32(D, W, Wp, modulus32, vl), vl);  // [0, 2q)
			}
		}
	}

	// Last stage (one block, t = n/2) with the N^{-1} scaling folded in:
	//   X' = (X + Y) * N^{-1},   Y' = (X - Y) * (W * N^{-1})
	const uint64_t t = n/2;
	const uint64_t n_inv = InverseMod(n, modulus);
	const uint64_t Wn = MultiplyMod(w_inv[k], n_inv, MultiplyFactor(n_inv, 64, modulus).BarrettFactor(), modulus);
	const uint32_t n_inv32 = static_cast<uint32_t>(n_inv);
	const uint32_t n_inv_p32 = static_cast<uint32_t>(MultiplyFactor(n_inv, 32, modulus).BarrettFactor());
	const uint32_t Wn32 = static_cast<uint32_t>(Wn);
	const uint32_t Wn_p32 = static_cast<uint32_t>(MultiplyFactor(Wn, 32, modulus).BarrettFactor());
	for (size_t j = 0, vl; j < t; j += vl){
		vl = rvv::SetVl<V>(t - j);
		V X = rvv::Load32<V>(result + j, vl);
		V Y = rvv::Load32<V>(result + j + t, vl);
		V Xr = rvv::MulModShoupLazy32(__riscv_vadd(X, Y, vl), n_inv32, n_inv_p32, modulus32, vl);                       // [0, 2q)
		V Yr = rvv::MulModShoupLazy32(__riscv_vadd(__riscv_vsub(X, Y, vl), twice_mod32, vl), Wn32, Wn_p32, modulus32, vl);  // [0, 2q)
		if (output_mod_factor == 1){
			Xr = rvv::ReduceFromTwice32(Xr, modulus32, vl);
			Yr = rvv::ReduceFromTwice32(Yr, modulus32, vl);
		}
		rvv::Store32(result + j, Xr, vl);
		rvv::Store32(result + j + t, Yr, vl);
	}
}

template <class V>
void ForwardTransformToBitReverseRVV64(
		uint64_t* result, const uint64_t* operand, uint64_t n, uint64_t modulus,
		const uint64_t* root_of_unity_powers,
		const uint64_t* precon_root_of_unity_powers, uint64_t input_mod_factor,
		uint64_t output_mod_factor) {

	if (result != operand){ //copy operand into result, then work in place.
		uint64_t *temp = result;
		for (size_t vl, m=n; m>0; m-=vl, operand+=vl, temp+=vl){
			vl = rvv::SetVl<V>(m);
			V v = rvv::Load<V>(operand, vl);
			rvv::Store<V>(temp, v, vl);
		}
	}

	for (uint64_t t = n/2, m = 1; m < n; m <<= 1, t >>= 1){      // log2(n) stages
		for (uint64_t i = 0; i < m; i++){                          // m blocks
			uint64_t W = root_of_unity_powers[m + i];
			uint64_t Wp = precon_root_of_unity_powers[m + i];
			uint64_t* x = result + 2*i*t;
			for (size_t j = 0, vl; j < t; j += vl){                   // t butterflies
				vl = rvv::SetVl<V>(t - j);
				V X = rvv::Load<V>(x + j, vl);
				V Y = rvv::Load<V>(x + j + t, vl);
				X = rvv::ReduceFromTwice(X, 2*modulus, vl);      // [0, 2q)
				V T  = rvv::MulModShoupLazy(Y, W, Wp, modulus, vl);         // in [0, 2q)
				V Xr = __riscv_vadd(X, T, vl);                                // [0, 4q)
				V Yr = __riscv_vsub(X, T, vl);
				Yr = __riscv_vadd(Yr, 2*modulus, vl);                         // [0, 4q)
				rvv::Store(x + j, Xr, vl);
				rvv::Store(x + j + t, Yr, vl);
			}
		}
	}

	if (output_mod_factor == 1) 
		for (size_t vl; n>0; n-=vl, result+=vl){
			vl = rvv::SetVl<V>(n);
			V v = rvv::Load<V>(result, vl);
			v = rvv::ReduceFromTwice(v, 2*modulus, vl);
			v = rvv::ReduceFromTwice(v, modulus, vl);
			rvv::Store(result, v, vl);
		}
}

template <class V>
void InverseTransformFromBitReverseRVV64(
		uint64_t* result, const uint64_t* operand, uint64_t n, uint64_t modulus,
		const uint64_t* inv_root_of_unity_powers,
		const uint64_t* precon_inv_root_of_unity_powers, uint64_t input_mod_factor,
		uint64_t output_mod_factor) {
	// Mirror of the RVV64 forward: Gentleman-Sande, each block's t butterflies
	// vectorized, twiddle table in stage order, Harvey lazy butterflies in
	// [0, 2q), and the N^{-1} scaling folded into the last stage.
	if (result != operand){ //copy operand into result, then work in place.
		uint64_t *temp = result;
		for (size_t vl, m=n; m>0; m-=vl, operand+=vl, temp+=vl){
			vl = rvv::SetVl<V>(m);
			V v = rvv::Load<V>(operand, vl);
			rvv::Store(temp, v, vl);
		}
	}

	const uint64_t twice_mod = 2*modulus;
	uint64_t k = 1;                                                    // next inverse twiddle
	for (uint64_t t = 1, m = n/2; m > 1; m >>= 1, t <<= 1){            // all stages but the last
		for (uint64_t i = 0; i < m; i++, k++){                          // m blocks
			uint64_t W = inv_root_of_unity_powers[k];
			uint64_t Wp = precon_inv_root_of_unity_powers[k];
			uint64_t* x = result + 2*i*t;
			for (size_t j = 0, vl; j < t; j += vl){                       // t butterflies
				vl = rvv::SetVl<V>(t - j);
				V X = rvv::Load<V>(x + j, vl);                             // [0, 2q)
				V Y = rvv::Load<V>(x + j + t, vl);                         // [0, 2q)
				V S = rvv::ReduceFromTwice(__riscv_vadd(X, Y, vl), twice_mod, vl);    // [0, 2q)
				V D = __riscv_vadd(__riscv_vsub(X, Y, vl), twice_mod, vl);            // [0, 4q)
				rvv::Store(x + j, S, vl);
				rvv::Store(x + j + t, rvv::MulModShoupLazy(D, W, Wp, modulus, vl), vl);  // [0, 2q)
			}
		}
	}

	// Last stage (one block, t = n/2) with the N^{-1} scaling folded in:
	//   X' = (X + Y) * N^{-1},   Y' = (X - Y) * (W * N^{-1})
	const uint64_t t = n/2;
	const uint64_t n_inv = InverseMod(n, modulus);
	const uint64_t n_inv_p = MultiplyFactor(n_inv, 64, modulus).BarrettFactor();
	const uint64_t Wn = MultiplyMod(inv_root_of_unity_powers[k], n_inv, n_inv_p, modulus);   // k == n - 1 here
	const uint64_t Wn_p = MultiplyFactor(Wn, 64, modulus).BarrettFactor();
	for (size_t j = 0, vl; j < t; j += vl){
		vl = rvv::SetVl<V>(t - j);
		V X = rvv::Load<V>(result + j, vl);
		V Y = rvv::Load<V>(result + j + t, vl);
		V Xr = rvv::MulModShoupLazy(__riscv_vadd(X, Y, vl), n_inv, n_inv_p, modulus, vl);                        // [0, 2q)
		V Yr = rvv::MulModShoupLazy(__riscv_vadd(__riscv_vsub(X, Y, vl), twice_mod, vl), Wn, Wn_p, modulus, vl); // [0, 2q)
		if (output_mod_factor == 1){
			Xr = rvv::ReduceFromTwice(Xr, modulus, vl);
			Yr = rvv::ReduceFromTwice(Yr, modulus, vl);
		}
		rvv::Store(result + j, Xr, vl);
		rvv::Store(result + j + t, Yr, vl);
	}
}

// Every lane type, so tests and benchmarks can compare LMULs; the public
// NTT uses the defaults from src/util/rvv-config.hpp. RVV32 on uint64_t
// storage stops at u32m4 (the narrowing load needs e64 at twice the LMUL).
template void ForwardTransformToBitReverseRVV32<uint64_t, vuint32m1_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void ForwardTransformToBitReverseRVV32<uint64_t, vuint32m2_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void ForwardTransformToBitReverseRVV32<uint64_t, vuint32m4_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void ForwardTransformToBitReverseRVV32<uint32_t, vuint32m1_t>(uint32_t*, const uint32_t*, uint64_t, uint64_t, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void ForwardTransformToBitReverseRVV32<uint32_t, vuint32m2_t>(uint32_t*, const uint32_t*, uint64_t, uint64_t, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void ForwardTransformToBitReverseRVV32<uint32_t, vuint32m4_t>(uint32_t*, const uint32_t*, uint64_t, uint64_t, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void ForwardTransformToBitReverseRVV32<uint32_t, vuint32m8_t>(uint32_t*, const uint32_t*, uint64_t, uint64_t, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void InverseTransformFromBitReverseRVV32<uint64_t, vuint32m1_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void InverseTransformFromBitReverseRVV32<uint64_t, vuint32m2_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void InverseTransformFromBitReverseRVV32<uint64_t, vuint32m4_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void InverseTransformFromBitReverseRVV32<uint32_t, vuint32m1_t>(uint32_t*, const uint32_t*, uint64_t, uint64_t, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void InverseTransformFromBitReverseRVV32<uint32_t, vuint32m2_t>(uint32_t*, const uint32_t*, uint64_t, uint64_t, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void InverseTransformFromBitReverseRVV32<uint32_t, vuint32m4_t>(uint32_t*, const uint32_t*, uint64_t, uint64_t, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void InverseTransformFromBitReverseRVV32<uint32_t, vuint32m8_t>(uint32_t*, const uint32_t*, uint64_t, uint64_t, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void ForwardTransformToBitReverseRVV64<vuint64m1_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void ForwardTransformToBitReverseRVV64<vuint64m2_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void ForwardTransformToBitReverseRVV64<vuint64m4_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void ForwardTransformToBitReverseRVV64<vuint64m8_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void InverseTransformFromBitReverseRVV64<vuint64m1_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void InverseTransformFromBitReverseRVV64<vuint64m2_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void InverseTransformFromBitReverseRVV64<vuint64m4_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void InverseTransformFromBitReverseRVV64<vuint64m8_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint64_t*, const uint64_t*, uint64_t, uint64_t);

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
