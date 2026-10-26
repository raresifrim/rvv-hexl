// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
//
// The strategy-independent pieces of the RVV NTT kernels: the per-block stage
// loops (each block's t butterflies as strips of vl <= t), which run the
// large-span stages, and every stage when no small-span strategy runs; plus
// the copy, the final reduction and the inverse's last stage with N^{-1}.
// RVV32: 32-bit lanes on either storage word (rvv::Load32 / Store32).
// RVV64: 64-bit lanes on uint64_t storage.
// Harvey lazy butterflies: forward values in [0, 4q), inverse in [0, 2q).
// FP64: the stages on a double scratch, signed ranges (forward (-2q, 2q),
// inverse [-q, q]), and the strips that convert in and out of it.

#pragma once

#include "util/cpu-features.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>
#include <stdint.h>

#include <type_traits>

#include "hexl/number-theory/number-theory.hpp"
#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

// ---------------------------------------------------------------------------
// RVV32
// ---------------------------------------------------------------------------

/// result = operand (n words); nothing when they alias. The source word may
/// differ from the destination's (values < 2^32): uint64_t -> uint32_t narrows,
/// uint32_t -> uint64_t widens (the 32-bit scratch of the RVV32 NTT on
/// uint64_t storage, ntt-rvv.cpp).
template <typename Word, class V, typename Src = Word>
inline void CopyRVV32(Word* result, const Src* operand, uint64_t n) {
	if constexpr (std::is_same_v<Word, Src>) {
		if (result == operand) return;
	}
	for (size_t vl; n > 0; n -= vl, operand += vl, result += vl){
		vl = rvv::SetVl<V>(n);
		V v = rvv::Load32<V>(operand, vl);   // plain vle32 or narrowing load, by Word
		rvv::Store32(result, v, vl);
	}
}

/// Forward (Cooley-Tukey) stages t = t_hi, t_hi/2, .. t_lo, in place,
/// per block. Values in [0, 4q) in and out.
template <typename Word, class V>
inline void ForwardStagesRVV32(Word* result, uint64_t n, uint32_t modulus32,
		const uint32_t* w, const uint32_t* w_precon,
		uint64_t t_hi, uint64_t t_lo) {
	for (uint64_t t = t_hi; t >= t_lo; t >>= 1){                  // stages (t_lo >= 1)
		const uint64_t m = n/(2*t);
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
}

/// [0, 4q) -> [0, q), in place.
template <typename Word, class V>
inline void ReduceFromFourRVV32(Word* result, uint64_t n, uint32_t modulus32) {
	for (size_t vl; n>0; n-=vl, result+=vl){
		vl = rvv::SetVl<V>(n);
		V v = rvv::Load32<V>(result, vl);
		v = rvv::ReduceFromTwice32(v, 2*modulus32, vl);
		v = rvv::ReduceFromTwice32(v, modulus32, vl);
		rvv::Store32(result, v, vl);
	}
}

/// Inverse (Gentleman-Sande) stages t = t_lo, 2 t_lo, .. t_hi, in place, per
/// block (t_hi <= n/4: the last stage is InverseLastStageRVV32). The stage
/// with m blocks reads w_inv[n - 2m + 1 ..]. Values in [0, 2q) in and out.
template <typename Word, class V>
inline void InverseStagesRVV32(Word* result, uint64_t n, uint32_t modulus32,
		const uint32_t* w_inv, const uint32_t* w_inv_precon,
		uint64_t t_lo, uint64_t t_hi) {
	const uint32_t twice_mod32 = 2*modulus32;
	for (uint64_t t = t_lo; t <= t_hi; t <<= 1){                       // stages (t_lo >= 1)
		const uint64_t m = n/(2*t);
		uint64_t k = n - 2*m + 1;                                       // this stage's first twiddle
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
}

/// Last inverse stage (one block, t = n/2) with the N^{-1} scaling folded in:
///   X' = (X + Y) * N^{-1},   Y' = (X - Y) * (W * N^{-1})
/// Output in [0, q) for output_mod_factor 1, [0, 2q) for 2.
template <typename Word, class V>
inline void InverseLastStageRVV32(Word* result, uint64_t n, uint64_t modulus,
		const uint32_t* w_inv, uint64_t output_mod_factor) {
	const uint32_t modulus32 = static_cast<uint32_t>(modulus);
	const uint32_t twice_mod32 = 2*modulus32;
	const uint64_t t = n/2;
	const uint64_t k = n - 1;                                          // the last inverse twiddle
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

// ---------------------------------------------------------------------------
// RVV64
// ---------------------------------------------------------------------------

/// result = operand (n words); nothing when they alias.
template <class V>
inline void CopyRVV64(uint64_t* result, const uint64_t* operand, uint64_t n) {
	if (result == operand) return;
	for (size_t vl; n > 0; n -= vl, operand += vl, result += vl){
		vl = rvv::SetVl<V>(n);
		V v = rvv::Load<V>(operand, vl);
		rvv::Store(result, v, vl);
	}
}

/// Forward stages t = t_hi .. t_lo, in place, per block; [0, 4q) in and out.
template <class V>
inline void ForwardStagesRVV64(uint64_t* result, uint64_t n, uint64_t modulus,
		const uint64_t* root_of_unity_powers,
		const uint64_t* precon_root_of_unity_powers,
		uint64_t t_hi, uint64_t t_lo) {
	for (uint64_t t = t_hi; t >= t_lo; t >>= 1){                  // stages (t_lo >= 1)
		const uint64_t m = n/(2*t);
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
}

/// [0, 4q) -> [0, q), in place.
template <class V>
inline void ReduceFromFourRVV64(uint64_t* result, uint64_t n, uint64_t modulus) {
	for (size_t vl; n>0; n-=vl, result+=vl){
		vl = rvv::SetVl<V>(n);
		V v = rvv::Load<V>(result, vl);
		v = rvv::ReduceFromTwice(v, 2*modulus, vl);
		v = rvv::ReduceFromTwice(v, modulus, vl);
		rvv::Store(result, v, vl);
	}
}

/// Inverse stages t = t_lo .. t_hi (t_hi <= n/4), in place, per block;
/// [0, 2q) in and out.
template <class V>
inline void InverseStagesRVV64(uint64_t* result, uint64_t n, uint64_t modulus,
		const uint64_t* inv_root_of_unity_powers,
		const uint64_t* precon_inv_root_of_unity_powers,
		uint64_t t_lo, uint64_t t_hi) {
	const uint64_t twice_mod = 2*modulus;
	for (uint64_t t = t_lo; t <= t_hi; t <<= 1){                       // stages (t_lo >= 1)
		const uint64_t m = n/(2*t);
		uint64_t k = n - 2*m + 1;                                       // this stage's first twiddle
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
}

/// Last inverse stage (t = n/2) with N^{-1} folded in; [0, q) or [0, 2q) out.
template <class V>
inline void InverseLastStageRVV64(uint64_t* result, uint64_t n, uint64_t modulus,
		const uint64_t* inv_root_of_unity_powers,
		uint64_t output_mod_factor) {
	const uint64_t twice_mod = 2*modulus;
	const uint64_t t = n/2;
	const uint64_t n_inv = InverseMod(n, modulus);
	const uint64_t n_inv_p = MultiplyFactor(n_inv, 64, modulus).BarrettFactor();
	const uint64_t Wn = MultiplyMod(inv_root_of_unity_powers[n - 1], n_inv, n_inv_p, modulus);   // the last inverse twiddle
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

// ---------------------------------------------------------------------------
// FP64 (ntt-rvv-fp64.hpp): the transform runs on a double scratch f (integers
// as doubles), with signed lazy ranges (rvv::FwdButterflyF64 /
// InvButterflyF64): forward (-2q, 2q), inverse [-q, q]. VF = vfloat64m1_t ..
// m8_t; the uint64_t side goes through rvv::UintOf<VF>.
// ---------------------------------------------------------------------------

/// First forward stage (one block, t = n/2) with the input conversion fused
/// in, operand into f: [0, F q) -> [0, 2q) in integer lanes (F =
/// input_mod_factor 1, 2 or 4; the multiply needs |y| <= 2q), converted, one
/// butterfly; (-2q, 2q) out. Replaces a separate conversion pass: every
/// strategy leaves the stage t = n/2 to the per-block loops (t_top <= n/2).
/// w, w_over_q: NttFp64Tables::w and w_over_q (this stage reads entry 1).
template <class VF>
inline void ForwardFirstStageRVVF64(double* f, const uint64_t* operand, uint64_t n, uint64_t modulus,
		const double* w, const double* w_over_q, uint64_t input_mod_factor) {
	using VU = rvv::UintOf<VF>;
	const size_t vlmax = rvv::VlMax<VU>();
	const VF Qv = rvv::SplatF64<VF>(static_cast<double>(modulus), vlmax);
	const VF Cv = rvv::SplatF64<VF>(0x1.8p52, vlmax);
	const double W = w[1], Wq = w_over_q[1];
	const uint64_t t = n/2;
	for (size_t j = 0, vl; j < t; j += vl){
		vl = rvv::SetVl<VU>(t - j);
		VU xu = rvv::Load<VU>(operand + j, vl);
		VU yu = rvv::Load<VU>(operand + j + t, vl);
		if (input_mod_factor == 4) {
			xu = rvv::ReduceFromTwice(xu, 2*modulus, vl);   // [0, 2q)
			yu = rvv::ReduceFromTwice(yu, 2*modulus, vl);
		}
		VF X = __riscv_vfcvt_f(xu, vl);
		VF Y = __riscv_vfcvt_f(yu, vl);
		rvv::FwdButterflyF64(X, Y, W, Wq, Qv, Cv, vl);   // (-2q, 2q)
		rvv::Store(f + j, X, vl);
		rvv::Store(f + j + t, Y, vl);
	}
}

/// Forward stages t = t_hi .. t_lo on f, in place, per block; (-2q, 2q) in
/// and out (none when t_hi < t_lo). w, w_over_q: NttFp64Tables::w and
/// w_over_q, bit-reversed as the integer table.
template <class VF>
inline void ForwardStagesRVVF64(double* f, uint64_t n, uint64_t modulus,
		const double* w, const double* w_over_q,
		uint64_t t_hi, uint64_t t_lo) {
	using VU = rvv::UintOf<VF>;
	const size_t vlmax = rvv::VlMax<VU>();
	const VF Qv = rvv::SplatF64<VF>(static_cast<double>(modulus), vlmax);
	const VF Cv = rvv::SplatF64<VF>(0x1.8p52, vlmax);
	for (uint64_t t = t_hi; t >= t_lo; t >>= 1){                  // stages (t_lo >= 1)
		const uint64_t m = n/(2*t);
		for (uint64_t i = 0; i < m; i++){                          // m blocks
			const double W = w[m + i];
			const double Wq = w_over_q[m + i];
			double* x = f + 2*i*t;
			for (size_t j = 0, vl; j < t; j += vl){                   // t butterflies
				vl = rvv::SetVl<VU>(t - j);
				VF X = rvv::LoadF64<VF>(x + j, vl);
				VF Y = rvv::LoadF64<VF>(x + j + t, vl);
				X = __riscv_vfsub(X, __riscv_vfsgnj(Qv, X, vl), vl);                 // [-q, q]
				VF h = __riscv_vfmul(Y, W, vl);                                       // Y W rounded
				VF l = __riscv_vfmsub(Y, W, h, vl);                                   // Y W - h, exact
				VF T = __riscv_vfsub(__riscv_vfmadd(Y, Wq, Cv, vl), Cv, vl);         // round(Y W / q)
				T = __riscv_vfadd(__riscv_vfnmsac(h, Qv, T, vl), l, vl);             // Y W mod q, (-q, q)
				VF Xr = __riscv_vfadd(X, T, vl);                                      // (-2q, 2q)
				VF Yr = __riscv_vfsub(X, T, vl);
				rvv::Store(x + j, Xr, vl);
				rvv::Store(x + j + t, Yr, vl);
			}
		}
	}
}

/// One strip of the forward output: (-2q, 2q) -> [0, q) for output_mod_factor
/// 1, [0, 2q) for 4. Folded to [-q, q) (strict above: |v| < 2q), + q, then
/// unsigned: no negative lanes.
template <class VF>
__attribute__((always_inline)) inline rvv::UintOf<VF> ForwardOutputF64(VF v, VF qv, uint64_t modulus,
		uint64_t output_mod_factor, size_t vl) {
	const rvv::UintOf<VF> u = __riscv_vfcvt_xu(__riscv_vfadd(rvv::ReduceFromTwiceF64(v, qv, vl), qv, vl), vl);  // [0, 2q)
	return output_mod_factor == 1 ? rvv::ReduceFromTwice(u, modulus, vl) : u;
}

/// result = f in the forward output range (ForwardOutputF64).
template <class VF>
inline void ForwardOutputRVVF64(uint64_t* result, const double* f, uint64_t n, uint64_t modulus,
		uint64_t output_mod_factor) {
	using VU = rvv::UintOf<VF>;
	const VF Qv = rvv::SplatF64<VF>(static_cast<double>(modulus), rvv::VlMax<VU>());
	for (size_t vl; n > 0; n -= vl, f += vl, result += vl){
		vl = rvv::SetVl<VU>(n);
		rvv::Store(result, ForwardOutputF64(rvv::LoadF64<VF>(f, vl), Qv, modulus, output_mod_factor, vl), vl);
	}
}

/// One strip of the inverse input: [0, 2q) (input_mod_factor 1 or 2) ->
/// [-q, q), the inverse butterflies' range.
template <class VF>
__attribute__((always_inline)) inline VF InverseInputF64(rvv::UintOf<VF> u, VF qv, size_t vl) {
	return __riscv_vfsub(__riscv_vfcvt_f(u, vl), qv, vl);
}

/// f = operand as FP64 in the inverse range: [0, 2q) -> [-q, q) (the input
/// pass when no small-span strategy runs).
template <class VF>
inline void InverseInputRVVF64(double* f, const uint64_t* operand, uint64_t n, uint64_t modulus) {
	using VU = rvv::UintOf<VF>;
	const VF Qv = rvv::SplatF64<VF>(static_cast<double>(modulus), rvv::VlMax<VU>());
	for (size_t vl; n > 0; n -= vl, operand += vl, f += vl){
		vl = rvv::SetVl<VU>(n);
		rvv::Store(f, InverseInputF64<VF>(rvv::Load<VU>(operand, vl), Qv, vl), vl);
	}
}

/// Inverse stages t = t_lo .. t_hi (t_hi <= n/4: the last stage is
/// InverseLastStageRVVF64) on f, in place, per block; [-q, q] in and out.
/// w_inv, w_inv_over_q: NttFp64Tables::w_inv and w_inv_over_q, stage order
/// as the integer table (the stage with m blocks reads [n - 2m + 1 ..]).
template <class VF>
inline void InverseStagesRVVF64(double* f, uint64_t n, uint64_t modulus,
		const double* w_inv, const double* w_inv_over_q,
		uint64_t t_lo, uint64_t t_hi) {
	using VU = rvv::UintOf<VF>;
	const size_t vlmax = rvv::VlMax<VU>();
	const VF Qv = rvv::SplatF64<VF>(static_cast<double>(modulus), vlmax);
	const VF Cv = rvv::SplatF64<VF>(0x1.8p52, vlmax);
	for (uint64_t t = t_lo; t <= t_hi; t <<= 1){                       // stages (t_lo >= 1)
		const uint64_t m = n/(2*t);
		uint64_t k = n - 2*m + 1;                                       // this stage's first twiddle
		for (uint64_t i = 0; i < m; i++, k++){                          // m blocks
			const double W = w_inv[k];
			const double Wq = w_inv_over_q[k];
			double* x = f + 2*i*t;
			for (size_t j = 0, vl; j < t; j += vl){                       // t butterflies
				vl = rvv::SetVl<VU>(t - j);
				VF X = rvv::LoadF64<VF>(x + j, vl);                        // [-q, q]
				VF Y = rvv::LoadF64<VF>(x + j + t, vl);                    // [-q, q]
				rvv::InvButterflyF64(X, Y, W, Wq, Qv, Cv, vl);             // [-q, q]
				rvv::Store(x + j, X, vl);
				rvv::Store(x + j + t, Y, vl);
			}
		}
	}
}

/// Last inverse stage (one block, t = n/2) with the N^{-1} scaling folded in,
/// f into result:
///   X' = (X + Y) * N^{-1},   Y' = (X - Y) * (W * N^{-1})
/// X + Y, X - Y in [-2q, 2q], each product in (-q, q), so result in [0, q)
/// (for output_mod_factor 1 and 2 alike).
template <class VF>
inline void InverseLastStageRVVF64(uint64_t* result, const double* f, uint64_t n, uint64_t modulus,
		double n_inv, double n_inv_over_q, double wn_inv, double wn_inv_over_q) {
	using VU = rvv::UintOf<VF>;
	const size_t vlmax = rvv::VlMax<VU>();
	const VF Qv = rvv::SplatF64<VF>(static_cast<double>(modulus), vlmax);
	const VF Cv = rvv::SplatF64<VF>(0x1.8p52, vlmax);
	const uint64_t t = n/2;
	for (size_t j = 0, vl; j < t; j += vl){
		vl = rvv::SetVl<VU>(t - j);
		VF X = rvv::LoadF64<VF>(f + j, vl);
		VF Y = rvv::LoadF64<VF>(f + j + t, vl);
		VF Xr = rvv::MulModF64(__riscv_vfadd(X, Y, vl), n_inv, n_inv_over_q, Qv, Cv, vl);     // (-q, q)
		VF Yr = rvv::MulModF64(__riscv_vfsub(X, Y, vl), wn_inv, wn_inv_over_q, Qv, Cv, vl);   // (-q, q)
		rvv::Store(result + j, rvv::ToUintModF64(Xr, modulus, vl), vl);                        // [0, q)
		rvv::Store(result + j + t, rvv::ToUintModF64(Yr, modulus, vl), vl);
	}
}

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
