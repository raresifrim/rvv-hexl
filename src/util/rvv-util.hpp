// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// RVV building blocks shared by the eltwise and NTT kernels.
// RISC-V counterpart of upstream's util/avx512-util.hpp.
//
// Only compiled when the TU has V enabled (HEXL_HAS_RVV).
//
// Parameterisation, fixed by the K3 measurements:
//   * SEW=e32 whenever the modulus allows it (q < 2^30): the vector multiplier
//     retires 8x more bits/cycle at e32 than at e64 on both clusters.
//   * LMUL=m4 for every kernel (measured on the K3, n = 4096):
//       - light kernels (Add/Sub/CmpAdd/CmpSubMod) are bound by per-strip
//         overhead: EltwiseAddMod m4 is 2.3-3.4x faster than m1 on the X100
//         and 2-3x on the A100.
//       - multiply kernels gain too: m4 is 1.5-2.4x faster than m1 on the X100
//         (MulModBarrett32 3.51 -> 1.59, MulModShoupLazy32 2.66 -> 1.13,
//         MulModBarrett 11.0 -> 6.4 cycles/elem) and the best LMUL there.
//       - m8 is slower than m4 on the X100. On the A100 it is 5-20% faster, but
//         it leaves only 4 register groups: kernels that keep more values live
//         (NTT butterflies, MultMod with input reduction) will spill.
//     Fractional LMUL only exposes the low part of a register.
//   * mf2 is the smallest PORTABLE fractional LMUL: X100 returns vl=0 and traps
//     on e32/mf4 and mf8. Never use mf4/mf8.
//   * Stay VLEN-agnostic (vsetvl every strip): X100 has VLEN=256, A100 1024.
//
// All arithmetic helpers below (add/sub, ReduceFromTwice, BarrettReduce, the
// Barrett and Shoup multiplies) are LMUL-generic templates: they use the
// overloaded intrinsics (__riscv_vadd(a, b, vl) etc.), which take the SEW/LMUL
// from the argument type, so the same helper serves u64m1 and u64m4. The
// kernels name their lane type V once (util/rvv-config.hpp) and use SetVl<V> /
// Load<V> / Store below, so no kernel intrinsic carries an LMUL suffix. The e64 helpers
// reject e32 vectors at compile time and vice versa. Load32 takes its lane
// type as a template argument (Load32<vuint32m4_t>(p, vl)), since a pointer
// carries no LMUL; Store32 deduces it from the vector.
//
// Storage is uint64_t (OpenFHE NATIVE_SIZE=64) or uint32_t (NATIVE_SIZE=32).
// The e32 kernels are templates on the storage word and read/write through
// Load32 / Store32 below, overloaded on the pointer type (shown at m4):
//   uint64_t: vle64 (e64,m8) -> vncvt.x.x.w -> e32,m4 ... vzext.vf2 -> vse64 (e64,m8)
//   uint32_t: vle32 (e32,m4) ...                                    ... vse32
// so the arithmetic is written once.

#pragma once

#include "util/cpu-features.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#include <type_traits>
#include <utility>


namespace intel {
namespace hexl {
namespace rvv {

/// Element type of an RVV vector type (uint64_t for vuint64m4_t, ...), taken
/// from the overloaded vmv.x.s. Used to keep the e64 and e32 helper families
/// apart at compile time.
template <class V>
using ElemT = decltype(__riscv_vmv_x(std::declval<V>()));

template <class V>
constexpr bool IsE64 = std::is_same_v<ElemT<V>, uint64_t>;
template <class V>
constexpr bool IsE32 = std::is_same_v<ElemT<V>, uint32_t>;

// ---------------------------------------------------------------------------
// Type-driven vsetvl / load / store. A kernel names its lane type V once (the
// defaults are in util/rvv-config.hpp) and these pick the intrinsic with V's
// SEW and LMUL; everything else uses the overloaded intrinsics, so moving a
// kernel to another LMUL is a change of V only.
// ---------------------------------------------------------------------------

/// @brief vsetvl for lane type V: min(n, VLMAX of V).
template <class V>
inline size_t SetVl(size_t n) {
  if constexpr (std::is_same_v<V, vuint64m1_t>) {
    return __riscv_vsetvl_e64m1(n);
  } else if constexpr (std::is_same_v<V, vuint64m2_t>) {
    return __riscv_vsetvl_e64m2(n);
  } else if constexpr (std::is_same_v<V, vuint64m4_t>) {
    return __riscv_vsetvl_e64m4(n);
  } else if constexpr (std::is_same_v<V, vuint64m8_t>) {
    return __riscv_vsetvl_e64m8(n);
  } else if constexpr (std::is_same_v<V, vuint32mf2_t>) {
    return __riscv_vsetvl_e32mf2(n);
  } else if constexpr (std::is_same_v<V, vuint32m1_t>) {
    return __riscv_vsetvl_e32m1(n);
  } else if constexpr (std::is_same_v<V, vuint32m2_t>) {
    return __riscv_vsetvl_e32m2(n);
  } else if constexpr (std::is_same_v<V, vuint32m4_t>) {
    return __riscv_vsetvl_e32m4(n);
  } else {
    static_assert(std::is_same_v<V, vuint32m8_t>, "rvv::SetVl<V>: V must be vuint64m1_t..m8_t or vuint32mf2_t..m8_t");
    return __riscv_vsetvl_e32m8(n);
  }
}

/// @brief Unit-stride load of vl elements as lane type V (e64 from uint64_t).
template <class V>
inline V Load(const uint64_t* p, size_t vl) {
  static_assert(IsE64<V>, "rvv::Load<V>(const uint64_t*): V must be vuint64m1_t..m8_t");
  if constexpr (std::is_same_v<V, vuint64m1_t>) {
    return __riscv_vle64_v_u64m1(p, vl);
  } else if constexpr (std::is_same_v<V, vuint64m2_t>) {
    return __riscv_vle64_v_u64m2(p, vl);
  } else if constexpr (std::is_same_v<V, vuint64m4_t>) {
    return __riscv_vle64_v_u64m4(p, vl);
  } else {
    return __riscv_vle64_v_u64m8(p, vl);
  }
}
/// @brief Unit-stride load of vl elements as lane type V (e32 from uint32_t).
template <class V>
inline V Load(const uint32_t* p, size_t vl) {
  static_assert(IsE32<V>, "rvv::Load<V>(const uint32_t*): V must be vuint32mf2_t..m8_t");
  if constexpr (std::is_same_v<V, vuint32mf2_t>) {
    return __riscv_vle32_v_u32mf2(p, vl);
  } else if constexpr (std::is_same_v<V, vuint32m1_t>) {
    return __riscv_vle32_v_u32m1(p, vl);
  } else if constexpr (std::is_same_v<V, vuint32m2_t>) {
    return __riscv_vle32_v_u32m2(p, vl);
  } else if constexpr (std::is_same_v<V, vuint32m4_t>) {
    return __riscv_vle32_v_u32m4(p, vl);
  } else {
    return __riscv_vle32_v_u32m8(p, vl);
  }
}

/// @brief Unit-stride store of vl elements (lane type deduced from v).
template <class V>
inline void Store(uint64_t* p, V v, size_t vl) {
  static_assert(IsE64<V>, "rvv::Store(uint64_t*, v): v must be an e64 vector");
  __riscv_vse64(p, v, vl);
}
template <class V>
inline void Store(uint32_t* p, V v, size_t vl) {
  static_assert(IsE32<V>, "rvv::Store(uint32_t*, v): v must be an e32 vector");
  __riscv_vse32(p, v, vl);
}

// ---------------------------------------------------------------------------
// e64 path (any modulus up to 62 bits). V = vuint64m1_t ... vuint64m8_t.
// ---------------------------------------------------------------------------

/// @brief (a + b) mod q for a, b in [0, q), q < 2^63.
/// s = a + b; r = vminu(s, s - q) (s - q wraps to a huge value when s < q).
template <class V>
inline V AddMod(V a, V b, uint64_t q, size_t vl) {
	static_assert(IsE64<V>, "rvv::AddMod takes e64 vectors; use AddMod32 for e32");
  	//add all elements inside the vectors
	V sum = __riscv_vadd(a, b, vl);
	//subtract the modulus from the element-wise sum
	V remainder = __riscv_vsub(sum, q, vl);
	//return the minimum between the sum and the modulus diff
	return __riscv_vminu(sum, remainder, vl);
}

/// @brief (a - b) mod q for a, b in [0, q).
template <class V>
inline V SubMod(V a, V b, uint64_t q, size_t vl) {
	static_assert(IsE64<V>, "rvv::SubMod takes e64 vectors; use SubMod32 for e32");
  	//sub all elements inside the vectors
	V diff = __riscv_vsub(a, b, vl);
	//add the modulus from the element-wise sum
	V remainder = __riscv_vadd(diff, q, vl);
	//return the minimum between the diff and the modulus sum
	return __riscv_vminu(diff, remainder, vl);
}

/// @brief (a + b) mod q for a in [0, q), scalar b in [0, q), q < 2^63.
/// s = a + b; r = vminu(s, s - q) (s - q wraps to a huge value when s < q).
template <class V>
inline V AddScalarMod(V a, uint64_t b, uint64_t q, size_t vl) {
	static_assert(IsE64<V>, "rvv::AddScalarMod takes e64 vectors; use AddScalarMod32 for e32");
  	//add all elements inside the vectors
	V sum = __riscv_vadd(a, b, vl);
	//subtract the modulus from the element-wise sum
	V remainder = __riscv_vsub(sum, q, vl);
	//return the minimum between the sum and the modulus diff
	return __riscv_vminu(sum, remainder, vl);
}

/// @brief (a - b) mod q for a in [0, q), scalar b in [0, q).
template <class V>
inline V SubScalarMod(V a, uint64_t b, uint64_t q, size_t vl) {
	static_assert(IsE64<V>, "rvv::SubScalarMod takes e64 vectors; use SubScalarMod32 for e32");
  	//sub all elements inside the vectors
	V diff = __riscv_vsub(a, b, vl);
	//add the modulus from the element-wise sum
	V remainder = __riscv_vadd(diff, q, vl);
	//return the minimum between the diff and the modulus sum
	return __riscv_vminu(diff, remainder, vl);
}

/// @brief Maps x in [0, 2q) to [0, q) (the "vminu" conditional subtract).
template <class V>
inline V ReduceFromTwice(V x, uint64_t q, size_t vl) {
	static_assert(IsE64<V>, "rvv::ReduceFromTwice takes e64 vectors");
	return __riscv_vminu(x, __riscv_vsub(x, q, vl), vl);
}

/// @brief x mod q for ANY 64-bit x (Barrett reduction), q >= 2. LMUL-generic.
/// q_barr = floor(2^64 / q) = MultiplyFactor(1, 64, q).BarrettFactor();
/// compute it once per kernel call, before the strip loop.
/// OutputModFactor 1 returns [0, q); 2 returns the lazy [0, 2q) (skips the
/// final correction), like the scalar BarrettReduce64<OutputModFactor>.
/// Used by EltwiseCmpSubMod, EltwiseReduceMod (input_mod_factor == q).
template <int OutputModFactor = 1, class V>
inline V BarrettReduce(V x, uint64_t q, uint64_t q_barr, size_t vl) {
	static_assert(IsE64<V>, "rvv::BarrettReduce takes e64 vectors; use BarrettReduce32 for e32");
	static_assert(OutputModFactor == 1 || OutputModFactor == 2, "OutputModFactor must be 1 or 2");
	//we already have q_barr as 2^k/q, we just need to multiply it by x and shift it by k bits
	//the trick is that mulhu already perform the multiplication + the shift so we can use that
	V Q = __riscv_vmulhu(x, q_barr, vl);
	//now we can compute Q*q using the standar vmul as we know this will never overflow the data type
	//Q = __riscv_vmul(Q, q, vl);
	//Q = __riscv_vsub(x, Q, vl);
	Q = __riscv_vnmsac(x, q, Q, vl); //= x - q*Q in one instruction with better measured performance 
	if constexpr (OutputModFactor == 1){
		return ReduceFromTwice(Q, q, vl);
	}
	return Q;
}

/// @brief Shoup multiplication by a precomputed operand, LAZY: returns
/// x * y mod q in [0, 2q). y_precon = floor(y * 2^64 / q).
/// This is the NTT butterfly multiply and the EltwiseFMAMod scalar multiply.
template <class V>
inline V MulModShoupLazy(V x, V y, V y_precon, uint64_t q, size_t vl) {
  static_assert(IsE64<V>, "rvv::MulModShoupLazy takes e64 vectors; use MulModShoupLazy32 for e32");
  V Q = __riscv_vmulhu(x, y_precon, vl);
  V r = __riscv_vnmsac(__riscv_vmul(x, y, vl), q, Q, vl);  // x*y - q*Q: the multiply-subtract in one instruction
  return r; 
}


/// @brief Same, with a scalar multiplier broadcast to every lane.
template <class V>
inline V MulModShoupLazy(V x, uint64_t y, uint64_t y_precon, uint64_t q, size_t vl) {
  static_assert(IsE64<V>, "rvv::MulModShoupLazy takes e64 vectors; use MulModShoupLazy32 for e32");
  V Q = __riscv_vmulhu(x, y_precon, vl);
  V r = __riscv_vnmsac(__riscv_vmul(x, y, vl), q, Q, vl);  // x*y - q*Q: the multiply-subtract in one instruction
  return r; 
}


/// @brief Full modular product of two VECTORS (no precomputed operand), as
/// needed by EltwiseMultMod. a, b < q < 2^61. Returns [0, q).
/// Method: upstream HEXL's pre-shift Barrett (EltwiseMultModNative), with
/// n = bit length of q:
///   shift = n - 2,   mu = floor(2^(n+62) / q)
///                       = MultiplyFactor(1ULL << shift, 64, q).BarrettFactor()
/// The kernel computes both ONCE per call (mu is a 128-by-64 division), never
/// per strip in here.
/// The quotient estimate is at most one short for q < 2^61 (it can be two
/// short for 62-bit q, measured), so one final correction suffices. shift == 0
/// only for q < 4, where hi == 0 (vsll by 64 shifts by 0 in RVV: harmless).
template <class V>
inline V MulModBarrett(V a, V b, uint64_t q, uint64_t mu, uint64_t shift, size_t vl) {
  static_assert(IsE64<V>, "rvv::MulModBarrett takes e64 vectors; use MulModBarrett32 for e32");
  V hi, lo;
  hi = __riscv_vmulhu(a, b, vl);
  lo = __riscv_vmul(a, b, vl);
  
  V c = __riscv_vor(
		   __riscv_vsrl(lo, shift, vl),
		   __riscv_vsll(hi, 64-shift, vl),
		   vl
		 ); 
  
  V Q = __riscv_vmulhu(c, mu, vl);
  
  V r = __riscv_vnmsac(lo, q, Q, vl);  // lo - q*Q in one instruction (8-33% faster than vmul + vsub, measured)
  
  return __riscv_vminu(
		  r, 
		  __riscv_vsub(r, q, vl),
		  vl);
}


// ---------------------------------------------------------------------------
// e32 path (q < 2^30): compute in 32-bit lanes, store in 64-bit words
// ---------------------------------------------------------------------------

/// @brief Loads vl values into 32-bit lanes of type V32. Overloaded on the
/// storage type so that one kernel template serves both OpenFHE word sizes:
///   uint64_t* (NATIVE_SIZE=64): each value < 2^32, narrow on load: vle64 at
///     TWICE the LMUL, then vncvt.x.x.w (u64m2 -> u32m1, ..., u64m8 -> u32m4).
///   uint32_t* (NATIVE_SIZE=32): plain vle32.
/// A pointer carries no LMUL, so the lane type is a template argument:
///   rvv::Load32<vuint32m4_t>(p, vl)      (default: vuint32m1_t)
/// From uint64_t storage V32 can be mf2..m4 (the e64 side needs twice the
/// LMUL, at most m8); from uint32_t storage mf2..m8. Measured on the K3, the
/// narrowing load reaches about 2x the bandwidth at u64m8 -> u32m4 than at
/// u64m2 -> u32m1 (X100 L1: 12.4 vs 5.8 B/cycle; A100: 17.0 vs 8.2).
template <class V32 = vuint32m1_t>
inline V32 Load32(const uint64_t* p, size_t vl) {
  static_assert(IsE32<V32>, "rvv::Load32<V32>: V32 must be an e32 vector type");
	//nice trick to load 32-bit lanes from wide 64-bit pointer, narrowing the values to fit
  if constexpr (std::is_same_v<V32, vuint32mf2_t>) {
    return __riscv_vncvt_x(__riscv_vle64_v_u64m1(p, vl), vl);
  } else if constexpr (std::is_same_v<V32, vuint32m1_t>) {
    return __riscv_vncvt_x(__riscv_vle64_v_u64m2(p, vl), vl);
  } else if constexpr (std::is_same_v<V32, vuint32m2_t>) {
    return __riscv_vncvt_x(__riscv_vle64_v_u64m4(p, vl), vl);
  } else {
    static_assert(std::is_same_v<V32, vuint32m4_t>,
                  "rvv::Load32 from uint64_t storage: V32 must be u32mf2..u32m4 (the e64 load needs twice the LMUL)");
    return __riscv_vncvt_x(__riscv_vle64_v_u64m8(p, vl), vl);
  }
}
template <class V32 = vuint32m1_t>
inline V32 Load32(const uint32_t* p, size_t vl) {
  static_assert(IsE32<V32>, "rvv::Load32<V32>: V32 must be an e32 vector type");
  if constexpr (std::is_same_v<V32, vuint32mf2_t>) {
    return __riscv_vle32_v_u32mf2(p, vl);
  } else if constexpr (std::is_same_v<V32, vuint32m1_t>) {
    return __riscv_vle32_v_u32m1(p, vl);
  } else if constexpr (std::is_same_v<V32, vuint32m2_t>) {
    return __riscv_vle32_v_u32m2(p, vl);
  } else if constexpr (std::is_same_v<V32, vuint32m4_t>) {
    return __riscv_vle32_v_u32m4(p, vl);
  } else {
    return __riscv_vle32_v_u32m8(p, vl);
  }
}

/// @brief Stores vl 32-bit lanes into the storage type (see Load32). The lane
/// type is deduced from v:
///   uint64_t*: widen with vzext.vf2 to twice the LMUL, then vse64 (V32 up to m4).
///   uint32_t*: plain vse32.
template <class V32>
inline void Store32(uint64_t* p, V32 v, size_t vl) {
  static_assert(IsE32<V32>, "rvv::Store32: v must be an e32 vector");
  static_assert(!std::is_same_v<V32, vuint32m8_t>,
                "rvv::Store32 to uint64_t storage: at most u32m4 (the e64 store needs twice the LMUL)");
  __riscv_vse64(p, __riscv_vzext_vf2(v, vl), vl);
}
template <class V32>
inline void Store32(uint32_t* p, V32 v, size_t vl) {
  static_assert(IsE32<V32>, "rvv::Store32: v must be an e32 vector");
  __riscv_vse32(p, v, vl);
}

// The Add/Sub helpers below are LMUL-generic: V = vuint32mf2_t ... vuint32m8_t.

/// @brief (a + b) mod q, 32-bit lanes.
template <class V>
inline V AddMod32(V a, V b, uint32_t q, size_t vl) {
	static_assert(IsE32<V>, "rvv::AddMod32 takes e32 vectors; use AddMod for e64");
  	//add all elements inside the vectors
	V sum = __riscv_vadd(a, b, vl);
	//subtract the modulus from the element-wise sum
	V remainder = __riscv_vsub(sum, q, vl);
	//return the minimum between the sum and the modulus diff
	return __riscv_vminu(sum, remainder, vl);
}

/// @brief (a - b) mod q, 32-bit lanes.
template <class V>
inline V SubMod32(V a, V b, uint32_t q, size_t vl) {
	static_assert(IsE32<V>, "rvv::SubMod32 takes e32 vectors; use SubMod for e64");
  	//sub all elements inside the vectors
	V diff = __riscv_vsub(a, b, vl);
	//add the modulus from the element-wise sum
	V remainder = __riscv_vadd(diff, q, vl);
	//return the minimum between the diff and the modulus sum
	return __riscv_vminu(diff, remainder, vl);
}

/// @brief (a + b) mod q, 32-bit lanes, scalar b.
template <class V>
inline V AddScalarMod32(V a, uint32_t b, uint32_t q, size_t vl) {
	static_assert(IsE32<V>, "rvv::AddScalarMod32 takes e32 vectors; use AddScalarMod for e64");
  	//add all elements inside the vectors
	V sum = __riscv_vadd(a, b, vl);
	//subtract the modulus from the element-wise sum
	V remainder = __riscv_vsub(sum, q, vl);
	//return the minimum between the sum and the modulus diff
	return __riscv_vminu(sum, remainder, vl);
}

/// @brief (a - b) mod q, 32-bit lanes, scalar b.
template <class V>
inline V SubScalarMod32(V a, uint32_t b, uint32_t q, size_t vl) {
	static_assert(IsE32<V>, "rvv::SubScalarMod32 takes e32 vectors; use SubScalarMod for e64");
  	//sub all elements inside the vectors
	V diff = __riscv_vsub(a, b, vl);
	//add the modulus from the element-wise sum
	V remainder = __riscv_vadd(diff, q, vl);
	//return the minimum between the diff and the modulus sum
	return __riscv_vminu(diff, remainder, vl);
}

/// @brief x mod q for ANY 32-bit x (Barrett reduction), 32-bit lanes, q >= 2.
/// LMUL-generic. q_barr = floor(2^32 / q) = MultiplyFactor(1, 32, q).BarrettFactor()
/// (< 2^32 for q >= 2); compute it once per kernel call, before the strip loop.
/// OutputModFactor 1 returns [0, q); 2 returns the lazy [0, 2q).
template <int OutputModFactor = 1, class V>
inline V BarrettReduce32(V x, uint32_t q, uint32_t q_barr, size_t vl) {
	static_assert(IsE32<V>, "rvv::BarrettReduce32 takes e32 vectors; use BarrettReduce for e64");
	static_assert(OutputModFactor == 1 || OutputModFactor == 2, "OutputModFactor must be 1 or 2");
  	//we already have q_barr as 2^k/q, we just need to multiply it by x and shift it by k bits
	//the trick is that mulhu already perform the multiplication + the shift so we can use that
	V Q = __riscv_vmulhu(x, q_barr, vl);
	//now we can compute Q*q using the standar vmul as we know this will never overflow the data type
	//Q = __riscv_vmul(Q, q, vl);
	//Q = __riscv_vsub(x, Q, vl);
	Q = __riscv_vnmsac(x, q, Q, vl); 
	if constexpr (OutputModFactor == 1){
		return __riscv_vminu(Q, __riscv_vsub(Q, q, vl), vl);	
	}
	return Q;
}

/// @brief Shoup lazy multiply in 32-bit lanes: [0, 2q).
/// y_precon = floor(y * 2^32 / q) (MultiplyFactor(y, 32, q)).
template <class V>
inline V MulModShoupLazy32(V x, V y, V y_precon, uint32_t q, size_t vl) {
  static_assert(IsE32<V>, "rvv::MulModShoupLazy32 takes e32 vectors; use MulModShoupLazy for e64");
  V Q = __riscv_vmulhu(x, y_precon, vl);
  V r = __riscv_vnmsac(__riscv_vmul(x, y, vl), q, Q, vl);  // x*y - q*Q: the multiply-subtract in one instruction
  return r; 
}


/// @brief Full modular product of two 32-bit vectors, a, b < q < 2^30. Returns [0, q).
/// Same pre-shift Barrett as the e64 MulModBarrett, at 32-bit words, with
/// n = bit length of q (n <= 30):
///   shift = n - 2,   mu = floor(2^(n+30) / q)
///                       = MultiplyFactor(1 << shift, 32, q).BarrettFactor()
/// The kernel computes both ONCE per call (a single 64-bit divu here).
template <class V>
inline V MulModBarrett32(V a, V b, uint32_t q, uint32_t mu, uint32_t shift, size_t vl) {
  static_assert(IsE32<V>, "rvv::MulModBarrett32 takes e32 vectors; use MulModBarrett for e64");
  V hi, lo;
  hi = __riscv_vmulhu(a, b, vl);
  lo = __riscv_vmul(a, b, vl);
  
  V c = __riscv_vor(
		   __riscv_vsrl(lo, shift, vl),
		   __riscv_vsll(hi, 32-shift, vl),
		   vl
		 ); 
  
  V Q = __riscv_vmulhu(c, mu, vl);
  
  V r = __riscv_vnmsac(lo, q, Q, vl);  // lo - q*Q in one instruction (8-33% faster than vmul + vsub, measured)
  
  r = __riscv_vminu(
		  r, 
		  __riscv_vsub(r, q, vl),
		  vl);
  
  if ((q >> 29) == 0) //if moduli is 30-bits or higher we need a double correction
    return r;
  else
    return __riscv_vminu(
		  r, 
		  __riscv_vsub(r, q, vl),
		  vl);
}


}  // namespace rvv
}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
