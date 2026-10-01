// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// RVV building blocks shared by the eltwise and NTT kernels.
// RISC-V counterpart of upstream's util/avx512-util.hpp.
//
// Only compiled when the TU has V enabled (HEXL_HAS_RVV). Everything here is a
// TODO(port-rvv) stub: the signatures fix the vocabulary the kernels are
// written in, the bodies are yours.
//
// Parameterisation, fixed by the K3 measurements:
//   * SEW=e32 whenever the modulus allows it (q < 2^30): the vector multiplier
//     retires 8x more bits/cycle at e32 than at e64 on both clusters.
//   * LMUL depends on what bounds the kernel:
//       - light kernels (Add/Sub/CmpAdd/CmpSubMod: 1-3 ALU ops per element)
//         are bound by per-strip overhead, so use LMUL=m4. Measured on
//         EltwiseAddMod, n = 1K-64K: 2.3-3.4x faster than m1 on the X100 and
//         2-3x on the A100. m8 is slower than m4 on the X100 and on the A100 at
//         e64 n >= 64K.
//       - multiply-bound kernels (Mult/FMA/NTT) keep LMUL=m1 for compute:
//         larger LMUL buys no multiplier throughput and costs registers.
//     Fractional LMUL only exposes the low part of a register.
//   * mf2 is the smallest PORTABLE fractional LMUL: X100 returns vl=0 and traps
//     on e32/mf4 and mf8. Never use mf4/mf8.
//   * Stay VLEN-agnostic (vsetvl every strip): X100 has VLEN=256, A100 1024.
//
// The Add/Sub/ReduceFromTwice helpers below are LMUL-generic templates: they
// use the overloaded intrinsics (__riscv_vadd(a, b, vl) etc.), which take the
// SEW/LMUL from the argument type, so the same helper serves u64m1 and u64m4.
// Only the kernel's vsetvl / vle / vse carry an explicit LMUL suffix. The
// e64 helpers reject e32 vectors at compile time and vice versa.
//
// Storage is uint64_t (OpenFHE NATIVE_SIZE=64) or uint32_t (NATIVE_SIZE=32).
// The e32 kernels are templates on the storage word and read/write through
// Load32 / Store32 below, overloaded on the pointer type:
//   uint64_t: vle64 (e64,m2) -> vncvt.x.x.w -> e32,m1 ... vzext.vf2 -> vse64
//   uint32_t: vle32 (e32,m1) ...                                    ... vse32
// so the arithmetic is written once. Alternative worth measuring for uint64_t
// storage: e64,m1 -> e32,mf2 (half the elements per op).

#pragma once

#include "util/cpu-features.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#include <type_traits>
#include <utility>

#include "util/not-implemented.hpp"

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
/// Hint: Q = vmulhu(x, q_barr) is floor(x/q) or one less, so
/// r = x - Q*q (vnmsac) is in [0, 2q) and never overflows (r <= x);
/// then ReduceFromTwice for OutputModFactor 1.
template <int OutputModFactor = 1, class V>
inline V BarrettReduce(V x, uint64_t q, uint64_t q_barr, size_t vl) {
	static_assert(IsE64<V>, "rvv::BarrettReduce takes e64 vectors; use BarrettReduce32 for e32");
	static_assert(OutputModFactor == 1 || OutputModFactor == 2, "OutputModFactor must be 1 or 2");
  HEXL_NOT_IMPLEMENTED();
}

/// @brief Shoup multiplication by a precomputed operand, LAZY: returns
/// x * y mod q in [0, 2q). y_precon = floor(y * 2^64 / q).
/// Hint: Q = vmulhu(x, y_precon); r = vmul(x, y) - vmul(Q, q).
/// This is the NTT butterfly multiply and the EltwiseFMAMod scalar multiply.
inline vuint64m1_t MulModShoupLazy(vuint64m1_t x, vuint64m1_t y,
                                   vuint64m1_t y_precon, uint64_t q,
                                   size_t vl) {
  HEXL_NOT_IMPLEMENTED();
}

/// @brief Same, with a scalar multiplier broadcast to every lane.
inline vuint64m1_t MulModShoupLazy(vuint64m1_t x, uint64_t y,
                                   uint64_t y_precon, uint64_t q, size_t vl) {
  HEXL_NOT_IMPLEMENTED();
}

/// @brief Full modular product of two VECTORS (no precomputed operand), as
/// needed by EltwiseMultMod. Returns [0, q).
/// Hint: needs the 128-bit product (vmul + vmulhu) and a Barrett reduction
/// with mu = floor(2^(2k) / q); see upstream EltwiseMultModNative for the
/// scalar algorithm with its exact shift amounts.
inline vuint64m1_t MulModBarrett(vuint64m1_t a, vuint64m1_t b, uint64_t q,
                                 uint64_t barrett_hi, uint64_t barrett_lo,
                                 size_t vl) {
  HEXL_NOT_IMPLEMENTED();
}

// ---------------------------------------------------------------------------
// e32 path (q < 2^30): compute in 32-bit lanes, store in 64-bit words
// ---------------------------------------------------------------------------

/// @brief Loads vl values into 32-bit lanes. Overloaded on the storage type so
/// that one kernel template serves both OpenFHE word sizes:
///   uint64_t* (NATIVE_SIZE=64): each value < 2^32, narrow on load.
///     Hint: __riscv_vle64_v_u64m2 + __riscv_vncvt_x_x_w_u32m1.
///   uint32_t* (NATIVE_SIZE=32): plain __riscv_vle32_v_u32m1.
inline vuint32m1_t Load32(const uint64_t* p, size_t vl) {
	//nice trick to load 32-bit lanes from wide 64-bit pointer, narrowing the values to fit
	return __riscv_vncvt_x_x_w_u32m1(__riscv_vle64_v_u64m2(p, vl), vl);
}
inline vuint32m1_t Load32(const uint32_t* p, size_t vl) {
	return __riscv_vle32_v_u32m1(p, vl);
}

/// @brief Stores vl 32-bit lanes into the storage type (see Load32).
///   uint64_t*: widen, __riscv_vzext_vf2_u64m2 + __riscv_vse64_v_u64m2.
///   uint32_t*: plain __riscv_vse32_v_u32m1.
inline void Store32(uint64_t* p, vuint32m1_t v, size_t vl) {
  	__riscv_vse64_v_u64m2(p, __riscv_vzext_vf2_u64m2(v, vl), vl);
}
inline void Store32(uint32_t* p, vuint32m1_t v, size_t vl) {
	__riscv_vse32_v_u32m1(p, v, vl);
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
/// Hint: same as BarrettReduce at e32: Q = vmulhu(x, q_barr), r = x - Q*q in
/// [0, 2q), then one vminu correction for OutputModFactor 1.
template <int OutputModFactor = 1, class V>
inline V BarrettReduce32(V x, uint32_t q, uint32_t q_barr, size_t vl) {
	static_assert(IsE32<V>, "rvv::BarrettReduce32 takes e32 vectors; use BarrettReduce for e64");
	static_assert(OutputModFactor == 1 || OutputModFactor == 2, "OutputModFactor must be 1 or 2");
  HEXL_NOT_IMPLEMENTED();
}

/// @brief Shoup lazy multiply in 32-bit lanes: [0, 2q).
/// y_precon = floor(y * 2^32 / q) (MultiplyFactor(y, 32, q)).
/// Hint: Q = vmulhu(x, y_precon) at e32; r = x*y - Q*q (all e32, wrapping).
/// This is the operation whose throughput the K3 datapath table measures.
inline vuint32m1_t MulModShoupLazy32(vuint32m1_t x, vuint32m1_t y,
                                     vuint32m1_t y_precon, uint32_t q,
                                     size_t vl) {
  HEXL_NOT_IMPLEMENTED();
}

/// @brief Full modular product of two 32-bit vectors, [0, q).
/// Hint: lo = vmul, hi = vmulhu (both e32, no widening needed), then a
/// two-word Barrett; or vwmulu to e64 and reduce there (measure both).
inline vuint32m1_t MulModBarrett32(vuint32m1_t a, vuint32m1_t b, uint32_t q,
                                   uint32_t barrett_factor, size_t vl) {
  HEXL_NOT_IMPLEMENTED();
}

}  // namespace rvv
}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
