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
// Parameterisation, fixed by the K3 measurements (SENTHIPoli A14/D01,
// "Directia de accelerare"):
//   * SEW=e32 whenever the modulus allows it (q < 2^30): the vector multiplier
//     retires 8x more bits/cycle at e32 than at e64 on both clusters.
//   * LMUL=m1 for both transfer and compute: larger LMUL buys no throughput,
//     and fractional LMUL only exposes the low part of a register.
//   * mf2 is the smallest PORTABLE fractional LMUL: X100 returns vl=0 and traps
//     on e32/mf4 and mf8. Never use mf4/mf8.
//   * Stay VLEN-agnostic (vsetvl every strip): X100 has VLEN=256, A100 1024.
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

#include "util/not-implemented.hpp"

namespace intel {
namespace hexl {
namespace rvv {

// ---------------------------------------------------------------------------
// e64 path (any modulus up to 62 bits)
// ---------------------------------------------------------------------------

/// @brief (a + b) mod q for a, b in [0, q), q < 2^63.
/// Hint: s = a + b; r = vminu(s, s - q) (s - q wraps to a huge value when s < q).
inline vuint64m1_t AddMod(vuint64m1_t a, vuint64m1_t b, uint64_t q,
                          size_t vl) {
  HEXL_NOT_IMPLEMENTED();
}

/// @brief (a - b) mod q for a, b in [0, q).
inline vuint64m1_t SubMod(vuint64m1_t a, vuint64m1_t b, uint64_t q,
                          size_t vl) {
  HEXL_NOT_IMPLEMENTED();
}

/// @brief Maps x in [0, 2q) to [0, q) (the "vminu" conditional subtract).
inline vuint64m1_t ReduceFromTwice(vuint64m1_t x, uint64_t q, size_t vl) {
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
  HEXL_NOT_IMPLEMENTED();
}
inline vuint32m1_t Load32(const uint32_t* p, size_t vl) {
  HEXL_NOT_IMPLEMENTED();
}

/// @brief Stores vl 32-bit lanes into the storage type (see Load32).
///   uint64_t*: widen, __riscv_vzext_vf2_u64m2 + __riscv_vse64_v_u64m2.
///   uint32_t*: plain __riscv_vse32_v_u32m1.
inline void Store32(uint64_t* p, vuint32m1_t v, size_t vl) {
  HEXL_NOT_IMPLEMENTED();
}
inline void Store32(uint32_t* p, vuint32m1_t v, size_t vl) {
  HEXL_NOT_IMPLEMENTED();
}

/// @brief (a + b) mod q, 32-bit lanes.
inline vuint32m1_t AddMod32(vuint32m1_t a, vuint32m1_t b, uint32_t q,
                            size_t vl) {
  HEXL_NOT_IMPLEMENTED();
}

/// @brief (a - b) mod q, 32-bit lanes.
inline vuint32m1_t SubMod32(vuint32m1_t a, vuint32m1_t b, uint32_t q,
                            size_t vl) {
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
