// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Scalar 128-bit helpers used by the inline number-theory functions and by the
// native (scalar) kernels. Shared by GCC and Clang; clang.hpp only swaps the
// loop-unroll pragmas. These are part of the upstream header API (OpenFHE and
// the upstream tests call them), so they are provided as-is, not as stubs.

#pragma once

#include <stdint.h>

#include "hexl/util/check.hpp"
#include "hexl/util/types.hpp"

namespace intel {
namespace hexl {

#if defined(HEXL_USE_GNU) || defined(HEXL_USE_CLANG)

/// @brief Returns the full 128-bit product x * y
inline uint128_t MultiplyUInt64(uint64_t x, uint64_t y) {
  return uint128_t(x) * uint128_t(y);
}

/// @brief Returns (input_hi * 2^64 + input_lo) mod modulus. Uses a 128-bit
/// division: slow, only meant for precomputation, never for inner loops.
inline uint64_t BarrettReduce128(uint64_t input_hi, uint64_t input_lo,
                                 uint64_t modulus) {
  HEXL_CHECK(modulus != 0, "modulus == 0")
  uint128_t n = (static_cast<uint128_t>(input_hi) << 64) |
                (static_cast<uint128_t>(input_lo));
  return static_cast<uint64_t>(n % modulus);
}

/// @brief Returns the low 64 bits of floor((x1 * 2^64 + x0) / y).
/// Used to build Barrett/Shoup precomputed factors.
inline uint64_t DivideUInt128UInt64Lo(uint64_t x1, uint64_t x0, uint64_t y) {
  uint128_t n =
      (static_cast<uint128_t>(x1) << 64) | (static_cast<uint128_t>(x0));
  uint128_t q = n / y;
  return static_cast<uint64_t>(q);
}

/// @brief Computes x * y as a (hi, lo) pair of 64-bit words
inline void MultiplyUInt64(uint64_t x, uint64_t y, uint64_t* prod_hi,
                           uint64_t* prod_lo) {
  uint128_t prod = MultiplyUInt64(x, y);
  *prod_hi = static_cast<uint64_t>(prod >> 64);
  *prod_lo = static_cast<uint64_t>(prod);
}

/// @brief Returns (x * y) >> BitShift. On RV64 with BitShift == 64 this is a
/// single `mulhu` instruction.
template <int BitShift>
inline uint64_t MultiplyUInt64Hi(uint64_t x, uint64_t y) {
  uint128_t product = MultiplyUInt64(x, y);
  return static_cast<uint64_t>(product >> BitShift);
}

/// @brief Returns the index of the most significant set bit (floor(log2)).
/// Upstream uses std::log2l, which on RISC-V is a software quad-float routine;
/// clz is exact and a single instruction with Zbb. input must be non-zero.
inline uint64_t MSB(uint64_t input) {
  return static_cast<uint64_t>(63 - __builtin_clzll(input));
}

#endif  // HEXL_USE_GNU || HEXL_USE_CLANG

#if defined(HEXL_USE_GNU)
#define HEXL_LOOP_UNROLL_4 _Pragma("GCC unroll 4")
#define HEXL_LOOP_UNROLL_8 _Pragma("GCC unroll 8")
#endif

}  // namespace hexl
}  // namespace intel
