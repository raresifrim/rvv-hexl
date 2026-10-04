// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Shared helpers for the HEXL microbenchmarks.
//
// These benches use ONLY the public HEXL API (the same API upstream Intel HEXL
// exposes), so the numbers are comparable with upstream's own benchmark suite.
//
// Parameters follow upstream HEXL's own benchmark suite (n = 1024/4096/16384,
// 45-bit NTT primes) plus the moduli the IPCEI protocol actually uses:
//   27 bits  binfhe/TFHE (OpenFHE STD128 sets, <= 28-bit q)  -> RVV e32 path
//   45 bits  upstream HEXL's published NTT benchmark modulus
//   49 bits  the IPCEI openfhe_ntt_bench / seal_ntt_bench prime
//   60 bits  BFV RNS limbs                                     -> RVV e64 path

#pragma once

#include <benchmark/benchmark.h>
#include <stdint.h>

#include <stdexcept>
#include <vector>

#include "hexl/hexl.hpp"
#include "oracle.hpp"  // prime generation independent of the library under test

namespace hexlbench {

using Vec = intel::hexl::AlignedVector64<uint64_t>;

/// NTT-friendly prime with `bits` bits (q = 1 mod 2N), computed by the test
/// oracle so a benchmark never depends on a library function being ported.
inline uint64_t Prime(size_t bits, uint64_t n) {
  return hexltest::oracle::NttPrime(bits, n);
}

inline Vec Random(size_t n, uint64_t bound) {
  auto v = hexltest::oracle::Random(n, bound);
  return Vec(v.begin(), v.end());
}

/// Same, for any storage word (the 32-bit benches use uint32_t).
template <typename Word>
inline intel::hexl::AlignedVector64<Word> RandomW(size_t n, uint64_t bound) {
  auto v = hexltest::oracle::Random(n, bound);
  return intel::hexl::AlignedVector64<Word>(v.begin(), v.end());
}

}  // namespace hexlbench

/// Runs `expr` once before timing. If it reaches a port stub (or throws for
/// any other reason) the benchmark is reported as skipped with the message,
/// instead of aborting the whole run.
#define HEXL_BENCH_PROBE(state, expr)          \
  try {                                        \
    expr;                                      \
  } catch (const std::exception& hexl_e_) {    \
    (state).SkipWithError(hexl_e_.what());     \
    return;                                    \
  }
