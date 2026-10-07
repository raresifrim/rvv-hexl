// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Scalar number theory. Nothing here is RISC-V specific and nothing is on a
// hot path (it runs once per NTT object / per parameter set), so write it for
// clarity and exactness, not speed. Start here: the NTT constructor and most
// tests depend on these functions.
//
// Contracts are documented in include/hexl/number-theory/number-theory.hpp.
// The tests (test/test-number-theory.cpp) check every function against an
// independent __int128 oracle in test/oracle.hpp.

#include "hexl/number-theory/number-theory.hpp"

#include "hexl/logging/logging.hpp"
#include "hexl/util/check.hpp"
#include "util/not-implemented.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

uint64_t InverseMod(uint64_t input, uint64_t modulus) {
  // extended Euclid on (input mod modulus, modulus).
  int64_t t_prev=0, t_cur=1;
  uint64_t r_prev=modulus, r_cur=input%modulus;
  while(r_cur != 0){
    int64_t k = r_prev / r_cur;
    uint64_t r_new = r_prev - k * r_cur;
    int64_t t_new = t_prev - k * t_cur;
    r_prev = r_cur;
    t_prev = t_cur;
    r_cur = r_new; 
    t_cur = t_new; 
  }
  HEXL_CHECK(r_prev == 1, "Previous remainder must be 1!");
  return static_cast<uint64_t>(t_prev + (t_prev < 0 ? modulus : 0U)); 
}

uint64_t AddUIntMod(uint64_t x, uint64_t y, uint64_t modulus) {
  //(x + y) mod modulus for x, y < modulus.
  uint64_t sum = x+y;
  return std::min(sum, sum-modulus);
}

uint64_t SubUIntMod(uint64_t x, uint64_t y, uint64_t modulus) {
  // (x - y) mod modulus for x, y < modulus.
  uint64_t diff = x-y;
  return std::min(diff, diff + modulus);
}

uint64_t PowMod(uint64_t base, uint64_t exp, uint64_t modulus) {
  // TODO(port): square-and-multiply with MultiplyMod; reduce base first.
  HEXL_NOT_IMPLEMENTED();
}

bool IsPrimitiveRoot(uint64_t root, uint64_t degree, uint64_t modulus) {
  // TODO(port): root == 0 -> false; otherwise, for power-of-two degree,
  // root is a primitive degree-th root iff root^(degree/2) == modulus - 1.
  HEXL_NOT_IMPLEMENTED();
}

uint64_t GeneratePrimitiveRoot(uint64_t degree, uint64_t modulus) {
  // TODO(port): pick random x in [0, q), set r = x^((q-1)/degree); r is a
  // degree-th root of unity; return it if IsPrimitiveRoot(r, degree, q).
  // Retry a bounded number of times (upstream: 200) and fail loudly.
  // Tip: std::mt19937_64 with a fixed seed keeps runs reproducible.
  HEXL_NOT_IMPLEMENTED();
}

uint64_t MinimalPrimitiveRoot(uint64_t degree, uint64_t modulus) {
  // TODO(port): r = GeneratePrimitiveRoot(degree, q). All primitive
  // degree-th roots are r^k for odd k, so walk r, r^3, r^5, ... (multiply by
  // r^2 each step, degree/2 steps) and return the smallest value seen.
  HEXL_NOT_IMPLEMENTED();
}

uint64_t ReverseBits(uint64_t x, uint64_t bit_width) {
  // TODO(port): reverse the low bit_width bits of x; bit_width == 0 -> 0.
  // (Only used when building twiddle tables; a simple loop is fine. With Zbb
  // there is also rev8/brev8-based tricks, and Zvbb has vbrev.v for vectors.)
  HEXL_NOT_IMPLEMENTED();
}

bool IsPrime(uint64_t n) {
  // TODO(port): deterministic Miller-Rabin. The witness set
  // {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37} is exact for all n < 2^64.
  // Trial-divide by those witnesses first (and return true when n equals one).
  HEXL_NOT_IMPLEMENTED();
}

std::vector<uint64_t> GeneratePrimes(size_t num_primes, size_t bit_size,
                                     bool prefer_small_primes,
                                     size_t ntt_size) {
  // TODO(port): candidates are q = 1 mod 2N inside [2^bit_size, 2^(bit_size+1)).
  //   prefer_small_primes: start at 2^bit_size + 1, step +2N, go up.
  //   otherwise:           start at the largest q = 1 mod 2N below
  //                        2^(bit_size+1), step -2N, go down.
  // Collect IsPrime() candidates in walk order until num_primes are found.
  // The tests compare the exact list against an oracle walking the same way.
  HEXL_NOT_IMPLEMENTED();
}

}  // namespace hexl
}  // namespace intel
