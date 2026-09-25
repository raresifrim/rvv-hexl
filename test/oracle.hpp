// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Independent ground truth for the tests. Deliberately naive and written only
// with unsigned __int128 arithmetic: nothing here calls into HEXL, so a bug in
// the library cannot hide by also being in the oracle. Validated by running the
// whole test suite against upstream Intel HEXL (make HEXL_IMPL=intel).

#pragma once

#include <stdint.h>

#include <random>
#include <vector>

namespace hexltest {

extern bool g_quick;  // --quick: smaller sweeps (spike)

namespace oracle {

using u128 = unsigned __int128;

inline uint64_t MulMod(uint64_t a, uint64_t b, uint64_t q) {
  return static_cast<uint64_t>((u128)a * b % q);
}
inline uint64_t AddMod(uint64_t a, uint64_t b, uint64_t q) {
  return static_cast<uint64_t>(((u128)a + b) % q);
}
inline uint64_t SubMod(uint64_t a, uint64_t b, uint64_t q) {
  return static_cast<uint64_t>(((u128)a % q + q - (u128)b % q) % q);
}
inline uint64_t PowMod(uint64_t b, uint64_t e, uint64_t q) {
  uint64_t r = 1 % q;
  b %= q;
  while (e) {
    if (e & 1) r = MulMod(r, b, q);
    b = MulMod(b, b, q);
    e >>= 1;
  }
  return r;
}
/// Inverse modulo a PRIME q (Fermat).
inline uint64_t InvModPrime(uint64_t a, uint64_t q) { return PowMod(a, q - 2, q); }

/// Deterministic Miller-Rabin, exact for all 64-bit n.
inline bool IsPrime(uint64_t n) {
  if (n < 2) return false;
  static const uint64_t small[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
  for (uint64_t p : small) {
    if (n == p) return true;
    if (n % p == 0) return false;
  }
  uint64_t d = n - 1;
  int s = 0;
  while ((d & 1) == 0) {
    d >>= 1;
    ++s;
  }
  for (uint64_t a : small) {
    uint64_t x = PowMod(a, d, n);
    if (x == 1 || x == n - 1) continue;
    bool composite = true;
    for (int r = 1; r < s; ++r) {
      x = MulMod(x, x, n);
      if (x == n - 1) {
        composite = false;
        break;
      }
    }
    if (composite) return false;
  }
  return true;
}

inline uint64_t ReverseBits(uint64_t x, uint64_t bits) {
  uint64_t r = 0;
  for (uint64_t i = 0; i < bits; ++i) {
    r = (r << 1) | ((x >> i) & 1);
  }
  return r;
}

/// Primes q = 1 mod 2N in [2^bits, 2^(bits+1)), in the walk order that
/// GeneratePrimes documents (ascending from 2^bits+1, or descending from the
/// top). Returns fewer than `count` if the range runs out.
inline std::vector<uint64_t> Primes(size_t count, size_t bits, bool small_first,
                                    uint64_t N) {
  std::vector<uint64_t> out;
  const uint64_t lo = (1ULL << bits) + 1;
  const uint64_t hi = (bits + 1 >= 64) ? ~0ULL : (1ULL << (bits + 1)) - 1;
  const uint64_t step = 2 * N;
  if (small_first) {
    for (uint64_t c = lo; c < hi && out.size() < count; c += step)
      if (IsPrime(c)) out.push_back(c);
  } else {
    uint64_t c = hi - (hi % step) + 1;
    for (; c > lo && out.size() < count; c -= step)
      if (IsPrime(c)) out.push_back(c);
  }
  return out;
}

/// Largest prime < x / smallest prime > x
inline uint64_t PrimeBelow(uint64_t x) {
  for (uint64_t c = x - 1;; --c) if (IsPrime(c)) return c;
}
inline uint64_t PrimeAbove(uint64_t x) {
  for (uint64_t c = x + 1;; ++c) if (IsPrime(c)) return c;
}

/// One NTT-friendly prime (q = 1 mod 2N) of the given bit size.
inline uint64_t NttPrime(size_t bits, uint64_t N, size_t index = 0) {
  return Primes(index + 1, bits, true, N).back();
}

/// Some primitive 2N-th root of unity mod q (smallest generator-derived one).
inline uint64_t PrimitiveRoot2N(uint64_t N, uint64_t q) {
  const uint64_t m = 2 * N;
  for (uint64_t g = 2;; ++g) {
    uint64_t w = PowMod(g, (q - 1) / m, q);
    if (PowMod(w, N, q) == q - 1) return w;  // order exactly 2N
  }
}

/// The numerically smallest primitive 2N-th root of unity.
inline uint64_t MinimalPrimitiveRoot2N(uint64_t N, uint64_t q) {
  const uint64_t w = PrimitiveRoot2N(N, q);
  const uint64_t w2 = MulMod(w, w, q);
  uint64_t cur = w, best = w;
  for (uint64_t k = 0; k < N; ++k) {  // w^1, w^3, ..., w^(2N-1)
    if (cur < best) best = cur;
    cur = MulMod(cur, w2, q);
  }
  return best;
}

/// Negacyclic forward NTT, O(N^2), HEXL output order:
///   out[i] = sum_j a[j] * w^((2*rev(i)+1)*j)  mod q
inline std::vector<uint64_t> ForwardNtt(const std::vector<uint64_t>& a,
                                        uint64_t q, uint64_t w) {
  const uint64_t N = a.size();
  uint64_t bits = 0;
  while ((1ULL << bits) < N) ++bits;
  std::vector<uint64_t> out(N);
  for (uint64_t i = 0; i < N; ++i) {
    const uint64_t x = PowMod(w, 2 * ReverseBits(i, bits) + 1, q);
    uint64_t acc = 0, xp = 1;
    for (uint64_t j = 0; j < N; ++j) {
      acc = AddMod(acc, MulMod(a[j] % q, xp, q), q);
      xp = MulMod(xp, x, q);
    }
    out[i] = acc;
  }
  return out;
}

/// Deterministic RNG shared by all tests (reproducible failures).
inline std::mt19937_64& Rng() {
  static std::mt19937_64 rng(0x5eed1234abcdULL);
  return rng;
}

/// n uniform values in [0, bound)
inline std::vector<uint64_t> Random(size_t n, uint64_t bound) {
  std::uniform_int_distribution<uint64_t> d(0, bound - 1);
  std::vector<uint64_t> v(n);
  for (auto& x : v) x = d(Rng());
  return v;
}

/// Converts an oracle vector (uint64_t) to the storage word under test.
template <typename Word>
inline std::vector<Word> As(const std::vector<uint64_t>& v) {
  return std::vector<Word>(v.begin(), v.end());
}

/// Exclusive bound for "any value of this word" (Random(n, AnyWord<W>())).
/// For uint64_t the RNG cannot take 2^64, so 2^64 - 1 is used.
template <typename Word>
constexpr uint64_t AnyWord() {
  return sizeof(Word) == 8 ? ~0ULL : (1ULL << (8 * sizeof(Word)));
}

/// Sizes that exercise every tail case of a strip-mined RVV loop
/// (VLMAX is 4/8 for e64/e32 at VLEN=256 and 16/32 at VLEN=1024).
inline std::vector<size_t> EltwiseSizes() {
  if (g_quick) return {1, 3, 8, 33, 257};
  return {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100,
          255, 256, 257, 1023, 1024, 1025, 4096, 16384 + 3};
}

}  // namespace oracle
}  // namespace hexltest
