// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// hexl/number-theory: every non-inline function vs the __int128 oracle.

#include "hexl/hexl.hpp"
#include "oracle.hpp"
#include "test.hpp"

using namespace intel::hexl;
namespace O = hexltest::oracle;

namespace {
// Moduli of every size class the port meets: tiny, binfhe (~2^27), the e32
// boundary (2^30), 32-bit, NTT bench (49), BFV (60), and the 62-bit maximum.
std::vector<uint64_t> TestModuli() {
  std::vector<uint64_t> qs = {2, 3, 5, 17, 65537};
  for (size_t bits : {27, 29, 30, 31, 32, 40, 49, 50, 52, 55, 59, 60, 61}) {
    qs.push_back(O::NttPrime(bits, 1024));
  }
  qs.push_back(O::PrimeBelow(1ULL << 62));  // largest 62-bit prime
  return qs;
}
}  // namespace

TEST(NumberTheory_ReverseBits) {
  CHECK_EQ(ReverseBits(0, 0), 0ULL);
  CHECK_EQ(ReverseBits(1, 1), 1ULL);
  CHECK_EQ(ReverseBits(0b0011, 4), 0b1100ULL);
  CHECK_EQ(ReverseBits(1, 64), 1ULL << 63);
  for (uint64_t bits = 1; bits <= 12; ++bits) {
    for (uint64_t x = 0; x < (1ULL << bits); ++x) {
      CHECK_EQ(ReverseBits(x, bits), O::ReverseBits(x, bits),
               << "x=" << x << " bits=" << bits);
    }
  }
}

TEST(NumberTheory_AddSubUIntMod) {
  for (uint64_t q : TestModuli()) {
    auto a = O::Random(200, q), b = O::Random(200, q);
    a.push_back(q - 1); b.push_back(q - 1);
    a.push_back(0);     b.push_back(q - 1);
    for (size_t i = 0; i < a.size(); ++i) {
      CHECK_EQ(AddUIntMod(a[i], b[i], q), O::AddMod(a[i], b[i], q), << "q=" << q);
      CHECK_EQ(SubUIntMod(a[i], b[i], q), O::SubMod(a[i], b[i], q), << "q=" << q);
    }
  }
}

TEST(NumberTheory_MultiplyMod) {
  for (uint64_t q : TestModuli()) {
    if (q >= (1ULL << 61)) continue;  // MultiplyMod requires q < 2^61 (pre-shift Barrett)
    auto a = O::Random(300, q), b = O::Random(300, q);
    a.push_back(q - 1); b.push_back(q - 1);
    for (size_t i = 0; i < a.size(); ++i) {
      CHECK_EQ(MultiplyMod(a[i], b[i], q), O::MulMod(a[i], b[i], q),
               << "a=" << a[i] << " b=" << b[i] << " q=" << q);
    }
  }
}

// MultiplyFactor has three paths (one 64-bit divu when the dividend fits,
// the floor(2^64/q) divu identity, and the generic 128-by-64 division): every
// path must equal floor(operand * 2^bit_shift / q) computed in __int128.
TEST(NumberTheory_MultiplyFactor) {
  using u128 = unsigned __int128;
  auto want = [](uint64_t operand, uint64_t bit_shift, uint64_t q) {
    return static_cast<uint64_t>((static_cast<u128>(operand) << bit_shift) / q);
  };
  std::vector<uint64_t> qs = TestModuli();
  for (int b = 1; b < 64; ++b) qs.push_back(1ULL << b);  // where the 2^64 identity adds 1
  for (uint64_t q : {1000ULL, (1ULL << 63) + 1, ~0ULL, 0xFFFFFFFFFFFFFFC5ULL}) qs.push_back(q);
  for (uint64_t q : qs) {
    // operand 1: the plain Barrett factors (2^64 identity path, and the
    // fits-in-64-bits path for 32 and 52)
    for (uint64_t shift : {32ULL, 52ULL, 64ULL}) {
      CHECK_EQ(MultiplyFactor(1, shift, q).BarrettFactor(), want(1, shift, q),
               << "operand=1 bit_shift=" << shift << " q=" << q);
    }
    // operand y < q: the Shoup factors. bit_shift 64 is the generic path;
    // bit_shift 32 fits in 64 bits while y < 2^32.
    auto ys = O::Random(200, q);
    ys[0] = q - 1;
    ys[1] = 0;
    for (uint64_t y : ys) {
      CHECK_EQ(MultiplyFactor(y, 64, q).BarrettFactor(), want(y, 64, q), << "y=" << y << " bit_shift=64 q=" << q);
      if (q <= (1ULL << 32)) {
        CHECK_EQ(MultiplyFactor(y, 32, q).BarrettFactor(), want(y, 32, q), << "y=" << y << " bit_shift=32 q=" << q);
      }
    }
  }
}

TEST(NumberTheory_MultiplyModShoup) {
  // MultiplyMod(x, y, y_precon, q) with y_precon = floor(y * 2^64 / q)
  for (uint64_t q : TestModuli()) {
    if (q >= (1ULL << 63)) continue;
    auto a = O::Random(300, q), b = O::Random(300, q);
    for (size_t i = 0; i < a.size(); ++i) {
      const uint64_t precon = MultiplyFactor(b[i], 64, q).BarrettFactor();
      CHECK_EQ(MultiplyMod(a[i], b[i], precon, q), O::MulMod(a[i], b[i], q),
               << "q=" << q);
    }
  }
}

namespace {
// MultiplyAddMod<Imf>(x, w, w_precon, y, q) == (x * w + y) mod q for any x,
// w < q, y < Imf * q, q < 2^61.
template <int Imf>
void CheckMultiplyAddMod(uint64_t q) {
  auto x = O::Random(300, O::AnyWord<uint64_t>());
  auto y = O::Random(300, Imf * q);
  auto ws = O::Random(300, q);
  x[0] = ~0ULL, y[0] = Imf * q - 1, ws[0] = q - 1;
  x[1] = q - 1, y[1] = 0, ws[1] = 0;
  x[2] = ~0ULL, y[2] = Imf * q - 1, ws[2] = 1;
  for (size_t i = 0; i < x.size(); ++i) {
    const uint64_t wp = MultiplyFactor(ws[i], 64, q).BarrettFactor();
    const uint64_t want = (O::MulMod(x[i] % q, ws[i], q) + y[i] % q) % q;
    CHECK_EQ(MultiplyAddMod<Imf>(x[i], ws[i], wp, y[i], q), want,
             << "imf=" << Imf << " q=" << q << " x=" << x[i] << " w=" << ws[i] << " y=" << y[i]);
  }
}
}  // namespace

TEST(NumberTheory_MultiplyAddMod) {
  for (uint64_t q : TestModuli()) {
    if (q >= (1ULL << 61)) continue;  // contract: q < 2^61
    CheckMultiplyAddMod<1>(q);
    CheckMultiplyAddMod<2>(q);
    CheckMultiplyAddMod<4>(q);
    CheckMultiplyAddMod<8>(q);
  }
}

TEST(NumberTheory_PowMod) {
  for (uint64_t q : TestModuli()) {
    auto base = O::Random(50, q), exp = O::Random(50, ~0ULL);
    exp.push_back(0); base.push_back(0);
    exp.push_back(1); base.push_back(q - 1);
    for (size_t i = 0; i < base.size(); ++i) {
      CHECK_EQ(PowMod(base[i], exp[i], q), O::PowMod(base[i], exp[i], q),
               << "q=" << q);
    }
  }
}

TEST(NumberTheory_InverseMod) {
  for (uint64_t q : TestModuli()) {
    if (q < 3) continue;
    auto a = O::Random(100, q);
    a.push_back(1); a.push_back(q - 1);
    for (uint64_t x : a) {
      if (x == 0) continue;
      const uint64_t inv = InverseMod(x, q);
      CHECK(inv < q);
      CHECK_EQ(O::MulMod(x, inv, q), 1ULL, << "x=" << x << " q=" << q);
    }
  }
  // Non-prime modulus: extended Euclid, not Fermat.
  CHECK_EQ(O::MulMod(7, InverseMod(7, 1ULL << 40), 1ULL << 40), 1ULL);
  CHECK_EQ(InverseMod(3, 10), 7ULL);
}

TEST(NumberTheory_IsPrime) {
  const uint64_t primes[] = {2, 3, 5, 37, 41, 65537, 2147483647ULL,
                             (1ULL << 61) - 1, O::PrimeBelow(1ULL << 62),
                             18446744073709551557ULL};  // largest 64-bit prime
  const uint64_t composites[] = {0, 1, 4, 561, 1105, 41041, 3215031751ULL,
                                 3825123056546413051ULL,  // strong pseudoprime to bases 2..23
                                 (1ULL << 62) - 1, 18446744073709551615ULL};
  for (uint64_t p : primes) CHECK_EQ(IsPrime(p), true, << "p=" << p);
  for (uint64_t c : composites) {
    if (c < 2) continue;  // upstream behaviour for 0/1 is unspecified
    CHECK_EQ(IsPrime(c), false, << "c=" << c);
  }
  auto r = O::Random(hexltest::g_quick ? 200 : 3000, 1ULL << 62);
  for (uint64_t x : r) {
    x |= 1;
    CHECK_EQ(IsPrime(x), O::IsPrime(x), << "x=" << x);
  }
}

TEST(NumberTheory_GeneratePrimes) {
  for (uint64_t N : {1ULL, 1024ULL, 4096ULL, 32768ULL}) {
    for (size_t bits : {20, 27, 30, 45, 49, 60, 62}) {
      if ((1ULL << bits) <= 2 * N) continue;
      for (bool small : {true, false}) {
        auto got = GeneratePrimes(3, bits, small, N);
        auto want = O::Primes(3, bits, small, N);
        CHECK_VEC_EQ(want, got, << "N=" << N << " bits=" << bits
                                << " small=" << small);
      }
    }
  }
}

TEST(NumberTheory_PrimitiveRoots) {
  for (uint64_t N : {2ULL, 16ULL, 1024ULL, 16384ULL}) {
    for (size_t bits : {27, 30, 49, 60}) {
      const uint64_t q = O::NttPrime(bits, N);
      const uint64_t deg = 2 * N;
      const uint64_t w = O::PrimitiveRoot2N(N, q);
      CHECK(IsPrimitiveRoot(w, deg, q));
      CHECK(!IsPrimitiveRoot(0, deg, q));
      CHECK(!IsPrimitiveRoot(1, deg, q));
      CHECK(!IsPrimitiveRoot(O::MulMod(w, w, q), deg, q));  // order N only

      const uint64_t g = GeneratePrimitiveRoot(deg, q);
      CHECK(O::PowMod(g, N, q) == q - 1);  // order exactly 2N

      CHECK_EQ(MinimalPrimitiveRoot(deg, q), O::MinimalPrimitiveRoot2N(N, q),
               << "N=" << N << " q=" << q);
    }
  }
}
