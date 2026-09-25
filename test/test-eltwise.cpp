// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// hexl/eltwise: all seven public functions, every size tail, every modulus
// class, every input/output mod factor, out-of-place AND in-place (OpenFHE
// calls almost everything with result == operand1).

#include "hexl/hexl.hpp"
#include "oracle.hpp"
#include "test.hpp"

using namespace intel::hexl;
namespace O = hexltest::oracle;
using Vec = std::vector<uint64_t>;

namespace {

/// Moduli with at most max_bits bits, covering the e32/e64 dispatch boundary.
std::vector<uint64_t> Moduli(size_t max_bits) {
  std::vector<uint64_t> qs = {2, 3, 17};
  for (size_t bits : {20, 27, 29}) qs.push_back(O::NttPrime(bits, 1));
  qs.push_back(O::PrimeBelow(1ULL << 30));  // last modulus on the e32 path
  qs.push_back(O::PrimeAbove(1ULL << 30));  // first modulus on the e64 path
  for (size_t bits : {31, 32, 40, 49, 50, 52, 59, 60, 61, 62}) {
    if (bits + 1 <= max_bits) qs.push_back(O::NttPrime(bits, 1));
  }
  if (max_bits >= 63) qs.push_back(O::PrimeBelow(1ULL << 63));
  return qs;
}

/// Runs f(result, n) both out-of-place (into a fresh buffer) and in-place
/// (into a copy of `in_place_src`), and checks both against `want`.
template <class F>
void CheckBothWays(const Vec& want, const Vec& in_place_src, F f,
                   const char* what, uint64_t q) {
  const size_t n = want.size();
  Vec out(n, 0xdeadbeefULL);
  f(out.data(), static_cast<const uint64_t*>(in_place_src.data()));
  CHECK_VEC_EQ(want, out, << what << " out-of-place n=" << n << " q=" << q);

  Vec inplace = in_place_src;
  f(inplace.data(), static_cast<const uint64_t*>(inplace.data()));
  CHECK_VEC_EQ(want, inplace, << what << " in-place n=" << n << " q=" << q);
}

}  // namespace

TEST(Eltwise_AddMod) {
  for (uint64_t q : Moduli(63)) {
    for (size_t n : O::EltwiseSizes()) {
      Vec a = O::Random(n, q), b = O::Random(n, q);
      a[0] = q - 1; b[0] = q - 1;  // largest possible sum
      const uint64_t s = O::Random(1, q)[0];
      Vec want_vv(n), want_vs(n);
      for (size_t i = 0; i < n; ++i) {
        want_vv[i] = O::AddMod(a[i], b[i], q);
        want_vs[i] = O::AddMod(a[i], s, q);
      }
      CheckBothWays(want_vv, a, [&](uint64_t* r, const uint64_t* x) {
        EltwiseAddMod(r, x, b.data(), n, q); }, "AddMod vv", q);
      CheckBothWays(want_vs, a, [&](uint64_t* r, const uint64_t* x) {
        EltwiseAddMod(r, x, s, n, q); }, "AddMod vs", q);
    }
  }
}

TEST(Eltwise_SubMod) {
  for (uint64_t q : Moduli(63)) {
    for (size_t n : O::EltwiseSizes()) {
      Vec a = O::Random(n, q), b = O::Random(n, q);
      a[0] = 0; b[0] = q - 1;  // largest borrow
      const uint64_t s = O::Random(1, q)[0];
      Vec want_vv(n), want_vs(n);
      for (size_t i = 0; i < n; ++i) {
        want_vv[i] = O::SubMod(a[i], b[i], q);
        want_vs[i] = O::SubMod(a[i], s, q);
      }
      CheckBothWays(want_vv, a, [&](uint64_t* r, const uint64_t* x) {
        EltwiseSubMod(r, x, b.data(), n, q); }, "SubMod vv", q);
      CheckBothWays(want_vs, a, [&](uint64_t* r, const uint64_t* x) {
        EltwiseSubMod(r, x, s, n, q); }, "SubMod vs", q);
    }
  }
}

TEST(Eltwise_MultMod) {
  for (uint64_t imf : {1ULL, 2ULL, 4ULL}) {
    for (uint64_t q : Moduli(62)) {
      if (imf * q >= (1ULL << 63)) continue;  // API precondition
      for (size_t n : O::EltwiseSizes()) {
        Vec a = O::Random(n, imf * q), b = O::Random(n, imf * q);
        a[0] = imf * q - 1; b[0] = imf * q - 1;
        Vec want(n);
        for (size_t i = 0; i < n; ++i) want[i] = O::MulMod(a[i], b[i], q);
        CheckBothWays(want, a, [&](uint64_t* r, const uint64_t* x) {
          EltwiseMultMod(r, x, b.data(), n, q, imf); }, "MultMod", q);
      }
    }
  }
}

TEST(Eltwise_FMAMod) {
  for (uint64_t imf : {1ULL, 2ULL, 4ULL, 8ULL}) {
    for (uint64_t q : Moduli(61)) {
      if (q >= (1ULL << 61)) continue;
      for (size_t n : O::EltwiseSizes()) {
        Vec a = O::Random(n, imf * q), c = O::Random(n, imf * q);
        const uint64_t s = O::Random(1, imf * q)[0];
        Vec want_add(n), want_mul(n);
        for (size_t i = 0; i < n; ++i) {
          want_mul[i] = O::MulMod(a[i] % q, s % q, q);
          want_add[i] = O::AddMod(want_mul[i], c[i] % q, q);
        }
        CheckBothWays(want_add, a, [&](uint64_t* r, const uint64_t* x) {
          EltwiseFMAMod(r, x, s, c.data(), n, q, imf); }, "FMAMod +arg3", q);
        CheckBothWays(want_mul, a, [&](uint64_t* r, const uint64_t* x) {
          EltwiseFMAMod(r, x, s, nullptr, n, q, imf); }, "FMAMod no arg3", q);
        // result aliasing arg3 (OpenFHE's hexldcrtpoly does this)
        Vec c2 = c;
        EltwiseFMAMod(c2.data(), a.data(), s, c2.data(), n, q, imf);
        CHECK_VEC_EQ(want_add, c2, << "FMAMod result==arg3 q=" << q);
      }
    }
  }
}

TEST(Eltwise_ReduceMod) {
  struct Case { uint64_t in; uint64_t out; };  // in == 0 means "modulus"
  const Case cases[] = {{0, 1}, {0, 2}, {2, 1}, {4, 1}, {4, 2}, {2, 2}};
  for (uint64_t q : Moduli(62)) {
    for (const Case& c : cases) {
      if (!c.in && q <= 4) continue;  // imf == q would alias the 2/4 codes
      const uint64_t imf = c.in ? c.in : q;
      if (c.in && imf * q >= (1ULL << 63)) continue;
      for (size_t n : O::EltwiseSizes()) {
        // "modulus" means arbitrary 64-bit input
        Vec a = c.in ? O::Random(n, imf * q) : O::Random(n, ~0ULL);
        Vec out(n);
        EltwiseReduceMod(out.data(), a.data(), n, q, imf, c.out);
        for (size_t i = 0; i < n; ++i) {
          // Any representative in [0, out*q) congruent mod q is correct.
          if (c.in == c.out) {
            CHECK_EQ(out[i], a[i], << "copy case i=" << i);
          } else {
            CHECK(out[i] < c.out * q);
            CHECK_EQ(out[i] % q, a[i] % q, << "i=" << i << " q=" << q
                     << " in=" << imf << " out=" << c.out);
          }
        }
        // In place is only defined when the factors differ (upstream asserts
        // input_mod_factor != output_mod_factor once operand == result).
        if (c.in != c.out) {
          Vec inplace = a;
          EltwiseReduceMod(inplace.data(), inplace.data(), n, q, imf, c.out);
          for (size_t i = 0; i < n; ++i) {
            CHECK(inplace[i] < c.out * q);
            CHECK_EQ(inplace[i] % q, a[i] % q);
          }
        }
      }
    }
  }
}

TEST(Eltwise_CmpAdd) {
  const CMPINT cmps[] = {CMPINT::EQ,  CMPINT::LT,  CMPINT::LE,  CMPINT::FALSE,
                         CMPINT::NE,  CMPINT::NLT, CMPINT::NLE, CMPINT::TRUE};
  for (CMPINT cmp : cmps) {
    for (size_t n : O::EltwiseSizes()) {
      const uint64_t range = 1000;  // small range so EQ actually hits
      Vec a = O::Random(n, range);
      const uint64_t bound = O::Random(1, range)[0];
      const uint64_t diff = 1 + O::Random(1, 1ULL << 40)[0];
      Vec want(n);
      for (size_t i = 0; i < n; ++i) {
        bool c = false;
        switch (cmp) {
          case CMPINT::EQ: c = a[i] == bound; break;
          case CMPINT::LT: c = a[i] < bound; break;
          case CMPINT::LE: c = a[i] <= bound; break;
          case CMPINT::FALSE: c = false; break;
          case CMPINT::NE: c = a[i] != bound; break;
          case CMPINT::NLT: c = a[i] >= bound; break;
          case CMPINT::NLE: c = a[i] > bound; break;
          case CMPINT::TRUE: c = true; break;
        }
        want[i] = c ? a[i] + diff : a[i];
      }
      CheckBothWays(want, a, [&](uint64_t* r, const uint64_t* x) {
        EltwiseCmpAdd(r, x, n, cmp, bound, diff); }, "CmpAdd",
        static_cast<uint64_t>(cmp));
    }
  }
}

TEST(Eltwise_CmpSubMod) {
  const CMPINT cmps[] = {CMPINT::EQ,  CMPINT::LT,  CMPINT::LE,  CMPINT::FALSE,
                         CMPINT::NE,  CMPINT::NLT, CMPINT::NLE, CMPINT::TRUE};
  for (uint64_t q : Moduli(62)) {
    if (q < 3) continue;
    for (CMPINT cmp : cmps) {
      for (size_t n : O::EltwiseSizes()) {
        // Mix of in-range, just-above-q and full 64-bit values; the bound sits
        // inside the data so every predicate splits the vector.
        Vec a = O::Random(n, 4 * q < q ? ~0ULL : 4 * q);
        for (size_t i = 0; i < n; i += 5) a[i] = O::Random(1, ~0ULL)[0];
        const uint64_t bound = a[n / 2];
        const uint64_t diff = 1 + O::Random(1, q - 1)[0];
        Vec want(n);
        for (size_t i = 0; i < n; ++i) {
          bool c = false;
          switch (cmp) {
            case CMPINT::EQ: c = a[i] == bound; break;
            case CMPINT::LT: c = a[i] < bound; break;
            case CMPINT::LE: c = a[i] <= bound; break;
            case CMPINT::FALSE: c = false; break;
            case CMPINT::NE: c = a[i] != bound; break;
            case CMPINT::NLT: c = a[i] >= bound; break;
            case CMPINT::NLE: c = a[i] > bound; break;
            case CMPINT::TRUE: c = true; break;
          }
          const uint64_t r = a[i] % q;
          want[i] = c ? O::SubMod(r, diff, q) : r;
        }
        CheckBothWays(want, a, [&](uint64_t* r, const uint64_t* x) {
          EltwiseCmpSubMod(r, x, n, q, cmp, bound, diff); }, "CmpSubMod", q);
      }
    }
  }
}

// The exact call pattern of OpenFHE's NativeVector::SwitchModulus
// (mubintvecnathexl.cpp) when moving to a smaller and to a larger modulus.
TEST(Eltwise_OpenFHESwitchModulusPattern) {
  const uint64_t om = O::NttPrime(49, 1024);  // old modulus
  for (uint64_t nm : {O::NttPrime(27, 1024), O::NttPrime(55, 1024)}) {
    const size_t n = 1024;
    Vec v = O::Random(n, om), got = v;
    if (nm < om) {
      EltwiseCmpSubMod(got.data(), got.data(), n, nm, CMPINT::NLE, om >> 1,
                       (om - nm) % nm);
    } else {
      EltwiseCmpAdd(got.data(), got.data(), n, CMPINT::NLE, om >> 1, nm - om);
    }
    for (size_t i = 0; i < n; ++i) {
      // centred lift: values above om/2 represent negatives
      const bool neg = v[i] > (om >> 1);
      uint64_t want;
      if (nm < om) {
        want = neg ? O::SubMod(v[i] % nm, (om - nm) % nm, nm) : v[i] % nm;
      } else {
        want = neg ? v[i] + (nm - om) : v[i];
      }
      CHECK_EQ(got[i], want, << "i=" << i << " nm=" << nm);
    }
  }
}
