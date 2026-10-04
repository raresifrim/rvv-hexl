// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// hexl/eltwise: all seven public functions, every size tail, every modulus
// class, every input/output mod factor, out-of-place AND in-place (OpenFHE
// calls almost everything with result == operand1).
//
// Every test is a template on the storage word. Eltwise_<Op> runs it with
// uint64_t (the upstream API; validated against Intel HEXL); Eltwise_<Op>_32
// runs the same checks with uint32_t (the rvv-hexl 32-bit extension, for
// OpenFHE NATIVE_SIZE=32). Only the modulus ranges differ, following each
// function's documented preconditions.

#include <type_traits>

#include "hexl/hexl.hpp"
#include "oracle.hpp"
#include "test.hpp"

using namespace intel::hexl;
namespace O = hexltest::oracle;

namespace {

template <typename Word>
constexpr bool Is32() {
  return std::is_same<Word, uint32_t>::value;
}

/// Candidate moduli: every size class, both sides of the e32/e64 dispatch
/// boundary (2^30) and of the 32-bit storage limit. Each test keeps the ones
/// its preconditions allow for the word under test.
std::vector<uint64_t> Candidates() {
  std::vector<uint64_t> qs = {2, 3, 17};
  for (size_t bits : {20, 27, 29}) qs.push_back(O::NttPrime(bits, 1));
  qs.push_back(O::PrimeBelow(1ULL << 30));  // last modulus on the e32 path
  qs.push_back(O::PrimeAbove(1ULL << 30));  // first one past it
  qs.push_back(O::NttPrime(31, 1));         // in [2^31, 2^32)
  qs.push_back(O::PrimeBelow(1ULL << 32));  // largest 32-bit modulus
  for (size_t bits : {32, 40, 49, 50, 52, 59, 60, 61, 62}) {
    qs.push_back(O::NttPrime(bits, 1));
  }
  qs.push_back(O::PrimeBelow(1ULL << 63));
  return qs;
}

/// Candidates q with pred(q) true.
template <class P>
std::vector<uint64_t> Moduli(P pred) {
  std::vector<uint64_t> out;
  for (uint64_t q : Candidates()) {
    if (pred(q)) out.push_back(q);
  }
  return out;
}

/// Runs f(result, operand) out-of-place (into a fresh buffer) and in-place
/// (into a copy of `src`), and checks both against `want`.
template <typename Word, class F>
void CheckBothWays(const std::vector<Word>& want, const std::vector<Word>& src,
                   F f, const char* what, uint64_t q) {
  const size_t n = want.size();
  std::vector<Word> out(n, static_cast<Word>(0xdeadbeefULL));
  f(out.data(), static_cast<const Word*>(src.data()));
  CHECK_VEC_EQ(want, out, << what << " out-of-place n=" << n << " q=" << q
                          << " word=" << 8 * sizeof(Word));

  std::vector<Word> inplace = src;
  f(inplace.data(), static_cast<const Word*>(inplace.data()));
  CHECK_VEC_EQ(want, inplace, << what << " in-place n=" << n << " q=" << q
                              << " word=" << 8 * sizeof(Word));
}

bool Compare(CMPINT cmp, uint64_t a, uint64_t bound) {
  switch (cmp) {
    case CMPINT::EQ: return a == bound;
    case CMPINT::LT: return a < bound;
    case CMPINT::LE: return a <= bound;
    case CMPINT::FALSE: return false;
    case CMPINT::NE: return a != bound;
    case CMPINT::NLT: return a >= bound;
    case CMPINT::NLE: return a > bound;
    case CMPINT::TRUE: return true;
  }
  return false;
}

const CMPINT kCmps[] = {CMPINT::EQ,  CMPINT::LT,  CMPINT::LE,  CMPINT::FALSE,
                        CMPINT::NE,  CMPINT::NLT, CMPINT::NLE, CMPINT::TRUE};

// ---------------------------------------------------------------------------

template <typename Word>
void RunAddSub(bool add) {
  // 64-bit: q < 2^63 (the sum fits). 32-bit: q <= 2^32 (values fit in Word).
  auto ok = [](uint64_t q) { return Is32<Word>() ? q <= (1ULL << 32) : q < (1ULL << 63); };
  for (uint64_t q : Moduli(ok)) {
    for (size_t n : O::EltwiseSizes()) {
      auto a64 = O::Random(n, q), b64 = O::Random(n, q);
      if (add) { a64[0] = q - 1; b64[0] = q - 1; }  // largest possible sum
      else     { a64[0] = 0;     b64[0] = q - 1; }  // largest borrow
      const uint64_t s = O::Random(1, q)[0];
      std::vector<Word> want_vv(n), want_vs(n);
      for (size_t i = 0; i < n; ++i) {
        want_vv[i] = static_cast<Word>(add ? O::AddMod(a64[i], b64[i], q) : O::SubMod(a64[i], b64[i], q));
        want_vs[i] = static_cast<Word>(add ? O::AddMod(a64[i], s, q) : O::SubMod(a64[i], s, q));
      }
      const auto a = O::As<Word>(a64), b = O::As<Word>(b64);
      if (add) {
        CheckBothWays(want_vv, a, [&](Word* r, const Word* x) {
          EltwiseAddMod(r, x, b.data(), n, q); }, "AddMod vv", q);
        CheckBothWays(want_vs, a, [&](Word* r, const Word* x) {
          EltwiseAddMod(r, x, s, n, q); }, "AddMod vs", q);
      } else {
        CheckBothWays(want_vv, a, [&](Word* r, const Word* x) {
          EltwiseSubMod(r, x, b.data(), n, q); }, "SubMod vv", q);
        CheckBothWays(want_vs, a, [&](Word* r, const Word* x) {
          EltwiseSubMod(r, x, s, n, q); }, "SubMod vs", q);
      }
    }
  }
}

template <typename Word>
void RunMultMod() {
  for (uint64_t imf : {1ULL, 2ULL, 4ULL}) {
    // 64-bit: q < 2^61 (the pre-shift Barrett; same bound as EltwiseFMAMod),
    // which also keeps imf*q < 2^63. 32-bit: imf*q <= 2^32.
    auto ok = [imf](uint64_t q) {
      // (written as a division: imf * q itself can wrap past 2^64)
      return Is32<Word>() ? q <= (1ULL << 32) / imf
                          : q < (1ULL << 61) && imf * q < (1ULL << 63);
    };
    for (uint64_t q : Moduli(ok)) {
      for (size_t n : O::EltwiseSizes()) {
        auto a64 = O::Random(n, imf * q), b64 = O::Random(n, imf * q);
        a64[0] = imf * q - 1; b64[0] = imf * q - 1;
        std::vector<Word> want(n);
        for (size_t i = 0; i < n; ++i) want[i] = static_cast<Word>(O::MulMod(a64[i], b64[i], q));
        const auto a = O::As<Word>(a64), b = O::As<Word>(b64);
        CheckBothWays(want, a, [&](Word* r, const Word* x) {
          EltwiseMultMod(r, x, b.data(), n, q, imf); }, "MultMod", q);
      }
    }
  }
}

template <typename Word>
void RunFMAMod() {
  for (uint64_t imf : {1ULL, 2ULL, 4ULL, 8ULL}) {
    // 64-bit: q < 2^61. 32-bit: imf*q <= 2^32.
    auto ok = [imf](uint64_t q) {
      return Is32<Word>() ? q <= (1ULL << 32) / imf : q < (1ULL << 61);
    };
    for (uint64_t q : Moduli(ok)) {
      for (size_t n : O::EltwiseSizes()) {
        auto a64 = O::Random(n, imf * q), c64 = O::Random(n, imf * q);
        const uint64_t s = O::Random(1, imf * q)[0];
        std::vector<Word> want_add(n), want_mul(n);
        for (size_t i = 0; i < n; ++i) {
          const uint64_t m = O::MulMod(a64[i] % q, s % q, q);
          want_mul[i] = static_cast<Word>(m);
          want_add[i] = static_cast<Word>(O::AddMod(m, c64[i] % q, q));
        }
        const auto a = O::As<Word>(a64), c = O::As<Word>(c64);
        CheckBothWays(want_add, a, [&](Word* r, const Word* x) {
          EltwiseFMAMod(r, x, s, c.data(), n, q, imf); }, "FMAMod +arg3", q);
        CheckBothWays(want_mul, a, [&](Word* r, const Word* x) {
          EltwiseFMAMod(r, x, s, static_cast<const Word*>(nullptr), n, q, imf); },
          "FMAMod no arg3", q);
        // result aliasing arg3 (OpenFHE's hexldcrtpoly does this)
        auto c2 = c;
        EltwiseFMAMod(c2.data(), a.data(), s, c2.data(), n, q, imf);
        CHECK_VEC_EQ(want_add, c2, << "FMAMod result==arg3 q=" << q);
      }
    }
  }
}

template <typename Word>
void RunReduceMod() {
  struct Case { uint64_t in; uint64_t out; };  // in == 0 means "modulus"
  const Case cases[] = {{0, 1}, {0, 2}, {2, 1}, {4, 1}, {4, 2}, {2, 2}};
  // 64-bit: q < 2^62. 32-bit: q <= 2^32.
  auto ok = [](uint64_t q) { return Is32<Word>() ? q <= (1ULL << 32) : q < (1ULL << 62); };
  for (uint64_t q : Moduli(ok)) {
    for (const Case& c : cases) {
      if (!c.in && q <= 4) continue;  // imf == q would alias the 2/4 codes
      const uint64_t imf = c.in ? c.in : q;
      if (c.in && (Is32<Word>() ? imf * q > (1ULL << 32) : imf * q >= (1ULL << 63))) continue;
      for (size_t n : O::EltwiseSizes()) {
        // "modulus" means any value of the word
        const auto a64 = c.in ? O::Random(n, imf * q) : O::Random(n, O::AnyWord<Word>());
        const auto a = O::As<Word>(a64);
        std::vector<Word> out(n);
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
          auto inplace = a;
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

template <typename Word>
void RunCmpAdd() {
  for (CMPINT cmp : kCmps) {
    for (size_t n : O::EltwiseSizes()) {
      const uint64_t range = 1000;  // small range so EQ actually hits
      const auto a64 = O::Random(n, range);
      const uint64_t bound = O::Random(1, range)[0];
      const uint64_t diff = 1 + O::Random(1, Is32<Word>() ? (1ULL << 20) : (1ULL << 40))[0];
      std::vector<Word> want(n);
      for (size_t i = 0; i < n; ++i) {
        want[i] = static_cast<Word>(Compare(cmp, a64[i], bound) ? a64[i] + diff : a64[i]);
      }
      CheckBothWays(want, O::As<Word>(a64), [&](Word* r, const Word* x) {
        EltwiseCmpAdd(r, x, n, cmp, bound, diff); }, "CmpAdd",
        static_cast<uint64_t>(cmp));
    }
  }
}

template <typename Word>
void RunCmpSubMod() {
  // 64-bit: q < 2^62. 32-bit: q <= 2^32.
  auto ok = [](uint64_t q) { return q >= 3 && (Is32<Word>() ? q <= (1ULL << 32) : q < (1ULL << 62)); };
  for (uint64_t q : Moduli(ok)) {
    for (CMPINT cmp : kCmps) {
      for (size_t n : O::EltwiseSizes()) {
        // Mix of in-range, just-above-q and full-width values; the bound sits
        // inside the data so every predicate splits the vector.
        const uint64_t span = (4 * q < q || 4 * q > O::AnyWord<Word>()) ? O::AnyWord<Word>() : 4 * q;
        auto a64 = O::Random(n, span);
        for (size_t i = 0; i < n; i += 5) a64[i] = O::Random(1, O::AnyWord<Word>())[0];
        const uint64_t bound = a64[n / 2];
        const uint64_t diff = 1 + O::Random(1, q - 1)[0];
        std::vector<Word> want(n);
        for (size_t i = 0; i < n; ++i) {
          const uint64_t r = a64[i] % q;
          want[i] = static_cast<Word>(Compare(cmp, a64[i], bound) ? O::SubMod(r, diff, q) : r);
        }
        CheckBothWays(want, O::As<Word>(a64), [&](Word* r, const Word* x) {
          EltwiseCmpSubMod(r, x, n, q, cmp, bound, diff); }, "CmpSubMod", q);
      }
    }
  }
}

// The exact call pattern of OpenFHE's NativeVector::SwitchModulus
// (mubintvecnathexl.cpp) when moving to a smaller and to a larger modulus.
// Moduli as OpenFHE uses them: <= 60 bits at NATIVE_SIZE=64, <= 28 at 32.
template <typename Word>
void RunSwitchModulusPattern() {
  const uint64_t om = O::NttPrime(Is32<Word>() ? 24 : 49, 1024);  // old modulus
  const uint64_t nms[] = {O::NttPrime(Is32<Word>() ? 18 : 27, 1024),
                          O::NttPrime(Is32<Word>() ? 27 : 55, 1024)};
  for (uint64_t nm : nms) {
    const size_t n = 1024;
    const auto v64 = O::Random(n, om);
    auto got = O::As<Word>(v64);
    if (nm < om) {
      EltwiseCmpSubMod(got.data(), got.data(), n, nm, CMPINT::NLE, om >> 1,
                       (om - nm) % nm);
    } else {
      EltwiseCmpAdd(got.data(), got.data(), n, CMPINT::NLE, om >> 1, nm - om);
    }
    for (size_t i = 0; i < n; ++i) {
      // centred lift: values above om/2 represent negatives
      const bool neg = v64[i] > (om >> 1);
      uint64_t want;
      if (nm < om) {
        want = neg ? O::SubMod(v64[i] % nm, (om - nm) % nm, nm) : v64[i] % nm;
      } else {
        want = neg ? v64[i] + (nm - om) : v64[i];
      }
      CHECK_EQ(static_cast<uint64_t>(got[i]), want, << "i=" << i << " nm=" << nm);
    }
  }
}

}  // namespace

TEST(Eltwise_AddMod) { RunAddSub<uint64_t>(true); }
TEST(Eltwise_SubMod) { RunAddSub<uint64_t>(false); }
TEST(Eltwise_MultMod) { RunMultMod<uint64_t>(); }
TEST(Eltwise_FMAMod) { RunFMAMod<uint64_t>(); }
TEST(Eltwise_ReduceMod) { RunReduceMod<uint64_t>(); }
TEST(Eltwise_CmpAdd) { RunCmpAdd<uint64_t>(); }
TEST(Eltwise_CmpSubMod) { RunCmpSubMod<uint64_t>(); }
TEST(Eltwise_OpenFHESwitchModulusPattern) { RunSwitchModulusPattern<uint64_t>(); }

#ifdef HEXL_RVV_HAS_32BIT_API
TEST(Eltwise_AddMod_32) { RunAddSub<uint32_t>(true); }
TEST(Eltwise_SubMod_32) { RunAddSub<uint32_t>(false); }
TEST(Eltwise_MultMod_32) { RunMultMod<uint32_t>(); }
TEST(Eltwise_FMAMod_32) { RunFMAMod<uint32_t>(); }
TEST(Eltwise_ReduceMod_32) { RunReduceMod<uint32_t>(); }
TEST(Eltwise_CmpAdd_32) { RunCmpAdd<uint32_t>(); }
TEST(Eltwise_CmpSubMod_32) { RunCmpSubMod<uint32_t>(); }
TEST(Eltwise_OpenFHESwitchModulusPattern_32) { RunSwitchModulusPattern<uint32_t>(); }
#endif
