// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// hexl/ntt: the NTT class, exactly as OpenFHE uses it and a bit beyond.

#include <thread>
#include <unordered_map>
#include <utility>

#include "hexl/hexl.hpp"
#include "oracle.hpp"
#include "test.hpp"

using namespace intel::hexl;
namespace O = hexltest::oracle;
using Vec = std::vector<uint64_t>;

namespace {

std::vector<uint64_t> Degrees(uint64_t max_n) {
  std::vector<uint64_t> ns;
  for (uint64_t n = 2; n <= max_n; n <<= 1) ns.push_back(n);
  return ns;
}

/// Modulus sizes around every dispatch boundary: binfhe (27), last/first e32
/// (29/30 vs 31), 32-bit, IPCEI NTT bench (49), BFV (60), maximum (62).
const size_t kModBits[] = {20, 27, 29, 30, 31, 32, 40, 49, 55, 60, 61};

uint64_t MaxN() { return hexltest::g_quick ? 1024 : 32768; }
uint64_t MaxOracleN() { return hexltest::g_quick ? 64 : 512; }

}  // namespace

// Forward NTT against the O(N^2) definition, using the constructor OpenFHE
// uses (explicit root). Also pins down the output ORDER (bit-reversed).
TEST(Ntt_ForwardMatchesDefinition) {
  for (uint64_t n : Degrees(MaxOracleN())) {
    for (size_t bits : kModBits) {
      const uint64_t q = O::NttPrime(bits, n);
      const uint64_t w = O::PrimitiveRoot2N(n, q);
      NTT ntt(n, q, w);
      const Vec x = O::Random(n, q);
      Vec got(n);
      ntt.ComputeForward(got.data(), x.data(), 1, 1);
      CHECK_VEC_EQ(O::ForwardNtt(x, q, w), got, << "n=" << n << " q=" << q);
    }
  }
}

// Constructor without a root must use the MINIMAL primitive 2N-th root.
TEST(Ntt_DefaultRootIsMinimal) {
  for (uint64_t n : {8ULL, 1024ULL}) {
    for (size_t bits : {27, 49, 60}) {
      const uint64_t q = O::NttPrime(bits, n);
      NTT ntt(n, q);
      const uint64_t w = O::MinimalPrimitiveRoot2N(n, q);
      CHECK_EQ(ntt.GetMinimalRootOfUnity(), w);
      CHECK_EQ(ntt.GetDegree(), n);
      CHECK_EQ(ntt.GetModulus(), q);
      if (n <= MaxOracleN()) {
        const Vec x = O::Random(n, q);
        Vec got(n);
        ntt.ComputeForward(got.data(), x.data(), 1, 1);
        CHECK_VEC_EQ(O::ForwardNtt(x, q, w), got, << "n=" << n << " q=" << q);
      }
    }
  }
}

TEST(Ntt_RoundTrip) {
  for (uint64_t n : Degrees(MaxN())) {
    for (size_t bits : kModBits) {
      const uint64_t q = O::NttPrime(bits, n);
      NTT ntt(n, q, O::PrimitiveRoot2N(n, q));
      const Vec x = O::Random(n, q);
      Vec y(n), z(n);
      ntt.ComputeForward(y.data(), x.data(), 1, 1);
      ntt.ComputeInverse(z.data(), y.data(), 1, 1);
      CHECK_VEC_EQ(x, z, << "n=" << n << " q=" << q);
    }
  }
}

// OpenFHE calls both transforms in place: ComputeForward(data, data, 1, 1).
TEST(Ntt_InPlaceEqualsOutOfPlace) {
  for (uint64_t n : {16ULL, 1024ULL, 8192ULL}) {
    for (size_t bits : {27, 49, 60}) {
      const uint64_t q = O::NttPrime(bits, n);
      NTT ntt(n, q, O::PrimitiveRoot2N(n, q));
      const Vec x = O::Random(n, q);
      Vec out(n), inplace = x;
      ntt.ComputeForward(out.data(), x.data(), 1, 1);
      ntt.ComputeForward(inplace.data(), inplace.data(), 1, 1);
      CHECK_VEC_EQ(out, inplace, << "forward n=" << n << " q=" << q);
      Vec back(n);
      ntt.ComputeInverse(back.data(), out.data(), 1, 1);
      ntt.ComputeInverse(inplace.data(), inplace.data(), 1, 1);
      CHECK_VEC_EQ(back, inplace, << "inverse n=" << n << " q=" << q);
      CHECK_VEC_EQ(x, back);
    }
  }
}

// Out-of-place calls must leave the operand untouched.
TEST(Ntt_OperandUnmodified) {
  const uint64_t n = 1024, q = O::NttPrime(27, n);
  NTT ntt(n, q);
  const Vec x = O::Random(n, q);
  Vec copy = x, out(n);
  ntt.ComputeForward(out.data(), copy.data(), 1, 1);
  CHECK_VEC_EQ(x, copy);
  const Vec out_before = out;
  Vec back(n);
  ntt.ComputeInverse(back.data(), out.data(), 1, 1);
  CHECK_VEC_EQ(out_before, out);
}

// input_mod_factor / output_mod_factor semantics (lazy reduction).
TEST(Ntt_ModFactors) {
  for (uint64_t n : {4ULL, 64ULL, 2048ULL}) {
    for (size_t bits : {27, 29, 30, 49, 60}) {
      const uint64_t q = O::NttPrime(bits, n);
      NTT ntt(n, q, O::PrimitiveRoot2N(n, q));
      const Vec x = O::Random(n, q);
      Vec ref(n);
      ntt.ComputeForward(ref.data(), x.data(), 1, 1);

      for (uint64_t imf : {1ULL, 2ULL, 4ULL}) {
        // same values mod q, spread over [0, imf*q)
        Vec xin = x;
        auto k = O::Random(n, imf);
        for (size_t i = 0; i < n; ++i) xin[i] += k[i] * q;
        for (uint64_t omf : {1ULL, 4ULL}) {
          Vec got(n);
          ntt.ComputeForward(got.data(), xin.data(), imf, omf);
          for (size_t i = 0; i < n; ++i) {
            CHECK(got[i] < omf * q);
            CHECK_EQ(got[i] % q, ref[i], << "fwd i=" << i << " imf=" << imf
                     << " omf=" << omf << " q=" << q);
          }
        }
      }
      for (uint64_t imf : {1ULL, 2ULL}) {
        Vec yin = ref;
        auto k = O::Random(n, imf);
        for (size_t i = 0; i < n; ++i) yin[i] += k[i] * q;
        for (uint64_t omf : {1ULL, 2ULL}) {
          Vec got(n);
          ntt.ComputeInverse(got.data(), yin.data(), imf, omf);
          for (size_t i = 0; i < n; ++i) {
            CHECK(got[i] < omf * q);
            CHECK_EQ(got[i] % q, x[i], << "inv i=" << i << " imf=" << imf
                     << " omf=" << omf << " q=" << q);
          }
        }
      }
    }
  }
}

// Convolution theorem: NTT turns negacyclic polynomial products into
// EltwiseMultMod. This is literally what a BFV/binfhe multiplication does.
TEST(Ntt_NegacyclicConvolution) {
  for (uint64_t n : {8ULL, 256ULL}) {
    for (size_t bits : {27, 60}) {
      const uint64_t q = O::NttPrime(bits, n);
      NTT ntt(n, q);
      const Vec a = O::Random(n, q), b = O::Random(n, q);
      Vec want(n, 0);  // schoolbook a*b mod (X^n + 1)
      for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = 0; j < n; ++j) {
          const uint64_t p = O::MulMod(a[i], b[j], q);
          const uint64_t k = (i + j) % n;
          want[k] = (i + j < n) ? O::AddMod(want[k], p, q) : O::SubMod(want[k], p, q);
        }
      }
      Vec fa(n), fb(n), prod(n), got(n);
      ntt.ComputeForward(fa.data(), a.data(), 1, 1);
      ntt.ComputeForward(fb.data(), b.data(), 1, 1);
      EltwiseMultMod(prod.data(), fa.data(), fb.data(), n, q, 1);
      ntt.ComputeInverse(got.data(), prod.data(), 1, 1);
      CHECK_VEC_EQ(want, got, << "n=" << n << " q=" << q);
    }
  }
}

// GetRootOfUnityPowers()[ReverseBits(i, logN)] == w^i (public table contract).
TEST(Ntt_RootOfUnityTable) {
  const uint64_t n = 1024, q = O::NttPrime(49, n);
  const uint64_t w = O::PrimitiveRoot2N(n, q);
  NTT ntt(n, q, w);
  const auto& t = ntt.GetRootOfUnityPowers();
  CHECK_EQ(t.size(), n);
  uint64_t wi = 1;
  for (uint64_t i = 0; i < n; ++i) {
    CHECK_EQ(t[O::ReverseBits(i, 10)], wi, << "i=" << i);
    wi = O::MulMod(wi, w, q);
  }
  const auto& p64 = ntt.GetPrecon64RootOfUnityPowers();
  CHECK_EQ(p64.size(), n);
  for (uint64_t k = 0; k < n; ++k) {
    CHECK_EQ(p64[k], static_cast<uint64_t>(((O::u128)t[k] << 64) / q));
  }
}

// The exact OpenFHE pattern (transformnathexl-impl.h): a map of NTT objects
// keyed by (N, q), default-constructed by operator[] and move-assigned.
TEST(Ntt_OpenFHEMapPattern) {
  struct HashPair {
    size_t operator()(const std::pair<uint64_t, uint64_t>& p) const {
      return std::hash<uint64_t>()(p.first) ^ (std::hash<uint64_t>()(p.second) << 1);
    }
  };
  std::unordered_map<std::pair<uint64_t, uint64_t>, NTT, HashPair> cache;
  for (uint64_t n : {1024ULL, 2048ULL}) {
    for (size_t bits : {27, 60}) {
      const uint64_t q = O::NttPrime(bits, n);
      const uint64_t w = O::PrimitiveRoot2N(n, q);
      NTT ntt(n, q, w);
      cache[{n, q}] = std::move(ntt);
      NTT* p = &cache.find({n, q})->second;
      Vec data = O::Random(n, q), orig = data;
      p->ComputeForward(data.data(), data.data(), 1, 1);
      p->ComputeInverse(data.data(), data.data(), 1, 1);
      CHECK_VEC_EQ(orig, data, << "n=" << n << " q=" << q);
    }
  }
}

#if !defined(__riscv) || defined(__linux__)
// OpenFHE runs ComputeForward on the SAME NTT object from many OpenMP threads.
// Any mutable scratch inside the object shows up here as corrupted output.
TEST(Ntt_ConcurrentUse) {
  const uint64_t n = 4096;
  for (size_t bits : {27, 60}) {
    const uint64_t q = O::NttPrime(bits, n);
    NTT ntt(n, q);
    const unsigned kThreads = 8;
    std::vector<Vec> inputs, want(kThreads);
    for (unsigned t = 0; t < kThreads; ++t) {
      inputs.push_back(O::Random(n, q));
      want[t].resize(n);
      ntt.ComputeForward(want[t].data(), inputs[t].data(), 1, 1);
    }
    std::vector<Vec> got(kThreads, Vec(n));
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < kThreads; ++t) {
      pool.emplace_back([&, t] {
        for (int rep = 0; rep < 50; ++rep) {
          Vec tmp = inputs[t];
          ntt.ComputeForward(tmp.data(), tmp.data(), 1, 1);
          got[t] = tmp;
        }
      });
    }
    for (auto& th : pool) th.join();
    for (unsigned t = 0; t < kThreads; ++t) {
      CHECK_VEC_EQ(want[t], got[t], << "thread " << t << " q=" << q);
    }
  }
}
#endif
