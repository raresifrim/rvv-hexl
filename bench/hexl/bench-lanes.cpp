// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// LMUL sweep of the finished RVV kernels: each kernel at every lane type
// (vuint64m1_t..m8_t for 64-bit storage, vuint32m1_t..m8_t for 32-bit
// storage), so the per-kernel defaults in src/util/rvv-config.hpp can be
// chosen from measurements, per cluster:
//
//   bench-hexl --benchmark_filter=BM_Lanes
//
// Calls the internal RVV kernels directly (needs -Isrc): the public API only
// runs the default lanes. Correctness of every lane is in test-rvv-lanes.cpp.

#include <stdint.h>

#include <type_traits>

#include "bench-common.hpp"

#if defined(__has_include)
#if __has_include("util/rvv-config.hpp")
#include "eltwise/eltwise-add-mod-internal.hpp"
#include "eltwise/eltwise-cmp-add-internal.hpp"
#include "eltwise/eltwise-cmp-sub-mod-internal.hpp"
#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "eltwise/eltwise-sub-mod-internal.hpp"
#endif
#endif

#ifdef HEXL_HAS_RVV

namespace {

using intel::hexl::CMPINT;
constexpr size_t kN = 4096;

/// 60-bit modulus for 64-bit storage, 27-bit (binfhe-sized, RVV e32 path) for 32-bit.
template <typename Word>
uint64_t LaneModulus() {
  return std::is_same_v<Word, uint64_t> ? hexlbench::Prime(60, 1024) : hexlbench::Prime(27, 1024);
}

template <typename Word, class V>
void BM_Lanes_AddModVV(benchmark::State& state) {
  const uint64_t q = LaneModulus<Word>();
  auto a = hexlbench::RandomW<Word>(kN, q), b = hexlbench::RandomW<Word>(kN, q), r = a;
  for (auto _ : state) {
    intel::hexl::EltwiseAddModRVV<Word, V>(r.data(), a.data(), b.data(), kN, q);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * kN);
}

template <typename Word, class V>
void BM_Lanes_AddModVS(benchmark::State& state) {
  const uint64_t q = LaneModulus<Word>(), s = q / 3;
  auto a = hexlbench::RandomW<Word>(kN, q), r = a;
  for (auto _ : state) {
    intel::hexl::EltwiseAddModRVV<Word, V>(r.data(), a.data(), s, kN, q);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * kN);
}

template <typename Word, class V>
void BM_Lanes_SubModVV(benchmark::State& state) {
  const uint64_t q = LaneModulus<Word>();
  auto a = hexlbench::RandomW<Word>(kN, q), b = hexlbench::RandomW<Word>(kN, q), r = a;
  for (auto _ : state) {
    intel::hexl::EltwiseSubModRVV<Word, V>(r.data(), a.data(), b.data(), kN, q);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * kN);
}

template <typename Word, class V>
void BM_Lanes_SubModVS(benchmark::State& state) {
  const uint64_t q = LaneModulus<Word>(), s = q / 3;
  auto a = hexlbench::RandomW<Word>(kN, q), r = a;
  for (auto _ : state) {
    intel::hexl::EltwiseSubModRVV<Word, V>(r.data(), a.data(), s, kN, q);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * kN);
}

/// OpenFHE's SwitchModulus shape: values below an old modulus, NLE against
/// half of it (about half the lanes take the add).
template <typename Word, class V>
void BM_Lanes_CmpAddNLE(benchmark::State& state) {
  const uint64_t om = std::is_same_v<Word, uint64_t> ? hexlbench::Prime(49, 1024) : hexlbench::Prime(24, 1024);
  const uint64_t nm = std::is_same_v<Word, uint64_t> ? hexlbench::Prime(55, 1024) : hexlbench::Prime(27, 1024);
  auto a = hexlbench::RandomW<Word>(kN, om), r = a;
  for (auto _ : state) {
    intel::hexl::EltwiseCmpAddRVV<Word, V>(r.data(), a.data(), kN, CMPINT::NLE, om >> 1, nm - om);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * kN);
}

template <typename Word, class V>
void BM_Lanes_CmpSubModNLE(benchmark::State& state) {
  const uint64_t om = std::is_same_v<Word, uint64_t> ? hexlbench::Prime(49, 1024) : hexlbench::Prime(27, 1024);
  const uint64_t nm = std::is_same_v<Word, uint64_t> ? hexlbench::Prime(27, 1024) : hexlbench::Prime(18, 1024);
  auto a = hexlbench::RandomW<Word>(kN, om), r = a;
  for (auto _ : state) {
    intel::hexl::EltwiseCmpSubModRVV<Word, V>(r.data(), a.data(), kN, nm, CMPINT::NLE, om >> 1, om % nm);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * kN);
}

/// MultMod, input_mod_factor 1. Word = uint64_t: EltwiseMultModRVV64 at a
/// 60-bit q; Word = uint32_t: EltwiseMultModRVV32 at a 27-bit q.
template <typename Word, class V>
void BM_Lanes_MultMod(benchmark::State& state) {
  const uint64_t q = LaneModulus<Word>();
  auto a = hexlbench::RandomW<Word>(kN, q), b = hexlbench::RandomW<Word>(kN, q), r = a;
  for (auto _ : state) {
    if constexpr (std::is_same_v<Word, uint64_t>) {
      intel::hexl::EltwiseMultModRVV64<1, V>(r.data(), a.data(), b.data(), kN, q);
    } else {
      intel::hexl::EltwiseMultModRVV32<Word, 1, V>(r.data(), a.data(), b.data(), kN, q);
    }
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * kN);
}

/// EltwiseMultModRVV32 on uint64_t storage (27-bit q, narrowing load and
/// widening store): the path EltwiseMultMod(uint64_t*) takes for q < 2^30.
template <class V>
void BM_Lanes_MultMod32FromU64(benchmark::State& state) {
  const uint64_t q = hexlbench::Prime(27, 1024);
  auto a = hexlbench::RandomW<uint64_t>(kN, q), b = hexlbench::RandomW<uint64_t>(kN, q), r = a;
  for (auto _ : state) {
    intel::hexl::EltwiseMultModRVV32<uint64_t, 1, V>(r.data(), a.data(), b.data(), kN, q);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * kN);
}

}  // namespace

#define HEXL_LANES(BM)                                  \
  BENCHMARK_TEMPLATE(BM, uint64_t, vuint64m1_t);        \
  BENCHMARK_TEMPLATE(BM, uint64_t, vuint64m2_t);        \
  BENCHMARK_TEMPLATE(BM, uint64_t, vuint64m4_t);        \
  BENCHMARK_TEMPLATE(BM, uint64_t, vuint64m8_t);        \
  BENCHMARK_TEMPLATE(BM, uint32_t, vuint32m1_t);        \
  BENCHMARK_TEMPLATE(BM, uint32_t, vuint32m2_t);        \
  BENCHMARK_TEMPLATE(BM, uint32_t, vuint32m4_t);        \
  BENCHMARK_TEMPLATE(BM, uint32_t, vuint32m8_t)

HEXL_LANES(BM_Lanes_AddModVV);
HEXL_LANES(BM_Lanes_AddModVS);
HEXL_LANES(BM_Lanes_SubModVV);
HEXL_LANES(BM_Lanes_SubModVS);
HEXL_LANES(BM_Lanes_CmpAddNLE);
HEXL_LANES(BM_Lanes_CmpSubModNLE);
HEXL_LANES(BM_Lanes_MultMod);
BENCHMARK_TEMPLATE(BM_Lanes_MultMod32FromU64, vuint32m1_t);
BENCHMARK_TEMPLATE(BM_Lanes_MultMod32FromU64, vuint32m2_t);
BENCHMARK_TEMPLATE(BM_Lanes_MultMod32FromU64, vuint32m4_t);

#endif  // HEXL_HAS_RVV
