// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Elementwise microbenchmarks, one per public function, at upstream HEXL's
// sizes (1024 / 4096 / 16384) and at both RVV dispatch classes (27-bit -> e32
// kernels, 60-bit -> e64 kernels). Bytes processed are reported so each
// result can be read against BM_MemcpyReference, the machine's own copy
// bandwidth at the same size: an eltwise kernel near that line is memory
// bound and no instruction selection will speed it up.

#include <cstring>

#include "bench-common.hpp"

using namespace intel::hexl;
using namespace hexlbench;

namespace {

void EltArgs(benchmark::internal::Benchmark* b) {
  for (int64_t bits : {27, 60}) {
    for (int64_t n : {1024, 4096, 16384}) b->Args({n, bits});
  }
  b->ArgNames({"n", "qbits"});
}

/// Streams `vectors` input/output arrays of n uint64_t per iteration.
void SetBytes(benchmark::State& state, uint64_t n, int vectors) {
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * n * 8 * vectors));
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
}

}  // namespace

// ---- reference: plain copy of the same amount of data ---------------------
static void BM_MemcpyReference(benchmark::State& state) {
  const uint64_t n = state.range(0);
  Vec a = Random(n, ~0ULL), r(n);
  for (auto _ : state) {
    std::memcpy(r.data(), a.data(), n * sizeof(uint64_t));
    benchmark::ClobberMemory();
  }
  SetBytes(state, n, 2);
}
BENCHMARK(BM_MemcpyReference)->Arg(1024)->Arg(4096)->Arg(16384)->ArgName("n");

// ---- add / sub --------------------------------------------------------------
static void BM_EltwiseAddModVV(benchmark::State& state) {
  const uint64_t n = state.range(0), q = Prime(state.range(1), 1);
  Vec a = Random(n, q), b = Random(n, q), r(n);
  HEXL_BENCH_PROBE(state, EltwiseAddMod(r.data(), a.data(), b.data(), n, q));
  for (auto _ : state) {
    EltwiseAddMod(r.data(), a.data(), b.data(), n, q);
    benchmark::ClobberMemory();
  }
  SetBytes(state, n, 3);
}
BENCHMARK(BM_EltwiseAddModVV)->Apply(EltArgs);

static void BM_EltwiseAddModVS(benchmark::State& state) {
  const uint64_t n = state.range(0), q = Prime(state.range(1), 1);
  Vec a = Random(n, q), r(n);
  const uint64_t s = q / 3;
  HEXL_BENCH_PROBE(state, EltwiseAddMod(r.data(), a.data(), s, n, q));
  for (auto _ : state) {
    EltwiseAddMod(r.data(), a.data(), s, n, q);
    benchmark::ClobberMemory();
  }
  SetBytes(state, n, 2);
}
BENCHMARK(BM_EltwiseAddModVS)->Apply(EltArgs);

static void BM_EltwiseSubModVV(benchmark::State& state) {
  const uint64_t n = state.range(0), q = Prime(state.range(1), 1);
  Vec a = Random(n, q), b = Random(n, q), r(n);
  HEXL_BENCH_PROBE(state, EltwiseSubMod(r.data(), a.data(), b.data(), n, q));
  for (auto _ : state) {
    EltwiseSubMod(r.data(), a.data(), b.data(), n, q);
    benchmark::ClobberMemory();
  }
  SetBytes(state, n, 3);
}
BENCHMARK(BM_EltwiseSubModVV)->Apply(EltArgs);

static void BM_EltwiseSubModVS(benchmark::State& state) {
  const uint64_t n = state.range(0), q = Prime(state.range(1), 1);
  Vec a = Random(n, q), r(n);
  const uint64_t s = q / 3;
  HEXL_BENCH_PROBE(state, EltwiseSubMod(r.data(), a.data(), s, n, q));
  for (auto _ : state) {
    EltwiseSubMod(r.data(), a.data(), s, n, q);
    benchmark::ClobberMemory();
  }
  SetBytes(state, n, 2);
}
BENCHMARK(BM_EltwiseSubModVS)->Apply(EltArgs);

// ---- multiply (the NTT-domain Hadamard product) -----------------------------
static void BM_EltwiseMultMod(benchmark::State& state) {
  const uint64_t n = state.range(0), q = Prime(state.range(1), 1);
  const uint64_t imf = state.range(2);
  Vec a = Random(n, imf * q), b = Random(n, imf * q), r(n);
  HEXL_BENCH_PROBE(state, EltwiseMultMod(r.data(), a.data(), b.data(), n, q, imf));
  for (auto _ : state) {
    EltwiseMultMod(r.data(), a.data(), b.data(), n, q, imf);
    benchmark::ClobberMemory();
  }
  SetBytes(state, n, 3);
}
BENCHMARK(BM_EltwiseMultMod)
    ->ArgsProduct({{1024, 4096, 16384}, {27, 50, 60}, {1, 4}})
    ->ArgNames({"n", "qbits", "imf"});

// ---- fused multiply-add (RNS basis switching, scalar ModMul) ----------------
static void BM_EltwiseFMAMod(benchmark::State& state) {
  const uint64_t n = state.range(0), q = Prime(state.range(1), 1);
  const bool add = state.range(2) != 0;
  Vec a = Random(n, q), c = Random(n, q), r(n);
  const uint64_t s = q / 3;
  const uint64_t* c_ptr = add ? c.data() : nullptr;
  HEXL_BENCH_PROBE(state, EltwiseFMAMod(r.data(), a.data(), s, c_ptr, n, q, 1));
  for (auto _ : state) {
    EltwiseFMAMod(r.data(), a.data(), s, c_ptr, n, q, 1);
    benchmark::ClobberMemory();
  }
  SetBytes(state, n, add ? 3 : 2);
}
BENCHMARK(BM_EltwiseFMAMod)
    ->ArgsProduct({{1024, 4096, 16384}, {27, 60}, {0, 1}})
    ->ArgNames({"n", "qbits", "add"});

// ---- reductions -------------------------------------------------------------
static void BM_EltwiseReduceMod(benchmark::State& state) {
  const uint64_t n = state.range(0), q = Prime(state.range(1), 1);
  const uint64_t imf = state.range(2) == 0 ? q : state.range(2);  // 0 = "modulus"
  Vec a = state.range(2) == 0 ? Random(n, ~0ULL) : Random(n, imf * q), r(n);
  HEXL_BENCH_PROBE(state, EltwiseReduceMod(r.data(), a.data(), n, q, imf, 1));
  for (auto _ : state) {
    EltwiseReduceMod(r.data(), a.data(), n, q, imf, 1);
    benchmark::ClobberMemory();
  }
  SetBytes(state, n, 2);
}
BENCHMARK(BM_EltwiseReduceMod)
    ->ArgsProduct({{1024, 4096, 16384}, {27, 60}, {0, 2, 4}})
    ->ArgNames({"n", "qbits", "imf"});

// ---- compare-and-adjust (OpenFHE SwitchModulus) ------------------------------
static void BM_EltwiseCmpAdd(benchmark::State& state) {
  const uint64_t n = state.range(0), q = Prime(state.range(1), 1);
  Vec a = Random(n, q), r(n);
  HEXL_BENCH_PROBE(state, EltwiseCmpAdd(r.data(), a.data(), n, CMPINT::NLE, q / 2, 12345));
  for (auto _ : state) {
    EltwiseCmpAdd(r.data(), a.data(), n, CMPINT::NLE, q / 2, 12345);
    benchmark::ClobberMemory();
  }
  SetBytes(state, n, 2);
}
BENCHMARK(BM_EltwiseCmpAdd)->Apply(EltArgs);

static void BM_EltwiseCmpSubMod(benchmark::State& state) {
  const uint64_t n = state.range(0), q = Prime(state.range(1), 1);
  Vec a = Random(n, 4 * q), r(n);
  HEXL_BENCH_PROBE(state, EltwiseCmpSubMod(r.data(), a.data(), n, q, CMPINT::NLE, q, q / 5));
  for (auto _ : state) {
    EltwiseCmpSubMod(r.data(), a.data(), n, q, CMPINT::NLE, q, q / 5);
    benchmark::ClobberMemory();
  }
  SetBytes(state, n, 2);
}
BENCHMARK(BM_EltwiseCmpSubMod)->Apply(EltArgs);
