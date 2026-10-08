// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// NTT microbenchmarks: in-place, fully reduced (1, 1) mod factors, i.e.
// exactly the call OpenFHE's HEXL backend makes.

#include "bench-common.hpp"

using namespace intel::hexl;
using namespace hexlbench;

namespace {

void NttArgs(benchmark::internal::Benchmark* b) {
	for (int64_t bits : {27, 45, 49, 60}) {
		for (int64_t n : {1024, 2048, 4096, 8192, 16384, 32768}) {
			b->Args({n, bits});
		}
	}
	b->ArgNames({"N", "qbits"})->Unit(benchmark::kMicrosecond);
}

void SetCounters(benchmark::State& state, uint64_t n) {
	state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
	state.counters["time_per_coeff"] = benchmark::Counter(
			static_cast<double>(state.iterations() * n),
			benchmark::Counter::kIsRate | benchmark::Counter::kInvert,
			benchmark::Counter::kIs1000);
}

}  // namespace

static void BM_NTTForward(benchmark::State& state) {
	const uint64_t n = state.range(0);
	const uint64_t q = Prime(state.range(1), n);
	Vec x = Random(n, q);
	NTT ntt;
	HEXL_BENCH_PROBE(state, (ntt = NTT(n, q), ntt.ComputeForward(x.data(), x.data(), 1, 1)));
	for (auto _ : state) {
		ntt.ComputeForward(x.data(), x.data(), 1, 1);
		benchmark::ClobberMemory();
	}
	SetCounters(state, n);
}
BENCHMARK(BM_NTTForward)->Apply(NttArgs);

static void BM_NTTInverse(benchmark::State& state) {
	const uint64_t n = state.range(0);
	const uint64_t q = Prime(state.range(1), n);
	Vec x = Random(n, q);
	NTT ntt;
	HEXL_BENCH_PROBE(state, (ntt = NTT(n, q), ntt.ComputeInverse(x.data(), x.data(), 1, 1)));
	for (auto _ : state) {
		ntt.ComputeInverse(x.data(), x.data(), 1, 1);
		benchmark::ClobberMemory();
	}
	SetCounters(state, n);
}
BENCHMARK(BM_NTTInverse)->Apply(NttArgs);

// Lazy output (output_mod_factor = 4): what a caller chaining NTTs with
// further lazy arithmetic would use. Shows the cost of the final reduction.
static void BM_NTTForwardLazy(benchmark::State& state) {
	const uint64_t n = state.range(0);
	const uint64_t q = Prime(state.range(1), n);
	Vec x = Random(n, q), y(n);
	NTT ntt;
	HEXL_BENCH_PROBE(state, (ntt = NTT(n, q), ntt.ComputeForward(y.data(), x.data(), 1, 4)));
	for (auto _ : state) {
		ntt.ComputeForward(y.data(), x.data(), 1, 4);
		benchmark::ClobberMemory();
	}
	SetCounters(state, n);
}
	BENCHMARK(BM_NTTForwardLazy)
->Args({1024, 27})->Args({16384, 45})->Args({16384, 60})
	->ArgNames({"N", "qbits"})->Unit(benchmark::kMicrosecond);

#ifdef HEXL_RVV_HAS_32BIT_API
	// ---- rvv-hexl 32-bit storage (OpenFHE NATIVE_SIZE=32) ----------------------
	// Same transforms on uint32_t data. Compare BM_NTTForward32/N:x/qbits:27 with
	// BM_NTTForward/N:x/qbits:27: identical arithmetic (both run the RVV e32
	// kernel), so the difference is the cost of 64-bit storage (narrow/widen and
	// twice the memory traffic).
	namespace {
	void NttArgs32(benchmark::internal::Benchmark* b) {
		for (int64_t n : {1024, 2048, 4096, 8192, 16384, 32768}) b->Args({n, 27});
		b->ArgNames({"N", "qbits"})->Unit(benchmark::kMicrosecond);
	}
	}  // namespace

static void BM_NTTForward32(benchmark::State& state) {
	const uint64_t n = state.range(0);
	const uint64_t q = Prime(state.range(1), n);
	auto x = RandomW<uint32_t>(n, q);
	NTT ntt;
	HEXL_BENCH_PROBE(state, (ntt = NTT(n, q), ntt.ComputeForward(x.data(), x.data(), 1, 1)));
	for (auto _ : state) {
		ntt.ComputeForward(x.data(), x.data(), 1, 1);
		benchmark::ClobberMemory();
	}
	SetCounters(state, n);
}
BENCHMARK(BM_NTTForward32)->Apply(NttArgs32);

static void BM_NTTInverse32(benchmark::State& state) {
	const uint64_t n = state.range(0);
	const uint64_t q = Prime(state.range(1), n);
	auto x = RandomW<uint32_t>(n, q);
	NTT ntt;
	HEXL_BENCH_PROBE(state, (ntt = NTT(n, q), ntt.ComputeInverse(x.data(), x.data(), 1, 1)));
	for (auto _ : state) {
		ntt.ComputeInverse(x.data(), x.data(), 1, 1);
		benchmark::ClobberMemory();
	}
	SetCounters(state, n);
}
BENCHMARK(BM_NTTInverse32)->Apply(NttArgs32);
#endif
