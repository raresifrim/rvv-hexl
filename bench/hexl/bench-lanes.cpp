// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// SPDX-License-Identifier: Apache-2.0
//
// LMUL sweep of the finished RVV kernels: each kernel at every lane type
// (vuint64m1_t..m8_t for 64-bit storage, vuint32m1_t..m8_t for 32-bit
// storage), so the per-kernel defaults in src/util/rvv-config.hpp can be
// chosen from measurements, per cluster:
//
//   bench-hexl --benchmark_filter=BM_Lanes
//
// The NTT kernels also at every small-span strategy (the third template
// argument, rvv-config.hpp NttSmallSpan*), so the default can be picked per
// platform: bench-hexl --benchmark_filter=BM_Lanes_Ntt
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
#include "eltwise/eltwise-mult-add-mod-internal.hpp"
#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "eltwise/eltwise-sub-mod-internal.hpp"
#include "hexl/ntt/ntt.hpp"
#include "ntt/ntt-internal.hpp"
#include "ntt/ntt-rvv-small-span.hpp"
#include "util/rvv-util.hpp"
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

/// EltwiseMultAddMod (rvv-hexl extension), accumulating in place: 64-bit
/// lanes at a 60-bit q (RVV64), 32-bit lanes at a 27-bit q (RVV32).
template <typename Word, class V>
void BM_Lanes_MultAddMod(benchmark::State& state) {
	const uint64_t q = LaneModulus<Word>();
	auto a = hexlbench::RandomW<Word>(kN, q), b = hexlbench::RandomW<Word>(kN, q), acc = hexlbench::RandomW<Word>(kN, q);
	for (auto _ : state) {
		if constexpr (std::is_same_v<Word, uint64_t> && intel::hexl::rvv::IsE64<V>) {
			intel::hexl::EltwiseMultAddModRVV64<1, V>(acc.data(), a.data(), b.data(), acc.data(), kN, q);
		} else {
			intel::hexl::EltwiseMultAddModRVV32<Word, 1, V>(acc.data(), a.data(), b.data(), acc.data(), kN, q);
		}
		benchmark::ClobberMemory();
	}
	state.SetItemsProcessed(state.iterations() * kN);
}

/// EltwiseMultAddModRVV32 on uint64_t storage (27-bit q, narrowing loads and
/// widening store), accumulating in place: the lane type cfg::MultAddMod32U64 picks.
template <class V>
void BM_Lanes_MultAddMod32FromU64(benchmark::State& state) {
	const uint64_t q = hexlbench::Prime(27, 1024);
	auto a = hexlbench::RandomW<uint64_t>(kN, q), b = hexlbench::RandomW<uint64_t>(kN, q), acc = hexlbench::RandomW<uint64_t>(kN, q);
	for (auto _ : state) {
		intel::hexl::EltwiseMultAddModRVV32<uint64_t, 1, V>(acc.data(), a.data(), b.data(), acc.data(), kN, q);
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

/// An NTT of size state.range(0) with the modulus its kernel serves (27-bit
/// for e32 lanes: RVV32 on either storage word; 60-bit for e64: RVV64), and
/// strategy S's small-span tables built for the lanes S runs at under V.
template <typename Word, class V, class S>
struct NttLane {
	static constexpr bool kE32 = intel::hexl::rvv::IsE32<V>;
	using E = std::conditional_t<kE32, uint32_t, uint64_t>;
	uint64_t n, q;
	intel::hexl::NTT ntt;
	intel::hexl::NttSmallSpanTables<S, E> tables;
	intel::hexl::AlignedVector64<Word> x;

	explicit NttLane(benchmark::State& state)
		: n(state.range(0)), q(hexlbench::Prime(kE32 ? 27 : 60, n)), ntt(n, q),
		tables(intel::hexl::BuildNttSmallSpanTables<S, E>(ntt, intel::hexl::SmallSpanLanes<S, V>())),
		x(hexlbench::RandomW<Word>(n, q)) {
			if (!std::is_same_v<S, intel::hexl::rvv::ntt::PerBlock> && tables.vl == 0)
				state.SetLabel("N too small for the strategy: per block");
		}

	/// In place, fully reduced (1, 1), so the output is the next input.
	void Forward() {
		if constexpr (kE32) {
			intel::hexl::ForwardTransformToBitReverseRVV32<Word, V, S>(
					x.data(), x.data(), n, q, ntt.GetRVV32RootOfUnityPowers().data(),
					ntt.GetRVV32PreconRootOfUnityPowers().data(), 1, 1, &tables);
		} else {
			intel::hexl::ForwardTransformToBitReverseRVV64<V, S>(
					x.data(), x.data(), n, q, ntt.GetRootOfUnityPowers().data(),
					ntt.GetPrecon64RootOfUnityPowers().data(), 1, 1, &tables);
		}
	}
	void Inverse() {
		if constexpr (kE32) {
			intel::hexl::InverseTransformFromBitReverseRVV32<Word, V, S>(
					x.data(), x.data(), n, q, ntt.GetRVV32InvRootOfUnityPowers().data(),
					ntt.GetRVV32PreconInvRootOfUnityPowers().data(), 1, 1, &tables);
		} else {
			intel::hexl::InverseTransformFromBitReverseRVV64<V, S>(
					x.data(), x.data(), n, q, ntt.GetInvRootOfUnityPowers().data(),
					ntt.GetPrecon64InvRootOfUnityPowers().data(), 1, 1, &tables);
		}
	}
};

template <typename Word, class V, class S>
void BM_Lanes_NttForward(benchmark::State& state) {
	NttLane<Word, V, S> k(state);
	for (auto _ : state) {
		k.Forward();
		benchmark::ClobberMemory();
	}
	state.SetItemsProcessed(state.iterations() * k.n);
}

template <typename Word, class V, class S>
void BM_Lanes_NttInverse(benchmark::State& state) {
	NttLane<Word, V, S> k(state);
	for (auto _ : state) {
		k.Inverse();
		benchmark::ClobberMemory();
	}
	state.SetItemsProcessed(state.iterations() * k.n);
}

// short names for the benchmark labels
using PerBlock = intel::hexl::rvv::ntt::PerBlock;
using FusedSlides = intel::hexl::rvv::ntt::FusedSlides;
#define HEXL_TRANSPOSED_NAME(S) using intel::hexl::rvv::ntt::S;
HEXL_RVV_NTT_TRANSPOSED_SHAPES(HEXL_TRANSPOSED_NAME);
#undef HEXL_TRANSPOSED_NAME

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
HEXL_LANES(BM_Lanes_MultAddMod);
BENCHMARK_TEMPLATE(BM_Lanes_MultMod32FromU64, vuint32m1_t);
BENCHMARK_TEMPLATE(BM_Lanes_MultMod32FromU64, vuint32m2_t);
BENCHMARK_TEMPLATE(BM_Lanes_MultMod32FromU64, vuint32m4_t);
BENCHMARK_TEMPLATE(BM_Lanes_MultAddMod32FromU64, vuint32m1_t);
BENCHMARK_TEMPLATE(BM_Lanes_MultAddMod32FromU64, vuint32m2_t);
BENCHMARK_TEMPLATE(BM_Lanes_MultAddMod32FromU64, vuint32m4_t);

// NTT: N = 1024 (binfhe) and 4096, every Transposed shape (its tile rows have
// their own LMUL, whatever V); N too small for a strategy falls back to per
// block, as the label says.
#define HEXL_NTT_LANES(BM, S)                                                   \
	BENCHMARK_TEMPLATE(BM, uint64_t, vuint64m1_t, S)->Arg(1024)->Arg(4096); \
BENCHMARK_TEMPLATE(BM, uint64_t, vuint64m2_t, S)->Arg(1024)->Arg(4096); \
BENCHMARK_TEMPLATE(BM, uint64_t, vuint64m4_t, S)->Arg(1024)->Arg(4096); \
BENCHMARK_TEMPLATE(BM, uint64_t, vuint64m8_t, S)->Arg(1024)->Arg(4096); \
BENCHMARK_TEMPLATE(BM, uint32_t, vuint32m1_t, S)->Arg(1024)->Arg(4096); \
BENCHMARK_TEMPLATE(BM, uint32_t, vuint32m2_t, S)->Arg(1024)->Arg(4096); \
BENCHMARK_TEMPLATE(BM, uint32_t, vuint32m4_t, S)->Arg(1024)->Arg(4096); \
BENCHMARK_TEMPLATE(BM, uint32_t, vuint32m8_t, S)->Arg(1024)->Arg(4096); \
BENCHMARK_TEMPLATE(BM, uint64_t, vuint32m1_t, S)->Arg(1024)->Arg(4096); \
BENCHMARK_TEMPLATE(BM, uint64_t, vuint32m2_t, S)->Arg(1024)->Arg(4096); \
BENCHMARK_TEMPLATE(BM, uint64_t, vuint32m4_t, S)->Arg(1024)->Arg(4096)

HEXL_NTT_LANES(BM_Lanes_NttForward, PerBlock);
HEXL_NTT_LANES(BM_Lanes_NttForward, FusedSlides);
#define HEXL_TRANSPOSED_FWD(S) HEXL_NTT_LANES(BM_Lanes_NttForward, S);
HEXL_RVV_NTT_TRANSPOSED_SHAPES(HEXL_TRANSPOSED_FWD);
#undef HEXL_TRANSPOSED_FWD
HEXL_NTT_LANES(BM_Lanes_NttInverse, PerBlock);
HEXL_NTT_LANES(BM_Lanes_NttInverse, FusedSlides);
#define HEXL_TRANSPOSED_INV(S) HEXL_NTT_LANES(BM_Lanes_NttInverse, S);
HEXL_RVV_NTT_TRANSPOSED_SHAPES(HEXL_TRANSPOSED_INV);
#undef HEXL_TRANSPOSED_INV

#endif  // HEXL_HAS_RVV
