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
// The FP64 paths at every lane type against the e64 integer kernels at their
// default lanes, at the same 49-bit prime: --benchmark_filter=F64 (NTT:
// BM_Lanes_NttF64*, eltwise: BM_Lanes_{MultMod,FMAMod,MultAddMod}F64).
//
// Calls the internal RVV kernels directly (needs -Isrc): the public API only
// runs the default lanes. Correctness of every lane is in test-rvv-lanes.cpp.

#include <stdint.h>

#include <memory>
#include <type_traits>
#include <vector>

#include "bench-common.hpp"

#if defined(__has_include)
#if __has_include("util/rvv-config.hpp")
#include "eltwise/eltwise-add-mod-internal.hpp"
#include "eltwise/eltwise-base-convert-internal.hpp"
#include "eltwise/eltwise-cmp-add-internal.hpp"
#include "eltwise/eltwise-cmp-sub-mod-internal.hpp"
#include "eltwise/eltwise-fma-mod-internal.hpp"
#include "eltwise/eltwise-mult-add-mod-internal.hpp"
#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "eltwise/eltwise-signed-digit-decompose-internal.hpp"
#include "eltwise/eltwise-sub-mod-internal.hpp"
#include "hexl/ntt/ntt.hpp"
#include "ntt/ntt-internal.hpp"
#include "ntt/ntt-rvv-fp64.hpp"
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

/// SignedDigitDecompose (rvv-hexl extension), one polynomial: 32-bit lanes at
/// the STD128 shape (27-bit q, g = 7, 3 digits kept), 64-bit lanes at a
/// 60-bit q (g = 20, 2 digits).
template <typename Word, class V>
void BM_Lanes_SignedDigitDecompose(benchmark::State& state) {
	constexpr bool kE64 = std::is_same_v<Word, uint64_t> && intel::hexl::rvv::IsE64<V>;
	const uint64_t q = LaneModulus<Word>(), g = kE64 ? 20 : 7, digits = kE64 ? 2 : 3;
	auto a = hexlbench::RandomW<Word>(kN, q);
	std::vector<intel::hexl::AlignedVector64<Word>> out(digits, intel::hexl::AlignedVector64<Word>(kN));
	std::vector<Word*> p(digits);
	for (uint64_t d = 0; d < digits; ++d) p[d] = out[d].data();
	for (auto _ : state) {
		if constexpr (kE64) {
			intel::hexl::SignedDigitDecomposeRVV64<V>(p.data(), a.data(), kN, q, g, digits);
		} else {
			intel::hexl::SignedDigitDecomposeRVV32<Word, V>(p.data(), a.data(), kN, q, g, digits);
		}
		benchmark::ClobberMemory();
	}
	state.SetItemsProcessed(state.iterations() * kN);
}

/// SignedDigitDecomposeRVV32 on uint64_t storage (27-bit q, narrowing load and
/// widening stores): the lane type cfg::SignedDigitDecompose32U64 picks.
template <class V>
void BM_Lanes_SignedDigitDecompose32FromU64(benchmark::State& state) {
	const uint64_t q = hexlbench::Prime(27, 1024);
	auto a = hexlbench::RandomW<uint64_t>(kN, q);
	std::vector<intel::hexl::AlignedVector64<uint64_t>> out(3, intel::hexl::AlignedVector64<uint64_t>(kN));
	std::vector<uint64_t*> p = {out[0].data(), out[1].data(), out[2].data()};
	for (auto _ : state) {
		intel::hexl::SignedDigitDecomposeRVV32<uint64_t, V>(p.data(), a.data(), kN, q, 7, 3);
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

/// The FP64 NTT path (ntt-rvv-fp64.hpp) at lane type VF, or with VF = void
/// the e64 integer kernel it replaces (rvv::cfg::Ntt64, NttSmallSpan64), both
/// at a 49-bit prime (inside [2^30, 2^50)), in place and fully reduced.
template <class VF>
struct NttF64Lane {
	using S64 = intel::hexl::rvv::cfg::NttSmallSpan64;
	uint64_t n, q;
	intel::hexl::NTT ntt;
	std::shared_ptr<const intel::hexl::NttFp64Tables> fp;
	intel::hexl::NttFp64IntSmallSpan tables;
	intel::hexl::AlignedVector64<uint64_t> x;

	explicit NttF64Lane(benchmark::State& state)
		: n(state.range(0)), q(hexlbench::Prime(49, n)), ntt(n, q), fp(intel::hexl::BuildNttFp64Tables(ntt)),
		tables(intel::hexl::BuildNttSmallSpanTables<S64, uint64_t>(
					ntt, intel::hexl::SmallSpanLanes<S64, intel::hexl::rvv::cfg::Ntt64>())),
		x(hexlbench::RandomW<uint64_t>(n, q)) {}

	void Forward() {
		if constexpr (std::is_void_v<VF>) {
			intel::hexl::ForwardTransformToBitReverseRVV64(x.data(), x.data(), n, q, ntt.GetRootOfUnityPowers().data(),
					ntt.GetPrecon64RootOfUnityPowers().data(), 1, 1, &tables);
		} else {
			intel::hexl::ForwardTransformToBitReverseRVVF64<VF>(x.data(), x.data(), n, q, *fp,
					ntt.GetRootOfUnityPowers().data(), ntt.GetPrecon64RootOfUnityPowers().data(), 1, 1, &tables);
		}
	}
	void Inverse() {
		if constexpr (std::is_void_v<VF>) {
			intel::hexl::InverseTransformFromBitReverseRVV64(x.data(), x.data(), n, q,
					ntt.GetInvRootOfUnityPowers().data(), ntt.GetPrecon64InvRootOfUnityPowers().data(), 1, 1, &tables);
		} else {
			intel::hexl::InverseTransformFromBitReverseRVVF64<VF>(x.data(), x.data(), n, q, *fp,
					ntt.GetInvRootOfUnityPowers().data(), ntt.GetPrecon64InvRootOfUnityPowers().data(), 1, 1, &tables);
		}
	}
};

template <class VF>
void BM_Lanes_NttF64Forward(benchmark::State& state) {
	NttF64Lane<VF> k(state);
	for (auto _ : state) {
		k.Forward();
		benchmark::ClobberMemory();
	}
	state.SetItemsProcessed(state.iterations() * k.n);
}

template <class VF>
void BM_Lanes_NttF64Inverse(benchmark::State& state) {
	NttF64Lane<VF> k(state);
	for (auto _ : state) {
		k.Inverse();
		benchmark::ClobberMemory();
	}
	state.SetItemsProcessed(state.iterations() * k.n);
}

/// The FP64 eltwise kernels at lane type VF, or with VF = void the e64 integer
/// kernels they replace (rvv::cfg::MultMod64 / FMAMod64 / MultAddMod64), at a
/// 49-bit prime (inside [2^30, 2^50)), input_mod_factor 1, n = kN.
constexpr size_t kF64Bits = 49;

template <class VF>
void BM_Lanes_MultModF64(benchmark::State& state) {
	const uint64_t q = hexlbench::Prime(kF64Bits, 1);
	auto a = hexlbench::RandomW<uint64_t>(kN, q), b = hexlbench::RandomW<uint64_t>(kN, q), r = a;
	for (auto _ : state) {
		if constexpr (std::is_void_v<VF>) {
			intel::hexl::EltwiseMultModRVV64<1>(r.data(), a.data(), b.data(), kN, q);
		} else {
			intel::hexl::EltwiseMultModRVVF64<1, VF>(r.data(), a.data(), b.data(), kN, q);
		}
		benchmark::ClobberMemory();
	}
	state.SetItemsProcessed(state.iterations() * kN);
}

/// EltwiseFMAMod with the addend (arg3 != nullptr), out of place.
template <class VF>
void BM_Lanes_FMAModF64(benchmark::State& state) {
	const uint64_t q = hexlbench::Prime(kF64Bits, 1);
	auto a = hexlbench::RandomW<uint64_t>(kN, q), c = hexlbench::RandomW<uint64_t>(kN, q), r = a;
	const uint64_t s = hexlbench::RandomW<uint64_t>(1, q)[0];
	for (auto _ : state) {
		if constexpr (std::is_void_v<VF>) {
			intel::hexl::EltwiseFMAModRVV64<1>(r.data(), a.data(), s, c.data(), kN, q);
		} else {
			intel::hexl::EltwiseFMAModRVVF64<1, VF>(r.data(), a.data(), s, c.data(), kN, q);
		}
		benchmark::ClobberMemory();
	}
	state.SetItemsProcessed(state.iterations() * kN);
}

/// EltwiseMultAddMod accumulating in place (result == arg3), as binfhe and
/// the key switching call it.
template <class VF>
void BM_Lanes_MultAddModF64(benchmark::State& state) {
	const uint64_t q = hexlbench::Prime(kF64Bits, 1);
	auto a = hexlbench::RandomW<uint64_t>(kN, q), b = hexlbench::RandomW<uint64_t>(kN, q),
	     acc = hexlbench::RandomW<uint64_t>(kN, q);
	for (auto _ : state) {
		if constexpr (std::is_void_v<VF>) {
			intel::hexl::EltwiseMultAddModRVV64<1>(acc.data(), a.data(), b.data(), acc.data(), kN, q);
		} else {
			intel::hexl::EltwiseMultAddModRVVF64<1, VF>(acc.data(), a.data(), b.data(), acc.data(), kN, q);
		}
		benchmark::ClobberMemory();
	}
	state.SetItemsProcessed(state.iterations() * kN);
}

/// BaseConvert's RVV kernel at lane type V (or, with V = void, the native
/// kernel), at blindloop ckks_openfhe's ModUp shape: N = 8192, a 36-bit digit
/// to 4 limbs (46, 37, 37, 59 bits).
template <class V>
void BM_Lanes_BaseConvert(benchmark::State& state) {
	namespace O = hexltest::oracle;
	const uint64_t n = 8192;
	const std::vector<uint64_t> q{hexlbench::Prime(36, n)},
	      p{hexlbench::Prime(46, n), hexlbench::Prime(37, n), O::NttPrime(37, n, 1), hexlbench::Prime(59, n)};
	std::vector<uint64_t> q_hat_inv{1}, q_hat_mod_p(p.size(), 1);  // L = 1: Q_0 = 1
	const auto args = intel::hexl::MakeBaseConvertArgs(q.data(), 1, p.data(), p.size(), q_hat_inv.data(),
			q_hat_mod_p.data());
	auto x = hexlbench::RandomW<uint64_t>(n, q[0]);
	std::vector<intel::hexl::AlignedVector64<uint64_t>> y(p.size(), intel::hexl::AlignedVector64<uint64_t>(n));
	const uint64_t* in[1] = {x.data()};
	std::vector<uint64_t*> out;
	for (auto& v : y) out.push_back(v.data());
	for (auto _ : state) {
		if constexpr (std::is_void_v<V>) {
			intel::hexl::BaseConvertNative(out.data(), in, n, args);
		} else {
			intel::hexl::BaseConvertRVV64<V>(out.data(), in, n, args);
		}
		benchmark::ClobberMemory();
	}
	state.SetItemsProcessed(state.iterations() * n);
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
HEXL_LANES(BM_Lanes_SignedDigitDecompose);
BENCHMARK_TEMPLATE(BM_Lanes_SignedDigitDecompose32FromU64, vuint32m1_t);
BENCHMARK_TEMPLATE(BM_Lanes_SignedDigitDecompose32FromU64, vuint32m2_t);
BENCHMARK_TEMPLATE(BM_Lanes_SignedDigitDecompose32FromU64, vuint32m4_t);

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

// FP64 path: void = the e64 integer baseline
#define HEXL_NTT_F64_LANES(BM)                                       \
	BENCHMARK_TEMPLATE(BM, void)->Arg(1024)->Arg(4096)->Arg(16384);         \
	BENCHMARK_TEMPLATE(BM, vfloat64m1_t)->Arg(1024)->Arg(4096)->Arg(16384); \
	BENCHMARK_TEMPLATE(BM, vfloat64m2_t)->Arg(1024)->Arg(4096)->Arg(16384); \
	BENCHMARK_TEMPLATE(BM, vfloat64m4_t)->Arg(1024)->Arg(4096)->Arg(16384); \
	BENCHMARK_TEMPLATE(BM, vfloat64m8_t)->Arg(1024)->Arg(4096)->Arg(16384)
HEXL_NTT_F64_LANES(BM_Lanes_NttF64Forward);
HEXL_NTT_F64_LANES(BM_Lanes_NttF64Inverse);
#undef HEXL_NTT_F64_LANES
// BaseConvert: void = the native kernel
BENCHMARK_TEMPLATE(BM_Lanes_BaseConvert, void);
BENCHMARK_TEMPLATE(BM_Lanes_BaseConvert, vuint64m1_t);
BENCHMARK_TEMPLATE(BM_Lanes_BaseConvert, vuint64m2_t);
BENCHMARK_TEMPLATE(BM_Lanes_BaseConvert, vuint64m4_t);
BENCHMARK_TEMPLATE(BM_Lanes_BaseConvert, vuint64m8_t);

#define HEXL_ELTWISE_F64_LANES(BM)       \
	BENCHMARK_TEMPLATE(BM, void);         \
	BENCHMARK_TEMPLATE(BM, vfloat64m1_t); \
	BENCHMARK_TEMPLATE(BM, vfloat64m2_t); \
	BENCHMARK_TEMPLATE(BM, vfloat64m4_t); \
	BENCHMARK_TEMPLATE(BM, vfloat64m8_t)
HEXL_ELTWISE_F64_LANES(BM_Lanes_MultModF64);
HEXL_ELTWISE_F64_LANES(BM_Lanes_FMAModF64);
HEXL_ELTWISE_F64_LANES(BM_Lanes_MultAddModF64);
#undef HEXL_ELTWISE_F64_LANES

#endif  // HEXL_HAS_RVV
