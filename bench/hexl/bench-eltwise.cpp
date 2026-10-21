// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// SPDX-License-Identifier: Apache-2.0
//
// Elementwise microbenchmarks, one per public function, at upstream HEXL's
// sizes (1024 / 4096 / 16384) and at both RVV dispatch classes (27-bit -> e32
// kernels, 60-bit -> e64 kernels). Bytes processed are reported so each
// result can be read against BM_MemcpyReference, the machine's own copy
// bandwidth at the same size: an eltwise kernel near that line is memory
// bound and no instruction selection will speed it up.

#include <algorithm>
#include <cstring>
#include <type_traits>
#include <vector>

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

#ifdef HEXL_RVV_HAS_MULT_ADD_MOD
	// ---- vector multiply-add (rvv-hexl extension; binfhe's accumulation) --------
	// acc = (a * b + acc) mod q in one call, against the two calls OpenFHE makes
	// without it (EltwiseMultMod into a temporary, then EltwiseAddMod).
	static void BM_EltwiseMultAddMod(benchmark::State& state) {
		const uint64_t n = state.range(0), q = Prime(state.range(1), 1);
		Vec a = Random(n, q), b = Random(n, q), acc = Random(n, q);
		HEXL_BENCH_PROBE(state, EltwiseMultAddMod(acc.data(), a.data(), b.data(), acc.data(), n, q, 1));
		for (auto _ : state) {
			EltwiseMultAddMod(acc.data(), a.data(), b.data(), acc.data(), n, q, 1);
			benchmark::ClobberMemory();
		}
		SetBytes(state, n, 4);
	}
	BENCHMARK(BM_EltwiseMultAddMod)
->ArgsProduct({{1024, 4096, 16384}, {27, 50, 60}})
	->ArgNames({"n", "qbits"});

	static void BM_EltwiseMultModThenAddMod(benchmark::State& state) {
		const uint64_t n = state.range(0), q = Prime(state.range(1), 1);
		Vec a = Random(n, q), b = Random(n, q), acc = Random(n, q), tmp(n);
		for (auto _ : state) {
			EltwiseMultMod(tmp.data(), a.data(), b.data(), n, q, 1);
			EltwiseAddMod(acc.data(), acc.data(), tmp.data(), n, q);
			benchmark::ClobberMemory();
		}
		SetBytes(state, n, 6);
	}
	BENCHMARK(BM_EltwiseMultModThenAddMod)
->ArgsProduct({{1024, 4096, 16384}, {27, 50, 60}})
	->ArgNames({"n", "qbits"});
#endif

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

#ifdef HEXL_RVV_HAS_32BIT_API
// ---- rvv-hexl 32-bit storage (OpenFHE NATIVE_SIZE=32) ----------------------
// The two eltwise ops OpenFHE's binfhe path uses most, on uint32_t data, at
// the same q as the 64-bit qbits:27 cases above: the difference is the
// storage cost (bytes per element halve, no narrow/widen).
static void BM_EltwiseMultMod32(benchmark::State& state) {
	const uint64_t n = state.range(0), q = Prime(state.range(1), 1);
	auto a = RandomW<uint32_t>(n, q), b = RandomW<uint32_t>(n, q), r = RandomW<uint32_t>(n, q);
	HEXL_BENCH_PROBE(state, EltwiseMultMod(r.data(), a.data(), b.data(), n, q, 1));
	for (auto _ : state) {
		EltwiseMultMod(r.data(), a.data(), b.data(), n, q, 1);
		benchmark::ClobberMemory();
	}
	state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * n * 4 * 3));
	state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
}
BENCHMARK(BM_EltwiseMultMod32)
	->ArgsProduct({{1024, 4096, 16384}, {27}})->ArgNames({"n", "qbits"});

	static void BM_EltwiseFMAMod32(benchmark::State& state) {
		const uint64_t n = state.range(0), q = Prime(state.range(1), 1);
		const bool add = state.range(2) != 0;
		auto a = RandomW<uint32_t>(n, q), c = RandomW<uint32_t>(n, q), r = RandomW<uint32_t>(n, q);
		const uint64_t s = q / 3;
		const uint32_t* c_ptr = add ? c.data() : nullptr;
		HEXL_BENCH_PROBE(state, EltwiseFMAMod(r.data(), a.data(), s, c_ptr, n, q, 1));
		for (auto _ : state) {
			EltwiseFMAMod(r.data(), a.data(), s, c_ptr, n, q, 1);
			benchmark::ClobberMemory();
		}
		state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * n * 4 * (add ? 3 : 2)));
		state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
	}
BENCHMARK(BM_EltwiseFMAMod32)
	->ArgsProduct({{1024, 4096, 16384}, {27}, {0, 1}})->ArgNames({"n", "qbits", "add"});

#ifdef HEXL_RVV_HAS_MULT_ADD_MOD
	static void BM_EltwiseMultAddMod32(benchmark::State& state) {
		const uint64_t n = state.range(0), q = Prime(state.range(1), 1);
		auto a = RandomW<uint32_t>(n, q), b = RandomW<uint32_t>(n, q), acc = RandomW<uint32_t>(n, q);
		HEXL_BENCH_PROBE(state, EltwiseMultAddMod(acc.data(), a.data(), b.data(), acc.data(), n, q, 1));
		for (auto _ : state) {
			EltwiseMultAddMod(acc.data(), a.data(), b.data(), acc.data(), n, q, 1);
			benchmark::ClobberMemory();
		}
		state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * n * 4 * 4));
		state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
	}
BENCHMARK(BM_EltwiseMultAddMod32)
	->ArgsProduct({{1024, 4096, 16384}, {27}})->ArgNames({"n", "qbits"});

	static void BM_EltwiseMultModThenAddMod32(benchmark::State& state) {
		const uint64_t n = state.range(0), q = Prime(state.range(1), 1);
		auto a = RandomW<uint32_t>(n, q), b = RandomW<uint32_t>(n, q), acc = RandomW<uint32_t>(n, q), tmp = acc;
		for (auto _ : state) {
			EltwiseMultMod(tmp.data(), a.data(), b.data(), n, q, 1);
			EltwiseAddMod(acc.data(), acc.data(), tmp.data(), n, q);
			benchmark::ClobberMemory();
		}
		state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * n * 4 * 6));
		state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
	}
BENCHMARK(BM_EltwiseMultModThenAddMod32)
	->ArgsProduct({{1024, 4096, 16384}, {27}})->ArgNames({"n", "qbits"});
#endif
#endif

#ifdef HEXL_RVV_HAS_SIGNED_DIGIT_DECOMPOSE
	// ---- signed gadget decomposition (rvv-hexl extension; binfhe's blind rotation) ----
	// One external product's worth at OpenFHE STD128 shapes: both accumulator
	// polynomials (n coefficients, 27-bit q) into 2 x digits polynomials of base
	// 2^g, interleaved as OpenFHE stores them; against the scalar loop OpenFHE
	// runs today (rgsw-acc.cpp, v1.5.1: coefficient outer, digits inner, each
	// digit added into its own zeroed polynomial), kept scalar with novector as
	// it is in OpenFHE, where NativePoly indexing stops the vectoriser.
	template <typename Word>
	struct SddSetup {
		uint64_t n, q, g, digits;
		intel::hexl::AlignedVector64<Word> a0, a1;
		std::vector<intel::hexl::AlignedVector64<Word>> out;
		std::vector<Word*> p0, p1;
		explicit SddSetup(benchmark::State& state)
			: n(state.range(0)), q(Prime(27, state.range(0))), g(state.range(1)), digits(state.range(2)),
			a0(RandomW<Word>(n, q)), a1(RandomW<Word>(n, q)),
			out(2 * digits, intel::hexl::AlignedVector64<Word>(n)), p0(digits), p1(digits) {
				for (uint64_t d = 0; d < digits; ++d) {
					p0[d] = out[2 * d].data();
					p1[d] = out[2 * d + 1].data();
				}
			}
	};

	template <typename Word>
	static void BM_SignedDigitDecompose(benchmark::State& state) {
		SddSetup<Word> s(state);
		HEXL_BENCH_PROBE(state, SignedDigitDecompose(s.p0.data(), s.a0.data(), s.n, s.q, s.g, s.digits));
		for (auto _ : state) {
			SignedDigitDecompose(s.p0.data(), s.a0.data(), s.n, s.q, s.g, s.digits);
			SignedDigitDecompose(s.p1.data(), s.a1.data(), s.n, s.q, s.g, s.digits);
			benchmark::ClobberMemory();
		}
		state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * 2 * s.n));
	}

	template <typename Word>
	static void BM_SignedDigitDecomposeOpenFHELoop(benchmark::State& state) {
		SddSetup<Word> s(state);
		using S = std::conditional_t<std::is_same_v<Word, uint32_t>, int32_t, int64_t>;
		using U = std::make_unsigned_t<S>;
		const S q = static_cast<S>(s.q), gb = static_cast<S>(s.g);
		const int up = static_cast<int>(sizeof(S) * 8 - s.g);
		const uint64_t q_half = s.q >> 1;
		for (auto _ : state) {
			for (auto& o : s.out) std::fill(o.begin(), o.end(), Word{0});  // OpenFHE's zeroed dct
#pragma GCC novector
			for (uint64_t k = 0; k < s.n; ++k) {
				S d0 = s.a0[k] < q_half ? static_cast<S>(s.a0[k]) : static_cast<S>(s.a0[k]) - q;
				S d1 = s.a1[k] < q_half ? static_cast<S>(s.a1[k]) : static_cast<S>(s.a1[k]) - q;
				S r0 = static_cast<S>(static_cast<U>(d0) << up) >> up;
				d0 = (d0 - r0) >> gb;
				S r1 = static_cast<S>(static_cast<U>(d1) << up) >> up;
				d1 = (d1 - r1) >> gb;
				for (uint64_t d = 0; d < 2 * s.digits; d += 2) {
					r0 = static_cast<S>(static_cast<U>(d0) << up) >> up;
					d0 = (d0 - r0) >> gb;
					if (r0 < 0) r0 += q;
					s.out[d][k] += static_cast<Word>(r0);
					r1 = static_cast<S>(static_cast<U>(d1) << up) >> up;
					d1 = (d1 - r1) >> gb;
					if (r1 < 0) r1 += q;
					s.out[d + 1][k] += static_cast<Word>(r1);
				}
			}
			benchmark::ClobberMemory();
		}
		state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * 2 * s.n));
	}

	// STD128 (N = 1024, g = 7, 3 digits kept), N = 2048, and a larger base
BENCHMARK_TEMPLATE(BM_SignedDigitDecompose, uint64_t)
	->Args({1024, 7, 3})->Args({2048, 7, 3})->Args({1024, 10, 2})->ArgNames({"n", "g", "digits"});
BENCHMARK_TEMPLATE(BM_SignedDigitDecomposeOpenFHELoop, uint64_t)
	->Args({1024, 7, 3})->Args({2048, 7, 3})->Args({1024, 10, 2})->ArgNames({"n", "g", "digits"});
#ifdef HEXL_RVV_HAS_32BIT_API
BENCHMARK_TEMPLATE(BM_SignedDigitDecompose, uint32_t)
	->Args({1024, 7, 3})->Args({2048, 7, 3})->Args({1024, 10, 2})->ArgNames({"n", "g", "digits"});
BENCHMARK_TEMPLATE(BM_SignedDigitDecomposeOpenFHELoop, uint32_t)
	->Args({1024, 7, 3})->Args({2048, 7, 3})->Args({1024, 10, 2})->ArgNames({"n", "g", "digits"});
#endif
#endif

#ifdef HEXL_RVV_HAS_BASE_CONVERT
	// ---- approximate RNS base conversion (rvv-hexl extension; OpenFHE HYBRID key switching) ----
	// The shapes of blindloop's ckks_openfhe (N = 8192, Q = 46 + 37 + 36 + 37, P = 60, one limb
	// per digit): shape 0 = ModUp of a 36-bit digit to the other limbs and P (1 -> 4), shape 1 =
	// ModDown of P to Q (1 -> 4), shape 2 = a 3-limb digit to 5 limbs (larger dnum). Against
	// OpenFHE's own loop (dcrtpoly-impl.h, v1.5.1, the HAVE_INT128 + OpenMP branch): Shoup for
	// y_i, 128-bit products summed, one 128-bit Barrett reduction per output limb.
	struct BaseConvSetup {
		uint64_t n;
		std::vector<uint64_t> q, p, q_hat_inv, q_hat_mod_p;
		std::vector<intel::hexl::AlignedVector64<uint64_t>> x, y;
		std::vector<const uint64_t*> in;
		std::vector<uint64_t*> out;
		explicit BaseConvSetup(benchmark::State& state) : n(state.range(0)) {
			const uint64_t N = n;
			switch (state.range(1)) {
				case 0: q = {Prime(36, N)}; p = {Prime(46, N), Prime(37, N), hexltest::oracle::NttPrime(37, N, 1), Prime(59, N)}; break;
				case 1: q = {Prime(59, N)}; p = {Prime(46, N), Prime(37, N), Prime(36, N), hexltest::oracle::NttPrime(37, N, 1)}; break;
				default: q = {Prime(46, N), Prime(37, N), Prime(36, N)};
					p = {hexltest::oracle::NttPrime(37, N, 1), Prime(59, N), hexltest::oracle::NttPrime(59, N, 1),
						hexltest::oracle::NttPrime(46, N, 1), hexltest::oracle::NttPrime(36, N, 1)};
			}
			const size_t L = q.size(), K = p.size();
			q_hat_inv.resize(L);
			q_hat_mod_p.resize(L * K);
			for (size_t i = 0; i < L; ++i) {
				uint64_t h = 1;
				for (size_t k = 0; k < L; ++k)
					if (k != i) h = hexltest::oracle::MulMod(h, q[k] % q[i], q[i]);
				q_hat_inv[i] = hexltest::oracle::InvModPrime(h, q[i]);
				for (size_t j = 0; j < K; ++j) {
					uint64_t hp = 1;
					for (size_t k = 0; k < L; ++k)
						if (k != i) hp = hexltest::oracle::MulMod(hp, q[k] % p[j], p[j]);
					q_hat_mod_p[i * K + j] = hp;
				}
			}
			for (size_t i = 0; i < L; ++i) x.push_back(RandomW<uint64_t>(n, q[i]));
			y.assign(K, intel::hexl::AlignedVector64<uint64_t>(n));
			for (auto& v : x) in.push_back(v.data());
			for (auto& v : y) out.push_back(v.data());
		}
	};

	static void BM_BaseConvert(benchmark::State& state) {
		BaseConvSetup s(state);
		const auto run = [&] {
			BaseConvert(s.out.data(), s.in.data(), s.n, s.q.data(), s.q.size(), s.p.data(), s.p.size(),
					s.q_hat_inv.data(), s.q_hat_mod_p.data());
		};
		HEXL_BENCH_PROBE(state, run());
		for (auto _ : state) {
			run();
			benchmark::ClobberMemory();
		}
		state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * s.n));
	}

	static void BM_BaseConvertOpenFHELoop(benchmark::State& state) {
		using u128 = unsigned __int128;
		BaseConvSetup s(state);
		const size_t L = s.q.size(), K = s.p.size();
		std::vector<uint64_t> precon(L);
		std::vector<u128> mu(K);  // OpenFHE's modpBarrettMu: floor(2^128 / p_j)
		for (size_t i = 0; i < L; ++i) precon[i] = MultiplyFactor(s.q_hat_inv[i], 64, s.q[i]).BarrettFactor();
		for (size_t j = 0; j < K; ++j) mu[j] = ~u128{0} / s.p[j];
		std::vector<u128> sum(K);
		for (auto _ : state) {
			for (uint64_t k = 0; k < s.n; ++k) {
				std::fill(sum.begin(), sum.end(), u128{0});
				for (size_t i = 0; i < L; ++i) {
					const uint64_t y = MultiplyMod(s.x[i][k], s.q_hat_inv[i], precon[i], s.q[i]);  // ModMulFastConst
					for (size_t j = 0; j < K; ++j) sum[j] += static_cast<u128>(y) * s.q_hat_mod_p[i * K + j];
				}
				for (size_t j = 0; j < K; ++j) {  // BarrettUint128ModUint64: (a * mu) >> 128, low word
					const u128 a = sum[j];
					const uint64_t a_lo = static_cast<uint64_t>(a), a_hi = static_cast<uint64_t>(a >> 64);
					const uint64_t m_lo = static_cast<uint64_t>(mu[j]), m_hi = static_cast<uint64_t>(mu[j] >> 64);
					const u128 mid = (static_cast<u128>(a_lo) * m_lo >> 64) + static_cast<u128>(a_lo) * m_hi +
						static_cast<u128>(a_hi) * m_lo;
					const uint64_t qt = a_hi * m_hi + static_cast<uint64_t>(mid >> 64);
					uint64_t r = a_lo - qt * s.p[j];
					while (r >= s.p[j]) r -= s.p[j];
					s.out[j][k] = r;
				}
			}
			benchmark::ClobberMemory();
		}
		state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * s.n));
	}

BENCHMARK(BM_BaseConvert)->ArgsProduct({{8192, 16384}, {0, 1, 2}})->ArgNames({"n", "shape"});
BENCHMARK(BM_BaseConvertOpenFHELoop)->ArgsProduct({{8192, 16384}, {0, 1, 2}})->ArgNames({"n", "shape"});
#endif
