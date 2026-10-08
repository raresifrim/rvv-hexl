// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// hexl/ntt: the NTT class, exactly as OpenFHE uses it and a bit beyond.
//
// Each test is a template on the storage word: Ntt_<Name> runs uint64_t (the
// upstream API, validated against Intel HEXL), Ntt_<Name>_32 the rvv-hexl
// 32-bit extension (OpenFHE NATIVE_SIZE=32) on moduli that fit its
// preconditions: q <= 2^32, mod_factor * q <= 2^32. q in [2^30, 2^32) is
// included on purpose: it takes the native path instead of RVV32.

#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "hexl/hexl.hpp"
#include "oracle.hpp"
#include "test.hpp"

using namespace intel::hexl;
namespace O = hexltest::oracle;

namespace {

std::vector<uint64_t> Degrees(uint64_t max_n) {
	std::vector<uint64_t> ns;
	for (uint64_t n = 2; n <= max_n; n <<= 1) ns.push_back(n);
	return ns;
}

template <typename Word>
constexpr bool Is32() {
	return std::is_same<Word, uint32_t>::value;
}

/// Modulus sizes around every dispatch boundary: binfhe (27), last/first e32
/// (29/30 vs 31), 32-bit, IPCEI NTT bench (49), BFV (60), maximum (62).
/// 32-bit storage: up to 31 bits (q < 2^32).
template <typename Word>
std::vector<size_t> ModBits() {
	if (Is32<Word>()) return {20, 27, 29, 30, 31};
	return {20, 27, 29, 30, 31, 32, 40, 49, 55, 60, 61};
}

/// The three moduli most tests use: binfhe-sized, and two larger ones.
template <typename Word>
std::vector<size_t> TypicalBits() {
	if (Is32<Word>()) return {27, 30};
	return {27, 49, 60};
}

uint64_t MaxN() { return hexltest::g_quick ? 1024 : 32768; }
uint64_t MaxOracleN() { return hexltest::g_quick ? 64 : 512; }


// Forward NTT against the O(N^2) definition, using the constructor OpenFHE
// uses (explicit root). Also pins down the output ORDER (bit-reversed).
template <typename Word>
void RunForwardMatchesDefinition() {
	for (uint64_t n : Degrees(MaxOracleN())) {
		for (size_t bits : ModBits<Word>()) {
			const uint64_t q = O::NttPrime(bits, n);
			const uint64_t w = O::PrimitiveRoot2N(n, q);
			NTT ntt(n, q, w);
			const auto x64 = O::Random(n, q);
			const auto x = O::As<Word>(x64);
			std::vector<Word> got(n);
			ntt.ComputeForward(got.data(), x.data(), 1, 1);
			CHECK_VEC_EQ(O::As<Word>(O::ForwardNtt(x64, q, w)), got,
					<< "n=" << n << " q=" << q);
		}
	}
}

// Constructor without a root must use the MINIMAL primitive 2N-th root.
template <typename Word>
void RunDefaultRootIsMinimal() {
	for (uint64_t n : {8ULL, 1024ULL}) {
		for (size_t bits : TypicalBits<Word>()) {
			const uint64_t q = O::NttPrime(bits, n);
			NTT ntt(n, q);
			const uint64_t w = O::MinimalPrimitiveRoot2N(n, q);
			CHECK_EQ(ntt.GetMinimalRootOfUnity(), w);
			CHECK_EQ(ntt.GetDegree(), n);
			CHECK_EQ(ntt.GetModulus(), q);
			if (n <= MaxOracleN()) {
				const auto x64 = O::Random(n, q);
				const auto x = O::As<Word>(x64);
				std::vector<Word> got(n);
				ntt.ComputeForward(got.data(), x.data(), 1, 1);
				CHECK_VEC_EQ(O::As<Word>(O::ForwardNtt(x64, q, w)), got,
						<< "n=" << n << " q=" << q);
			}
		}
	}
}

template <typename Word>
void RunRoundTrip() {
	for (uint64_t n : Degrees(MaxN())) {
		for (size_t bits : ModBits<Word>()) {
			const uint64_t q = O::NttPrime(bits, n);
			NTT ntt(n, q, O::PrimitiveRoot2N(n, q));
			const auto x = O::As<Word>(O::Random(n, q));
			std::vector<Word> y(n), z(n);
			ntt.ComputeForward(y.data(), x.data(), 1, 1);
			ntt.ComputeInverse(z.data(), y.data(), 1, 1);
			CHECK_VEC_EQ(x, z, << "n=" << n << " q=" << q);
		}
	}
}

// OpenFHE calls both transforms in place: ComputeForward(data, data, 1, 1).
template <typename Word>
void RunInPlaceEqualsOutOfPlace() {
	for (uint64_t n : {16ULL, 1024ULL, 8192ULL}) {
		for (size_t bits : TypicalBits<Word>()) {
			const uint64_t q = O::NttPrime(bits, n);
			NTT ntt(n, q, O::PrimitiveRoot2N(n, q));
			const auto x = O::As<Word>(O::Random(n, q));
			std::vector<Word> out(n), inplace = x;
			ntt.ComputeForward(out.data(), x.data(), 1, 1);
			ntt.ComputeForward(inplace.data(), inplace.data(), 1, 1);
			CHECK_VEC_EQ(out, inplace, << "forward n=" << n << " q=" << q);
			std::vector<Word> back(n);
			ntt.ComputeInverse(back.data(), out.data(), 1, 1);
			ntt.ComputeInverse(inplace.data(), inplace.data(), 1, 1);
			CHECK_VEC_EQ(back, inplace, << "inverse n=" << n << " q=" << q);
			CHECK_VEC_EQ(x, back);
		}
	}
}

// Out-of-place calls must leave the operand untouched.
template <typename Word>
void RunOperandUnmodified() {
	const uint64_t n = 1024, q = O::NttPrime(27, n);
	NTT ntt(n, q);
	const auto x = O::As<Word>(O::Random(n, q));
	std::vector<Word> copy = x, out(n);
	ntt.ComputeForward(out.data(), copy.data(), 1, 1);
	CHECK_VEC_EQ(x, copy);
	const std::vector<Word> out_before = out;
	std::vector<Word> back(n);
	ntt.ComputeInverse(back.data(), out.data(), 1, 1);
	CHECK_VEC_EQ(out_before, out);
}

// input_mod_factor / output_mod_factor semantics (lazy reduction). For 32-bit
// storage only the combinations with mod_factor * q <= 2^32 are valid.
template <typename Word>
void RunModFactors() {
	const std::vector<size_t> bits_list =
		Is32<Word>() ? std::vector<size_t>{27, 29, 30}
	: std::vector<size_t>{27, 29, 30, 49, 60};
	auto fits = [](uint64_t factor, uint64_t q) {
		return !Is32<Word>() || factor * q <= (1ULL << 32);
	};
	for (uint64_t n : {4ULL, 64ULL, 2048ULL}) {
		for (size_t bits : bits_list) {
			const uint64_t q = O::NttPrime(bits, n);
			NTT ntt(n, q, O::PrimitiveRoot2N(n, q));
			const auto x64 = O::Random(n, q);
			std::vector<Word> ref(n);
			ntt.ComputeForward(ref.data(), O::As<Word>(x64).data(), 1, 1);

			for (uint64_t imf : {1ULL, 2ULL, 4ULL}) {
				// same values mod q, spread over [0, imf*q)
				auto xin64 = x64;
				auto k = O::Random(n, imf);
				for (size_t i = 0; i < n; ++i) xin64[i] += k[i] * q;
				const auto xin = O::As<Word>(xin64);
				for (uint64_t omf : {1ULL, 4ULL}) {
					if (!fits(imf, q) || !fits(omf, q)) continue;
					std::vector<Word> got(n);
					ntt.ComputeForward(got.data(), xin.data(), imf, omf);
					for (size_t i = 0; i < n; ++i) {
						CHECK(got[i] < omf * q);
						CHECK_EQ(got[i] % q, ref[i], << "fwd i=" << i << " imf=" << imf
								<< " omf=" << omf << " q=" << q);
					}
				}
			}
			for (uint64_t imf : {1ULL, 2ULL}) {
				std::vector<uint64_t> yin64(ref.begin(), ref.end());
				auto k = O::Random(n, imf);
				for (size_t i = 0; i < n; ++i) yin64[i] += k[i] * q;
				const auto yin = O::As<Word>(yin64);
				for (uint64_t omf : {1ULL, 2ULL}) {
					if (!fits(imf, q) || !fits(omf, q)) continue;
					std::vector<Word> got(n);
					ntt.ComputeInverse(got.data(), yin.data(), imf, omf);
					for (size_t i = 0; i < n; ++i) {
						CHECK(got[i] < omf * q);
						CHECK_EQ(got[i] % q, x64[i], << "inv i=" << i << " imf=" << imf
								<< " omf=" << omf << " q=" << q);
					}
				}
			}
		}
	}
}

// Convolution theorem: NTT turns negacyclic polynomial products into
// EltwiseMultMod. This is literally what a BFV/binfhe multiplication does.
template <typename Word>
void RunNegacyclicConvolution() {
	for (uint64_t n : {8ULL, 256ULL}) {
		for (size_t bits : {size_t{27}, Is32<Word>() ? size_t{30} : size_t{60}}) {
			const uint64_t q = O::NttPrime(bits, n);
			NTT ntt(n, q);
			const auto a64 = O::Random(n, q), b64 = O::Random(n, q);
			std::vector<uint64_t> want(n, 0);  // schoolbook a*b mod (X^n + 1)
			for (uint64_t i = 0; i < n; ++i) {
				for (uint64_t j = 0; j < n; ++j) {
					const uint64_t p = O::MulMod(a64[i], b64[j], q);
					const uint64_t k = (i + j) % n;
					want[k] = (i + j < n) ? O::AddMod(want[k], p, q) : O::SubMod(want[k], p, q);
				}
			}
			const auto a = O::As<Word>(a64), b = O::As<Word>(b64);
			std::vector<Word> fa(n), fb(n), prod(n), got(n);
			ntt.ComputeForward(fa.data(), a.data(), 1, 1);
			ntt.ComputeForward(fb.data(), b.data(), 1, 1);
			EltwiseMultMod(prod.data(), fa.data(), fb.data(), n, q, 1);
			ntt.ComputeInverse(got.data(), prod.data(), 1, 1);
			CHECK_VEC_EQ(O::As<Word>(want), got, << "n=" << n << " q=" << q);
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
template <typename Word>
void RunOpenFHEMapPattern() {
	struct HashPair {
		size_t operator()(const std::pair<uint64_t, uint64_t>& p) const {
			return std::hash<uint64_t>()(p.first) ^ (std::hash<uint64_t>()(p.second) << 1);
		}
	};
	std::unordered_map<std::pair<uint64_t, uint64_t>, NTT, HashPair> cache;
	for (uint64_t n : {1024ULL, 2048ULL}) {
		for (size_t bits : {size_t{27}, Is32<Word>() ? size_t{28} : size_t{60}}) {
			const uint64_t q = O::NttPrime(bits, n);
			const uint64_t w = O::PrimitiveRoot2N(n, q);
			NTT ntt(n, q, w);
			cache[{n, q}] = std::move(ntt);
			NTT* p = &cache.find({n, q})->second;
			auto data = O::As<Word>(O::Random(n, q));
			const auto orig = data;
			p->ComputeForward(data.data(), data.data(), 1, 1);
			p->ComputeInverse(data.data(), data.data(), 1, 1);
			CHECK_VEC_EQ(orig, data, << "n=" << n << " q=" << q);
		}
	}
}

#if !defined(__riscv) || defined(__linux__)
// OpenFHE runs ComputeForward on the SAME NTT object from many OpenMP threads.
// Any mutable scratch inside the object shows up here as corrupted output.
template <typename Word>
void RunConcurrentUse() {
	const uint64_t n = 4096;
	for (size_t bits : {size_t{27}, Is32<Word>() ? size_t{30} : size_t{60}}) {
		const uint64_t q = O::NttPrime(bits, n);
		NTT ntt(n, q);
		const unsigned kThreads = 8;
		std::vector<std::vector<Word>> inputs, want(kThreads);
		for (unsigned t = 0; t < kThreads; ++t) {
			inputs.push_back(O::As<Word>(O::Random(n, q)));
			want[t].resize(n);
			ntt.ComputeForward(want[t].data(), inputs[t].data(), 1, 1);
		}
		std::vector<std::vector<Word>> got(kThreads, std::vector<Word>(n));
		std::vector<std::thread> pool;
		for (unsigned t = 0; t < kThreads; ++t) {
			pool.emplace_back([&, t] {
					for (int rep = 0; rep < 50; ++rep) {
					auto tmp = inputs[t];
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

#ifdef HEXL_RVV_PORT
// OpenFHE at NATIVE_SIZE=32 constructs NTT(N, q, root) with uint32_t arguments.
// Upstream's unconstrained allocator-template constructor wins that overload
// resolution and does not compile; rvv-hexl constrains it. This must pick the
// (degree, q, root_of_unity) constructor.
TEST(Ntt_Uint32ConstructorArgs) {
	const uint32_t n = 1024;
	const uint32_t q = static_cast<uint32_t>(O::NttPrime(27, n));
	const uint32_t w = static_cast<uint32_t>(O::PrimitiveRoot2N(n, q));
	NTT ntt(n, q, w);
	CHECK_EQ(ntt.GetMinimalRootOfUnity(), uint64_t{w});
	CHECK_EQ(ntt.GetDegree(), uint64_t{n});
}
#endif

}  // namespace

TEST(Ntt_ForwardMatchesDefinition) { RunForwardMatchesDefinition<uint64_t>(); }
TEST(Ntt_DefaultRootIsMinimal) { RunDefaultRootIsMinimal<uint64_t>(); }
TEST(Ntt_RoundTrip) { RunRoundTrip<uint64_t>(); }
TEST(Ntt_InPlaceEqualsOutOfPlace) { RunInPlaceEqualsOutOfPlace<uint64_t>(); }
TEST(Ntt_OperandUnmodified) { RunOperandUnmodified<uint64_t>(); }
TEST(Ntt_ModFactors) { RunModFactors<uint64_t>(); }
TEST(Ntt_NegacyclicConvolution) { RunNegacyclicConvolution<uint64_t>(); }
TEST(Ntt_OpenFHEMapPattern) { RunOpenFHEMapPattern<uint64_t>(); }
#if !defined(__riscv) || defined(__linux__)
TEST(Ntt_ConcurrentUse) { RunConcurrentUse<uint64_t>(); }
#endif

#ifdef HEXL_RVV_HAS_32BIT_API
TEST(Ntt_ForwardMatchesDefinition_32) { RunForwardMatchesDefinition<uint32_t>(); }
TEST(Ntt_DefaultRootIsMinimal_32) { RunDefaultRootIsMinimal<uint32_t>(); }
TEST(Ntt_RoundTrip_32) { RunRoundTrip<uint32_t>(); }
TEST(Ntt_InPlaceEqualsOutOfPlace_32) { RunInPlaceEqualsOutOfPlace<uint32_t>(); }
TEST(Ntt_OperandUnmodified_32) { RunOperandUnmodified<uint32_t>(); }
TEST(Ntt_ModFactors_32) { RunModFactors<uint32_t>(); }
TEST(Ntt_NegacyclicConvolution_32) { RunNegacyclicConvolution<uint32_t>(); }
TEST(Ntt_OpenFHEMapPattern_32) { RunOpenFHEMapPattern<uint32_t>(); }
#if !defined(__riscv) || defined(__linux__)
TEST(Ntt_ConcurrentUse_32) { RunConcurrentUse<uint32_t>(); }
#endif
#endif
