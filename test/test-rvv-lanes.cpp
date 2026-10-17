// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// The finished RVV kernels at every lane type (LMUL m1/m2/m4/m8, both storage
// words), checked against the __int128 oracle. The public API runs only the
// default lanes of src/util/rvv-config.hpp; these tests make sure every other
// instantiation (the ones bench-hexl BM_Lanes_* times) is correct too.
// The NTT kernels additionally at every small-span strategy (PerBlock,
// FusedSlides, every Transposed shape), against the reference transforms.
// Also rvv::SetVl / Load / Store at every LMUL.
//
//   hexl-tests RvvLanes
//
// Internal headers: compiled only into RVV builds (needs -Isrc, which the
// Makefile adds; the tests are empty for ISA=scalar).

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "oracle.hpp"
#include "test.hpp"

#if defined(__has_include)
#if __has_include("util/rvv-util.hpp")
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
#include "util/util-internal.hpp"
#endif
#endif

#ifdef HEXL_HAS_RVV

namespace O = hexltest::oracle;
namespace rvv = intel::hexl::rvv;
using intel::hexl::CMPINT;

namespace {

const CMPINT kAllCmps[] = {CMPINT::EQ, CMPINT::LT,  CMPINT::LE,  CMPINT::FALSE,
	CMPINT::NE, CMPINT::NLT, CMPINT::NLE, CMPINT::TRUE};

// every tail at VLEN 256 and 1024 for m1..m8, without the full eltwise sweep
const size_t kSizes[] = {1, 3, 7, 8, 9, 17, 31, 33, 63, 65, 129, 257, 1023, 1025, 4099};

/// Moduli per storage word, inside what the kernels accept on the RVV path:
/// 64-bit add/sub q < 2^63, CmpSubMod q <= 2^63; 32-bit RVV path q < 2^30.
template <typename Word>
std::vector<uint64_t> LaneModuli() {
	if constexpr (std::is_same_v<Word, uint64_t>) {
		return {3, 17, O::NttPrime(27, 1), O::NttPrime(49, 1), O::NttPrime(60, 1), O::PrimeBelow(1ULL << 62)};
	} else {
		return {3, 17, O::NttPrime(20, 1), O::NttPrime(27, 1), O::PrimeBelow(1ULL << 30)};
	}
}

template <typename Word>
std::vector<Word> AsWord(const std::vector<uint64_t>& v) {
	return std::vector<Word>(v.begin(), v.end());
}

template <typename Word, class V>
void CheckAddSub(const char* lane) {
	for (uint64_t q : LaneModuli<Word>()) {
		for (size_t n : kSizes) {
			auto a64 = O::Random(n, q), b64 = O::Random(n, q);
			a64[0] = q - 1;
			b64[0] = q - 1;
			const uint64_t s = O::Random(1, q)[0];
			const auto a = AsWord<Word>(a64), b = AsWord<Word>(b64);
			std::vector<Word> add(n), sub(n), adds(n), subs(n);
			intel::hexl::EltwiseAddModRVV<Word, V>(add.data(), a.data(), b.data(), n, q);
			intel::hexl::EltwiseSubModRVV<Word, V>(sub.data(), a.data(), b.data(), n, q);
			intel::hexl::EltwiseAddModRVV<Word, V>(adds.data(), a.data(), s, n, q);
			intel::hexl::EltwiseSubModRVV<Word, V>(subs.data(), a.data(), s, n, q);
			for (size_t i = 0; i < n; ++i) {
				CHECK_EQ(static_cast<uint64_t>(add[i]), O::AddMod(a64[i], b64[i], q), << "AddMod vv " << lane << " n=" << n << " q=" << q);
				CHECK_EQ(static_cast<uint64_t>(sub[i]), O::SubMod(a64[i], b64[i], q), << "SubMod vv " << lane << " n=" << n << " q=" << q);
				CHECK_EQ(static_cast<uint64_t>(adds[i]), O::AddMod(a64[i], s, q), << "AddMod vs " << lane << " n=" << n << " q=" << q);
				CHECK_EQ(static_cast<uint64_t>(subs[i]), O::SubMod(a64[i], s, q), << "SubMod vs " << lane << " n=" << n << " q=" << q);
			}
		}
	}
}

template <typename Word, class V>
void CheckCmpAdd(const char* lane) {
	const uint64_t word_bound = O::AnyWord<Word>();
	for (CMPINT cmp : kAllCmps) {
		for (size_t n : kSizes) {
			auto a64 = O::Random(n, word_bound);
			const uint64_t bound = a64[n / 2];
			for (size_t i = 0; i < n; i += 3) a64[i] = bound;  // ties for EQ/LE/NLT
			const uint64_t diff = 1 + O::Random(1, std::is_same_v<Word, uint64_t> ? (1ULL << 40) : (1ULL << 20))[0];
			const auto a = AsWord<Word>(a64);
			std::vector<Word> got(n);
			intel::hexl::EltwiseCmpAddRVV<Word, V>(got.data(), a.data(), n, cmp, bound, diff);
			for (size_t i = 0; i < n; ++i) {
				const Word want = static_cast<Word>(intel::hexl::Compare(cmp, a64[i], bound) ? a64[i] + diff : a64[i]);
				CHECK_EQ(got[i], want, << "CmpAdd " << lane << " cmp=" << static_cast<int>(cmp) << " n=" << n << " i=" << i);
			}
		}
	}
}

template <typename Word, class V>
void CheckCmpSubMod(const char* lane) {
	const uint64_t word_bound = O::AnyWord<Word>();
	for (uint64_t q : LaneModuli<Word>()) {
		for (CMPINT cmp : kAllCmps) {
			for (size_t n : kSizes) {
				auto a64 = O::Random(n, word_bound);  // any word: the kernel reduces
				const uint64_t bound = a64[n / 2];
				for (size_t i = 0; i < n; i += 3) a64[i] = bound;
				const uint64_t diff = 1 + O::Random(1, q - 1)[0];
				const auto a = AsWord<Word>(a64);
				std::vector<Word> got(n);
				intel::hexl::EltwiseCmpSubModRVV<Word, V>(got.data(), a.data(), n, q, cmp, bound, diff);
				for (size_t i = 0; i < n; ++i) {
					const uint64_t r = a64[i] % q;
					const uint64_t want = intel::hexl::Compare(cmp, a64[i], bound) ? O::SubMod(r, diff, q) : r;
					CHECK_EQ(static_cast<uint64_t>(got[i]), want,
							<< "CmpSubMod " << lane << " cmp=" << static_cast<int>(cmp) << " n=" << n << " q=" << q << " i=" << i);
				}
			}
		}
	}
}

/// MultMod at input_mod_factor 1, 2, 4 (inputs anywhere in [0, imf * q)).
/// Word = uint64_t with e64 lanes runs EltwiseMultModRVV64 (q < 2^61, incl.
/// q = 3 where the pre-shift is 0); Word = uint32_t runs EltwiseMultModRVV32
/// (q < 2^30, incl. the 30-bit double-correction case).
template <typename Word, int Imf, class V>
void CheckMultModImf(const char* lane, uint64_t q) {
	for (size_t n : kSizes) {
		auto a64 = O::Random(n, Imf * q), b64 = O::Random(n, Imf * q);
		a64[0] = Imf * q - 1;
		b64[0] = Imf * q - 1;
		const auto a = AsWord<Word>(a64), b = AsWord<Word>(b64);
		std::vector<Word> got(n);
		if constexpr (std::is_same_v<V, vuint64m1_t> || std::is_same_v<V, vuint64m2_t> ||
				std::is_same_v<V, vuint64m4_t> || std::is_same_v<V, vuint64m8_t>) {
			intel::hexl::EltwiseMultModRVV64<Imf, V>(got.data(), a.data(), b.data(), n, q);
		} else {
			intel::hexl::EltwiseMultModRVV32<Word, Imf, V>(got.data(), a.data(), b.data(), n, q);
		}
		for (size_t i = 0; i < n; ++i) {
			CHECK_EQ(static_cast<uint64_t>(got[i]), O::MulMod(a64[i] % q, b64[i] % q, q),
					<< "MultMod " << lane << " imf=" << Imf << " n=" << n << " q=" << q << " i=" << i);
		}
	}
}

template <typename Word, class V>
void CheckMultMod(const char* lane) {
	constexpr bool kE64 = std::is_same_v<Word, uint64_t> && !std::is_same_v<V, vuint32m1_t> &&
		!std::is_same_v<V, vuint32m2_t> && !std::is_same_v<V, vuint32m4_t>;
	const std::vector<uint64_t> moduli =
		kE64 ? std::vector<uint64_t>{3, 17, O::NttPrime(27, 1), O::NttPrime(49, 1), O::NttPrime(60, 1), O::PrimeBelow(1ULL << 61)}
	: std::vector<uint64_t>{3, 17, O::NttPrime(20, 1), O::NttPrime(27, 1), O::PrimeBelow(1ULL << 30)};
	for (uint64_t q : moduli) {
		CheckMultModImf<Word, 1, V>(lane, q);
		CheckMultModImf<Word, 2, V>(lane, q);
		CheckMultModImf<Word, 4, V>(lane, q);
	}
}

/// EltwiseMultAddMod's RVV kernels at input_mod_factor 1, 2, 4 (all three
/// inputs anywhere in [0, imf * q)), out of place and accumulating in place
/// (result == arg3). e64 lanes: RVV64 (q < 2^61); e32 lanes: RVV32 (q < 2^30).
template <typename Word, int Imf, class V>
void CheckMultAddModImf(const char* lane, uint64_t q) {
	constexpr bool kE64 = rvv::IsE64<V>;
	for (size_t n : kSizes) {
		auto a64 = O::Random(n, Imf * q), b64 = O::Random(n, Imf * q), c64 = O::Random(n, Imf * q);
		a64[0] = Imf * q - 1; b64[0] = Imf * q - 1; c64[0] = Imf * q - 1;
		const auto a = AsWord<Word>(a64), b = AsWord<Word>(b64), c = AsWord<Word>(c64);
		std::vector<Word> got(n), acc = c;
		if constexpr (kE64) {
			intel::hexl::EltwiseMultAddModRVV64<Imf, V>(got.data(), a.data(), b.data(), c.data(), n, q);
			intel::hexl::EltwiseMultAddModRVV64<Imf, V>(acc.data(), a.data(), b.data(), acc.data(), n, q);
		} else {
			intel::hexl::EltwiseMultAddModRVV32<Word, Imf, V>(got.data(), a.data(), b.data(), c.data(), n, q);
			intel::hexl::EltwiseMultAddModRVV32<Word, Imf, V>(acc.data(), a.data(), b.data(), acc.data(), n, q);
		}
		for (size_t i = 0; i < n; ++i) {
			const uint64_t want = O::AddMod(O::MulMod(a64[i] % q, b64[i] % q, q), c64[i] % q, q);
			CHECK_EQ(static_cast<uint64_t>(got[i]), want, << "MultAddMod " << lane << " imf=" << Imf << " n=" << n << " q=" << q << " i=" << i);
			CHECK_EQ(static_cast<uint64_t>(acc[i]), want, << "MultAddMod (result==arg3) " << lane << " imf=" << Imf << " n=" << n << " q=" << q << " i=" << i);
		}
	}
}

template <typename Word, class V>
void CheckMultAddMod(const char* lane) {
	constexpr bool kE64 = rvv::IsE64<V>;
	const std::vector<uint64_t> moduli =
		kE64 ? std::vector<uint64_t>{3, 17, O::NttPrime(27, 1), O::NttPrime(49, 1), O::NttPrime(60, 1), O::PrimeBelow(1ULL << 61)}
	: std::vector<uint64_t>{3, 17, O::NttPrime(20, 1), O::NttPrime(27, 1), O::PrimeBelow(1ULL << 30)};
	for (uint64_t q : moduli) {
		CheckMultAddModImf<Word, 1, V>(lane, q);
		CheckMultAddModImf<Word, 2, V>(lane, q);
		CheckMultAddModImf<Word, 4, V>(lane, q);
	}
}

/// SetVl<V> must equal the matching vsetvl; Load<V>/Store must round-trip
/// exactly vl elements per strip (guard words past n stay untouched).
template <class V, typename Word>
void CheckSetVlLoadStore(const char* lane, size_t vlmax) {
	for (size_t n : kSizes) {
		CHECK_EQ(rvv::SetVl<V>(n), n < vlmax ? n : vlmax, << "SetVl " << lane << " n=" << n);
		const Word guard = static_cast<Word>(0xA5A5A5A5A5A5A5A5ULL);
		auto src = AsWord<Word>(O::Random(n, O::AnyWord<Word>()));
		std::vector<Word> dst(n + 8, guard), want(src);
		want.resize(n + 8, guard);
		for (size_t i = 0, vl; i < n; i += vl) {
			vl = rvv::SetVl<V>(n - i);
			rvv::Store(&dst[i], rvv::Load<V>(&src[i], vl), vl);
		}
		CHECK_VEC_EQ(want, dst, << "Load/Store round trip " << lane << " n=" << n);
	}
}

/// rvv::LoadStrided / StoreStrided at lane type V on storage Word: lane k is
/// p[k * stride] (elements; positive, negative, and zero for loads), at vl 1,
/// about half of VLMAX and VLMAX; a store writes exactly those words. e32
/// lanes on uint64_t storage carry values below 2^32 (as the NTT does).
template <class V, typename Word>
void CheckLoadStoreStrided(const char* lane) {
	using E = rvv::ElemT<V>;
	const size_t vlmax = rvv::VlMax<V>();
	const uint64_t bound = rvv::IsE32<V> ? (uint64_t{1} << 32) : O::AnyWord<Word>();
	const Word guard = static_cast<Word>(0xA5A5A5A5A5A5A5A5ULL);
	for (ptrdiff_t stride : {1, 2, 3, 7, 512, -1, -3, 0}) {
		for (size_t vl : {size_t{1}, vlmax / 2 + 1, vlmax}) {
			const size_t mag = static_cast<size_t>(stride < 0 ? -stride : stride);
			const size_t span = (vl - 1) * mag + 1;
			const size_t base = stride < 0 ? span - 1 : 0;  // lane 0's word
			const auto src = AsWord<Word>(O::Random(span, bound));
			const V v = rvv::LoadStrided<V>(src.data() + base, stride, vl);
			std::vector<E> got(vl);
			rvv::Store(got.data(), v, vl);
			std::vector<E> want(vl);
			for (size_t k = 0; k < vl; ++k) want[k] = static_cast<E>(src[base + k * stride]);
			CHECK_VEC_EQ(want, got, << "LoadStrided " << lane << " stride=" << stride << " vl=" << vl);
			if (stride == 0) continue;  // a zero-stride store writes one word vl times
			std::vector<Word> dst(span, guard), dwant(span, guard);
			rvv::StoreStrided(dst.data() + base, stride, v, vl);
			for (size_t k = 0; k < vl; ++k) dwant[base + k * stride] = static_cast<Word>(got[k]);
			CHECK_VEC_EQ(dwant, dst, << "StoreStrided " << lane << " stride=" << stride << " vl=" << vl);
		}
	}
}

/// rvv::FwdButterfly / InvButterfly (and the 32 variants) at lane type V, with
/// per-lane and broadcast twiddles, against the exact butterflies: outputs
/// congruent and inside Harvey's lazy ranges (forward [0, 4q) in and out,
/// inputs up to 4q - 1; inverse [0, 2q)).
template <typename Word, class V>
void CheckButterflies(const char* lane) {
	using E = rvv::ElemT<V>;
	constexpr bool k32 = rvv::IsE32<V>;
	constexpr unsigned kBits = sizeof(E) * 8;
	const size_t vl = rvv::VlMax<V>();
	const std::vector<uint64_t> moduli = k32 ? std::vector<uint64_t>{O::NttPrime(19, 1024), O::Primes(1, 29, false, 1024)[0]}
	: std::vector<uint64_t>{O::NttPrime(30, 1024), O::Primes(1, 61, false, 1024)[0]};
	auto vec = [&](const std::vector<uint64_t>& a) { return rvv::Load<V>(AsWord<E>(a).data(), vl); };
	auto out = [&](V v) { std::vector<E> r(vl); rvv::Store(r.data(), v, vl); return r; };
	for (uint64_t q : moduli) {
		for (int round = 0; round < 20; ++round) {
			for (bool lanes : {true, false}) {  // per-lane twiddles, or one broadcast
				const auto w = O::Random(vl, q);
				std::vector<uint64_t> wp(vl);
				for (size_t k = 0; k < vl; ++k) wp[k] = static_cast<uint64_t>((static_cast<unsigned __int128>(w[k]) << kBits) / q);
				auto tw = [&](size_t k) { return lanes ? w[k] : w[0]; };
				// forward
				std::vector<uint64_t> x = O::Random(vl, 4 * q), y = O::Random(vl, 4 * q);
				x[0] = 4 * q - 1;
				y[0] = 4 * q - 1;
				V X = vec(x), Y = vec(y);
				if constexpr (k32) {
					if (lanes) rvv::FwdButterfly32(X, Y, vec(w), vec(wp), static_cast<uint32_t>(q), vl);
					else rvv::FwdButterfly32(X, Y, static_cast<uint32_t>(w[0]), static_cast<uint32_t>(wp[0]), static_cast<uint32_t>(q), vl);
				} else {
					if (lanes) rvv::FwdButterfly(X, Y, vec(w), vec(wp), q, vl);
					else rvv::FwdButterfly(X, Y, w[0], wp[0], q, vl);
				}
				auto gx = out(X), gy = out(Y);
				for (size_t k = 0; k < vl; ++k) {
					const uint64_t t = O::MulMod(y[k] % q, tw(k), q);
					CHECK_EQ(gx[k] < 4 * q && gx[k] % q == O::AddMod(x[k] % q, t, q), true,
							<< "FwdButterfly x " << lane << " q=" << q << " k=" << k << " got " << uint64_t{gx[k]});
					CHECK_EQ(gy[k] < 4 * q && gy[k] % q == O::SubMod(x[k] % q, t, q), true,
							<< "FwdButterfly y " << lane << " q=" << q << " k=" << k << " got " << uint64_t{gy[k]});
				}
				// inverse
				x = O::Random(vl, 2 * q);
				y = O::Random(vl, 2 * q);
				x[0] = 2 * q - 1;
				y[0] = 0;
				X = vec(x);
				Y = vec(y);
				if constexpr (k32) {
					if (lanes) rvv::InvButterfly32(X, Y, vec(w), vec(wp), static_cast<uint32_t>(q), vl);
					else rvv::InvButterfly32(X, Y, static_cast<uint32_t>(w[0]), static_cast<uint32_t>(wp[0]), static_cast<uint32_t>(q), vl);
				} else {
					if (lanes) rvv::InvButterfly(X, Y, vec(w), vec(wp), q, vl);
					else rvv::InvButterfly(X, Y, w[0], wp[0], q, vl);
				}
				gx = out(X);
				gy = out(Y);
				for (size_t k = 0; k < vl; ++k) {
					CHECK_EQ(gx[k] < 2 * q && gx[k] % q == O::AddMod(x[k] % q, y[k] % q, q), true,
							<< "InvButterfly x " << lane << " q=" << q << " k=" << k << " got " << uint64_t{gx[k]});
					CHECK_EQ(gy[k] < 2 * q && gy[k] % q == O::MulMod(O::SubMod(x[k] % q, y[k] % q, q), tw(k), q), true,
							<< "InvButterfly y " << lane << " q=" << q << " k=" << k << " got " << uint64_t{gy[k]});
				}
			}
		}
	}
}

/// One NTT check, shared by every lane and strategy: the inputs and the
/// fully reduced reference output (ReferenceForward/InverseTransform*, the
/// textbook transforms that test-ntt.cpp checks against the definition).
struct NttCase {
	std::shared_ptr<const intel::hexl::NTT> ntt;
	uint64_t n, q, imf, omf;
	bool forward, in_place;
	std::vector<uint64_t> in, want;
};

/// N = 2 .. 4096 at three moduli and all mod factors; N = 16384 and 65536 at
/// the largest modulus, (1, 1) and lazy (every strategy at large N). e32 lanes: q < 2^30; e64: q up to 2^62.
const std::vector<NttCase>& NttCases(bool e32) {
	static std::vector<NttCase> cases[2];
	std::vector<NttCase>& cs = cases[e32];
	if (!cs.empty()) return cs;
	std::vector<uint64_t> sizes;
	for (uint64_t n = 2; n <= (hexltest::g_quick ? 1024 : 4096); n <<= 1) sizes.push_back(n);
	if (!hexltest::g_quick) sizes.insert(sizes.end(), {16384, 65536});
	for (uint64_t n : sizes) {
		const bool big = n > 4096;
		std::vector<uint64_t> moduli =
			e32 ? std::vector<uint64_t>{O::NttPrime(19, n), O::NttPrime(26, n), O::Primes(1, 29, false, n)[0]}
		: std::vector<uint64_t>{O::Primes(1, 30, true, n)[0], O::NttPrime(49, n), O::Primes(1, 61, false, n)[0]};
		if (big) moduli.erase(moduli.begin(), moduli.end() - 1);
		for (uint64_t q : moduli) {
			auto ntt = std::make_shared<const intel::hexl::NTT>(n, q);
			const std::vector<std::pair<uint64_t, uint64_t>> fwd_factors =
				big ? std::vector<std::pair<uint64_t, uint64_t>>{{1, 1}, {4, 4}}
			: std::vector<std::pair<uint64_t, uint64_t>>{{1, 1}, {2, 4}, {4, 1}, {4, 4}};
			const std::vector<std::pair<uint64_t, uint64_t>> inv_factors =
				big ? std::vector<std::pair<uint64_t, uint64_t>>{{1, 1}, {2, 2}}
			: std::vector<std::pair<uint64_t, uint64_t>>{{1, 1}, {2, 2}, {1, 2}, {2, 1}};
			for (bool forward : {true, false}) {
				bool in_place = true;  // alternates, starting out of place
				for (auto [imf, omf] : forward ? fwd_factors : inv_factors) {
					in_place = !in_place;
					NttCase c{ntt, n, q, imf, omf, forward, in_place, O::Random(n, imf * q), {}};
					c.in[0] = imf * q - 1;
					c.want = c.in;
					for (auto& x : c.want) x %= q;
					if (forward) {
						intel::hexl::ReferenceForwardTransformToBitReverse(c.want.data(), n, q, ntt->GetRootOfUnityPowers().data());
					} else {
						intel::hexl::ReferenceInverseTransformFromBitReverse(c.want.data(), n, q, ntt->GetInvRootOfUnityPowers().data());
					}
					cs.push_back(std::move(c));
				}
			}
		}
	}
	return cs;
}

/// The RVV32 (e32 lanes, either storage word) or RVV64 (e64 lanes) forward and
/// inverse kernels at lane type V and small-span strategy S, with S's tables
/// built for VLMAX of V. Exact for output_mod_factor 1, else congruent and
/// below the bound (4q forward, 2q inverse).
template <typename Word, class V, class S>
void CheckNtt(const char* lane, const char* strategy) {
	constexpr bool kE32 = rvv::IsE32<V>;
	static_assert(kE32 || std::is_same_v<Word, uint64_t>, "e64 lanes need uint64_t storage");
	using E = std::conditional_t<kE32, uint32_t, uint64_t>;
	const uint64_t vl = intel::hexl::SmallSpanLanes<S, V>();  // the lanes S runs at
	const intel::hexl::NTT* built = nullptr;
	intel::hexl::NttSmallSpanTables<S, E> tables;
	for (const NttCase& c : NttCases(kE32)) {
		const intel::hexl::NTT& ntt = *c.ntt;
		if (&ntt != built) {  // the tables once per (N, q)
			tables = intel::hexl::BuildNttSmallSpanTables<S, E>(ntt, vl);
			built = &ntt;
			CHECK_EQ(tables.vl, intel::hexl::SmallSpanRuns<S>(c.n, vl) ? vl : 0,
					<< "tables " << lane << " " << strategy << " n=" << c.n);
		}
		const std::vector<Word> in = AsWord<Word>(c.in);
		std::vector<Word> out = c.in_place ? in : std::vector<Word>(c.n);
		const Word* operand = c.in_place ? out.data() : in.data();
		if constexpr (kE32) {
			if (c.forward) {
				intel::hexl::ForwardTransformToBitReverseRVV32<Word, V, S>(
						out.data(), operand, c.n, c.q, ntt.GetRVV32RootOfUnityPowers().data(),
						ntt.GetRVV32PreconRootOfUnityPowers().data(), c.imf, c.omf, &tables);
			} else {
				intel::hexl::InverseTransformFromBitReverseRVV32<Word, V, S>(
						out.data(), operand, c.n, c.q, ntt.GetRVV32InvRootOfUnityPowers().data(),
						ntt.GetRVV32PreconInvRootOfUnityPowers().data(), c.imf, c.omf, &tables);
			}
		} else {
			if (c.forward) {
				intel::hexl::ForwardTransformToBitReverseRVV64<V, S>(
						out.data(), operand, c.n, c.q, ntt.GetRootOfUnityPowers().data(),
						ntt.GetPrecon64RootOfUnityPowers().data(), c.imf, c.omf, &tables);
			} else {
				intel::hexl::InverseTransformFromBitReverseRVV64<V, S>(
						out.data(), operand, c.n, c.q, ntt.GetInvRootOfUnityPowers().data(),
						ntt.GetPrecon64InvRootOfUnityPowers().data(), c.imf, c.omf, &tables);
			}
		}
		const char* dir = c.forward ? "forward" : "inverse";
		if (!c.in_place) CHECK_VEC_EQ(AsWord<Word>(c.in), in, << "operand modified " << dir << " " << lane << " " << strategy << " n=" << c.n);
		const uint64_t bound = (c.forward ? (c.omf == 1 ? 1 : 4) : c.omf) * c.q;
		for (uint64_t i = 0; i < c.n; ++i) {
			const uint64_t got = out[i];
			CHECK_EQ(got < bound && got % c.q == c.want[i], true,
					<< dir << " " << lane << " " << strategy << " n=" << c.n << " q=" << c.q << " imf=" << c.imf
					<< " omf=" << c.omf << (c.in_place ? " in place" : "") << " i=" << i << ": got " << got
					<< ", want " << c.want[i] << " (mod q), below " << bound);
		}
	}
}

/// The small-span layouts, independent of the hardware VL: a scalar emulation
/// of each strategy's small stages that reads its twiddles exactly where the
/// layout (src/ntt/ntt-rvv-small-span.hpp) says the vector body will, for VL
/// 2 .. 256, against the reference transforms. Also every Shoup factor, and,
/// for FusedSlides, that both lanes of a butterfly pair hold the same twiddle.
template <class S, typename E>
void CheckSmallSpanLayout(uint64_t n, uint64_t vl, uint64_t q, const char* name) {
	constexpr bool kFused = std::is_same_v<S, intel::hexl::rvv::ntt::FusedSlides>;
	constexpr uint64_t kT = [] {  // Transposed tile rows
		if constexpr (intel::hexl::rvv::ntt::kIsTransposed<S>) return uint64_t{S::kRows};
		else return uint64_t{0};
	}();
	constexpr unsigned kBits = sizeof(E) * 8;
	const intel::hexl::NTT ntt(n, q);
	const auto tw = intel::hexl::BuildNttSmallSpanTables<S, E>(ntt, vl);
	CHECK_EQ(tw.vl, vl, << name << " n=" << n);
	CHECK_EQ(tw.t_top, kFused ? vl : kT, << name << " t_top n=" << n << " vl=" << vl);
	auto check_precon = [&](const std::vector<E>& t, const std::vector<E>& p) {
		CHECK_EQ(t.size(), p.size(), << name << " n=" << n << " vl=" << vl);
		for (size_t k = 0; k < t.size(); ++k) {
			const E want = static_cast<E>((static_cast<unsigned __int128>(t[k]) << kBits) / q);
			CHECK_EQ(p[k], want, << name << " Shoup factor n=" << n << " vl=" << vl << " k=" << k);
		}
	};
	check_precon(tw.fwd, tw.fwd_precon);
	check_precon(tw.inv, tw.inv_precon);
	const uint64_t stages = __builtin_ctzll(vl), rows = n / vl;
	auto ct = [q](uint64_t& x, uint64_t& y, uint64_t w) {  // Cooley-Tukey, exact
		const uint64_t wy = O::MulMod(y, w, q);
		y = O::SubMod(x, wy, q);
		x = O::AddMod(x, wy, q);
	};
	auto gs = [q](uint64_t& x, uint64_t& y, uint64_t w) {  // Gentleman-Sande, exact
		const uint64_t d = O::SubMod(x, y, q);
		x = O::AddMod(x, y, q);
		y = O::MulMod(d, w, q);
	};

	// forward: large stages from the NTT table, small stages from the layout
	std::vector<uint64_t> x = O::Random(n, q), want = x;
	intel::hexl::ReferenceForwardTransformToBitReverse(want.data(), n, q, ntt.GetRootOfUnityPowers().data());
	const auto& W = ntt.GetRootOfUnityPowers();
	for (uint64_t t = n / 2; t >= tw.t_top; t >>= 1)
		for (uint64_t i = 0, m = n / (2 * t); i < m; ++i)
			for (uint64_t j = 2 * i * t; j < 2 * i * t + t; ++j) ct(x[j], x[j + t], W[m + i]);
	if constexpr (kFused) {
		for (uint64_t c = 0; c < rows; ++c)
			for (uint64_t t = vl / 2, s = 0; t >= 1; t >>= 1, ++s)
				for (uint64_t l = 0; l < vl; ++l) {
					if (l & t) continue;
					const uint64_t k = (c * stages + s) * vl + l;
					CHECK_EQ(tw.fwd[k + t], tw.fwd[k], << name << " pair twiddle n=" << n << " vl=" << vl);
					ct(x[c * vl + l], x[c * vl + l + t], tw.fwd[k]);
				}
	} else {
		for (uint64_t c = 0; c < rows / kT; ++c)  // tile by tile, the stages t < kT
			for (uint64_t t = kT / 2; t >= 1; t >>= 1)
				for (uint64_t g = 0; g < kT / (2 * t); ++g)
					for (uint64_t r = 2 * g * t; r < 2 * g * t + t; ++r)
						for (uint64_t k = 0; k < vl; ++k) {
							const uint64_t e = c * kT + r + k * rows;
							ct(x[e], x[e + t], tw.fwd[(c * (kT - 1) + kT / (2 * t) - 1 + g) * vl + k]);
						}
		CHECK_EQ(n / kT * (kT - 1), tw.fwd.size(), << name << " n=" << n << " vl=" << vl);
	}
	CHECK_VEC_EQ(want, x, << "forward " << name << " n=" << n << " vl=" << vl);

	// inverse: small stages from the layout, large stages from the NTT table, N^-1
	x = O::Random(n, q);
	want = x;
	intel::hexl::ReferenceInverseTransformFromBitReverse(want.data(), n, q, ntt.GetInvRootOfUnityPowers().data());
	if constexpr (kFused) {
		for (uint64_t c = 0; c < rows; ++c)
			for (uint64_t t = 1, s = 0; t < vl; t <<= 1, ++s)
				for (uint64_t l = 0; l < vl; ++l) {
					if (l & t) continue;
					const uint64_t k = (c * stages + s) * vl + l;
					CHECK_EQ(tw.inv[k + t], tw.inv[k], << name << " pair twiddle n=" << n << " vl=" << vl);
					gs(x[c * vl + l], x[c * vl + l + t], tw.inv[k]);
				}
	} else {
		for (uint64_t c = 0; c < rows / kT; ++c)
			for (uint64_t t = 1; t < kT; t <<= 1)
				for (uint64_t g = 0; g < kT / (2 * t); ++g)
					for (uint64_t r = 2 * g * t; r < 2 * g * t + t; ++r)
						for (uint64_t k = 0; k < vl; ++k) {
							const uint64_t e = c * kT + r + k * rows;
							gs(x[e], x[e + t], tw.inv[(c * (kT - 1) + kT - kT / t + g) * vl + k]);
						}
		CHECK_EQ(n / kT * (kT - 1), tw.inv.size(), << name << " n=" << n << " vl=" << vl);
	}
	const auto& Wi = ntt.GetInvRootOfUnityPowers();
	for (uint64_t t = tw.t_top; t <= n / 2; t <<= 1)
		for (uint64_t i = 0, m = n / (2 * t); i < m; ++i)
			for (uint64_t j = 2 * i * t; j < 2 * i * t + t; ++j) gs(x[j], x[j + t], Wi[n - 2 * m + 1 + i]);
	const uint64_t n_inv = O::InvModPrime(n, q);
	for (auto& v : x) v = O::MulMod(v, n_inv, q);
	CHECK_VEC_EQ(want, x, << "inverse " << name << " n=" << n << " vl=" << vl);
}

template <class S>
void CheckSmallSpanLayouts(const char* name) {
	for (uint64_t vl = 2; vl <= 256; vl <<= 1) {
		for (uint64_t n = 2; n <= (hexltest::g_quick ? 1024 : 65536); n <<= 1) {
			if (!intel::hexl::SmallSpanRuns<S>(n, vl) || (n > 4096 && n != vl * vl)) continue;
			CheckSmallSpanLayout<S, uint32_t>(n, vl, O::NttPrime(26, n), name);
			CheckSmallSpanLayout<S, uint64_t>(n, vl, O::Primes(1, 61, false, n)[0], name);
		}
	}
}

}  // namespace

// The lane list once, for every kernel test below.
#define HEXL_FOR_EACH_LANE(CHECK)                    \
	CHECK<uint64_t, vuint64m1_t>("u64m1");             \
	CHECK<uint64_t, vuint64m2_t>("u64m2");             \
	CHECK<uint64_t, vuint64m4_t>("u64m4");             \
	CHECK<uint64_t, vuint64m8_t>("u64m8");             \
	CHECK<uint32_t, vuint32m1_t>("u32m1");             \
	CHECK<uint32_t, vuint32m2_t>("u32m2");             \
	CHECK<uint32_t, vuint32m4_t>("u32m4");             \
	CHECK<uint32_t, vuint32m8_t>("u32m8")

TEST(RvvLanes_SetVlLoadStore) {
	CheckSetVlLoadStore<vuint64m1_t, uint64_t>("u64m1", __riscv_vsetvlmax_e64m1());
	CheckSetVlLoadStore<vuint64m2_t, uint64_t>("u64m2", __riscv_vsetvlmax_e64m2());
	CheckSetVlLoadStore<vuint64m4_t, uint64_t>("u64m4", __riscv_vsetvlmax_e64m4());
	CheckSetVlLoadStore<vuint64m8_t, uint64_t>("u64m8", __riscv_vsetvlmax_e64m8());
	CheckSetVlLoadStore<vuint32mf2_t, uint32_t>("u32mf2", __riscv_vsetvlmax_e32mf2());
	CheckSetVlLoadStore<vuint32m1_t, uint32_t>("u32m1", __riscv_vsetvlmax_e32m1());
	CheckSetVlLoadStore<vuint32m2_t, uint32_t>("u32m2", __riscv_vsetvlmax_e32m2());
	CheckSetVlLoadStore<vuint32m4_t, uint32_t>("u32m4", __riscv_vsetvlmax_e32m4());
	CheckSetVlLoadStore<vuint32m8_t, uint32_t>("u32m8", __riscv_vsetvlmax_e32m8());
}

TEST(RvvLanes_LoadStoreStrided) {
	CheckLoadStoreStrided<vuint64m1_t, uint64_t>("u64m1");
	CheckLoadStoreStrided<vuint64m2_t, uint64_t>("u64m2");
	CheckLoadStoreStrided<vuint64m4_t, uint64_t>("u64m4");
	CheckLoadStoreStrided<vuint64m8_t, uint64_t>("u64m8");
	CheckLoadStoreStrided<vuint32mf2_t, uint32_t>("u32mf2");
	CheckLoadStoreStrided<vuint32m1_t, uint32_t>("u32m1");
	CheckLoadStoreStrided<vuint32m2_t, uint32_t>("u32m2");
	CheckLoadStoreStrided<vuint32m4_t, uint32_t>("u32m4");
	CheckLoadStoreStrided<vuint32m8_t, uint32_t>("u32m8");
	CheckLoadStoreStrided<vuint32mf2_t, uint64_t>("u64->u32mf2");
	CheckLoadStoreStrided<vuint32m1_t, uint64_t>("u64->u32m1");
	CheckLoadStoreStrided<vuint32m2_t, uint64_t>("u64->u32m2");
	CheckLoadStoreStrided<vuint32m4_t, uint64_t>("u64->u32m4");
}

TEST(RvvLanes_Butterflies) { HEXL_FOR_EACH_LANE(CheckButterflies); }

TEST(RvvLanes_AddSubMod) { HEXL_FOR_EACH_LANE(CheckAddSub); }
TEST(RvvLanes_CmpAdd) { HEXL_FOR_EACH_LANE(CheckCmpAdd); }
TEST(RvvLanes_CmpSubMod) { HEXL_FOR_EACH_LANE(CheckCmpSubMod); }
TEST(RvvLanes_MultMod) {
	HEXL_FOR_EACH_LANE(CheckMultMod);
	// EltwiseMultModRVV32 on uint64_t storage (narrowing load): u32m1..m4
	CheckMultMod<uint64_t, vuint32m1_t>("u64->u32m1");
	CheckMultMod<uint64_t, vuint32m2_t>("u64->u32m2");
	CheckMultMod<uint64_t, vuint32m4_t>("u64->u32m4");
}

TEST(RvvLanes_MultAddMod) {
	HEXL_FOR_EACH_LANE(CheckMultAddMod);
	// RVV32 on uint64_t storage (narrowing load): u32m1..m4
	CheckMultAddMod<uint64_t, vuint32m1_t>("u64->u32m1");
	CheckMultAddMod<uint64_t, vuint32m2_t>("u64->u32m2");
	CheckMultAddMod<uint64_t, vuint32m4_t>("u64->u32m4");
}

// The NTT lanes: RVV64 at u64m1..m8, RVV32 on uint32_t storage at u32m1..m8
// and on uint64_t storage at u32m1..m4.
#define HEXL_FOR_EACH_NTT_LANE(S, NAME)                       \
	CheckNtt<uint64_t, vuint64m1_t, S>("u64m1", NAME);          \
	CheckNtt<uint64_t, vuint64m2_t, S>("u64m2", NAME);          \
	CheckNtt<uint64_t, vuint64m4_t, S>("u64m4", NAME);          \
	CheckNtt<uint64_t, vuint64m8_t, S>("u64m8", NAME);          \
	CheckNtt<uint32_t, vuint32m1_t, S>("u32m1", NAME);          \
	CheckNtt<uint32_t, vuint32m2_t, S>("u32m2", NAME);          \
	CheckNtt<uint32_t, vuint32m4_t, S>("u32m4", NAME);          \
	CheckNtt<uint32_t, vuint32m8_t, S>("u32m8", NAME);          \
	CheckNtt<uint64_t, vuint32m1_t, S>("u64->u32m1", NAME);     \
	CheckNtt<uint64_t, vuint32m2_t, S>("u64->u32m2", NAME);     \
	CheckNtt<uint64_t, vuint32m4_t, S>("u64->u32m4", NAME)

TEST(RvvLanes_NttSmallSpanLayouts) {
	CheckSmallSpanLayouts<intel::hexl::rvv::ntt::FusedSlides>("FusedSlides");
#define HEXL_TRANSPOSED_LAYOUT(S) CheckSmallSpanLayouts<intel::hexl::rvv::ntt::S>(#S);
	HEXL_RVV_NTT_TRANSPOSED_SHAPES(HEXL_TRANSPOSED_LAYOUT);
#undef HEXL_TRANSPOSED_LAYOUT
}

TEST(RvvLanes_NttPerBlock) { HEXL_FOR_EACH_NTT_LANE(intel::hexl::rvv::ntt::PerBlock, "PerBlock"); }
TEST(RvvLanes_NttFusedSlides) { HEXL_FOR_EACH_NTT_LANE(intel::hexl::rvv::ntt::FusedSlides, "FusedSlides"); }
#define HEXL_TRANSPOSED_TEST(S) \
	TEST(RvvLanes_Ntt##S) { HEXL_FOR_EACH_NTT_LANE(intel::hexl::rvv::ntt::S, #S); }
HEXL_RVV_NTT_TRANSPOSED_SHAPES(HEXL_TRANSPOSED_TEST);
#undef HEXL_TRANSPOSED_TEST

#endif  // HEXL_HAS_RVV
