// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// The finished RVV kernels at every lane type (LMUL m1/m2/m4/m8, both storage
// words), checked against the __int128 oracle. The public API runs only the
// default lanes of src/util/rvv-config.hpp; these tests make sure every other
// instantiation (the ones bench-hexl BM_Lanes_* times) is correct too.
// Also rvv::SetVl / Load / Store at every LMUL.
//
//   hexl-tests RvvLanes
//
// Internal headers: compiled only into RVV builds (needs -Isrc, which the
// Makefile adds; the tests are empty for ISA=scalar).

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "oracle.hpp"
#include "test.hpp"

#if defined(__has_include)
#if __has_include("util/rvv-util.hpp")
#include "eltwise/eltwise-add-mod-internal.hpp"
#include "eltwise/eltwise-cmp-add-internal.hpp"
#include "eltwise/eltwise-cmp-sub-mod-internal.hpp"
#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "eltwise/eltwise-sub-mod-internal.hpp"
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

#endif  // HEXL_HAS_RVV
