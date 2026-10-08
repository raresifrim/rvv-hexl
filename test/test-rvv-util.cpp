// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// src/util/rvv-util.hpp: one test per helper (each overload separately), so a
// helper can be checked the moment it is written, before any kernel uses it.
//
//   hexl-tests RvvUtil                  all of them
//   hexl-tests RvvUtil_MulModShoup      one family
//
// Every test runs the helper strip by strip (vsetvl per strip, so VLEN-agnostic)
// over the same size sweep as the eltwise tests (every tail at VLEN 128..1024),
// feeds it through plain vle/vse intrinsics (never through another helper, so a
// failure points at exactly one function), and compares with the __int128
// oracle. Edge values (q - 1, 0, the largest allowed input) sit at the front of
// every vector. Lazy helpers are checked for both the residue and the range.
//
// Internal header: compiled only into RVV builds (needs -Isrc, which the
// Makefile adds; the tests are empty for ISA=scalar).

#include <cstdint>
#include <type_traits>
#include <vector>

#include "hexl/number-theory/number-theory.hpp"
#include "oracle.hpp"
#include "test.hpp"

#if defined(__has_include)
#if __has_include("util/rvv-util.hpp")
#include "util/rvv-util.hpp"
#endif
#endif

#ifdef HEXL_HAS_RVV

namespace O = hexltest::oracle;
namespace rvv = intel::hexl::rvv;
using u128 = unsigned __int128;

namespace {

// ===========================================================================
// Barrett factor conventions. The two MulModBarrett* signatures take
// precomputed factors whose exact definition is the implementer's choice (the
// kernels that call them compute the factors too). EDIT THESE TWO FUNCTIONS if
// your implementation expects a different convention; nothing else depends on
// them.
// ===========================================================================

/// e64: upstream HEXL's pre-shift Barrett (EltwiseMultModNative), q < 2^61.
/// With n = bit length of q: shift = n - 2 and mu = floor(2^(n+62) / q), i.e.
/// MultiplyFactor(1 << shift, 64, q).BarrettFactor(), which is what the kernel
/// calls (checked in RvvUtil_MulModBarrett). Computed here independently.
inline void Barrett64Factors(uint64_t q, uint64_t* mu, uint64_t* shift) {
	const int n = 64 - __builtin_clzll(q);
	*shift = static_cast<uint64_t>(n - 2);
	*mu = static_cast<uint64_t>((static_cast<u128>(1) << (n + 62)) / q);
}

/// e32: the same pre-shift Barrett at 32-bit words, q < 2^30. With n = bit
/// length of q: shift = n - 2 and mu = floor(2^(n+30) / q) (< 2^32), i.e.
/// MultiplyFactor(1 << shift, 32, q).BarrettFactor() (checked in the test).
inline void Barrett32Factors(uint64_t q, uint32_t* mu, uint32_t* shift) {
	const int n = 64 - __builtin_clzll(q);
	*shift = static_cast<uint32_t>(n - 2);
	*mu = static_cast<uint32_t>((uint64_t{1} << (n + 30)) / q);
}

// ---------------------------------------------------------------------------

/// Shoup precomputation: floor(y * 2^bits / q), y < q.
inline uint64_t ShoupPrecon(uint64_t y, uint64_t q, int bits) {
	return static_cast<uint64_t>((static_cast<u128>(y) << bits) / q);
}

/// Moduli for the e64 helpers: every size class up to the header's contract.
std::vector<uint64_t> Moduli64(uint64_t max_exclusive) {
	std::vector<uint64_t> qs = {2, 3, 17};
	for (size_t bits : {20, 27, 29, 31, 32, 40, 49, 52, 59, 60, 61}) qs.push_back(O::NttPrime(bits, 1));
	qs.push_back(O::PrimeBelow(1ULL << 62));
	qs.push_back(O::PrimeBelow(1ULL << 63));
	std::vector<uint64_t> out;
	for (uint64_t q : qs) {
		if (q < max_exclusive) out.push_back(q);
	}
	return out;
}

/// Moduli for the e32 helpers: q < 2^30 (the e32 dispatch bound).
std::vector<uint64_t> Moduli32() {
	std::vector<uint64_t> qs = {2, 3, 17};
	for (size_t bits : {12, 20, 27, 28, 29}) qs.push_back(O::NttPrime(bits, 1));
	qs.push_back(O::PrimeBelow(1ULL << 30));
	return qs;
}

/// n values in [0, bound) with edge values at the front.
std::vector<uint64_t> WithEdges(size_t n, uint64_t bound, std::initializer_list<uint64_t> edges) {
	auto v = O::Random(n, bound);
	size_t i = 0;
	for (uint64_t e : edges) {
		if (i < n) v[i++] = e;
	}
	return v;
}

/// f(offset, vl) for each strip of n elements at the given element width.
template <class F>
void Strips64(size_t n, F f) {
	for (size_t i = 0, vl; i < n; i += vl) {
		vl = __riscv_vsetvl_e64m1(n - i);
		f(i, vl);
	}
}
template <class F>
void Strips32(size_t n, F f) {
	for (size_t i = 0, vl; i < n; i += vl) {
		vl = __riscv_vsetvl_e32m1(n - i);
		f(i, vl);
	}
}

std::vector<uint32_t> To32(const std::vector<uint64_t>& v) { return O::As<uint32_t>(v); }

}  // namespace

// ===========================================================================
// e64 helpers
// ===========================================================================

TEST(RvvUtil_AddMod) {
	for (uint64_t q : Moduli64(1ULL << 63)) {
		for (size_t n : O::EltwiseSizes()) {
			auto a = WithEdges(n, q, {q - 1, 0, q - 1}), b = WithEdges(n, q, {q - 1, 0, 0});
			std::vector<uint64_t> want(n), got(n);
			for (size_t i = 0; i < n; ++i) want[i] = O::AddMod(a[i], b[i], q);
			Strips64(n, [&](size_t i, size_t vl) {
					auto r = rvv::AddMod(__riscv_vle64_v_u64m1(&a[i], vl), __riscv_vle64_v_u64m1(&b[i], vl), q, vl);
					__riscv_vse64_v_u64m1(&got[i], r, vl);
					});
			CHECK_VEC_EQ(want, got, << "rvv::AddMod n=" << n << " q=" << q);
		}
	}
}

TEST(RvvUtil_SubMod) {
	for (uint64_t q : Moduli64(1ULL << 63)) {
		for (size_t n : O::EltwiseSizes()) {
			auto a = WithEdges(n, q, {0, q - 1, 0}), b = WithEdges(n, q, {q - 1, 0, 0});
			std::vector<uint64_t> want(n), got(n);
			for (size_t i = 0; i < n; ++i) want[i] = O::SubMod(a[i], b[i], q);
			Strips64(n, [&](size_t i, size_t vl) {
					auto r = rvv::SubMod(__riscv_vle64_v_u64m1(&a[i], vl), __riscv_vle64_v_u64m1(&b[i], vl), q, vl);
					__riscv_vse64_v_u64m1(&got[i], r, vl);
					});
			CHECK_VEC_EQ(want, got, << "rvv::SubMod n=" << n << " q=" << q);
		}
	}
}

TEST(RvvUtil_ReduceFromTwice) {
	for (uint64_t q : Moduli64(1ULL << 63)) {
		for (size_t n : O::EltwiseSizes()) {
			auto x = WithEdges(n, 2 * q, {2 * q - 1, q, q - 1, 0});  // x in [0, 2q)
			std::vector<uint64_t> want(n), got(n);
			for (size_t i = 0; i < n; ++i) want[i] = x[i] % q;
			Strips64(n, [&](size_t i, size_t vl) {
					__riscv_vse64_v_u64m1(&got[i], rvv::ReduceFromTwice(__riscv_vle64_v_u64m1(&x[i], vl), q, vl), vl);
					});
			CHECK_VEC_EQ(want, got, << "rvv::ReduceFromTwice n=" << n << " q=" << q);
		}
	}
}

// Barrett: x mod q for ANY 64-bit x, at m1 and m4 (the helper is LMUL-generic),
// both output ranges. Also pins the factor convention to MultiplyFactor, which
// is what the kernels will call.
namespace {
/// The full 64-bit input range, edge values first (only those that don't wrap).
std::vector<uint64_t> BarrettInputs64(size_t n, uint64_t q) {
	const uint64_t top = ~0ULL, kq = top / q * q;  // kq: largest multiple of q
	std::vector<uint64_t> edges = {0, 1, q - 1, q, top, top - 1, kq, kq - 1};
	if (q < top) edges.push_back(q + 1);
	if (q <= top / 2) {
		edges.push_back(2 * q - 1);
		edges.push_back(2 * q);
	}
	auto v = O::Random(n, top);
	for (size_t i = 0; i < edges.size() && i < n; ++i) v[i] = edges[i];
	return v;
}

template <int OutputModFactor>
std::vector<uint64_t> RunBarrett64(const std::vector<uint64_t>& x, uint64_t q, uint64_t q_barr, bool m4) {
	const size_t n = x.size();
	std::vector<uint64_t> got(n);
	for (size_t i = 0, vl; i < n; i += vl) {
		if (m4) {
			vl = __riscv_vsetvl_e64m4(n - i);
			__riscv_vse64_v_u64m4(&got[i], rvv::BarrettReduce<OutputModFactor>(__riscv_vle64_v_u64m4(&x[i], vl), q, q_barr, vl), vl);
		} else {
			vl = __riscv_vsetvl_e64m1(n - i);
			__riscv_vse64_v_u64m1(&got[i], rvv::BarrettReduce<OutputModFactor>(__riscv_vle64_v_u64m1(&x[i], vl), q, q_barr, vl), vl);
		}
	}
	return got;
}
}  // namespace

TEST(RvvUtil_BarrettReduce) {
	// every helper modulus, plus q >= 2^63 (Barrett needs no bound on q) and a non-prime
	std::vector<uint64_t> qs = Moduli64(~0ULL);
	for (uint64_t q : {1000ULL, 1ULL << 63, 0xFFFFFFFFFFFFFFC5ULL}) qs.push_back(q);
	for (uint64_t q : qs) {
		const uint64_t q_barr = static_cast<uint64_t>((static_cast<u128>(1) << 64) / q);  // floor(2^64/q)
		CHECK_EQ(intel::hexl::MultiplyFactor(1, 64, q).BarrettFactor(), q_barr, << "factor convention q=" << q);
		for (size_t n : O::EltwiseSizes()) {
			const auto x = BarrettInputs64(n, q);
			std::vector<uint64_t> want(n);
			for (size_t i = 0; i < n; ++i) want[i] = x[i] % q;
			for (bool m4 : {false, true}) {
				const char* lmul = m4 ? "u64m4" : "u64m1";
				CHECK_VEC_EQ(want, RunBarrett64<1>(x, q, q_barr, m4),
						<< "rvv::BarrettReduce<1> " << lmul << " n=" << n << " q=" << q);
				const auto lazy = RunBarrett64<2>(x, q, q_barr, m4);
				for (size_t i = 0; i < n; ++i) {
					CHECK_EQ(lazy[i] % q, want[i], << "rvv::BarrettReduce<2> " << lmul << " residue, x=" << x[i] << " q=" << q);
					CHECK_EQ(static_cast<u128>(lazy[i]) < 2 * static_cast<u128>(q), true,
							<< "rvv::BarrettReduce<2> " << lmul << " range [0, 2q), x=" << x[i] << " got=" << lazy[i] << " q=" << q);
				}
			}
		}
	}
}

// Shoup lazy: r = x*y mod q, r in [0, 2q), for ANY 64-bit x (the NTT feeds [0, 4q)).
namespace {
void CheckShoupLazy64(const std::vector<uint64_t>& x, const std::vector<uint64_t>& y,
		const std::vector<uint64_t>& got, uint64_t q, const char* what) {
	for (size_t i = 0; i < x.size(); ++i) {
		CHECK_EQ(got[i] % q, O::MulMod(x[i] % q, y[i], q),
				<< what << " residue at [" << i << "] x=" << x[i] << " y=" << y[i] << " q=" << q);
		CHECK(got[i] < 2 * q);
	}
}
}  // namespace

TEST(RvvUtil_MulModShoupLazy) {
	for (uint64_t q : Moduli64(1ULL << 63)) {
		for (size_t n : O::EltwiseSizes()) {
			const uint64_t x4q = (q < (1ULL << 62)) ? 4 * q - 1 : ~0ULL;
			auto x = WithEdges(n, O::AnyWord<uint64_t>(), {~0ULL, x4q, q - 1, 0});
			auto y = WithEdges(n, q, {q - 1, q - 1, q - 1, q - 1});
			std::vector<uint64_t> yp(n), got(n);
			for (size_t i = 0; i < n; ++i) yp[i] = ShoupPrecon(y[i], q, 64);
			Strips64(n, [&](size_t i, size_t vl) {
					auto r = rvv::MulModShoupLazy(__riscv_vle64_v_u64m1(&x[i], vl), __riscv_vle64_v_u64m1(&y[i], vl),
							__riscv_vle64_v_u64m1(&yp[i], vl), q, vl);
					__riscv_vse64_v_u64m1(&got[i], r, vl);
					});
			CheckShoupLazy64(x, y, got, q, "rvv::MulModShoupLazy (vector y)");
		}
	}
}

TEST(RvvUtil_MulModShoupLazyScalar) {
	for (uint64_t q : Moduli64(1ULL << 63)) {
		for (uint64_t ys : {q - 1, uint64_t{1}, O::Random(1, q)[0]}) {
			const uint64_t yp = ShoupPrecon(ys, q, 64);
			for (size_t n : O::EltwiseSizes()) {
				auto x = WithEdges(n, O::AnyWord<uint64_t>(), {~0ULL, q - 1, 0});
				std::vector<uint64_t> y(n, ys), got(n);
				Strips64(n, [&](size_t i, size_t vl) {
						__riscv_vse64_v_u64m1(&got[i], rvv::MulModShoupLazy(__riscv_vle64_v_u64m1(&x[i], vl), ys, yp, q, vl), vl);
						});
				CheckShoupLazy64(x, y, got, q, "rvv::MulModShoupLazy (scalar y)");
			}
		}
	}
}

// Fused Shoup multiply-add: (x*w + y) mod q for ANY x, scalar w < q and y in
// [0, InputModFactor * q), at every InputModFactor, at m1 and m4.
namespace {
template <int Imf>
	void CheckMulAddModShoup(uint64_t q) {
		for (uint64_t w : {q - 1, uint64_t{1}, uint64_t{0}, O::Random(1, q)[0]}) {
			const uint64_t wp = ShoupPrecon(w, q, 64);
			for (size_t n : O::EltwiseSizes()) {
				auto x = WithEdges(n, O::AnyWord<uint64_t>(), {~0ULL, ~0ULL, q - 1, 0});
				auto y = WithEdges(n, Imf * q, {Imf * q - 1, 0, Imf * q - 1, q});
				std::vector<uint64_t> want(n), got1(n), got4(n);
				for (size_t i = 0; i < n; ++i) want[i] = (O::MulMod(x[i] % q, w, q) + y[i] % q) % q;
				Strips64(n, [&](size_t i, size_t vl) {
						auto r = rvv::MulAddModShoup<Imf>(__riscv_vle64_v_u64m1(&x[i], vl), w, wp, __riscv_vle64_v_u64m1(&y[i], vl), q, vl);
						__riscv_vse64_v_u64m1(&got1[i], r, vl);
						});
				for (size_t i = 0, vl; i < n; i += vl) {
					vl = __riscv_vsetvl_e64m4(n - i);
					auto r = rvv::MulAddModShoup<Imf>(__riscv_vle64_v_u64m4(&x[i], vl), w, wp, __riscv_vle64_v_u64m4(&y[i], vl), q, vl);
					__riscv_vse64_v_u64m4(&got4[i], r, vl);
				}
				CHECK_VEC_EQ(want, got1, << "rvv::MulAddModShoup<" << Imf << "><u64m1> n=" << n << " q=" << q << " w=" << w);
				CHECK_VEC_EQ(want, got4, << "rvv::MulAddModShoup<" << Imf << "><u64m4> n=" << n << " q=" << q << " w=" << w);
			}
		}
	}
}  // namespace

TEST(RvvUtil_MulAddModShoup) {
	for (uint64_t q : Moduli64(1ULL << 61)) {  // header: q < 2^61, so 8q fits
		CheckMulAddModShoup<1>(q);
		CheckMulAddModShoup<2>(q);
		CheckMulAddModShoup<4>(q);
		CheckMulAddModShoup<8>(q);
	}
}

TEST(RvvUtil_MulModBarrett) {
	for (uint64_t q : Moduli64(1ULL << 61)) {  // header: e64 path, q < 2^61
		uint64_t mu, shift;
		Barrett64Factors(q, &mu, &shift);
		CHECK_EQ(intel::hexl::MultiplyFactor(1ULL << shift, 64, q).BarrettFactor(), mu,
				<< "factor convention q=" << q);
		for (size_t n : O::EltwiseSizes()) {
			auto a = WithEdges(n, q, {q - 1, 0, q - 1}), b = WithEdges(n, q, {q - 1, q - 1, 0});
			std::vector<uint64_t> want(n), got(n);
			for (size_t i = 0; i < n; ++i) want[i] = O::MulMod(a[i], b[i], q);
			Strips64(n, [&](size_t i, size_t vl) {
					auto r = rvv::MulModBarrett(__riscv_vle64_v_u64m1(&a[i], vl), __riscv_vle64_v_u64m1(&b[i], vl), q, mu,
							shift, vl);
					__riscv_vse64_v_u64m1(&got[i], r, vl);
					});
			CHECK_VEC_EQ(want, got, << "rvv::MulModBarrett n=" << n << " q=" << q
					<< " (factors: see Barrett64Factors in this file)");
		}
	}
}

// ===========================================================================
// e32 helpers: storage conversion
// ===========================================================================

TEST(RvvUtil_Load32_FromUint64) {  // narrow on load: e64 storage, values < 2^32
	for (size_t n : O::EltwiseSizes()) {
		auto src = WithEdges(n, 1ULL << 32, {(1ULL << 32) - 1, 0, 1});
		std::vector<uint32_t> got(n);
		Strips32(n, [&](size_t i, size_t vl) { __riscv_vse32_v_u32m1(&got[i], rvv::Load32(&src[i], vl), vl); });
		CHECK_VEC_EQ(To32(src), got, << "rvv::Load32(const uint64_t*) n=" << n);
	}
}

TEST(RvvUtil_Load32_FromUint32) {
	for (size_t n : O::EltwiseSizes()) {
		auto src = To32(WithEdges(n, 1ULL << 32, {(1ULL << 32) - 1, 0, 1}));
		std::vector<uint32_t> got(n);
		Strips32(n, [&](size_t i, size_t vl) { __riscv_vse32_v_u32m1(&got[i], rvv::Load32(&src[i], vl), vl); });
		CHECK_VEC_EQ(src, got, << "rvv::Load32(const uint32_t*) n=" << n);
	}
}

// Stores must zero-extend (upper 32 bits cleared) and must not write past vl:
// the destination is pre-filled with a sentinel and has guard elements at the end.
TEST(RvvUtil_Store32_ToUint64) {
	const uint64_t kSentinel = 0xA5A5A5A5A5A5A5A5ULL;
	for (size_t n : O::EltwiseSizes()) {
		auto v = To32(WithEdges(n, 1ULL << 32, {(1ULL << 32) - 1, 0, 1}));
		std::vector<uint64_t> dst(n + 8, kSentinel), want(n + 8, kSentinel);
		for (size_t i = 0; i < n; ++i) want[i] = v[i];
		Strips32(n, [&](size_t i, size_t vl) { rvv::Store32(&dst[i], __riscv_vle32_v_u32m1(&v[i], vl), vl); });
		CHECK_VEC_EQ(want, dst, << "rvv::Store32(uint64_t*) n=" << n << " (incl. 8 guard words past n)");
	}
}

TEST(RvvUtil_Store32_ToUint32) {
	const uint32_t kSentinel = 0xA5A5A5A5U;
	for (size_t n : O::EltwiseSizes()) {
		auto v = To32(WithEdges(n, 1ULL << 32, {(1ULL << 32) - 1, 0, 1}));
		std::vector<uint32_t> dst(n + 8, kSentinel), want(n + 8, kSentinel);
		for (size_t i = 0; i < n; ++i) want[i] = v[i];
		Strips32(n, [&](size_t i, size_t vl) { rvv::Store32(&dst[i], __riscv_vle32_v_u32m1(&v[i], vl), vl); });
		CHECK_VEC_EQ(want, dst, << "rvv::Store32(uint32_t*) n=" << n << " (incl. 8 guard words past n)");
	}
}

// ===========================================================================
// e32 helpers: arithmetic (q < 2^30)
// ===========================================================================

TEST(RvvUtil_AddMod32) {
	for (uint64_t q : Moduli32()) {
		for (size_t n : O::EltwiseSizes()) {
			auto a = To32(WithEdges(n, q, {q - 1, 0, q - 1})), b = To32(WithEdges(n, q, {q - 1, 0, 0}));
			std::vector<uint32_t> want(n), got(n);
			for (size_t i = 0; i < n; ++i) want[i] = static_cast<uint32_t>(O::AddMod(a[i], b[i], q));
			Strips32(n, [&](size_t i, size_t vl) {
					auto r = rvv::AddMod32(__riscv_vle32_v_u32m1(&a[i], vl), __riscv_vle32_v_u32m1(&b[i], vl),
							static_cast<uint32_t>(q), vl);
					__riscv_vse32_v_u32m1(&got[i], r, vl);
					});
			CHECK_VEC_EQ(want, got, << "rvv::AddMod32 n=" << n << " q=" << q);
		}
	}
}

TEST(RvvUtil_SubMod32) {
	for (uint64_t q : Moduli32()) {
		for (size_t n : O::EltwiseSizes()) {
			auto a = To32(WithEdges(n, q, {0, q - 1, 0})), b = To32(WithEdges(n, q, {q - 1, 0, 0}));
			std::vector<uint32_t> want(n), got(n);
			for (size_t i = 0; i < n; ++i) want[i] = static_cast<uint32_t>(O::SubMod(a[i], b[i], q));
			Strips32(n, [&](size_t i, size_t vl) {
					auto r = rvv::SubMod32(__riscv_vle32_v_u32m1(&a[i], vl), __riscv_vle32_v_u32m1(&b[i], vl),
							static_cast<uint32_t>(q), vl);
					__riscv_vse32_v_u32m1(&got[i], r, vl);
					});
			CHECK_VEC_EQ(want, got, << "rvv::SubMod32 n=" << n << " q=" << q);
		}
	}
}

TEST(RvvUtil_ReduceFromTwice32) {
	for (uint64_t q : Moduli32()) {
		for (size_t n : O::EltwiseSizes()) {
			auto x = To32(WithEdges(n, 2 * q, {2 * q - 1, q, q - 1, 0}));  // x in [0, 2q)
			std::vector<uint32_t> want(n), got(n);
			for (size_t i = 0; i < n; ++i) want[i] = static_cast<uint32_t>(x[i] % q);
			Strips32(n, [&](size_t i, size_t vl) {
					__riscv_vse32_v_u32m1(&got[i], rvv::ReduceFromTwice32(__riscv_vle32_v_u32m1(&x[i], vl), static_cast<uint32_t>(q), vl), vl);
					});
			CHECK_VEC_EQ(want, got, << "rvv::ReduceFromTwice32 n=" << n << " q=" << q);
		}
	}
}

// Barrett, 32-bit lanes: x mod q for ANY 32-bit x, at m1 and m4, both output ranges.
namespace {
std::vector<uint32_t> BarrettInputs32(size_t n, uint64_t q) {
	const uint64_t top = 0xFFFFFFFFULL, kq = top / q * q;
	std::vector<uint64_t> edges = {0, 1, q - 1, q, top, top - 1, kq, kq - 1};
	if (q < top) edges.push_back(q + 1);
	if (2 * q <= top) {
		edges.push_back(2 * q - 1);
		edges.push_back(2 * q);
	}
	auto v = O::Random(n, top + 1);  // [0, 2^32)
	for (size_t i = 0; i < edges.size() && i < n; ++i) v[i] = edges[i];
	return To32(v);
}

template <int OutputModFactor>
std::vector<uint32_t> RunBarrett32(const std::vector<uint32_t>& x, uint32_t q, uint32_t q_barr, bool m4) {
	const size_t n = x.size();
	std::vector<uint32_t> got(n);
	for (size_t i = 0, vl; i < n; i += vl) {
		if (m4) {
			vl = __riscv_vsetvl_e32m4(n - i);
			__riscv_vse32_v_u32m4(&got[i], rvv::BarrettReduce32<OutputModFactor>(__riscv_vle32_v_u32m4(&x[i], vl), q, q_barr, vl), vl);
		} else {
			vl = __riscv_vsetvl_e32m1(n - i);
			__riscv_vse32_v_u32m1(&got[i], rvv::BarrettReduce32<OutputModFactor>(__riscv_vle32_v_u32m1(&x[i], vl), q, q_barr, vl), vl);
		}
	}
	return got;
}
}  // namespace

TEST(RvvUtil_BarrettReduce32) {
	// the e32 dispatch range (q < 2^30), plus larger 32-bit moduli and a non-prime
	std::vector<uint64_t> qs = Moduli32();
	for (uint64_t q : std::vector<uint64_t>{1000, 1ULL << 31, O::PrimeBelow(1ULL << 32)}) qs.push_back(q);
	for (uint64_t q : qs) {
		const uint32_t q32 = static_cast<uint32_t>(q);
		const uint32_t q_barr = static_cast<uint32_t>((1ULL << 32) / q);  // floor(2^32/q)
		CHECK_EQ(intel::hexl::MultiplyFactor(1, 32, q).BarrettFactor(), static_cast<uint64_t>(q_barr),
				<< "factor convention q=" << q);
		for (size_t n : O::EltwiseSizes()) {
			const auto x = BarrettInputs32(n, q);
			std::vector<uint32_t> want(n);
			for (size_t i = 0; i < n; ++i) want[i] = static_cast<uint32_t>(x[i] % q);
			for (bool m4 : {false, true}) {
				const char* lmul = m4 ? "u32m4" : "u32m1";
				CHECK_VEC_EQ(want, RunBarrett32<1>(x, q32, q_barr, m4),
						<< "rvv::BarrettReduce32<1> " << lmul << " n=" << n << " q=" << q);
				const auto lazy = RunBarrett32<2>(x, q32, q_barr, m4);
				for (size_t i = 0; i < n; ++i) {
					CHECK_EQ(lazy[i] % q32, want[i], << "rvv::BarrettReduce32<2> " << lmul << " residue, x=" << x[i] << " q=" << q);
					CHECK_EQ(static_cast<uint64_t>(lazy[i]) < 2 * q, true,
							<< "rvv::BarrettReduce32<2> " << lmul << " range [0, 2q), x=" << x[i] << " got=" << lazy[i] << " q=" << q);
				}
			}
		}
	}
}

// Shoup lazy, 32-bit lanes: r = x*y mod q in [0, 2q) for any 32-bit x.
TEST(RvvUtil_MulModShoupLazy32) {
	for (uint64_t q : Moduli32()) {
		for (size_t n : O::EltwiseSizes()) {
			auto x = To32(WithEdges(n, 1ULL << 32, {(1ULL << 32) - 1, 4 * q - 1, q - 1, 0}));
			auto y = To32(WithEdges(n, q, {q - 1, q - 1, q - 1, q - 1}));
			std::vector<uint32_t> yp(n), got(n);
			for (size_t i = 0; i < n; ++i) yp[i] = static_cast<uint32_t>(ShoupPrecon(y[i], q, 32));
			Strips32(n, [&](size_t i, size_t vl) {
					auto r = rvv::MulModShoupLazy32(__riscv_vle32_v_u32m1(&x[i], vl), __riscv_vle32_v_u32m1(&y[i], vl),
							__riscv_vle32_v_u32m1(&yp[i], vl), static_cast<uint32_t>(q), vl);
					__riscv_vse32_v_u32m1(&got[i], r, vl);
					});
			for (size_t i = 0; i < n; ++i) {
				CHECK_EQ(got[i] % q, O::MulMod(x[i] % q, y[i], q),
						<< "rvv::MulModShoupLazy32 residue at [" << i << "] x=" << x[i] << " y=" << y[i] << " q=" << q);
				CHECK(got[i] < 2 * q);
			}
		}
	}
}

// Shoup lazy, 32-bit lanes, scalar multiplier: r = x*y mod q in [0, 2q) for
// any 32-bit x and a fixed y < q. At m1 and m4.
TEST(RvvUtil_MulModShoupLazy32Scalar) {
	for (uint64_t q : Moduli32()) {
		const uint32_t q32 = static_cast<uint32_t>(q);
		for (uint64_t ys : {q - 1, uint64_t{1}, uint64_t{0}, O::Random(1, q)[0]}) {
			const uint32_t y32 = static_cast<uint32_t>(ys), yp = static_cast<uint32_t>(ShoupPrecon(ys, q, 32));
			for (size_t n : O::EltwiseSizes()) {
				auto x = To32(WithEdges(n, 1ULL << 32, {(1ULL << 32) - 1, 4 * q - 1, q - 1, 0}));
				std::vector<uint32_t> got1(n), got4(n);
				Strips32(n, [&](size_t i, size_t vl) {
						__riscv_vse32_v_u32m1(&got1[i], rvv::MulModShoupLazy32(__riscv_vle32_v_u32m1(&x[i], vl), y32, yp, q32, vl), vl);
						});
				for (size_t i = 0, vl; i < n; i += vl) {
					vl = __riscv_vsetvl_e32m4(n - i);
					__riscv_vse32_v_u32m4(&got4[i], rvv::MulModShoupLazy32(__riscv_vle32_v_u32m4(&x[i], vl), y32, yp, q32, vl), vl);
				}
				for (size_t i = 0; i < n; ++i) {
					CHECK_EQ(got1[i] % q, O::MulMod(x[i] % q, ys, q),
							<< "rvv::MulModShoupLazy32 (scalar y) residue at [" << i << "] x=" << x[i] << " y=" << ys << " q=" << q);
					CHECK(got1[i] < 2 * q);
					CHECK_EQ(got4[i], got1[i], << "rvv::MulModShoupLazy32<u32m4> (scalar y) at [" << i << "] q=" << q);
				}
			}
		}
	}
}

// Fused Shoup multiply-add, 32-bit lanes: any 32-bit x, scalar w < q, y in
// [0, InputModFactor * q) with InputModFactor * q <= 2^32. At m1 and m4.
namespace {
template <int Imf>
	void CheckMulAddModShoup32(uint64_t q) {
		if (Imf * q > (1ULL << 32)) return;  // outside the contract (e.g. imf 8 needs q < 2^29)
		const uint32_t q32 = static_cast<uint32_t>(q);
		for (uint64_t w : {q - 1, uint64_t{1}, uint64_t{0}, O::Random(1, q)[0]}) {
			const uint32_t w32 = static_cast<uint32_t>(w), wp = static_cast<uint32_t>(ShoupPrecon(w, q, 32));
			for (size_t n : O::EltwiseSizes()) {
				auto x = To32(WithEdges(n, 1ULL << 32, {(1ULL << 32) - 1, (1ULL << 32) - 1, q - 1, 0}));
				auto y = To32(WithEdges(n, Imf * q, {Imf * q - 1, 0, Imf * q - 1, q}));
				std::vector<uint32_t> want(n), got1(n), got4(n);
				for (size_t i = 0; i < n; ++i) want[i] = static_cast<uint32_t>((O::MulMod(x[i] % q, w, q) + y[i] % q) % q);
				Strips32(n, [&](size_t i, size_t vl) {
						auto r = rvv::MulAddModShoup32<Imf>(__riscv_vle32_v_u32m1(&x[i], vl), w32, wp, __riscv_vle32_v_u32m1(&y[i], vl), q32, vl);
						__riscv_vse32_v_u32m1(&got1[i], r, vl);
						});
				for (size_t i = 0, vl; i < n; i += vl) {
					vl = __riscv_vsetvl_e32m4(n - i);
					auto r = rvv::MulAddModShoup32<Imf>(__riscv_vle32_v_u32m4(&x[i], vl), w32, wp, __riscv_vle32_v_u32m4(&y[i], vl), q32, vl);
					__riscv_vse32_v_u32m4(&got4[i], r, vl);
				}
				CHECK_VEC_EQ(want, got1, << "rvv::MulAddModShoup32<" << Imf << "><u32m1> n=" << n << " q=" << q << " w=" << w);
				CHECK_VEC_EQ(want, got4, << "rvv::MulAddModShoup32<" << Imf << "><u32m4> n=" << n << " q=" << q << " w=" << w);
			}
		}
	}
}  // namespace

TEST(RvvUtil_MulAddModShoup32) {
	for (uint64_t q : Moduli32()) {
		CheckMulAddModShoup32<1>(q);
		CheckMulAddModShoup32<2>(q);
		CheckMulAddModShoup32<4>(q);
		CheckMulAddModShoup32<8>(q);
	}
}

TEST(RvvUtil_MulModBarrett32) {
	// 30-bit primes with inputs where the quotient estimate is TWO short (found by
	// search; random inputs hit this only ~2 in a million): with a single final
	// correction the helper returns a value in [q, 2q) for these.
	const std::vector<std::pair<uint64_t, std::vector<std::pair<uint64_t, uint64_t>>>> two_short = {
		{1011494717, {{1008325232, 1001574391}, {1010685420, 1003203244}, {1010268970, 1003690734}}},
		{931089161, {{917661731, 929557181}}}};
	std::vector<uint64_t> qs = Moduli32();
	for (const auto& h : two_short) qs.push_back(h.first);
	for (uint64_t q : qs) {
		uint32_t mu, shift;
		Barrett32Factors(q, &mu, &shift);
		CHECK_EQ(intel::hexl::MultiplyFactor(1ULL << shift, 32, q).BarrettFactor(), static_cast<uint64_t>(mu),
				<< "factor convention q=" << q);
		for (size_t n : O::EltwiseSizes()) {
			auto a64 = WithEdges(n, q, {q - 1, 0, q - 1}), b64 = WithEdges(n, q, {q - 1, q - 1, 0});
			for (const auto& h : two_short) {
				if (h.first != q) continue;
				for (size_t j = 0; j < h.second.size() && 3 + j < n; ++j) {
					a64[3 + j] = h.second[j].first;
					b64[3 + j] = h.second[j].second;
				}
			}
			auto a = To32(a64), b = To32(b64);
			std::vector<uint32_t> want(n), got(n);
			for (size_t i = 0; i < n; ++i) want[i] = static_cast<uint32_t>(O::MulMod(a[i], b[i], q));
			Strips32(n, [&](size_t i, size_t vl) {
					auto r = rvv::MulModBarrett32(__riscv_vle32_v_u32m1(&a[i], vl), __riscv_vle32_v_u32m1(&b[i], vl),
							static_cast<uint32_t>(q), mu, shift, vl);
					__riscv_vse32_v_u32m1(&got[i], r, vl);
					});
			CHECK_VEC_EQ(want, got, << "rvv::MulModBarrett32 n=" << n << " q=" << q
					<< " (factors: see Barrett32Factors in this file)");
		}
	}
}

// ===========================================================================
// LMUL-generic Add/Sub helpers at m4, the LMUL the light kernels use (the tests
// above instantiate them at m1). Also the direct test of the scalar forms.
// ===========================================================================

TEST(RvvUtil_AddSub_m4) {
	for (uint64_t q : Moduli64(1ULL << 63)) {
		for (size_t n : O::EltwiseSizes()) {
			auto a = WithEdges(n, q, {q - 1, 0, q - 1, 0}), b = WithEdges(n, q, {q - 1, 0, 0, q - 1});
			auto x = WithEdges(n, 2 * q, {2 * q - 1, q, q - 1, 0});  // x in [0, 2q)
			for (uint64_t s : {q - 1, O::Random(1, q)[0]}) {
				std::vector<uint64_t> w_add(n), w_sub(n), w_adds(n), w_subs(n), w_red(n);
				for (size_t i = 0; i < n; ++i) {
					w_add[i] = O::AddMod(a[i], b[i], q);
					w_sub[i] = O::SubMod(a[i], b[i], q);
					w_adds[i] = O::AddMod(a[i], s, q);
					w_subs[i] = O::SubMod(a[i], s, q);
					w_red[i] = x[i] % q;
				}
				std::vector<uint64_t> g_add(n), g_sub(n), g_adds(n), g_subs(n), g_red(n);
				for (size_t i = 0, vl; i < n; i += vl) {
					vl = __riscv_vsetvl_e64m4(n - i);
					vuint64m4_t va = __riscv_vle64_v_u64m4(&a[i], vl), vb = __riscv_vle64_v_u64m4(&b[i], vl);
					__riscv_vse64_v_u64m4(&g_add[i], rvv::AddMod(va, vb, q, vl), vl);
					__riscv_vse64_v_u64m4(&g_sub[i], rvv::SubMod(va, vb, q, vl), vl);
					__riscv_vse64_v_u64m4(&g_adds[i], rvv::AddScalarMod(va, s, q, vl), vl);
					__riscv_vse64_v_u64m4(&g_subs[i], rvv::SubScalarMod(va, s, q, vl), vl);
					__riscv_vse64_v_u64m4(&g_red[i], rvv::ReduceFromTwice(__riscv_vle64_v_u64m4(&x[i], vl), q, vl), vl);
				}
				CHECK_VEC_EQ(w_add, g_add, << "rvv::AddMod<u64m4> n=" << n << " q=" << q);
				CHECK_VEC_EQ(w_sub, g_sub, << "rvv::SubMod<u64m4> n=" << n << " q=" << q);
				CHECK_VEC_EQ(w_adds, g_adds, << "rvv::AddScalarMod<u64m4> n=" << n << " q=" << q << " b=" << s);
				CHECK_VEC_EQ(w_subs, g_subs, << "rvv::SubScalarMod<u64m4> n=" << n << " q=" << q << " b=" << s);
				CHECK_VEC_EQ(w_red, g_red, << "rvv::ReduceFromTwice<u64m4> n=" << n << " q=" << q);
			}
		}
	}
}

TEST(RvvUtil_AddSub32_m4) {
	for (uint64_t q : Moduli32()) {
		for (size_t n : O::EltwiseSizes()) {
			auto a = To32(WithEdges(n, q, {q - 1, 0, q - 1, 0})), b = To32(WithEdges(n, q, {q - 1, 0, 0, q - 1}));
			auto x = To32(WithEdges(n, 2 * q, {2 * q - 1, q, q - 1, 0}));  // x in [0, 2q)
			const uint32_t q32 = static_cast<uint32_t>(q);
			for (uint64_t s : {q - 1, O::Random(1, q)[0]}) {
				const uint32_t s32 = static_cast<uint32_t>(s);
				std::vector<uint32_t> w_add(n), w_sub(n), w_adds(n), w_subs(n), w_red(n);
				for (size_t i = 0; i < n; ++i) {
					w_add[i] = static_cast<uint32_t>(O::AddMod(a[i], b[i], q));
					w_sub[i] = static_cast<uint32_t>(O::SubMod(a[i], b[i], q));
					w_adds[i] = static_cast<uint32_t>(O::AddMod(a[i], s, q));
					w_subs[i] = static_cast<uint32_t>(O::SubMod(a[i], s, q));
					w_red[i] = static_cast<uint32_t>(x[i] % q);
				}
				std::vector<uint32_t> g_add(n), g_sub(n), g_adds(n), g_subs(n), g_red(n);
				for (size_t i = 0, vl; i < n; i += vl) {
					vl = __riscv_vsetvl_e32m4(n - i);
					vuint32m4_t va = __riscv_vle32_v_u32m4(&a[i], vl), vb = __riscv_vle32_v_u32m4(&b[i], vl);
					__riscv_vse32_v_u32m4(&g_add[i], rvv::AddMod32(va, vb, q32, vl), vl);
					__riscv_vse32_v_u32m4(&g_sub[i], rvv::SubMod32(va, vb, q32, vl), vl);
					__riscv_vse32_v_u32m4(&g_adds[i], rvv::AddScalarMod32(va, s32, q32, vl), vl);
					__riscv_vse32_v_u32m4(&g_subs[i], rvv::SubScalarMod32(va, s32, q32, vl), vl);
					__riscv_vse32_v_u32m4(&g_red[i], rvv::ReduceFromTwice32(__riscv_vle32_v_u32m4(&x[i], vl), q32, vl), vl);
				}
				CHECK_VEC_EQ(w_add, g_add, << "rvv::AddMod32<u32m4> n=" << n << " q=" << q);
				CHECK_VEC_EQ(w_sub, g_sub, << "rvv::SubMod32<u32m4> n=" << n << " q=" << q);
				CHECK_VEC_EQ(w_adds, g_adds, << "rvv::AddScalarMod32<u32m4> n=" << n << " q=" << q << " b=" << s);
				CHECK_VEC_EQ(w_subs, g_subs, << "rvv::SubScalarMod32<u32m4> n=" << n << " q=" << q << " b=" << s);
				CHECK_VEC_EQ(w_red, g_red, << "rvv::ReduceFromTwice32<u32m4> n=" << n << " q=" << q);
			}
		}
	}
}

// ===========================================================================
// Load32 / Store32 at every LMUL: uint64_t storage (mf2..m4) and uint32_t
// storage (mf2..m8), round trips and the cross-storage conversion, with guard
// words past n.
// ===========================================================================

namespace {
template <class V32>
	size_t Vl32(size_t n) {
		if constexpr (std::is_same_v<V32, vuint32mf2_t>) return __riscv_vsetvl_e32mf2(n);
		else if constexpr (std::is_same_v<V32, vuint32m1_t>) return __riscv_vsetvl_e32m1(n);
		else if constexpr (std::is_same_v<V32, vuint32m2_t>) return __riscv_vsetvl_e32m2(n);
		else if constexpr (std::is_same_v<V32, vuint32m4_t>) return __riscv_vsetvl_e32m4(n);
		else return __riscv_vsetvl_e32m8(n);
	}

template <class V32, bool kWithU64>
	void CheckLoadStore32(const char* lmul) {
		constexpr uint64_t kGuard64 = 0xA5A5A5A5A5A5A5A5ULL;
		constexpr uint32_t kGuard32 = 0xA5A5A5A5u;
		for (size_t n : O::EltwiseSizes()) {
			auto src64 = WithEdges(n, 1ULL << 32, {(1ULL << 32) - 1, 0, 1});  // values < 2^32
			auto src32 = To32(src64);
			std::vector<uint64_t> back64(n + 8, kGuard64), want64(src64);
			std::vector<uint32_t> back32(n + 8, kGuard32), from64(n + 8, kGuard32), want32(src32);
			want64.resize(n + 8, kGuard64);
			want32.resize(n + 8, kGuard32);
			for (size_t i = 0, vl; i < n; i += vl) {
				vl = Vl32<V32>(n - i);
				rvv::Store32(&back32[i], rvv::Load32<V32>(&src32[i], vl), vl);    // u32 -> e32 -> u32
				if constexpr (kWithU64) {
					rvv::Store32(&back64[i], rvv::Load32<V32>(&src64[i], vl), vl);  // u64 -> e32 -> u64
					rvv::Store32(&from64[i], rvv::Load32<V32>(&src64[i], vl), vl);  // u64 -> e32 -> u32
				}
			}
			CHECK_VEC_EQ(want32, back32, << "Load32/Store32 uint32_t round trip at " << lmul << " n=" << n);
			if constexpr (kWithU64) {
				CHECK_VEC_EQ(want64, back64, << "Load32/Store32 uint64_t round trip at " << lmul << " n=" << n);
				CHECK_VEC_EQ(want32, from64, << "Load32(uint64_t*) -> Store32(uint32_t*) at " << lmul << " n=" << n);
			}
		}
	}
}  // namespace

TEST(RvvUtil_Load32Store32_AllLmul) {
	CheckLoadStore32<vuint32mf2_t, true>("u32mf2");
	CheckLoadStore32<vuint32m1_t, true>("u32m1");
	CheckLoadStore32<vuint32m2_t, true>("u32m2");
	CheckLoadStore32<vuint32m4_t, true>("u32m4");
	CheckLoadStore32<vuint32m8_t, false>("u32m8");  // uint32_t storage only
}

// ===========================================================================
// The multiply helpers at m4, the LMUL the kernels use (the tests above
// instantiate them at m1).
// ===========================================================================

TEST(RvvUtil_MulMod_m4) {
	for (uint64_t q : Moduli64(1ULL << 61)) {  // Barrett needs q < 2^61 (Shoup is tested to 2^63 above)
		uint64_t mu, shift;
		Barrett64Factors(q, &mu, &shift);
		const uint64_t ys = O::Random(1, q)[0], yps = ShoupPrecon(ys, q, 64);
		for (size_t n : O::EltwiseSizes()) {
			auto x = WithEdges(n, O::AnyWord<uint64_t>(), {~0ULL, 4 * q - 1, q - 1, 0});
			auto y = WithEdges(n, q, {q - 1, q - 1, q - 1, q - 1});
			auto a = WithEdges(n, q, {q - 1, 0, q - 1}), b = WithEdges(n, q, {q - 1, q - 1, 0});
			std::vector<uint64_t> yp(n), ysv(n, ys), g_sv(n), g_ss(n), g_bar(n), w_bar(n);
			for (size_t i = 0; i < n; ++i) {
				yp[i] = ShoupPrecon(y[i], q, 64);
				w_bar[i] = O::MulMod(a[i], b[i], q);
			}
			for (size_t i = 0, vl; i < n; i += vl) {
				vl = __riscv_vsetvl_e64m4(n - i);
				vuint64m4_t vx = __riscv_vle64_v_u64m4(&x[i], vl);
				__riscv_vse64_v_u64m4(&g_sv[i], rvv::MulModShoupLazy(vx, __riscv_vle64_v_u64m4(&y[i], vl),
							__riscv_vle64_v_u64m4(&yp[i], vl), q, vl), vl);
				__riscv_vse64_v_u64m4(&g_ss[i], rvv::MulModShoupLazy(vx, ys, yps, q, vl), vl);
				__riscv_vse64_v_u64m4(&g_bar[i], rvv::MulModBarrett(__riscv_vle64_v_u64m4(&a[i], vl),
							__riscv_vle64_v_u64m4(&b[i], vl), q, mu, shift, vl), vl);
			}
			CheckShoupLazy64(x, y, g_sv, q, "rvv::MulModShoupLazy<u64m4> (vector y)");
			CheckShoupLazy64(x, ysv, g_ss, q, "rvv::MulModShoupLazy<u64m4> (scalar y)");
			CHECK_VEC_EQ(w_bar, g_bar, << "rvv::MulModBarrett<u64m4> n=" << n << " q=" << q);
		}
	}
}

TEST(RvvUtil_MulMod32_m4) {
	// the 30-bit "two short" inputs from RvvUtil_MulModBarrett32, one per modulus
	const std::vector<std::pair<uint64_t, std::pair<uint64_t, uint64_t>>> two_short = {
		{1011494717, {1008325232, 1001574391}}, {931089161, {917661731, 929557181}}};
	std::vector<uint64_t> qs = Moduli32();
	for (const auto& h : two_short) qs.push_back(h.first);
	for (uint64_t q : qs) {
		uint32_t mu, shift;
		Barrett32Factors(q, &mu, &shift);
		const uint32_t q32 = static_cast<uint32_t>(q);
		for (size_t n : O::EltwiseSizes()) {
			auto x = To32(WithEdges(n, 1ULL << 32, {(1ULL << 32) - 1, 4 * q - 1, q - 1, 0}));
			auto y = To32(WithEdges(n, q, {q - 1, q - 1, q - 1, q - 1}));
			auto a64 = WithEdges(n, q, {q - 1, 0, q - 1}), b64 = WithEdges(n, q, {q - 1, q - 1, 0});
			for (const auto& h : two_short) {
				if (h.first == q && n > 3) {
					a64[3] = h.second.first;
					b64[3] = h.second.second;
				}
			}
			auto a = To32(a64), b = To32(b64);
			std::vector<uint32_t> yp(n), g_sh(n), g_bar(n), w_bar(n);
			for (size_t i = 0; i < n; ++i) {
				yp[i] = static_cast<uint32_t>(ShoupPrecon(y[i], q, 32));
				w_bar[i] = static_cast<uint32_t>(O::MulMod(a[i], b[i], q));
			}
			for (size_t i = 0, vl; i < n; i += vl) {
				vl = __riscv_vsetvl_e32m4(n - i);
				__riscv_vse32_v_u32m4(&g_sh[i], rvv::MulModShoupLazy32(__riscv_vle32_v_u32m4(&x[i], vl), __riscv_vle32_v_u32m4(&y[i], vl),
							__riscv_vle32_v_u32m4(&yp[i], vl), q32, vl), vl);
				__riscv_vse32_v_u32m4(&g_bar[i], rvv::MulModBarrett32(__riscv_vle32_v_u32m4(&a[i], vl),
							__riscv_vle32_v_u32m4(&b[i], vl), q32, mu, shift, vl), vl);
			}
			for (size_t i = 0; i < n; ++i) {
				CHECK_EQ(g_sh[i] % q, O::MulMod(x[i] % q, y[i], q),
						<< "rvv::MulModShoupLazy32<u32m4> residue at [" << i << "] q=" << q);
				CHECK(g_sh[i] < 2 * q);
			}
			CHECK_VEC_EQ(w_bar, g_bar, << "rvv::MulModBarrett32<u32m4> n=" << n << " q=" << q);
		}
	}
}

#endif  // HEXL_HAS_RVV
