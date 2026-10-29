// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
//
// RVV kernels for EltwiseDotProductMod (rvv-hexl extension). Per strip of vl
// elements the results go two at a time (m, m + 1: M = 2 for key switching):
//   init each sum to the old result (accumulating without post) or 0;
//   for each term i: load a[i] once for both results and reduce it from
//     [0, 4q) only when input_mod_factor > 1, then fold the product into each
//     sum with a fused multiply-add (MulAddModBarrett / MulAddModBarrett32, the
//     addend inside the Barrett step), so every sum stays in [0, q);
//   multiply by the post strip when given (adding the old result in the same
//     step when accumulating), one store per result.
// RVVF64 does the same with the Harvey product, the addend folded into the
// quotient estimate (sums as doubles in (-q, q)), converting back once per
// result. Next step: lazy sums across the terms (bounds in the internal header).

#include "eltwise/eltwise-dot-product-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

template <typename Word, class V>
void DotProductModRVV32(Word* const* result, uint64_t num_results,
		const Word* const* a, uint64_t num_terms, const Word* const* b,
		const Word* post, uint64_t n, uint64_t modulus,
		uint64_t input_mod_factor, bool accumulate) {
	static_assert(rvv::IsE32<V>, "DotProductModRVV32 takes e32 lanes");

	const uint32_t q = static_cast<uint32_t>(modulus);
	const uint32_t shift = MSB(modulus) - 1;
	const uint32_t mu = MultiplyFactor(uint32_t{1} << shift, 32, modulus).BarrettFactor();
	const bool reduce4 = input_mod_factor >= 4, reduce2 = input_mod_factor >= 2;

	// results m and m + 1 (when two), sharing each a[i] strip
	const auto pair = [&](uint64_t m, bool two, size_t off, size_t e) {
		const auto init = [&](uint64_t r) {
			return accumulate && !post ? rvv::Load32<V>(result[r] + off, e) : rvv::Splat<V>(0, e);
		};
		V s0 = init(m);
		V s1 = two ? init(m + 1) : s0;
		for (uint64_t i = 0; i < num_terms; ++i) {
			V x = rvv::Load32<V>(a[i] + off, e);
			if (reduce4) x = rvv::ReduceFromTwice32(x, 2*q, e);  // [0, 4q) -> [0, 2q): 4q <= 2^32
			if (reduce2) x = rvv::ReduceFromTwice32(x, q, e);

			s0 = rvv::MulAddModBarrett32(x, rvv::Load32<V>(b[i*num_results + m] + off, e), s0, q, mu, shift, e);
			if (two) s1 = rvv::MulAddModBarrett32(x, rvv::Load32<V>(b[i*num_results + m + 1] + off, e), s1, q, mu, shift, e);
		}

		const auto finish = [&](V s, uint64_t r) {
			if (post) {
				V postv = rvv::Load32<V>(post + off, e);
				s = accumulate ? rvv::MulAddModBarrett32(s, postv, rvv::Load32<V>(result[r] + off, e), q, mu, shift, e)
					: rvv::MulModBarrett32(s, postv, q, mu, shift, e);
			}
			rvv::Store32<V>(result[r] + off, s, e);
		};
		finish(s0, m);
		if (two) finish(s1, m + 1);
	};

	for (size_t e, off = 0; off < n; off += e) {
		e = rvv::SetVl<V>(n - off);
		for (uint64_t m = 0; m < num_results; m += 2) pair(m, m + 1 < num_results, off, e);
	}
}

template <class VF>
void DotProductModRVVF64(uint64_t* const* result, uint64_t num_results,
		const uint64_t* const* a, uint64_t num_terms, const uint64_t* const* b,
		const uint64_t* post, uint64_t n, uint64_t modulus,
		uint64_t input_mod_factor, bool accumulate) {
	static_assert(rvv::IsE64<rvv::UintOf<VF>>, "DotProductModRVVF64 takes f64 lanes");
	using VU = rvv::UintOf<VF>;

	const double qd = static_cast<double>(modulus);
	const double qi = 1.0 / qd;
	const double C = 0x1.8p52;
	const bool reduce4 = input_mod_factor >= 4, reduce2 = input_mod_factor >= 2;

	// x y mod q in (-q, q) for x, y in [0, q) (util/fp64-mod.hpp), as EltwiseMultModRVVF64
	const auto mulmod = [qd, qi, C](VF x, VF y, size_t vl) {
		VF h = __riscv_vfmul(x, y, vl);                                                  // x y rounded
		VF l = __riscv_vfmsub(x, y, h, vl);                                              // x y - h, exact
		VF t = __riscv_vfsub(__riscv_vfadd(__riscv_vfmul(h, qi, vl), C, vl), C, vl);    // round(h / q)
		return __riscv_vfadd(__riscv_vfnmsac(h, qd, t, vl), l, vl);                      // x y - t q
	};
	// x y + c mod q in (-q, q) for x, y in [0, q), |c| < q, as EltwiseMultAddModRVVF64
	const auto muladdmod = [qd, qi, C](VF x, VF y, VF c, size_t vl) {
		VF h = __riscv_vfmul(x, y, vl);                                                  // x y rounded
		VF l = __riscv_vfmsub(x, y, h, vl);                                              // x y - h, exact
		VF s = __riscv_vfadd(h, c, vl);                                                  // ~ x y + c, for the estimate
		VF t = __riscv_vfsub(__riscv_vfadd(__riscv_vfmul(s, qi, vl), C, vl), C, vl);    // round((x y + c) / q)
		return __riscv_vfadd(__riscv_vfadd(__riscv_vfnmsac(h, qd, t, vl), l, vl), c, vl);  // x y + c - t q
	};
	const auto load = [](const uint64_t* p, size_t vl) { return __riscv_vfcvt_f(rvv::Load<VU>(p, vl), vl); };

	// results m and m + 1 (when two), sharing each a[i] strip
	const auto pair = [&](uint64_t m, bool two, size_t off, size_t e) {
		const auto init = [&](uint64_t r) {
			return accumulate && !post ? load(result[r] + off, e) : rvv::SplatF64<VF>(0.0, e);
		};
		VF s0 = init(m);  // the sums stay in (-q, q)
		VF s1 = two ? init(m + 1) : s0;
		for (uint64_t i = 0; i < num_terms; ++i) {
			VU x = rvv::Load<VU>(a[i] + off, e);
			if (reduce4) x = rvv::ReduceFromTwice(x, 2*modulus, e);
			if (reduce2) x = rvv::ReduceFromTwice(x, modulus, e);
			VF xf = __riscv_vfcvt_f(x, e);

			s0 = muladdmod(xf, load(b[i*num_results + m] + off, e), s0, e);
			if (two) s1 = muladdmod(xf, load(b[i*num_results + m + 1] + off, e), s1, e);
		}

		const auto finish = [&](VF s, uint64_t r) {
			if (post) {
				VF postv = load(post + off, e);
				s = accumulate ? muladdmod(s, postv, load(result[r] + off, e), e) : mulmod(s, postv, e);
			}
			rvv::Store(result[r] + off, rvv::ToUintModF64(s, modulus, e), e);  // (-q, q) -> [0, q)
		};
		finish(s0, m);
		if (two) finish(s1, m + 1);
	};

	for (size_t e, off = 0; off < n; off += e) {
		e = rvv::SetVl<VU>(n - off);
		for (uint64_t m = 0; m < num_results; m += 2) pair(m, m + 1 < num_results, off, e);
	}
}

template <class V>
void DotProductModRVV64(uint64_t* const* result, uint64_t num_results,
		const uint64_t* const* a, uint64_t num_terms, const uint64_t* const* b,
		const uint64_t* post, uint64_t n, uint64_t modulus,
		uint64_t input_mod_factor, bool accumulate) {
	static_assert(rvv::IsE64<V>, "DotProductModRVV64 takes e64 lanes");

	const uint64_t shift = MSB(modulus) - 1;
	const uint64_t mu = MultiplyFactor(uint64_t(1U) << shift, 64, modulus).BarrettFactor();
	const bool reduce4 = input_mod_factor >= 4, reduce2 = input_mod_factor >= 2;

	// results m and m + 1 (when two), sharing each a[i] strip
	const auto pair = [&](uint64_t m, bool two, size_t off, size_t e) {
		const auto init = [&](uint64_t r) {
			return accumulate && !post ? rvv::Load<V>(result[r] + off, e) : rvv::Splat<V>(0, e);
		};
		V s0 = init(m);
		V s1 = two ? init(m + 1) : s0;
		for (uint64_t i = 0; i < num_terms; ++i) {
			V x = rvv::Load<V>(a[i] + off, e);
			if (reduce4) x = rvv::ReduceFromTwice(x, 2*modulus, e);
			if (reduce2) x = rvv::ReduceFromTwice(x, modulus, e);

			s0 = rvv::MulAddModBarrett(x, rvv::Load<V>(b[i*num_results + m] + off, e), s0, modulus, mu, shift, e);
			if (two) s1 = rvv::MulAddModBarrett(x, rvv::Load<V>(b[i*num_results + m + 1] + off, e), s1, modulus, mu, shift, e);
		}

		const auto finish = [&](V s, uint64_t r) {
			if (post) {
				V postv = rvv::Load<V>(post + off, e);
				s = accumulate ? rvv::MulAddModBarrett(s, postv, rvv::Load<V>(result[r] + off, e), modulus, mu, shift, e)
					: rvv::MulModBarrett(s, postv, modulus, mu, shift, e);
			}
			rvv::Store(result[r] + off, s, e);
		};
		finish(s0, m);
		if (two) finish(s1, m + 1);
	};

	for (size_t e, off = 0; off < n; off += e) {
		e = rvv::SetVl<V>(n - off);
		for (uint64_t m = 0; m < num_results; m += 2) pair(m, m + 1 < num_results, off, e);
	}
}

#define HEXL_DOT_RVV32(Word, V)                                                                    \
	template void DotProductModRVV32<Word, V>(Word* const*, uint64_t, const Word* const*, uint64_t, \
			const Word* const*, const Word*, uint64_t, uint64_t, uint64_t, bool)
#define HEXL_DOT_RVV64(NAME, V)                                                                       \
	template void NAME<V>(uint64_t* const*, uint64_t, const uint64_t* const*, uint64_t,                \
			const uint64_t* const*, const uint64_t*, uint64_t, uint64_t, uint64_t, bool)

HEXL_DOT_RVV32(uint64_t, vuint32m1_t);
HEXL_DOT_RVV32(uint64_t, vuint32m2_t);
HEXL_DOT_RVV32(uint64_t, vuint32m4_t);
HEXL_DOT_RVV32(uint32_t, vuint32m1_t);
HEXL_DOT_RVV32(uint32_t, vuint32m2_t);
HEXL_DOT_RVV32(uint32_t, vuint32m4_t);
HEXL_DOT_RVV32(uint32_t, vuint32m8_t);
HEXL_DOT_RVV64(DotProductModRVV64, vuint64m1_t);
HEXL_DOT_RVV64(DotProductModRVV64, vuint64m2_t);
HEXL_DOT_RVV64(DotProductModRVV64, vuint64m4_t);
HEXL_DOT_RVV64(DotProductModRVV64, vuint64m8_t);
HEXL_DOT_RVV64(DotProductModRVVF64, vfloat64m1_t);
HEXL_DOT_RVV64(DotProductModRVVF64, vfloat64m2_t);
HEXL_DOT_RVV64(DotProductModRVVF64, vfloat64m4_t);
HEXL_DOT_RVV64(DotProductModRVVF64, vfloat64m8_t);

#undef HEXL_DOT_RVV32
#undef HEXL_DOT_RVV64

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
