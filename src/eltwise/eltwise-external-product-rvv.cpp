// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
//
// RVV kernels for EltwiseExternalProductMod (rvv-hexl extension), the same shape as the
// dot-product kernels. Per strip of vl elements and per output j (0, 1):
//   init the key1 sum to the old acc_j (accumulating without mono1) or 0, the key2 sum to 0;
//   for each digit i: load the digit strip and fold its products with key1[2i + j] and
//     key2[2i + j] into the two sums with a fused multiply-add (MulAddModBarrett32 /
//     MulAddModBarrett, the addend inside the Barrett step), so every sum stays in [0, q);
//   multiply the key1 sum by mono1 (adding the old acc_j in the same step when
//     accumulating), the key2 sum by mono2 with the key1 result as the addend, one store.
// The digit strip is loaded once per j (the second load hits L1). Next step: lazy sums
// across the digits, as the native kernel (bounds in the internal header).

#include "eltwise/eltwise-external-product-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

template <typename Word, class V>
void ExternalProductModRVV32(Word* acc0, Word* acc1,
		const Word* const* digits, uint64_t num_digits,
		const Word* const* key1, const Word* mono1,
		const Word* const* key2, const Word* mono2,
		uint64_t n, uint64_t modulus, bool accumulate) {
	static_assert(rvv::IsE32<V>, "ExternalProductModRVV32 takes e32 lanes");

	Word* acc[2] = {acc0, acc1};
	const uint32_t q = static_cast<uint32_t>(modulus);
	const uint32_t shift = MSB(modulus) - 1;
	const uint32_t mu = MultiplyFactor(uint32_t{1} << shift, 32, modulus).BarrettFactor();

	for (size_t e, off = 0; off < n; off += e) {
		e = rvv::SetVl<V>(n - off);
		for (int j = 0; j < 2; ++j) {
			V s1 = accumulate && !mono1 ? rvv::Load32<V>(acc[j] + off, e) : rvv::Splat<V>(0, e);
			V s2 = rvv::Splat<V>(0, e);
			for (uint64_t i = 0; i < num_digits; ++i) {
				V d = rvv::Load32<V>(digits[i] + off, e);
				s1 = rvv::MulAddModBarrett32(d, rvv::Load32<V>(key1[2 * i + j] + off, e), s1, q, mu, shift, e);
				if (key2) s2 = rvv::MulAddModBarrett32(d, rvv::Load32<V>(key2[2 * i + j] + off, e), s2, q, mu, shift, e);
			}

			if (mono1) {
				V m1 = rvv::Load32<V>(mono1 + off, e);
				s1 = accumulate ? rvv::MulAddModBarrett32(s1, m1, rvv::Load32<V>(acc[j] + off, e), q, mu, shift, e)
					: rvv::MulModBarrett32(s1, m1, q, mu, shift, e);
			}
			if (key2) {
				s1 = mono2 ? rvv::MulAddModBarrett32(s2, rvv::Load32<V>(mono2 + off, e), s1, q, mu, shift, e)
					: rvv::AddMod32(s1, s2, q, e);
			}

			rvv::Store32<V>(acc[j] + off, s1, e);
		}
	}
}

template <class V>
void ExternalProductModRVV64(uint64_t* acc0, uint64_t* acc1,
		const uint64_t* const* digits, uint64_t num_digits,
		const uint64_t* const* key1, const uint64_t* mono1,
		const uint64_t* const* key2, const uint64_t* mono2,
		uint64_t n, uint64_t modulus, bool accumulate) {
	static_assert(rvv::IsE64<V>, "ExternalProductModRVV64 takes e64 lanes");

	uint64_t* acc[2] = {acc0, acc1};
	const uint64_t shift = MSB(modulus) - 1;
	const uint64_t mu = MultiplyFactor(uint64_t(1U) << shift, 64, modulus).BarrettFactor();

	for (size_t e, off = 0; off < n; off += e) {
		e = rvv::SetVl<V>(n - off);
		const V zero = rvv::Splat<V>(0, e);
		for (int j = 0; j < 2; ++j) {
			V s1 = accumulate && !mono1 ? rvv::Load<V>(acc[j] + off, e) : zero;
			V s2 = zero;
			for (uint64_t i = 0; i < num_digits; ++i) {
				V d = rvv::Load<V>(digits[i] + off, e);
				s1 = rvv::MulAddModBarrett(d, rvv::Load<V>(key1[2 * i + j] + off, e), s1, modulus, mu, shift, e);
				if (key2) s2 = rvv::MulAddModBarrett(d, rvv::Load<V>(key2[2 * i + j] + off, e), s2, modulus, mu, shift, e);
			}

			// MulAddModBarrett with a zero addend for the plain products: up to 62-bit q
			// (MulModBarrett's single correction holds below 2^61 only)
			if (mono1) {
				s1 = rvv::MulAddModBarrett(s1, rvv::Load<V>(mono1 + off, e),
						accumulate ? rvv::Load<V>(acc[j] + off, e) : zero, modulus, mu, shift, e);
			}
			if (key2) {
				s1 = mono2 ? rvv::MulAddModBarrett(s2, rvv::Load<V>(mono2 + off, e), s1, modulus, mu, shift, e)
					: rvv::AddMod(s1, s2, modulus, e);
			}

			rvv::Store(acc[j] + off, s1, e);
		}
	}
}

#define HEXL_EXT_PROD_RVV32(Word, V)                                                          \
	template void ExternalProductModRVV32<Word, V>(Word*, Word*, const Word* const*, uint64_t,  \
			const Word* const*, const Word*, const Word* const*, const Word*, uint64_t, uint64_t, bool)
#define HEXL_EXT_PROD_RVV64(V)                                                                 \
	template void ExternalProductModRVV64<V>(uint64_t*, uint64_t*, const uint64_t* const*,      \
			uint64_t, const uint64_t* const*, const uint64_t*, const uint64_t* const*,           \
			const uint64_t*, uint64_t, uint64_t, bool)

HEXL_EXT_PROD_RVV32(uint64_t, vuint32m1_t);
HEXL_EXT_PROD_RVV32(uint64_t, vuint32m2_t);
HEXL_EXT_PROD_RVV32(uint64_t, vuint32m4_t);
HEXL_EXT_PROD_RVV32(uint32_t, vuint32m1_t);
HEXL_EXT_PROD_RVV32(uint32_t, vuint32m2_t);
HEXL_EXT_PROD_RVV32(uint32_t, vuint32m4_t);
HEXL_EXT_PROD_RVV32(uint32_t, vuint32m8_t);
HEXL_EXT_PROD_RVV64(vuint64m1_t);
HEXL_EXT_PROD_RVV64(vuint64m2_t);
HEXL_EXT_PROD_RVV64(vuint64m4_t);
HEXL_EXT_PROD_RVV64(vuint64m8_t);

#undef HEXL_EXT_PROD_RVV32
#undef HEXL_EXT_PROD_RVV64

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
