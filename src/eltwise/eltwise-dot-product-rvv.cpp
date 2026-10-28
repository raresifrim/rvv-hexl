// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
//
// RVV kernels for EltwiseDotProductMod (rvv-hexl extension). Planned shape, per
// strip of vl elements and per result m (or all M results at once, sharing the
// a[i] loads, when the register budget allows: M = 2 for key switching):
//   for each term i: load a[i] (reduce from [0, 4q) when input_mod_factor > 1)
//     and b[i M + m], add the lazy product into an accumulator register,
//     correct every few terms (see the internal header for the bounds);
//   multiply by the post strip when given, add the old result when
//     accumulating, correct to [0, q), one store per result.
// RVVF64 converts a and b to double per term but keeps the sums as doubles
// (Harvey products in (-q, q)), converting back once per result: the
// conversions (2 elements per cycle on the X100) are what limits the FP64 path.

#include "eltwise/eltwise-dot-product-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

// STUB: calls the native kernel until the RVV body is written.
template <typename Word, class V>
void DotProductModRVV32(Word* const* result, uint64_t num_results,
		const Word* const* a, uint64_t num_terms, const Word* const* b,
		const Word* post, uint64_t n, uint64_t modulus,
		uint64_t input_mod_factor, bool accumulate) {
	static_assert(rvv::IsE32<V>, "DotProductModRVV32 takes e32 lanes");
	DotProductModNative<Word>(result, num_results, a, num_terms, b, post, n, modulus, input_mod_factor, accumulate);
}

// STUB: calls the native kernel until the RVV body is written.
template <class VF>
void DotProductModRVVF64(uint64_t* const* result, uint64_t num_results,
		const uint64_t* const* a, uint64_t num_terms, const uint64_t* const* b,
		const uint64_t* post, uint64_t n, uint64_t modulus,
		uint64_t input_mod_factor, bool accumulate) {
	static_assert(rvv::IsE64<rvv::UintOf<VF>>, "DotProductModRVVF64 takes f64 lanes");
	DotProductModNative<uint64_t>(result, num_results, a, num_terms, b, post, n, modulus, input_mod_factor,
			accumulate);
}

// STUB: calls the native kernel until the RVV body is written.
template <class V>
void DotProductModRVV64(uint64_t* const* result, uint64_t num_results,
		const uint64_t* const* a, uint64_t num_terms, const uint64_t* const* b,
		const uint64_t* post, uint64_t n, uint64_t modulus,
		uint64_t input_mod_factor, bool accumulate) {
	static_assert(rvv::IsE64<V>, "DotProductModRVV64 takes e64 lanes");
	DotProductModNative<uint64_t>(result, num_results, a, num_terms, b, post, n, modulus, input_mod_factor,
			accumulate);
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
