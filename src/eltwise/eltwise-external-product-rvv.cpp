// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
//
// RVV kernels for EltwiseExternalProductMod (rvv-hexl extension). Planned shape,
// per strip of vl elements (CGGI, k digit polynomials):
//   load digits[0..k-1] once;
//   for each of the four sums (key1 / key2, j = 0 / 1): load the key strip,
//     add its Barrett product with the digit into an accumulator register,
//     products left lazy ([0, 2q) at e32, one correction per sum);
//   multiply each sum by its monomial strip, add both into acc_j (+ the old
//     acc_j when accumulating), correct to [0, q), one store per acc_j.
// Memory per element: 5k + 6 loads and stores against 16k + 12 for OpenFHE's
// call sequence; 4 full reductions against 4(k + 1).

#include "eltwise/eltwise-external-product-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

// STUB: calls the native kernel until the RVV body is written.
template <typename Word, class V>
void ExternalProductModRVV32(Word* acc0, Word* acc1,
		const Word* const* digits, uint64_t num_digits,
		const Word* const* key1, const Word* mono1,
		const Word* const* key2, const Word* mono2,
		uint64_t n, uint64_t modulus, bool accumulate) {
	static_assert(rvv::IsE32<V>, "ExternalProductModRVV32 takes e32 lanes");
	ExternalProductModNative<Word>(acc0, acc1, digits, num_digits, key1, mono1, key2, mono2,
			n, modulus, accumulate);
}

// STUB: calls the native kernel until the RVV body is written.
template <class V>
void ExternalProductModRVV64(uint64_t* acc0, uint64_t* acc1,
		const uint64_t* const* digits, uint64_t num_digits,
		const uint64_t* const* key1, const uint64_t* mono1,
		const uint64_t* const* key2, const uint64_t* mono2,
		uint64_t n, uint64_t modulus, bool accumulate) {
	static_assert(rvv::IsE64<V>, "ExternalProductModRVV64 takes e64 lanes");
	ExternalProductModNative<uint64_t>(acc0, acc1, digits, num_digits, key1, mono1, key2, mono2,
			n, modulus, accumulate);
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
