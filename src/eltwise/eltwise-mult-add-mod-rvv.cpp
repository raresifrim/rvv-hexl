// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// SPDX-License-Identifier: Apache-2.0
//
// RVV kernels for EltwiseMultAddMod (rvv-hexl extension):
//   result[i] = (arg1[i] * arg2[i] + arg3[i]) mod modulus
// Fused body plan, per strip: load arg1, arg2 (and reduce each from
// [0, InputModFactor * q) as EltwiseMultMod does), Barrett product left lazy
// in [0, 2q) (rvv::MulModBarrett / MulModBarrett32 without its last
// correction), load arg3 and reduce it to [0, q), add (< 3q), then
// ReduceFromTwice(2q) and ReduceFromTwice(q). One store; result may alias
// any input.

#include "eltwise/eltwise-mult-add-mod-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>


#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

template <typename Word, int InputModFactor, class V>
void EltwiseMultAddModRVV32(Word* result, const Word* arg1, const Word* arg2,
		const Word* arg3, uint64_t n, uint64_t modulus) {
	const uint32_t shift = MSB(modulus) - 1;
	const uint32_t mu = MultiplyFactor(uint32_t{1} << shift, 32, modulus).BarrettFactor();
	for(size_t vl; n > 0; n-=vl, result+=vl, arg1+=vl, arg2+=vl, arg3+=vl){
		vl = rvv::SetVl<V>(n);
		V v1 = rvv::Load32<V>(arg1, vl);
		V v2 = rvv::Load32<V>(arg2, vl);
		V v3 = rvv::Load32<V>(arg3, vl);
		if constexpr (InputModFactor >= 4) { 
			v1 = rvv::ReduceFromTwice32(v1, static_cast<uint32_t>(2*modulus), vl);
			v2 = rvv::ReduceFromTwice32(v2, static_cast<uint32_t>(2*modulus), vl);
			v3 = rvv::ReduceFromTwice32(v3, static_cast<uint32_t>(2*modulus), vl);
		}
		if constexpr (InputModFactor >= 2) { 
			v1 = rvv::ReduceFromTwice32(v1, static_cast<uint32_t>(modulus), vl);
			v2 = rvv::ReduceFromTwice32(v2, static_cast<uint32_t>(modulus), vl);
			v3 = rvv::ReduceFromTwice32(v3, static_cast<uint32_t>(modulus), vl);

		} 
		V vr = rvv::MulAddModBarrett32<V>(v1, v2, v3, modulus, mu, shift, vl);
		rvv::Store32<V>(result, vr, vl);
	}
}

template <int InputModFactor, class V>
void EltwiseMultAddModRVV64(uint64_t* result, const uint64_t* arg1,
		const uint64_t* arg2, const uint64_t* arg3, uint64_t n,
		uint64_t modulus) {

	const uint64_t shift = MSB(modulus) - 1;
	const uint64_t mu = MultiplyFactor(uint64_t{1} << shift, 64, modulus).BarrettFactor();
	for(size_t vl; n > 0; n-=vl, result+=vl, arg1+=vl, arg2+=vl, arg3+=vl){
		vl = rvv::SetVl<V>(n);
		V v1 = rvv::Load<V>(arg1, vl);
		V v2 = rvv::Load<V>(arg2, vl);
		V v3 = rvv::Load<V>(arg3, vl);
		if constexpr (InputModFactor >= 4) { 
			v1 = rvv::ReduceFromTwice(v1, 2*modulus, vl);
			v2 = rvv::ReduceFromTwice(v2, 2*modulus, vl);
			v3 = rvv::ReduceFromTwice(v3, 2*modulus, vl);
		}
		if constexpr (InputModFactor >= 2) { 
			v1 = rvv::ReduceFromTwice(v1, modulus, vl);
			v2 = rvv::ReduceFromTwice(v2, modulus, vl);
			v3 = rvv::ReduceFromTwice(v3, modulus, vl);
		} 
		V vr = rvv::MulModBarrett<V>(v1, v2, modulus, mu, shift, vl);
		vr = rvv::AddMod<V>(vr, v3, modulus, vl);
		rvv::Store<V>(result, vr, vl);
	}
}

// Every lane type and input factor, so tests and benchmarks can compare them;
// the public function uses the defaults from src/util/rvv-config.hpp. On
// uint64_t storage the e32 lanes stop at u32m4 (the narrowing load).
#define HEXL_MULT_ADD_RVV32(Word, Imf, V)                                            \
	template void EltwiseMultAddModRVV32<Word, Imf, V>(Word*, const Word*,     \
			const Word*, const Word*, uint64_t, uint64_t)
#define HEXL_MULT_ADD_RVV64(Imf, V)                                                  \
	template void EltwiseMultAddModRVV64<Imf, V>(uint64_t*, const uint64_t*,   \
			const uint64_t*, const uint64_t*, uint64_t, uint64_t)
#define HEXL_MULT_ADD_LANES(Imf)                          \
	HEXL_MULT_ADD_RVV32(uint64_t, Imf, vuint32m1_t);  \
	HEXL_MULT_ADD_RVV32(uint64_t, Imf, vuint32m2_t);  \
	HEXL_MULT_ADD_RVV32(uint64_t, Imf, vuint32m4_t);  \
	HEXL_MULT_ADD_RVV32(uint32_t, Imf, vuint32m1_t);  \
	HEXL_MULT_ADD_RVV32(uint32_t, Imf, vuint32m2_t);  \
	HEXL_MULT_ADD_RVV32(uint32_t, Imf, vuint32m4_t);  \
	HEXL_MULT_ADD_RVV32(uint32_t, Imf, vuint32m8_t);  \
	HEXL_MULT_ADD_RVV64(Imf, vuint64m1_t);            \
	HEXL_MULT_ADD_RVV64(Imf, vuint64m2_t);            \
	HEXL_MULT_ADD_RVV64(Imf, vuint64m4_t);            \
	HEXL_MULT_ADD_RVV64(Imf, vuint64m8_t)

HEXL_MULT_ADD_LANES(1);
HEXL_MULT_ADD_LANES(2);
HEXL_MULT_ADD_LANES(4);

#undef HEXL_MULT_ADD_LANES
#undef HEXL_MULT_ADD_RVV64
#undef HEXL_MULT_ADD_RVV32

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
