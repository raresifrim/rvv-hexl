// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// RVV kernels for EltwiseMultMod (the NTT-domain Hadamard product).

#include "eltwise/eltwise-mult-mod-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include <type_traits>

#include "util/rvv-util.hpp"
#include "hexl/number-theory/number-theory.hpp"

namespace intel {
namespace hexl {

template <typename Word, int InputModFactor, class V>
	void EltwiseMultModRVV32(Word* result, const Word* operand1, const Word* operand2, uint64_t n, uint64_t modulus) {
		const uint32_t shift = MSB(modulus) - 1;
		const uint32_t mu = MultiplyFactor(uint32_t{1} << shift, 32, modulus).BarrettFactor();
		for(size_t vl; n > 0; n-=vl, result+=vl, operand1+=vl, operand2+=vl){
			vl = rvv::SetVl<V>(n);
			V v1 = rvv::Load32<V>(operand1, vl);
			V v2 = rvv::Load32<V>(operand2, vl);
			if constexpr (InputModFactor >= 4) { 
				v1 = rvv::ReduceFromTwice32(v1, static_cast<uint32_t>(2*modulus), vl);
				v2 = rvv::ReduceFromTwice32(v2, static_cast<uint32_t>(2*modulus), vl);
			}
			if constexpr (InputModFactor >= 2) { 
				v1 = rvv::ReduceFromTwice32(v1, static_cast<uint32_t>(modulus), vl);
				v2 = rvv::ReduceFromTwice32(v2, static_cast<uint32_t>(modulus), vl);
			} 
			V vr = rvv::MulModBarrett32<V>(v1, v2, modulus, mu, shift, vl);
			rvv::Store32<V>(result, vr, vl);
		}
	}

template <int InputModFactor, class V>
	void EltwiseMultModRVV64(uint64_t* result, const uint64_t* operand1,
			const uint64_t* operand2, uint64_t n,
			uint64_t modulus) {
		const uint64_t shift = MSB(modulus) - 1;
		const uint64_t mu = MultiplyFactor(uint64_t{1} << shift, 64, modulus).BarrettFactor();
		for(size_t vl; n > 0; n-=vl, result+=vl, operand1+=vl, operand2+=vl){
			vl = rvv::SetVl<V>(n);
			V v1 = rvv::Load<V>(operand1, vl);
			V v2 = rvv::Load<V>(operand2, vl);
			if constexpr (InputModFactor >= 4) { 
				v1 = rvv::ReduceFromTwice(v1, 2*modulus, vl);
				v2 = rvv::ReduceFromTwice(v2, 2*modulus, vl); 
			}
			if constexpr (InputModFactor >= 2) { 
				v1 = rvv::ReduceFromTwice(v1, modulus, vl);
				v2 = rvv::ReduceFromTwice(v2, modulus, vl);
			} 
			V vr = rvv::MulModBarrett<V>(v1, v2, modulus, mu, shift, vl);
			rvv::Store<V>(result, vr, vl);
		}
	}

template void EltwiseMultModRVV32<uint64_t, 1, vuint32m1_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint64_t, 1, vuint32m2_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint64_t, 1, vuint32m4_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint64_t, 2, vuint32m1_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint64_t, 2, vuint32m2_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint64_t, 2, vuint32m4_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint64_t, 4, vuint32m1_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint64_t, 4, vuint32m2_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint64_t, 4, vuint32m4_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint32_t, 1, vuint32m1_t>(uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint32_t, 1, vuint32m2_t>(uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint32_t, 1, vuint32m4_t>(uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint32_t, 1, vuint32m8_t>(uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint32_t, 2, vuint32m1_t>(uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint32_t, 2, vuint32m2_t>(uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint32_t, 2, vuint32m4_t>(uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint32_t, 2, vuint32m8_t>(uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint32_t, 4, vuint32m1_t>(uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint32_t, 4, vuint32m2_t>(uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint32_t, 4, vuint32m4_t>(uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint32_t, 4, vuint32m8_t>(uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV64<1, vuint64m1_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV64<1, vuint64m2_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV64<1, vuint64m4_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV64<1, vuint64m8_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV64<2, vuint64m1_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV64<2, vuint64m2_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV64<2, vuint64m4_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV64<2, vuint64m8_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV64<4, vuint64m1_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV64<4, vuint64m2_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV64<4, vuint64m4_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV64<4, vuint64m8_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
