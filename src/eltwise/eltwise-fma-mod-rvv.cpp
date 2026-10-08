// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// RVV kernels for EltwiseFMAMod.

#include "eltwise/eltwise-fma-mod-internal.hpp"
#include <algorithm>                              // std::min
#include "hexl/number-theory/number-theory.hpp"   // MultiplyFactor

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include <type_traits>

#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

template <typename Word, int InputModFactor, class V>
	void EltwiseFMAModRVV32(Word* result, const Word* arg1, uint64_t arg2, const Word* arg3, uint64_t n, uint64_t modulus) {
		uint64_t w = arg2;
		if constexpr (InputModFactor == 8) w = std::min(w, w - 4 * modulus);
		if constexpr (InputModFactor >= 4) w = std::min(w, w - 2 * modulus);
		if constexpr (InputModFactor >= 2) w = std::min(w, w - modulus);
		uint64_t w_precon = MultiplyFactor(w, 32, modulus).BarrettFactor();
		if(arg3 != nullptr){
			for(size_t vl; n>0; n-=vl, result+=vl, arg1+=vl, arg3+=vl){
				vl = rvv::SetVl<V>(n);
				V x = rvv::Load32<V>(arg1, vl);
				V y = rvv::Load32<V>(arg3, vl);
				V z = rvv::MulAddModShoup32<InputModFactor, V>(x, static_cast<uint32_t>(w), static_cast<uint32_t>(w_precon), y, static_cast<uint32_t>(modulus), vl);
				rvv::Store32<V>(result, z, vl);	
			}
		}
		else{
			for(size_t vl; n>0; n-=vl, result+=vl, arg1+=vl){
				vl = rvv::SetVl<V>(n);
				V x = rvv::Load32<V>(arg1, vl); 
				V z = rvv::MulModShoupLazy32<V>(x, static_cast<uint32_t>(w), static_cast<uint32_t>(w_precon), static_cast<uint32_t>(modulus), vl);
				z= rvv::ReduceFromTwice32(z, static_cast<uint32_t>(modulus), vl);
				rvv::Store32<V>(result, z, vl);	
			}
		}
	}

template <int InputModFactor, class V>
	void EltwiseFMAModRVV64(uint64_t* result, const uint64_t* arg1, uint64_t arg2,
			const uint64_t* arg3, uint64_t n, uint64_t modulus) {
		uint64_t w = arg2;
		if constexpr (InputModFactor == 8) w = std::min(w, w - 4 * modulus);
		if constexpr (InputModFactor >= 4) w = std::min(w, w - 2 * modulus);
		if constexpr (InputModFactor >= 2) w = std::min(w, w - modulus);
		uint64_t w_precon = MultiplyFactor(w, 64, modulus).BarrettFactor();
		if(arg3 != nullptr){
			for(size_t vl; n>0; n-=vl, result+=vl, arg1+=vl, arg3+=vl){
				vl = rvv::SetVl<V>(n);
				V x = rvv::Load<V>(arg1, vl);
				V y = rvv::Load<V>(arg3, vl);
				V z = rvv::MulAddModShoup<InputModFactor, V>(x, w, w_precon, y, modulus, vl);
				rvv::Store<V>(result, z, vl);	
			}
		}
		else{
			for(size_t vl; n>0; n-=vl, result+=vl, arg1+=vl){
				vl = rvv::SetVl<V>(n);
				V x = rvv::Load<V>(arg1, vl); 
				V z = rvv::MulModShoupLazy<V>(x, w, w_precon, modulus, vl);
				z= rvv::ReduceFromTwice(z, modulus, vl);
				rvv::Store<V>(result, z, vl);	
			}
		}
	}

template void EltwiseFMAModRVV32<uint64_t, 1>(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t);
template void EltwiseFMAModRVV32<uint64_t, 2>(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t);
template void EltwiseFMAModRVV32<uint64_t, 4>(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t);
template void EltwiseFMAModRVV32<uint64_t, 8>(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t);
template void EltwiseFMAModRVV32<uint32_t, 1>(uint32_t*, const uint32_t*, uint64_t, const uint32_t*, uint64_t, uint64_t);
template void EltwiseFMAModRVV32<uint32_t, 2>(uint32_t*, const uint32_t*, uint64_t, const uint32_t*, uint64_t, uint64_t);
template void EltwiseFMAModRVV32<uint32_t, 4>(uint32_t*, const uint32_t*, uint64_t, const uint32_t*, uint64_t, uint64_t);
template void EltwiseFMAModRVV32<uint32_t, 8>(uint32_t*, const uint32_t*, uint64_t, const uint32_t*, uint64_t, uint64_t);
template void EltwiseFMAModRVV64<1>(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t);
template void EltwiseFMAModRVV64<2>(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t);
template void EltwiseFMAModRVV64<4>(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t);
template void EltwiseFMAModRVV64<8>(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t);

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
