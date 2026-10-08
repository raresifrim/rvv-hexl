// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// RVV kernel for EltwiseReduceMod. Compiled to nothing without V.

#include "eltwise/eltwise-reduce-mod-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include <type_traits>

#include "hexl/number-theory/number-theory.hpp"
#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

template <typename Word, class V>
	void EltwiseReduceModRVV(Word* result, const Word* operand, uint64_t n, uint64_t modulus, uint64_t input_mod_factor, uint64_t output_mod_factor) {

		if (input_mod_factor == modulus && output_mod_factor == 1){
			uint64_t q_barr; 
			if constexpr (std::is_same_v<Word, uint64_t>) 
				q_barr = MultiplyFactor(1, 64, modulus).BarrettFactor();
			else	
				q_barr = MultiplyFactor(1, 32, modulus).BarrettFactor();
			for(size_t vl; n>0; n-=vl, operand+=vl, result+=vl){
				vl = rvv::SetVl<V>(n);
				V vs = rvv::Load<V>(operand, vl);
				V vd;
				if constexpr (std::is_same_v<Word,uint64_t>)
					vd = rvv::BarrettReduce<1>(vs, modulus, q_barr, vl);
				else
					vd = rvv::BarrettReduce32<1>(vs, static_cast<uint32_t>(modulus), static_cast<uint32_t>(q_barr), vl);
				rvv::Store<V>(result, vd, vl);
			}
		}
		else if (input_mod_factor == modulus && output_mod_factor == 2){
			uint64_t q_barr; 
			if constexpr (std::is_same_v<Word, uint64_t>) 
				q_barr = MultiplyFactor(1, 64, modulus).BarrettFactor();
			else	
				q_barr = MultiplyFactor(1, 32, modulus).BarrettFactor();
			for(size_t vl; n>0; n-=vl, operand+=vl, result+=vl){
				vl = rvv::SetVl<V>(n);
				V vs = rvv::Load<V>(operand, vl);
				V vd;
				if constexpr (std::is_same_v<Word,uint64_t>)
					vd = rvv::BarrettReduce<2>(vs, modulus, q_barr, vl);
				else
					vd = rvv::BarrettReduce32<2>(vs, static_cast<uint32_t>(modulus), static_cast<uint32_t>(q_barr), vl);
				rvv::Store<V>(result, vd, vl);
			}  
		}
		else if (input_mod_factor == 4 && output_mod_factor == 1) { 
			for(size_t vl; n>0; n-=vl, operand+=vl, result+=vl){
				vl = rvv::SetVl<V>(n);
				V vs = rvv::Load<V>(operand, vl);
				V vd;
				if constexpr (std::is_same_v<Word,uint64_t>) {
					vd = rvv::ReduceFromTwice(vs, 2*modulus, vl);
					vd = rvv::ReduceFromTwice(vd, modulus, vl);
				} else {
					vd = rvv::ReduceFromTwice32(vs, static_cast<uint32_t>(2*modulus), vl);
					vd = rvv::ReduceFromTwice32(vd, static_cast<uint32_t>(modulus), vl);
				}
				rvv::Store<V>(result, vd, vl);
			}
		}
		else if (input_mod_factor == 4 && output_mod_factor == 2) { 
			for(size_t vl; n>0; n-=vl, operand+=vl, result+=vl){
				vl = rvv::SetVl<V>(n);
				V vs = rvv::Load<V>(operand, vl);
				V vd;
				if constexpr (std::is_same_v<Word,uint64_t>)
					vd = rvv::ReduceFromTwice(vs, 2*modulus, vl);
				else
					vd = rvv::ReduceFromTwice32(vs, static_cast<uint32_t>(2*modulus), vl);
				rvv::Store<V>(result, vd, vl);
			} 
		}
		else if (input_mod_factor == 2 && output_mod_factor == 1) { 
			for(size_t vl; n>0; n-=vl, operand+=vl, result+=vl){
				vl = rvv::SetVl<V>(n);
				V vs = rvv::Load<V>(operand, vl);
				V vd;
				if constexpr (std::is_same_v<Word,uint64_t>)
					vd = rvv::ReduceFromTwice(vs, modulus, vl);
				else
					vd = rvv::ReduceFromTwice32(vs, static_cast<uint32_t>(modulus), vl);
				rvv::Store<V>(result, vd, vl);
			}  
		}
	}

template void EltwiseReduceModRVV<uint64_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, uint64_t, uint64_t);
template void EltwiseReduceModRVV<uint32_t>(uint32_t*, const uint32_t*, uint64_t, uint64_t, uint64_t, uint64_t);

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
