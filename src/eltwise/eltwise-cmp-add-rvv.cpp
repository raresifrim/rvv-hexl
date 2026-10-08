// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// RVV kernel for EltwiseCmpAdd. Compiled to nothing without V.

#include "eltwise/eltwise-cmp-add-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include <type_traits>

#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

template <typename Word, class V>
	void EltwiseCmpAddRVV(Word* result, const Word* operand1, uint64_t n, CMPINT cmp, uint64_t bound, uint64_t diff) { 
		switch (cmp) {
			case CMPINT::EQ: {
						 if constexpr (std::is_same_v<Word, uint64_t>) {
							 for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
								 vl = rvv::SetVl<V>(n);
								 V v = rvv::Load<V>(operand1, vl);
								 auto mask = __riscv_vmseq(v, bound, vl);
								 V sum  = __riscv_vadd_mu(mask, v, v, diff, vl);
								 rvv::Store(result, sum, vl);	
							 }
						 } else { 
							 const uint32_t bound32 = static_cast<uint32_t>(bound);
							 const uint32_t diff32  = static_cast<uint32_t>(diff);
							 for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
								 vl = rvv::SetVl<V>(n);
								 V v = rvv::Load<V>(operand1, vl);
								 auto mask = __riscv_vmseq(v, bound32, vl);
								 V sum  = __riscv_vadd_mu(mask, v, v, diff32, vl);
								 rvv::Store(result, sum, vl);	
							 }
						 }
						 break;
					 }
			case CMPINT::LT:{
						if constexpr (std::is_same_v<Word, uint64_t>) {
							for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
								vl = rvv::SetVl<V>(n);
								V v = rvv::Load<V>(operand1, vl);
								auto mask = __riscv_vmsltu(v, bound, vl);
								V sum  = __riscv_vadd_mu(mask, v, v, diff, vl);
								rvv::Store(result, sum, vl);	
							}
						} else { 
							const uint32_t bound32 = static_cast<uint32_t>(bound);
							const uint32_t diff32  = static_cast<uint32_t>(diff);
							for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
								vl = rvv::SetVl<V>(n);
								V v = rvv::Load<V>(operand1, vl);
								auto mask = __riscv_vmsltu(v, bound32, vl);
								V sum  = __riscv_vadd_mu(mask, v, v, diff32, vl);
								rvv::Store(result, sum, vl);	
							}
						}
						break;
					}    
			case CMPINT::LE:{
						if constexpr (std::is_same_v<Word, uint64_t>) {
							for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
								vl = rvv::SetVl<V>(n);
								V v = rvv::Load<V>(operand1, vl);
								auto mask = __riscv_vmsleu(v, bound, vl);
								V sum  = __riscv_vadd_mu(mask, v, v, diff, vl);
								rvv::Store(result, sum, vl);	
							}
						} else { 
							const uint32_t bound32 = static_cast<uint32_t>(bound);
							const uint32_t diff32  = static_cast<uint32_t>(diff);
							for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
								vl = rvv::SetVl<V>(n);
								V v = rvv::Load<V>(operand1, vl);
								auto mask = __riscv_vmsleu(v, bound32, vl);
								V sum  = __riscv_vadd_mu(mask, v, v, diff32, vl);
								rvv::Store(result, sum, vl);	
							}
						}
						break;
					}    
			case CMPINT::NE:{
						if constexpr (std::is_same_v<Word, uint64_t>) {
							for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
								vl = rvv::SetVl<V>(n);
								V v = rvv::Load<V>(operand1, vl);
								auto mask = __riscv_vmsne(v, bound, vl);
								V sum  = __riscv_vadd_mu(mask, v, v, diff, vl);
								rvv::Store(result, sum, vl);	
							}
						} else { 
							const uint32_t bound32 = static_cast<uint32_t>(bound);
							const uint32_t diff32  = static_cast<uint32_t>(diff);
							for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
								vl = rvv::SetVl<V>(n);
								V v = rvv::Load<V>(operand1, vl);
								auto mask = __riscv_vmsne(v, bound32, vl);
								V sum  = __riscv_vadd_mu(mask, v, v, diff32, vl);
								rvv::Store(result, sum, vl);	
							}
						}
						break;
					}
			case CMPINT::NLT:{
						 if constexpr (std::is_same_v<Word, uint64_t>) {
							 for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
								 vl = rvv::SetVl<V>(n);
								 V v = rvv::Load<V>(operand1, vl);
								 auto mask = __riscv_vmsgeu(v, bound, vl);
								 V sum  = __riscv_vadd_mu(mask, v, v, diff, vl);
								 rvv::Store(result, sum, vl);	
							 }
						 } else { 
							 const uint32_t bound32 = static_cast<uint32_t>(bound);
							 const uint32_t diff32  = static_cast<uint32_t>(diff);
							 for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
								 vl = rvv::SetVl<V>(n);
								 V v = rvv::Load<V>(operand1, vl);
								 auto mask = __riscv_vmsgeu(v, bound32, vl);
								 V sum  = __riscv_vadd_mu(mask, v, v, diff32, vl);
								 rvv::Store(result, sum, vl);	
							 }
						 }
						 break;
					 }
			case CMPINT::NLE:{
						 if constexpr (std::is_same_v<Word, uint64_t>) {
							 for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
								 vl = rvv::SetVl<V>(n);
								 V v = rvv::Load<V>(operand1, vl);
								 auto mask = __riscv_vmsgtu(v, bound, vl);
								 V sum  = __riscv_vadd_mu(mask, v, v, diff, vl);
								 rvv::Store(result, sum, vl);	
							 }
						 } else { 
							 const uint32_t bound32 = static_cast<uint32_t>(bound);
							 const uint32_t diff32  = static_cast<uint32_t>(diff);
							 for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
								 vl = rvv::SetVl<V>(n);
								 V v = rvv::Load<V>(operand1, vl);
								 auto mask = __riscv_vmsgtu(v, bound32, vl);
								 V sum  = __riscv_vadd_mu(mask, v, v, diff32, vl);
								 rvv::Store(result, sum, vl);	
							 }
						 }
						 break;
					 }
			case CMPINT::TRUE:{
						  if constexpr (std::is_same_v<Word, uint64_t>) {
							  for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
								  vl = rvv::SetVl<V>(n);
								  V v = rvv::Load<V>(operand1, vl);	
								  V sum  = __riscv_vadd(v, diff, vl);
								  rvv::Store(result, sum, vl);	
							  }
						  } else { 
							  const uint32_t diff32  = static_cast<uint32_t>(diff);
							  for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
								  vl = rvv::SetVl<V>(n);
								  V v = rvv::Load<V>(operand1, vl);	
								  V sum  = __riscv_vadd(v, diff32, vl);
								  rvv::Store(result, sum, vl);	
							  }
						  }
						  break;
					  }
			case CMPINT::FALSE:{
						   // no element changes: copy only when not in place (pointer compare)
						   if (result != operand1) {
							   if constexpr (std::is_same_v<Word, uint64_t>) {
								   for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
									   vl = rvv::SetVl<V>(n);
									   V v = rvv::Load<V>(operand1, vl);		
									   rvv::Store(result, v, vl);	
								   }
							   } else { 
								   for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
									   vl = rvv::SetVl<V>(n);
									   V v = rvv::Load<V>(operand1, vl);		
									   rvv::Store(result, v, vl);	
								   }
							   }
						   }
						   break;
					   }
			default:{
					if constexpr (std::is_same_v<Word, uint64_t>) {
						for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
							vl = rvv::SetVl<V>(n);
							V v = rvv::Load<V>(operand1, vl);	
							V sum  = __riscv_vadd(v, diff, vl);
							rvv::Store(result, sum, vl);	
						}
					} else { 
						const uint32_t diff32  = static_cast<uint32_t>(diff);
						for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
							vl = rvv::SetVl<V>(n);
							V v = rvv::Load<V>(operand1, vl);	
							V sum  = __riscv_vadd(v, diff32, vl);
							rvv::Store(result, sum, vl);	
						}
					}
					break;
				}  
		}
	}

template void EltwiseCmpAddRVV<uint64_t, vuint64m1_t>(uint64_t*, const uint64_t*, uint64_t, CMPINT, uint64_t, uint64_t);
template void EltwiseCmpAddRVV<uint64_t, vuint64m2_t>(uint64_t*, const uint64_t*, uint64_t, CMPINT, uint64_t, uint64_t);
template void EltwiseCmpAddRVV<uint64_t, vuint64m4_t>(uint64_t*, const uint64_t*, uint64_t, CMPINT, uint64_t, uint64_t);
template void EltwiseCmpAddRVV<uint64_t, vuint64m8_t>(uint64_t*, const uint64_t*, uint64_t, CMPINT, uint64_t, uint64_t);
template void EltwiseCmpAddRVV<uint32_t, vuint32m1_t>(uint32_t*, const uint32_t*, uint64_t, CMPINT, uint64_t, uint64_t);
template void EltwiseCmpAddRVV<uint32_t, vuint32m2_t>(uint32_t*, const uint32_t*, uint64_t, CMPINT, uint64_t, uint64_t);
template void EltwiseCmpAddRVV<uint32_t, vuint32m4_t>(uint32_t*, const uint32_t*, uint64_t, CMPINT, uint64_t, uint64_t);
template void EltwiseCmpAddRVV<uint32_t, vuint32m8_t>(uint32_t*, const uint32_t*, uint64_t, CMPINT, uint64_t, uint64_t);

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
