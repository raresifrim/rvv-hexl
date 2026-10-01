// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// RVV kernel for EltwiseCmpAdd. Compiled to nothing without V.

#include "eltwise/eltwise-cmp-add-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include <type_traits>

#include "util/not-implemented.hpp"
#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

template <typename Word>
void EltwiseCmpAddRVV(Word* result, const Word* operand1, uint64_t n, CMPINT cmp, uint64_t bound, uint64_t diff) { 
  switch (cmp) {
    case CMPINT::EQ: {
	if constexpr (std::is_same_v<Word, uint64_t>) {
      		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = __riscv_vsetvl_e64m4(n);
			vuint64m4_t v = __riscv_vle64_v_u64m4(operand1, vl);
			vbool16_t mask = __riscv_vmseq_vx_u64m4_b16(v, bound, vl);
			vuint64m4_t sum  = __riscv_vadd_vx_u64m4_mu(mask, v, v, diff, vl);
			__riscv_vse64_v_u64m4(result, sum, vl);	
      		}
	} else { 
		const uint32_t bound32 = static_cast<uint32_t>(bound);
                const uint32_t diff32  = static_cast<uint32_t>(diff);
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = __riscv_vsetvl_e32m4(n);
			vuint32m4_t v = __riscv_vle32_v_u32m4(operand1, vl);
			vbool8_t mask = __riscv_vmseq_vx_u32m4_b8(v, bound32, vl);
			vuint32m4_t sum  = __riscv_vadd_vx_u32m4_mu(mask, v, v, diff32, vl);
			__riscv_vse32_v_u32m4(result, sum, vl);	
      		}
	}
      	break;
    }
    case CMPINT::LT:{
	if constexpr (std::is_same_v<Word, uint64_t>) {
      		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = __riscv_vsetvl_e64m4(n);
			vuint64m4_t v = __riscv_vle64_v_u64m4(operand1, vl);
			vbool16_t mask = __riscv_vmsltu_vx_u64m4_b16(v, bound, vl);
			vuint64m4_t sum  = __riscv_vadd_vx_u64m4_mu(mask, v, v, diff, vl);
			__riscv_vse64_v_u64m4(result, sum, vl);	
      		}
	} else { 
		const uint32_t bound32 = static_cast<uint32_t>(bound);
                const uint32_t diff32  = static_cast<uint32_t>(diff);
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = __riscv_vsetvl_e32m4(n);
			vuint32m4_t v = __riscv_vle32_v_u32m4(operand1, vl);
			vbool8_t mask = __riscv_vmsltu_vx_u32m4_b8(v, bound32, vl);
			vuint32m4_t sum  = __riscv_vadd_vx_u32m4_mu(mask, v, v, diff32, vl);
			__riscv_vse32_v_u32m4(result, sum, vl);	
      		}
	}
      	break;
    }    
    case CMPINT::LE:{
	if constexpr (std::is_same_v<Word, uint64_t>) {
      		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = __riscv_vsetvl_e64m4(n);
			vuint64m4_t v = __riscv_vle64_v_u64m4(operand1, vl);
			vbool16_t mask = __riscv_vmsleu_vx_u64m4_b16(v, bound, vl);
			vuint64m4_t sum  = __riscv_vadd_vx_u64m4_mu(mask, v, v, diff, vl);
			__riscv_vse64_v_u64m4(result, sum, vl);	
      		}
	} else { 
		const uint32_t bound32 = static_cast<uint32_t>(bound);
                const uint32_t diff32  = static_cast<uint32_t>(diff);
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = __riscv_vsetvl_e32m4(n);
			vuint32m4_t v = __riscv_vle32_v_u32m4(operand1, vl);
			vbool8_t mask = __riscv_vmsleu_vx_u32m4_b8(v, bound32, vl);
			vuint32m4_t sum  = __riscv_vadd_vx_u32m4_mu(mask, v, v, diff32, vl);
			__riscv_vse32_v_u32m4(result, sum, vl);	
      		}
	}
      	break;
    }    
    case CMPINT::NE:{
	if constexpr (std::is_same_v<Word, uint64_t>) {
      		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = __riscv_vsetvl_e64m4(n);
			vuint64m4_t v = __riscv_vle64_v_u64m4(operand1, vl);
			vbool16_t mask = __riscv_vmsne_vx_u64m4_b16(v, bound, vl);
			vuint64m4_t sum  = __riscv_vadd_vx_u64m4_mu(mask, v, v, diff, vl);
			__riscv_vse64_v_u64m4(result, sum, vl);	
      		}
	} else { 
		const uint32_t bound32 = static_cast<uint32_t>(bound);
                const uint32_t diff32  = static_cast<uint32_t>(diff);
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = __riscv_vsetvl_e32m4(n);
			vuint32m4_t v = __riscv_vle32_v_u32m4(operand1, vl);
			vbool8_t mask = __riscv_vmsne_vx_u32m4_b8(v, bound32, vl);
			vuint32m4_t sum  = __riscv_vadd_vx_u32m4_mu(mask, v, v, diff32, vl);
			__riscv_vse32_v_u32m4(result, sum, vl);	
      		}
	}
      	break;
    }
    case CMPINT::NLT:{
	if constexpr (std::is_same_v<Word, uint64_t>) {
      		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = __riscv_vsetvl_e64m4(n);
			vuint64m4_t v = __riscv_vle64_v_u64m4(operand1, vl);
			vbool16_t mask = __riscv_vmsgeu_vx_u64m4_b16(v, bound, vl);
			vuint64m4_t sum  = __riscv_vadd_vx_u64m4_mu(mask, v, v, diff, vl);
			__riscv_vse64_v_u64m4(result, sum, vl);	
      		}
	} else { 
		const uint32_t bound32 = static_cast<uint32_t>(bound);
                const uint32_t diff32  = static_cast<uint32_t>(diff);
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = __riscv_vsetvl_e32m4(n);
			vuint32m4_t v = __riscv_vle32_v_u32m4(operand1, vl);
			vbool8_t mask = __riscv_vmsgeu_vx_u32m4_b8(v, bound32, vl);
			vuint32m4_t sum  = __riscv_vadd_vx_u32m4_mu(mask, v, v, diff32, vl);
			__riscv_vse32_v_u32m4(result, sum, vl);	
      		}
	}
      	break;
    }
    case CMPINT::NLE:{
	if constexpr (std::is_same_v<Word, uint64_t>) {
      		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = __riscv_vsetvl_e64m4(n);
			vuint64m4_t v = __riscv_vle64_v_u64m4(operand1, vl);
			vbool16_t mask = __riscv_vmsgtu_vx_u64m4_b16(v, bound, vl);
			vuint64m4_t sum  = __riscv_vadd_vx_u64m4_mu(mask, v, v, diff, vl);
			__riscv_vse64_v_u64m4(result, sum, vl);	
      		}
	} else { 
		const uint32_t bound32 = static_cast<uint32_t>(bound);
                const uint32_t diff32  = static_cast<uint32_t>(diff);
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = __riscv_vsetvl_e32m4(n);
			vuint32m4_t v = __riscv_vle32_v_u32m4(operand1, vl);
			vbool8_t mask = __riscv_vmsgtu_vx_u32m4_b8(v, bound32, vl);
			vuint32m4_t sum  = __riscv_vadd_vx_u32m4_mu(mask, v, v, diff32, vl);
			__riscv_vse32_v_u32m4(result, sum, vl);	
      		}
	}
      	break;
    }
    case CMPINT::TRUE:{
	if constexpr (std::is_same_v<Word, uint64_t>) {
      		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = __riscv_vsetvl_e64m4(n);
			vuint64m4_t v = __riscv_vle64_v_u64m4(operand1, vl);	
			vuint64m4_t sum  = __riscv_vadd_vx_u64m4(v, diff, vl);
			__riscv_vse64_v_u64m4(result, sum, vl);	
      		}
	} else { 
		const uint32_t diff32  = static_cast<uint32_t>(diff);
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = __riscv_vsetvl_e32m4(n);
			vuint32m4_t v = __riscv_vle32_v_u32m4(operand1, vl);	
			vuint32m4_t sum  = __riscv_vadd_vx_u32m4(v, diff32, vl);
			__riscv_vse32_v_u32m4(result, sum, vl);	
      		}
	}
      	break;
    }
    case CMPINT::FALSE:{
	// no element changes: copy only when not in place (pointer compare)
	if (result != operand1) {
		if constexpr (std::is_same_v<Word, uint64_t>) {
	      		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
	        		vl = __riscv_vsetvl_e64m4(n);
				vuint64m4_t v = __riscv_vle64_v_u64m4(operand1, vl);		
				__riscv_vse64_v_u64m4(result, v, vl);	
	      		}
		} else { 
			for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
	        		vl = __riscv_vsetvl_e32m4(n);
				vuint32m4_t v = __riscv_vle32_v_u32m4(operand1, vl);		
				__riscv_vse32_v_u32m4(result, v, vl);	
	      		}
		}
	}
      	break;
    }
    default:{
	if constexpr (std::is_same_v<Word, uint64_t>) {
      		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = __riscv_vsetvl_e64m4(n);
			vuint64m4_t v = __riscv_vle64_v_u64m4(operand1, vl);	
			vuint64m4_t sum  = __riscv_vadd_vx_u64m4(v, diff, vl);
			__riscv_vse64_v_u64m4(result, sum, vl);	
      		}
	} else { 
		const uint32_t diff32  = static_cast<uint32_t>(diff);
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = __riscv_vsetvl_e32m4(n);
			vuint32m4_t v = __riscv_vle32_v_u32m4(operand1, vl);	
			vuint32m4_t sum  = __riscv_vadd_vx_u32m4(v, diff32, vl);
			__riscv_vse32_v_u32m4(result, sum, vl);	
      		}
	}
      	break;
    }  
  }
}

template void EltwiseCmpAddRVV<uint64_t>(uint64_t*, const uint64_t*, uint64_t, CMPINT, uint64_t, uint64_t);
template void EltwiseCmpAddRVV<uint32_t>(uint32_t*, const uint32_t*, uint64_t, CMPINT, uint64_t, uint64_t);

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
