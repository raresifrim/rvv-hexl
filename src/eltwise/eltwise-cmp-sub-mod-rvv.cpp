// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// RVV kernel for EltwiseCmpSubMod. Compiled to nothing without V.

#include "eltwise/eltwise-cmp-sub-mod-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include <type_traits>

#include "hexl/number-theory/number-theory.hpp"
#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

template <typename Word, class V>
void EltwiseCmpSubModRVV(Word* result, const Word* operand1, uint64_t n, uint64_t modulus, CMPINT cmp, uint64_t bound, uint64_t diff) { 
  switch (cmp) {
    case CMPINT::EQ: {
	if constexpr (std::is_same_v<Word, uint64_t>) {
      		const uint64_t q_barr = MultiplyFactor(1, 64, modulus).BarrettFactor();  // floor(2^64/q)
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = rvv::SetVl<V>(n);
			V v = rvv::Load<V>(operand1, vl);	
			auto mask = __riscv_vmseq(v, bound, vl);
			v = rvv::BarrettReduce(v, modulus, q_barr, vl);
			V d = __riscv_vsub_mu(mask, v, v, diff, vl);
			V r = __riscv_vadd(d, modulus, vl);	
			d = __riscv_vminu(d, r, vl);
			rvv::Store(result, d, vl);	
      		}
	} else {
		const uint32_t q_barr = static_cast<uint32_t>(MultiplyFactor(1, 32, modulus).BarrettFactor());  // floor(2^32/q)
		const uint32_t bound32 = static_cast<uint32_t>(bound);
                const uint32_t diff32  = static_cast<uint32_t>(diff);
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = rvv::SetVl<V>(n);
			V v = rvv::Load<V>(operand1, vl);
			auto mask = __riscv_vmseq(v, bound32, vl);
			v = rvv::BarrettReduce32(v, static_cast<uint32_t>(modulus), q_barr, vl);
			V d = __riscv_vsub_mu(mask, v, v, diff32, vl);
			V r = __riscv_vadd(d, static_cast<uint32_t>(modulus), vl);	
			d = __riscv_vminu(d, r, vl);
			rvv::Store(result, d, vl);	
      		}
	}
      	break;
    }
    case CMPINT::LT: {
	if constexpr (std::is_same_v<Word, uint64_t>) {
      		const uint64_t q_barr = MultiplyFactor(1, 64, modulus).BarrettFactor();  // floor(2^64/q)
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = rvv::SetVl<V>(n);
			V v = rvv::Load<V>(operand1, vl);	
			auto mask = __riscv_vmsltu(v, bound, vl);
			v = rvv::BarrettReduce(v, modulus, q_barr, vl);
			V d = __riscv_vsub_mu(mask, v, v, diff, vl);
			V r = __riscv_vadd(d, modulus, vl);	
			d = __riscv_vminu(d, r, vl);
			rvv::Store(result, d, vl);	
      		}
	} else {
		const uint32_t q_barr = static_cast<uint32_t>(MultiplyFactor(1, 32, modulus).BarrettFactor());  // floor(2^32/q)
		const uint32_t bound32 = static_cast<uint32_t>(bound);
                const uint32_t diff32  = static_cast<uint32_t>(diff);
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = rvv::SetVl<V>(n);
			V v = rvv::Load<V>(operand1, vl);
			auto mask = __riscv_vmsltu(v, bound32, vl);
			v = rvv::BarrettReduce32(v, static_cast<uint32_t>(modulus), q_barr, vl);
			V d = __riscv_vsub_mu(mask, v, v, diff32, vl);
			V r = __riscv_vadd(d, static_cast<uint32_t>(modulus), vl);	
			d = __riscv_vminu(d, r, vl);
			rvv::Store(result, d, vl);	
      		}
	}
      	break;
    }
    case CMPINT::LE: {
	if constexpr (std::is_same_v<Word, uint64_t>) {
      		const uint64_t q_barr = MultiplyFactor(1, 64, modulus).BarrettFactor();  // floor(2^64/q)
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = rvv::SetVl<V>(n);
			V v = rvv::Load<V>(operand1, vl);	
			auto mask = __riscv_vmsleu(v, bound, vl);
			v = rvv::BarrettReduce(v, modulus, q_barr, vl);
			V d = __riscv_vsub_mu(mask, v, v, diff, vl);
			V r = __riscv_vadd(d, modulus, vl);	
			d = __riscv_vminu(d, r, vl);
			rvv::Store(result, d, vl);	
      		}
	} else {
		const uint32_t q_barr = static_cast<uint32_t>(MultiplyFactor(1, 32, modulus).BarrettFactor());  // floor(2^32/q)
		const uint32_t bound32 = static_cast<uint32_t>(bound);
                const uint32_t diff32  = static_cast<uint32_t>(diff);
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = rvv::SetVl<V>(n);
			V v = rvv::Load<V>(operand1, vl);
			auto mask = __riscv_vmsleu(v, bound32, vl);
			v = rvv::BarrettReduce32(v, static_cast<uint32_t>(modulus), q_barr, vl);
			V d = __riscv_vsub_mu(mask, v, v, diff32, vl);
			V r = __riscv_vadd(d, static_cast<uint32_t>(modulus), vl);	
			d = __riscv_vminu(d, r, vl);
			rvv::Store(result, d, vl);	
      		}
	}
      	break;
    }
    case CMPINT::NE: {
	if constexpr (std::is_same_v<Word, uint64_t>) {
      		const uint64_t q_barr = MultiplyFactor(1, 64, modulus).BarrettFactor();  // floor(2^64/q)
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = rvv::SetVl<V>(n);
			V v = rvv::Load<V>(operand1, vl);	
			auto mask = __riscv_vmsne(v, bound, vl);
			v = rvv::BarrettReduce(v, modulus, q_barr, vl);
			V d = __riscv_vsub_mu(mask, v, v, diff, vl);
			V r = __riscv_vadd(d, modulus, vl);	
			d = __riscv_vminu(d, r, vl);
			rvv::Store(result, d, vl);	
      		}
	} else {
		const uint32_t q_barr = static_cast<uint32_t>(MultiplyFactor(1, 32, modulus).BarrettFactor());  // floor(2^32/q)
		const uint32_t bound32 = static_cast<uint32_t>(bound);
                const uint32_t diff32  = static_cast<uint32_t>(diff);
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = rvv::SetVl<V>(n);
			V v = rvv::Load<V>(operand1, vl);
			auto mask = __riscv_vmsne(v, bound32, vl);
			v = rvv::BarrettReduce32(v, static_cast<uint32_t>(modulus), q_barr, vl);
			V d = __riscv_vsub_mu(mask, v, v, diff32, vl);
			V r = __riscv_vadd(d, static_cast<uint32_t>(modulus), vl);	
			d = __riscv_vminu(d, r, vl);
			rvv::Store(result, d, vl);	
      		}
	}
      	break;
    }
    case CMPINT::NLT: {
	if constexpr (std::is_same_v<Word, uint64_t>) {
      		const uint64_t q_barr = MultiplyFactor(1, 64, modulus).BarrettFactor();  // floor(2^64/q)
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = rvv::SetVl<V>(n);
			V v = rvv::Load<V>(operand1, vl);	
			auto mask = __riscv_vmsgeu(v, bound, vl);
			v = rvv::BarrettReduce(v, modulus, q_barr, vl);
			V d = __riscv_vsub_mu(mask, v, v, diff, vl);
			V r = __riscv_vadd(d, modulus, vl);	
			d = __riscv_vminu(d, r, vl);
			rvv::Store(result, d, vl);	
      		}
	} else {
		const uint32_t q_barr = static_cast<uint32_t>(MultiplyFactor(1, 32, modulus).BarrettFactor());  // floor(2^32/q)
		const uint32_t bound32 = static_cast<uint32_t>(bound);
                const uint32_t diff32  = static_cast<uint32_t>(diff);
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = rvv::SetVl<V>(n);
			V v = rvv::Load<V>(operand1, vl);
			auto mask = __riscv_vmsgeu(v, bound32, vl);
			v = rvv::BarrettReduce32(v, static_cast<uint32_t>(modulus), q_barr, vl);
			V d = __riscv_vsub_mu(mask, v, v, diff32, vl);
			V r = __riscv_vadd(d, static_cast<uint32_t>(modulus), vl);	
			d = __riscv_vminu(d, r, vl);
			rvv::Store(result, d, vl);	
      		}
	}
      	break;
    }
    case CMPINT::NLE: {
	if constexpr (std::is_same_v<Word, uint64_t>) {
      		const uint64_t q_barr = MultiplyFactor(1, 64, modulus).BarrettFactor();  // floor(2^64/q)
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = rvv::SetVl<V>(n);
			V v = rvv::Load<V>(operand1, vl);	
			auto mask = __riscv_vmsgtu(v, bound, vl);
			v = rvv::BarrettReduce(v, modulus, q_barr, vl);
			V d = __riscv_vsub_mu(mask, v, v, diff, vl);
			V r = __riscv_vadd(d, modulus, vl);	
			d = __riscv_vminu(d, r, vl);
			rvv::Store(result, d, vl);	
      		}
	} else {
		const uint32_t q_barr = static_cast<uint32_t>(MultiplyFactor(1, 32, modulus).BarrettFactor());  // floor(2^32/q)
		const uint32_t bound32 = static_cast<uint32_t>(bound);
                const uint32_t diff32  = static_cast<uint32_t>(diff);
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = rvv::SetVl<V>(n);
			V v = rvv::Load<V>(operand1, vl);
			auto mask = __riscv_vmsgtu(v, bound32, vl);
			v = rvv::BarrettReduce32(v, static_cast<uint32_t>(modulus), q_barr, vl);
			V d = __riscv_vsub_mu(mask, v, v, diff32, vl);
			V r = __riscv_vadd(d, static_cast<uint32_t>(modulus), vl);	
			d = __riscv_vminu(d, r, vl);
			rvv::Store(result, d, vl);	
      		}
	}
      	break;
    }
    case CMPINT::TRUE: {
	// every element: reduce mod q, then subtract diff mod q
	if constexpr (std::is_same_v<Word, uint64_t>) {
      		const uint64_t q_barr = MultiplyFactor(1, 64, modulus).BarrettFactor();  // floor(2^64/q)
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = rvv::SetVl<V>(n);
			V v = rvv::Load<V>(operand1, vl);	
			v = rvv::BarrettReduce(v, modulus, q_barr, vl);
			V d = __riscv_vsub(v, diff, vl);
			V r = __riscv_vadd(d, modulus, vl);	
			d = __riscv_vminu(d, r, vl);
			rvv::Store(result, d, vl);	
      		}
	} else {
		const uint32_t q_barr = static_cast<uint32_t>(MultiplyFactor(1, 32, modulus).BarrettFactor());  // floor(2^32/q)
                const uint32_t diff32  = static_cast<uint32_t>(diff);
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = rvv::SetVl<V>(n);
			V v = rvv::Load<V>(operand1, vl);
			v = rvv::BarrettReduce32(v, static_cast<uint32_t>(modulus), q_barr, vl);
			V d = __riscv_vsub(v, diff32, vl);
			V r = __riscv_vadd(d, static_cast<uint32_t>(modulus), vl);	
			d = __riscv_vminu(d, r, vl);
			rvv::Store(result, d, vl);	
      		}
	}
      	break;
    }
    case CMPINT::FALSE: {
	// no element is shifted, but every element is still reduced mod q
	if constexpr (std::is_same_v<Word, uint64_t>) {
      		const uint64_t q_barr = MultiplyFactor(1, 64, modulus).BarrettFactor();  // floor(2^64/q)
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = rvv::SetVl<V>(n);
			V v = rvv::Load<V>(operand1, vl);	
			v = rvv::BarrettReduce(v, modulus, q_barr, vl);
			rvv::Store(result, v, vl);	
      		}
	} else {
		const uint32_t q_barr = static_cast<uint32_t>(MultiplyFactor(1, 32, modulus).BarrettFactor());  // floor(2^32/q)
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = rvv::SetVl<V>(n);
			V v = rvv::Load<V>(operand1, vl);
			v = rvv::BarrettReduce32(v, static_cast<uint32_t>(modulus), q_barr, vl);
			rvv::Store(result, v, vl);	
      		}
	}
      	break;
    }
    default: {
	// not a valid CMPINT; the scalar Compare() treats unknown values as TRUE
	if constexpr (std::is_same_v<Word, uint64_t>) {
      		const uint64_t q_barr = MultiplyFactor(1, 64, modulus).BarrettFactor();  // floor(2^64/q)
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = rvv::SetVl<V>(n);
			V v = rvv::Load<V>(operand1, vl);	
			v = rvv::BarrettReduce(v, modulus, q_barr, vl);
			V d = __riscv_vsub(v, diff, vl);
			V r = __riscv_vadd(d, modulus, vl);	
			d = __riscv_vminu(d, r, vl);
			rvv::Store(result, d, vl);	
      		}
	} else {
		const uint32_t q_barr = static_cast<uint32_t>(MultiplyFactor(1, 32, modulus).BarrettFactor());  // floor(2^32/q)
                const uint32_t diff32  = static_cast<uint32_t>(diff);
		for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
        		vl = rvv::SetVl<V>(n);
			V v = rvv::Load<V>(operand1, vl);
			v = rvv::BarrettReduce32(v, static_cast<uint32_t>(modulus), q_barr, vl);
			V d = __riscv_vsub(v, diff32, vl);
			V r = __riscv_vadd(d, static_cast<uint32_t>(modulus), vl);	
			d = __riscv_vminu(d, r, vl);
			rvv::Store(result, d, vl);	
      		}
	}
      	break;
    }
  }
}

template void EltwiseCmpSubModRVV<uint64_t, vuint64m1_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, CMPINT, uint64_t, uint64_t);
template void EltwiseCmpSubModRVV<uint64_t, vuint64m2_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, CMPINT, uint64_t, uint64_t);
template void EltwiseCmpSubModRVV<uint64_t, vuint64m4_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, CMPINT, uint64_t, uint64_t);
template void EltwiseCmpSubModRVV<uint64_t, vuint64m8_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, CMPINT, uint64_t, uint64_t);
template void EltwiseCmpSubModRVV<uint32_t, vuint32m1_t>(uint32_t*, const uint32_t*, uint64_t, uint64_t, CMPINT, uint64_t, uint64_t);
template void EltwiseCmpSubModRVV<uint32_t, vuint32m2_t>(uint32_t*, const uint32_t*, uint64_t, uint64_t, CMPINT, uint64_t, uint64_t);
template void EltwiseCmpSubModRVV<uint32_t, vuint32m4_t>(uint32_t*, const uint32_t*, uint64_t, uint64_t, CMPINT, uint64_t, uint64_t);
template void EltwiseCmpSubModRVV<uint32_t, vuint32m8_t>(uint32_t*, const uint32_t*, uint64_t, uint64_t, CMPINT, uint64_t, uint64_t);

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
