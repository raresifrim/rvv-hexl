// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// RVV kernels for EltwiseAddMod. Compiled to nothing without V.

#include "eltwise/eltwise-add-mod-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include <type_traits>

#include "util/not-implemented.hpp"
#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

template <typename Word>
void EltwiseAddModRVV(Word* result, const Word* operand1, const Word* operand2,
               uint64_t n, uint64_t modulus) {
  // strip-mined loop. The natural lane width follows the storage (a plain add/sub is load/store bound, so no conversions).
  // LMUL=m4: the kernel is bound by per-strip overhead, and m4 measured 2.3-3.4x faster than m1.
  if constexpr (std::is_same_v<Word, uint64_t>) {
       for (size_t vl; n > 0; n -= vl, operand1 += vl, operand2 += vl, result += vl) {
	       vl = __riscv_vsetvl_e64m4(n);   // vle64 x2 -> rvv::AddMod -> vse64
	       vuint64m4_t v1 = __riscv_vle64_v_u64m4(operand1, vl);
	       vuint64m4_t v2 = __riscv_vle64_v_u64m4(operand2, vl);
	       vuint64m4_t sum = rvv::AddMod(v1, v2, modulus, vl);
	       __riscv_vse64_v_u64m4(result, sum, vl);
       }
  } else { // uint32_t storage, q < 2^30 (the dispatcher guarantees it, so q fits the e32 helpers)
        for (size_t vl; n > 0; n -= vl, operand1 += vl, operand2 += vl, result += vl) {
	       vl = __riscv_vsetvl_e32m4(n);   // vle32 x2 -> rvv::AddMod32 -> vse32
	       vuint32m4_t v1 = __riscv_vle32_v_u32m4(operand1, vl);
	       vuint32m4_t v2 = __riscv_vle32_v_u32m4(operand2, vl);
	       vuint32m4_t sum = rvv::AddMod32(v1, v2, static_cast<uint32_t>(modulus), vl);
	       __riscv_vse32_v_u32m4(result, sum, vl);
       }
  }
  
}

template <typename Word>
void EltwiseAddModRVV(Word* result, const Word* operand1, uint64_t operand2,
               uint64_t n, uint64_t modulus) {
  //as above with operand2 broadcast (the .vx instruction forms). 
  if constexpr (std::is_same_v<Word, uint64_t>) {
       for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
	       vl = __riscv_vsetvl_e64m4(n);   // vle64 x1 -> rvv::AddScalarMod -> vse64
	       vuint64m4_t v = __riscv_vle64_v_u64m4(operand1, vl);
	       vuint64m4_t sum = rvv::AddScalarMod(v, operand2, modulus, vl);
	       __riscv_vse64_v_u64m4(result, sum, vl);
       }
  } else { // uint32_t storage, q < 2^30 (operand2 < q, so both fit in 32 bits)
        for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
	       vl = __riscv_vsetvl_e32m4(n);   // vle32 x1 -> rvv::AddScalarMod32 -> vse32
	       vuint32m4_t v = __riscv_vle32_v_u32m4(operand1, vl);
	       vuint32m4_t sum = rvv::AddScalarMod32(v, static_cast<uint32_t>(operand2), static_cast<uint32_t>(modulus), vl);
	       __riscv_vse32_v_u32m4(result, sum, vl);
       }
  }
}

template void EltwiseAddModRVV<uint64_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseAddModRVV<uint32_t>(uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void EltwiseAddModRVV<uint64_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, uint64_t);
template void EltwiseAddModRVV<uint32_t>(uint32_t*, const uint32_t*, uint64_t, uint64_t, uint64_t);

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
