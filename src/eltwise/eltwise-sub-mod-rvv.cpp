// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// RVV kernels for EltwiseSubMod. Compiled to nothing without V.

#include "eltwise/eltwise-sub-mod-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include <type_traits>

#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

template <typename Word, class V>
void EltwiseSubModRVV(Word* result, const Word* operand1, const Word* operand2,
               uint64_t n, uint64_t modulus) {
  // strip-mined loop. The natural lane width follows the storage (a plain add/sub is load/store bound, so no conversions), and
  // LMUL from the lane type V, like EltwiseAddModRVV (default m4, see rvv-config.hpp):
  if constexpr (std::is_same_v<Word, uint64_t>) {
       for (size_t vl; n > 0; n -= vl, operand1 += vl, operand2 += vl, result += vl) {
	       vl = rvv::SetVl<V>(n);   // vle64 x2 -> rvv::SubMod -> vse64
	       V v1 = rvv::Load<V>(operand1, vl);
	       V v2 = rvv::Load<V>(operand2, vl);
	       V diff = rvv::SubMod(v1, v2, modulus, vl);
	       rvv::Store(result, diff, vl);
       }
     } else {  // uint32_t storage, q < 2^30
  //     ... __riscv_vsetvl_e32m4 / vle32 / rvv::SubMod32 / vse32 ...
  	for (size_t vl; n > 0; n -= vl, operand1 += vl, operand2 += vl, result += vl) {
	       vl = rvv::SetVl<V>(n);   // vle32 x2 -> rvv::SubMod32 -> vse32
	       V v1 = rvv::Load<V>(operand1, vl);
	       V v2 = rvv::Load<V>(operand2, vl);
	       V diff = rvv::SubMod32(v1, v2, static_cast<uint32_t>(modulus), vl);
	       rvv::Store(result, diff, vl);
       }
     }
}

template <typename Word, class V>
void EltwiseSubModRVV(Word* result, const Word* operand1, uint64_t operand2,
               uint64_t n, uint64_t modulus) {
  
    if constexpr (std::is_same_v<Word, uint64_t>) {
       for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
	       vl = rvv::SetVl<V>(n);   // vle64 -> rvv::SubMod -> vse64
	       V v = rvv::Load<V>(operand1, vl);
	       V diff = rvv::SubScalarMod(v, operand2, modulus, vl);
	       rvv::Store(result, diff, vl);
       }
     } else {  // uint32_t storage, q < 2^30
  //     ... __riscv_vsetvl_e32m4 / vle32 / rvv::SubMod32 / vse32 ...
  	for (size_t vl; n > 0; n -= vl, operand1 += vl, result += vl) {
	       vl = rvv::SetVl<V>(n);   // vle32 -> rvv::SubMod32 -> vse32
	       V v = rvv::Load<V>(operand1, vl);
	       V diff = rvv::SubScalarMod32(v, static_cast<uint32_t>(operand2), static_cast<uint32_t>(modulus), vl);
	       rvv::Store(result, diff, vl);
       }
     }	
}

template void EltwiseSubModRVV<uint64_t, vuint64m1_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseSubModRVV<uint64_t, vuint64m2_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseSubModRVV<uint64_t, vuint64m4_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseSubModRVV<uint64_t, vuint64m8_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseSubModRVV<uint32_t, vuint32m1_t>(uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void EltwiseSubModRVV<uint32_t, vuint32m2_t>(uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void EltwiseSubModRVV<uint32_t, vuint32m4_t>(uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void EltwiseSubModRVV<uint32_t, vuint32m8_t>(uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void EltwiseSubModRVV<uint64_t, vuint64m1_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, uint64_t);
template void EltwiseSubModRVV<uint64_t, vuint64m2_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, uint64_t);
template void EltwiseSubModRVV<uint64_t, vuint64m4_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, uint64_t);
template void EltwiseSubModRVV<uint64_t, vuint64m8_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, uint64_t);
template void EltwiseSubModRVV<uint32_t, vuint32m1_t>(uint32_t*, const uint32_t*, uint64_t, uint64_t, uint64_t);
template void EltwiseSubModRVV<uint32_t, vuint32m2_t>(uint32_t*, const uint32_t*, uint64_t, uint64_t, uint64_t);
template void EltwiseSubModRVV<uint32_t, vuint32m4_t>(uint32_t*, const uint32_t*, uint64_t, uint64_t, uint64_t);
template void EltwiseSubModRVV<uint32_t, vuint32m8_t>(uint32_t*, const uint32_t*, uint64_t, uint64_t, uint64_t);

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
