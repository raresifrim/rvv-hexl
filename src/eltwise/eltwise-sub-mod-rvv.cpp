// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// RVV kernels for EltwiseSubMod. Compiled to nothing without V.

#include "eltwise/eltwise-sub-mod-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include <type_traits>

#include "util/not-implemented.hpp"
#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

template <typename Word>
void EltwiseSubModRVV(Word* result, const Word* operand1, const Word* operand2,
               uint64_t n, uint64_t modulus) {
  // TODO(port-rvv): strip-mined loop. The natural lane width follows the
  // storage (a plain add/sub is load/store bound, so no conversions):
  //   if constexpr (std::is_same_v<Word, uint64_t>) {
  //     for (size_t vl; n > 0; n -= vl, operand1 += vl, operand2 += vl, result += vl) {
  //       vl = __riscv_vsetvl_e64m1(n);   // vle64 x2 -> rvv::SubMod -> vse64
  //     }
  //   } else {                              // uint32_t storage, q < 2^30
  //     ... __riscv_vsetvl_e32m1 / vle32 / rvv::SubMod32 / vse32 ...
  //   }
  HEXL_NOT_IMPLEMENTED();
}

template <typename Word>
void EltwiseSubModRVV(Word* result, const Word* operand1, uint64_t operand2,
               uint64_t n, uint64_t modulus) {
  // TODO(port-rvv): as above with operand2 broadcast (the .vx instruction forms).
  HEXL_NOT_IMPLEMENTED();
}

template void EltwiseSubModRVV<uint64_t>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseSubModRVV<uint32_t>(uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void EltwiseSubModRVV<uint64_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, uint64_t);
template void EltwiseSubModRVV<uint32_t>(uint32_t*, const uint32_t*, uint64_t, uint64_t, uint64_t);

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
