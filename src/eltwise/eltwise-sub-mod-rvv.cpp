// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// RVV kernels for EltwiseSubMod. Compiled to nothing without V.

#include "eltwise/eltwise-sub-mod-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include "util/not-implemented.hpp"
#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

void EltwiseSubModRVV(uint64_t* result, const uint64_t* operand1,
                       const uint64_t* operand2, uint64_t n, uint64_t modulus) {
  // TODO(port-rvv): strip-mined loop, e64/m1 (a plain add/sub gains nothing
  // from narrowing to e32: it is load/store bound, measure before trying):
  //   for (size_t vl; n > 0; n -= vl, operand1 += vl, operand2 += vl, result += vl) {
  //     vl = __riscv_vsetvl_e64m1(n);
  //     ... vle64 x2 -> rvv::SubMod -> vse64 ...
  //   }
  HEXL_NOT_IMPLEMENTED();
}

void EltwiseSubModRVV(uint64_t* result, const uint64_t* operand1,
                       uint64_t operand2, uint64_t n, uint64_t modulus) {
  // TODO(port-rvv): as above with operand2 broadcast (the .vx instruction forms).
  HEXL_NOT_IMPLEMENTED();
}

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
