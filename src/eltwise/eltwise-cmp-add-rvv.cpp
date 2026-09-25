// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// RVV kernel for EltwiseCmpAdd.

#include "eltwise/eltwise-cmp-add-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include "util/not-implemented.hpp"
#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

void EltwiseCmpAddRVV(uint64_t* result, const uint64_t* operand1,
                        uint64_t n, CMPINT cmp, uint64_t bound,
                        uint64_t diff) {
  // TODO(port-rvv): e64/m1. One vmsXX.vx per predicate (EQ->vmseq, LT->vmsltu,
  //   LE->vmsleu, NE->vmsne, NLT->vmsgeu == !LT, NLE->vmsgtu == !LE) producing a
  //   mask, then a masked vadd.vx (mask-undisturbed, merge with the input).
  HEXL_NOT_IMPLEMENTED();
}

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
