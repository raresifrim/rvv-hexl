// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// RVV kernel for EltwiseCmpSubMod.

#include "eltwise/eltwise-cmp-sub-mod-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include "util/not-implemented.hpp"
#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

void EltwiseCmpSubModRVV(uint64_t* result, const uint64_t* operand1,
                        uint64_t n, uint64_t modulus, CMPINT cmp,
                        uint64_t bound, uint64_t diff) {
  // TODO(port-rvv): mask from vmsXX.vx on the raw values, vectorised Barrett
  //   reduction to [0,q), then a masked rvv::SubMod with diff broadcast.
  HEXL_NOT_IMPLEMENTED();
}

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
