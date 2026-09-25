// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// RVV kernel for EltwiseReduceMod.

#include "eltwise/eltwise-reduce-mod-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include "util/not-implemented.hpp"
#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

void EltwiseReduceModRVV(uint64_t* result, const uint64_t* operand, uint64_t n,
                         uint64_t modulus, uint64_t input_mod_factor,
                         uint64_t output_mod_factor) {
  // TODO(port-rvv): e64/m1. The 2q/4q cases are vminu chains
  //   (rvv::ReduceFromTwice); the "input_mod_factor == modulus" case is a
  //   vectorised Barrett: Q = vmulhu(x, q_barr); r = x - Q*q; then vminu.
  //   Hoist the branch on (input_mod_factor, output_mod_factor) out of the loop.
  HEXL_NOT_IMPLEMENTED();
}

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
