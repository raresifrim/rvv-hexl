// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// RVV kernel for EltwiseReduceMod. Compiled to nothing without V.

#include "eltwise/eltwise-reduce-mod-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include <type_traits>

#include "util/not-implemented.hpp"
#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

template <typename Word>
void EltwiseReduceModRVV(Word* result, const Word* operand, uint64_t n, uint64_t modulus, uint64_t input_mod_factor, uint64_t output_mod_factor) {
  // TODO(port-rvv): the 2q/4q cases are vminu chains (rvv::ReduceFromTwice);
  //   the "input_mod_factor == modulus" case is a vectorised Barrett:
  //   Q = vmulhu(x, q_barr); r = x - Q*q; then vminu. e64 lanes for
  //   Word = uint64_t, e32 lanes for Word = uint32_t (q < 2^30 there; use a
  //   32-bit Barrett factor). Hoist the branch on the mod factors out of the loop.
  HEXL_NOT_IMPLEMENTED();
}

template void EltwiseReduceModRVV<uint64_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, uint64_t, uint64_t);
template void EltwiseReduceModRVV<uint32_t>(uint32_t*, const uint32_t*, uint64_t, uint64_t, uint64_t, uint64_t);

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
