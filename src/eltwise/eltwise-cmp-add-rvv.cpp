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
  // TODO(port-rvv): SEW follows Word (e64 / e32). One vmsXX.vx per predicate
  //   (EQ->vmseq, LT->vmsltu, LE->vmsleu, NE->vmsne, NLT->vmsgeu, NLE->vmsgtu)
  //   producing a mask, then a masked vadd.vx (mask-undisturbed).
  HEXL_NOT_IMPLEMENTED();
}

template void EltwiseCmpAddRVV<uint64_t>(uint64_t*, const uint64_t*, uint64_t, CMPINT, uint64_t, uint64_t);
template void EltwiseCmpAddRVV<uint32_t>(uint32_t*, const uint32_t*, uint64_t, CMPINT, uint64_t, uint64_t);

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
