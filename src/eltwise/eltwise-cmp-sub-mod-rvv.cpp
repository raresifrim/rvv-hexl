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
#include "util/not-implemented.hpp"
#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

template <typename Word>
void EltwiseCmpSubModRVV(Word* result, const Word* operand1, uint64_t n, uint64_t modulus, CMPINT cmp, uint64_t bound, uint64_t diff) {
  // TODO(port-rvv): mask from vmsXX.vx on the raw values, vectorised Barrett
  //   reduction to [0,q), then a masked rvv::SubMod (e64) / rvv::SubMod32 (e32)
  //   with diff broadcast. SEW follows Word.
  //   Barrett: rvv::BarrettReduce (e64) / rvv::BarrettReduce32 (e32), with the
  //   factor computed ONCE before the strip loop:
  //     q_barr = MultiplyFactor(1, 64, modulus).BarrettFactor();  // floor(2^64/q)
  //     q_barr = MultiplyFactor(1, 32, modulus).BarrettFactor();  // floor(2^32/q), e32
  //   Switch on cmp outside the loop (one loop per CMPINT, or a mask builder).
  HEXL_NOT_IMPLEMENTED();
}

template void EltwiseCmpSubModRVV<uint64_t>(uint64_t*, const uint64_t*, uint64_t, uint64_t, CMPINT, uint64_t, uint64_t);
template void EltwiseCmpSubModRVV<uint32_t>(uint32_t*, const uint32_t*, uint64_t, uint64_t, CMPINT, uint64_t, uint64_t);

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
