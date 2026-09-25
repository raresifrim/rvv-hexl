// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// RVV kernels for EltwiseMultMod (the NTT-domain Hadamard product).

#include "eltwise/eltwise-mult-mod-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include "util/not-implemented.hpp"
#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

template <int InputModFactor>
void EltwiseMultModRVV32(uint64_t* result, const uint64_t* operand1,
                         const uint64_t* operand2, uint64_t n,
                         uint64_t modulus) {
  // TODO(port-rvv): the binfhe case (q ~ 2^27..2^28).
  //   vl = __riscv_vsetvl_e32m1(n) per strip; rvv::LoadNarrow both operands;
  //   reduce inputs to [0, q) if InputModFactor > 1; rvv::MulModBarrett32;
  //   rvv::StoreWiden. There is no fixed multiplier, so Shoup does not apply.
  HEXL_NOT_IMPLEMENTED();
}

template <int InputModFactor>
void EltwiseMultModRVV64(uint64_t* result, const uint64_t* operand1,
                         const uint64_t* operand2, uint64_t n,
                         uint64_t modulus) {
  // TODO(port-rvv): the BFV case (60-bit primes), e64/m1, rvv::MulModBarrett.
  //   Expect ~1 element/cycle from the e64 multiplier on K3: compare against
  //   the native path (HEXL_DISABLE_RVV=1) before assuming RVV wins here.
  HEXL_NOT_IMPLEMENTED();
}

template void EltwiseMultModRVV32<1>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<2>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<4>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV64<1>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV64<2>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV64<4>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
