// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// RVV kernels for EltwiseMultMod (the NTT-domain Hadamard product).

#include "eltwise/eltwise-mult-mod-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include <type_traits>

#include "util/not-implemented.hpp"
#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

template <typename Word, int InputModFactor>
void EltwiseMultModRVV32(Word* result, const Word* operand1, const Word* operand2, uint64_t n, uint64_t modulus) {
  // TODO(port-rvv): the binfhe case (q ~ 2^27..2^28), written once for both
  //   storage types: vl = __riscv_vsetvl_e32m4(n) per strip (m4: MulModBarrett32
  //   measured 2.2x faster than at m1 on the X100);
  //   rvv::Load32<vuint32m4_t>(operand1, vl) / (operand2, vl) (the overload
  //   picks narrow-from-u64m8 or plain vle32 from Word); reduce inputs to [0, q)
  //   if InputModFactor > 1; rvv::MulModBarrett32; rvv::Store32(result, ..).
  //   There is no fixed multiplier, so Shoup does not apply.
  HEXL_NOT_IMPLEMENTED();
}

template <int InputModFactor>
void EltwiseMultModRVV64(uint64_t* result, const uint64_t* operand1,
                         const uint64_t* operand2, uint64_t n,
                         uint64_t modulus) {
  // TODO(port-rvv): the BFV case (60-bit primes), e64/m4, rvv::MulModBarrett.
  //   Expect ~1 element/cycle from the e64 multiplier on K3: compare against
  //   the native path (HEXL_DISABLE_RVV=1) before assuming RVV wins here.
  HEXL_NOT_IMPLEMENTED();
}

template void EltwiseMultModRVV32<uint64_t, 1>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint64_t, 2>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint64_t, 4>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint32_t, 1>(uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint32_t, 2>(uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV32<uint32_t, 4>(uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV64<1>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV64<2>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
template void EltwiseMultModRVV64<4>(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
