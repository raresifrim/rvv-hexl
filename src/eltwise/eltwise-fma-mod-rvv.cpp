// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// RVV kernels for EltwiseFMAMod.

#include "eltwise/eltwise-fma-mod-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include <type_traits>

#include "util/not-implemented.hpp"
#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

template <typename Word, int InputModFactor>
void EltwiseFMAModRVV32(Word* result, const Word* arg1, uint64_t arg2, const Word* arg3, uint64_t n, uint64_t modulus) {
  // TODO(port-rvv): scalar multiplier => Shoup: precompute
  //   w = arg2 mod q and w_precon = MultiplyFactor(w, 32, q).BarrettFactor()
  //   once, then per strip rvv::Load32 (either storage), rvv::MulModShoupLazy32
  //   (.vx forms), reduce [0,2q) -> [0,q), add arg3 with rvv::AddMod32,
  //   rvv::Store32. Split the arg3 == nullptr case into its own loop rather
  //   than testing it per strip.
  HEXL_NOT_IMPLEMENTED();
}

template <int InputModFactor>
void EltwiseFMAModRVV64(uint64_t* result, const uint64_t* arg1, uint64_t arg2,
                        const uint64_t* arg3, uint64_t n, uint64_t modulus) {
  // TODO(port-rvv): as above with 64-bit lanes and a 64-bit Shoup factor.
  HEXL_NOT_IMPLEMENTED();
}

template void EltwiseFMAModRVV32<uint64_t, 1>(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t);
template void EltwiseFMAModRVV32<uint64_t, 2>(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t);
template void EltwiseFMAModRVV32<uint64_t, 4>(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t);
template void EltwiseFMAModRVV32<uint64_t, 8>(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t);
template void EltwiseFMAModRVV32<uint32_t, 1>(uint32_t*, const uint32_t*, uint64_t, const uint32_t*, uint64_t, uint64_t);
template void EltwiseFMAModRVV32<uint32_t, 2>(uint32_t*, const uint32_t*, uint64_t, const uint32_t*, uint64_t, uint64_t);
template void EltwiseFMAModRVV32<uint32_t, 4>(uint32_t*, const uint32_t*, uint64_t, const uint32_t*, uint64_t, uint64_t);
template void EltwiseFMAModRVV32<uint32_t, 8>(uint32_t*, const uint32_t*, uint64_t, const uint32_t*, uint64_t, uint64_t);
template void EltwiseFMAModRVV64<1>(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t);
template void EltwiseFMAModRVV64<2>(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t);
template void EltwiseFMAModRVV64<4>(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t);
template void EltwiseFMAModRVV64<8>(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, uint64_t, uint64_t);

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
