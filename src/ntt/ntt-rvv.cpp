// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// RVV NTT kernels: the RISC-V counterpart of upstream's fwd-ntt-avx512.cpp /
// inv-ntt-avx512.cpp, and the actual deliverable of "Etapa 1".
//
// Suggested order of work, straight from the D01 plan (each step measured on
// both K3 clusters before the next, against HEXL_DISABLE_RVV=1 and against
// stock OpenFHE):
//   1. Flatten the stage loop nest so the compiler / you get ONE inner loop
//      per stage over n/2 butterflies (est. ~1.24x end-to-end on its own).
//   2. Vectorise the final stages (t < VL) ACROSS blocks instead of running
//      them scalar: strided loads (vlse32 with stride 2t), segment loads
//      (vlseg2e32/vlseg4e32 for t = 1, 2) or replicated twiddle tables. The
//      microbenchmark already showed ~2x on X100 for this alone; on A100
//      avoid strided access (in-register transposes instead).
//   3. Block in the vector register file: radix-4 / radix-8 butterflies do
//      2-3 stages per pass over memory, raising arithmetic intensity toward
//      the roofline balance point (A100's 4x larger VRF gains the most).
//   4. Only then hand-tune instruction selection.
// Parameters: SEW=e32 whenever q < 2^30, LMUL=m1, never below mf2.

#include "ntt/ntt-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include "hexl/number-theory/number-theory.hpp"
#include "util/not-implemented.hpp"
#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

void ForwardTransformToBitReverseRVV32(uint64_t* result,
                                       const uint64_t* operand, uint64_t n,
                                       uint64_t modulus, const uint32_t* w,
                                       const uint32_t* w_precon,
                                       uint64_t input_mod_factor,
                                       uint64_t output_mod_factor) {
  // TODO(port-rvv): the binfhe hot path (N = 1024/2048, q ~ 2^27).
  //   * Narrow the 64-bit input to 32-bit lanes once (in the first stage, not
  //     a separate pass), keep [0, 4q) < 2^32 lazily through the stages, and
  //     widen back in the last stage.
  //   * No scratch member in the NTT object (concurrent callers). If you need
  //     a 32-bit working buffer use a thread_local AlignedVector64<uint32_t>
  //     sized N, or work directly on result.
  //   * Butterfly: rvv::MulModShoupLazy32 + adds/subs with vminu reductions.
  HEXL_NOT_IMPLEMENTED();
}

void InverseTransformFromBitReverseRVV32(uint64_t* result,
                                         const uint64_t* operand, uint64_t n,
                                         uint64_t modulus, const uint32_t* w_inv,
                                         const uint32_t* w_inv_precon,
                                         uint64_t input_mod_factor,
                                         uint64_t output_mod_factor) {
  // TODO(port-rvv): mirror of the forward kernel. Here the SMALL-t stages come
  // first, so step 2 of the plan applies to the first stages instead.
  HEXL_NOT_IMPLEMENTED();
}

void ForwardTransformToBitReverseRVV64(
    uint64_t* result, const uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* root_of_unity_powers,
    const uint64_t* precon_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor) {
  // TODO(port-rvv): 64-bit lanes for 49/60-bit primes (BFV). The e64
  // multiplier retires 1 element/cycle on K3, so this is where the proposed
  // modular-multiply ISA extension (Etapa 2) would pay; until then, measure
  // it honestly against the native path.
  HEXL_NOT_IMPLEMENTED();
}

void InverseTransformFromBitReverseRVV64(
    uint64_t* result, const uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor) {
  // TODO(port-rvv): mirror of the 64-bit forward kernel.
  HEXL_NOT_IMPLEMENTED();
}

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
