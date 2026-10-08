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
// Parameters: SEW=e32 whenever q < 2^30; LMUL m4 where a stage's butterfly
// span allows it (the Shoup multiply alone measured 1.5-2.4x over m1 on the
// X100), never below mf2. m8 leaves 4 register groups: a butterfly keeps more
// values live, so expect spills there. Lane types: rvv::cfg::Ntt64 / Ntt32 in
// util/rvv-config.hpp, with rvv::SetVl<V> / rvv::Load<V> / rvv::Store.

#include "ntt/ntt-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include <type_traits>

#include "hexl/number-theory/number-theory.hpp"
#include "util/not-implemented.hpp"
#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

template <typename Word>
	void ForwardTransformToBitReverseRVV32(Word* result,
			const Word* operand, uint64_t n,
			uint64_t modulus, const uint32_t* w,
			const uint32_t* w_precon,
			uint64_t input_mod_factor,
			uint64_t output_mod_factor) {
		// TODO(port-rvv): the binfhe hot path (N = 1024/2048, q ~ 2^27), written
		//   once for both storage types:
		//   * Read the input through rvv::Load32 in the first stage and write the
		//     output through rvv::Store32 in the last stage (not as separate
		//     passes): for Word = uint64_t that is the narrow/widen, for uint32_t a
		//     plain vle32/vse32. Keep [0, 4q) < 2^32 lazily through the stages.
		//   * The middle stages work in place on 32-bit data. For Word = uint32_t
		//     that is result itself; for uint64_t you need a 32-bit working copy
		//     (thread_local, see below) or do every stage through Load32/Store32:
		//       if constexpr (std::is_same_v<Word, uint32_t>) { /* in place */ }
		//       else { /* 32-bit scratch */ }
		//   * No scratch member in the NTT object (concurrent callers). If you need
		//     a 32-bit working buffer use a thread_local AlignedVector64<uint32_t>
		//     sized N, or work directly on result.
		//   * Butterfly: rvv::MulModShoupLazy32 + adds/subs with vminu reductions.
		HEXL_NOT_IMPLEMENTED();
	}

template <typename Word>
	void InverseTransformFromBitReverseRVV32(Word* result,
			const Word* operand, uint64_t n,
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

template void ForwardTransformToBitReverseRVV32<uint64_t>(
		uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint32_t*,
		const uint32_t*, uint64_t, uint64_t);
template void ForwardTransformToBitReverseRVV32<uint32_t>(
		uint32_t*, const uint32_t*, uint64_t, uint64_t, const uint32_t*,
		const uint32_t*, uint64_t, uint64_t);
template void InverseTransformFromBitReverseRVV32<uint64_t>(
		uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint32_t*,
		const uint32_t*, uint64_t, uint64_t);
template void InverseTransformFromBitReverseRVV32<uint32_t>(
		uint32_t*, const uint32_t*, uint64_t, uint64_t, const uint32_t*,
		const uint32_t*, uint64_t, uint64_t);

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
