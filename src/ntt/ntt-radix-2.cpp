// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Native (portable scalar) NTT. This is simultaneously
//   * the fallback whenever RVV is absent or HEXL_DISABLE_RVV=1,
//   * the "base RISC-V, no RVV" data point (make ISA=scalar),
//   * the correctness reference the RVV kernels are diffed against.
// Write it first and make every NTT test pass with it before touching RVV.

#include <cstring>

#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "ntt/ntt-internal.hpp"
#include "util/not-implemented.hpp"

namespace intel {
namespace hexl {

template <typename Word>
	void ForwardTransformToBitReverseRadix2(
			Word* result, const Word* operand, uint64_t n, uint64_t modulus,
			const uint64_t* root_of_unity_powers,
			const uint64_t* precon_root_of_unity_powers, uint64_t input_mod_factor,
			uint64_t output_mod_factor) {
		// TODO(port): Cooley-Tukey, decimation in time, natural -> bit-reversed.
		//
		//   if (result != operand) copy operand into result, then work in place.
		//   for (t = n/2, m = 1; m < n; m <<= 1, t >>= 1)      // log2(n) stages
		//     for (i = 0; i < m; i++)                          // m blocks
		//       W = root_of_unity_powers[m + i]; Wp = precon_...[m + i];
		//       for (j = 2*i*t; j < 2*i*t + t; j++)            // t butterflies
		//         X = result[j], Y = result[j + t]
		//         Harvey butterfly, values kept in [0, 4q):
		//           X  = (X >= 2q) ? X - 2q : X
		//           T  = MultiplyModLazy<64>(Y, W, Wp, q)      // in [0, 2q)
		//           result[j]     = X + T                      // [0, 4q)
		//           result[j + t] = X - T + 2q                 // [0, 4q)
		//   Finally reduce to [0, output_mod_factor * q) (4 = leave lazy, 1 = two
		//   conditional subtractions). input_mod_factor 1/2/4 all fit the [0, 4q)
		//   invariant, so it needs no extra work at the start.
		//
		//   This loop nest (two consecutive inner loops over the same data) is the
		//   one GCC refuses to auto-vectorise in OpenFHE (D01, "Etapa 1", item 1).
		//   Keep it simple here; the restructuring belongs in the RVV kernel.
		//
		//   Word = uint32_t: q can be anything up to 2^32 here (the RVV path takes
		//   q < 2^30), so the lazy [0, 4q) values do NOT fit in 32 bits. Load each
		//   pair into uint64_t, do the butterfly in 64-bit, store back. Only the
		//   final output_mod_factor * q <= 2^32 is guaranteed by the caller.
		HEXL_NOT_IMPLEMENTED();
	}

template <typename Word>
	void InverseTransformFromBitReverseRadix2(
			Word* result, const Word* operand, uint64_t n, uint64_t modulus,
			const uint64_t* inv_root_of_unity_powers,
			const uint64_t* precon_inv_root_of_unity_powers, uint64_t input_mod_factor,
			uint64_t output_mod_factor) {
		// TODO(port): Gentleman-Sande, decimation in frequency, bit-reversed ->
		// natural, then multiply every element by N^{-1} mod q.
		//
		//   for (m = n/2, t = 1; m >= 1; m >>= 1, t <<= 1)
		//     for (i = 0; i < m; i++)
		//       W = next inverse twiddle (table order is whatever
		//           NTT::ComputeRootOfUnityPowers stored; upstream: sequential)
		//       for (j over the t butterflies of block i)
		//         X, Y in [0, 2q):
		//           result[j]     = X + Y reduced to [0, 2q)
		//           result[j + t] = MultiplyModLazy<64>(X - Y + 2q, W, Wp, q)
		//   Fold the N^{-1} scaling into the last stage (saves one full pass):
		//   the last stage multiplies by N^{-1} and by W * N^{-1} (precompute both
		//   and their Shoup factors), then reduce to output_mod_factor.
		//   Word = uint32_t: same remark as the forward kernel (64-bit temporaries).
		HEXL_NOT_IMPLEMENTED();
	}

void ReferenceForwardTransformToBitReverse(
		uint64_t* operand, uint64_t n, uint64_t modulus,
		const uint64_t* root_of_unity_powers) {
	// TODO(port): same loop nest as the radix-2 forward, but with fully reduced
	// arithmetic (MultiplyMod / AddUIntMod / SubUIntMod) and no lazy tricks.
	HEXL_NOT_IMPLEMENTED();
}

void ReferenceInverseTransformFromBitReverse(
		uint64_t* operand, uint64_t n, uint64_t modulus,
		const uint64_t* inv_root_of_unity_powers) {
	// TODO(port): textbook Gentleman-Sande + N^{-1} scaling, fully reduced.
	HEXL_NOT_IMPLEMENTED();
}

// Both storage types are dispatched from ntt.cpp.
template void ForwardTransformToBitReverseRadix2<uint64_t>(
		uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint64_t*,
		const uint64_t*, uint64_t, uint64_t);
template void ForwardTransformToBitReverseRadix2<uint32_t>(
		uint32_t*, const uint32_t*, uint64_t, uint64_t, const uint64_t*,
		const uint64_t*, uint64_t, uint64_t);
template void InverseTransformFromBitReverseRadix2<uint64_t>(
		uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint64_t*,
		const uint64_t*, uint64_t, uint64_t);
template void InverseTransformFromBitReverseRadix2<uint32_t>(
		uint32_t*, const uint32_t*, uint64_t, uint64_t, const uint64_t*,
		const uint64_t*, uint64_t, uint64_t);

}  // namespace hexl
}  // namespace intel
