// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Small-span stages, strategy rvv::ntt::FusedSlides. Contract, terms and the
// twiddle layout: ntt-rvv-small-span.hpp. The kernels call these only when
// tables.vl == VLMAX of V (and N >= 2 VL), so a chunk is exactly one register.
//
// Plan, per chunk of VL contiguous elements (one vle / vse for all small
// stages together; Load32 / Store32 for RVV32, either storage word):
//   for each small stage t (forward VL/2 .. 1, inverse 1 .. VL/2):
//     Y  = vslidedown(v, t)              lane l (l & t == 0) gets its partner
//     W  = per-lane twiddles, Wp = Shoup factors (next VL entries of the table)
//     X', Y' = butterfly on all lanes    only the X lanes (l & t == 0) count
//     v  = vslideup(X', Y', t)           Y' back to lanes l + t
//     v  = vmerge(v, X', mask)           mask = X lanes, from vid: (l & t) == 0;
//                                        only needed for t < VL/2
// Forward: the chunk arrives in [0, 4q) (after the large stages); reduce to
// [0, q) before the store when output_mod_factor == 1. Inverse: load the chunk
// from operand, store it to result (the copy pass disappears).
// Measured in the layout study (whole forward NTT, cycles per butterfly,
// N = 1024): X100 e32 m4 2.64 (transposed 2.04), A100 e32 m4 2.57 (best).

#pragma once

#include "ntt/ntt-rvv-small-span.hpp"
#include "ntt/ntt-rvv-stages.hpp"

#ifdef HEXL_HAS_RVV

namespace intel {
namespace hexl {

/// Forward small-span stages t = VL/2 .. 1 on result (in place), then the
/// output range of output_mod_factor.
template <typename Word, class V>
void ForwardFusedSlidesRVV32(Word* result, uint64_t n,
		uint64_t modulus, const uint32_t* w,
		const uint32_t* w_precon,
		const NttSmallSpanTables<rvv::ntt::FusedSlides, uint32_t>& tables,
		uint64_t output_mod_factor) {
	// Stub: the per-block loops until the fused body is written.
	const uint32_t modulus32 = static_cast<uint32_t>(modulus);
	ForwardStagesRVV32<Word, V>(result, n, modulus32, w, w_precon, tables.vl/2, 1);
	if (output_mod_factor == 1) ReduceFromFourRVV32<Word, V>(result, n, modulus32);
}

/// Inverse small-span stages t = 1 .. VL/2, from operand into result (they
/// may alias).
template <typename Word, class V>
void InverseFusedSlidesRVV32(Word* result,
		const Word* operand, uint64_t n, uint64_t modulus,
		const uint32_t* w_inv, const uint32_t* w_inv_precon,
		const NttSmallSpanTables<rvv::ntt::FusedSlides, uint32_t>& tables) {
	// Stub: copy, then the per-block loops until the fused body is written.
	CopyRVV32<Word, V>(result, operand, n);
	InverseStagesRVV32<Word, V>(result, n, static_cast<uint32_t>(modulus), w_inv, w_inv_precon, 1, tables.vl/2);
}

/// Forward small-span stages, 64-bit lanes.
template <class V>
void ForwardFusedSlidesRVV64(uint64_t* result, uint64_t n,
		uint64_t modulus, const uint64_t* root_of_unity_powers,
		const uint64_t* precon_root_of_unity_powers,
		const NttSmallSpanTables<rvv::ntt::FusedSlides, uint64_t>& tables,
		uint64_t output_mod_factor) {
	// Stub: the per-block loops until the fused body is written.
	ForwardStagesRVV64<V>(result, n, modulus, root_of_unity_powers, precon_root_of_unity_powers, tables.vl/2, 1);
	if (output_mod_factor == 1) ReduceFromFourRVV64<V>(result, n, modulus);
}

/// Inverse small-span stages, 64-bit lanes.
template <class V>
void InverseFusedSlidesRVV64(uint64_t* result,
		const uint64_t* operand, uint64_t n, uint64_t modulus,
		const uint64_t* inv_root_of_unity_powers,
		const uint64_t* precon_inv_root_of_unity_powers,
		const NttSmallSpanTables<rvv::ntt::FusedSlides, uint64_t>& tables) {
	// Stub: copy, then the per-block loops until the fused body is written.
	CopyRVV64<V>(result, operand, n);
	InverseStagesRVV64<V>(result, n, modulus, inv_root_of_unity_powers, precon_inv_root_of_unity_powers, 1, tables.vl/2);
}

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
