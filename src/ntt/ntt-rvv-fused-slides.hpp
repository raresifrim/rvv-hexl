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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wregister"

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

	const register uint32_t modulus32 = static_cast<uint32_t>(modulus);
	const register uint64_t vl = tables.vl;
	const register uint32_t* tw  = tables.fwd.data();
	const register uint32_t* twp = tables.fwd_precon.data();

	V lane; //get id of each lane
	if constexpr (std::is_same_v<V, vuint32m1_t>) {
		lane = __riscv_vid_v_u32m1(vl);
	} else if constexpr (std::is_same_v<V, vuint32m2_t>) {
		lane = __riscv_vid_v_u32m2(vl);
	} else if constexpr (std::is_same_v<V, vuint32m4_t>) {
		lane = __riscv_vid_v_u32m4(vl);
	} else if constexpr (std::is_same_v<V, vuint32m8_t>) {
		lane = __riscv_vid_v_u32m8(vl);
	}

	for (uint64_t c = 0; c < n/vl; ++c){
		V r = rvv::Load32<V>(result + c * vl, vl);

		//perform first stage outside inner loop as its doesn't require merging
		V W = rvv::Load32<V>(tw, vl); tw += vl;
		V Wp = rvv::Load32<V>(twp, vl); twp += vl;
		V x = r;
		V y = __riscv_vslidedown(r, vl/2, vl);
		rvv::FwdButterfly32(x, y, W, Wp, modulus32, vl);
		r = __riscv_vslideup(x, y, vl/2, vl);
		//rest of the stages will always require a merge
		for (uint64_t t = vl/4; t >= 1; t >>= 1, tw += vl, twp += vl){
			V W = rvv::Load32<V>(tw, vl);
			V Wp = rvv::Load32<V>(twp, vl);
			auto mask = __riscv_vmseq(__riscv_vand(lane, t, vl), 0, vl); //mask based on lane id and current stage

			V x = r;
			V y = __riscv_vslidedown(r, t, vl);
			rvv::FwdButterfly32(x, y, W, Wp, modulus32, vl);
			r = __riscv_vslideup(x, y, t, vl);
			r = __riscv_vmerge(r, x, mask, vl);
		}

		if (output_mod_factor == 1) {
			r = rvv::ReduceFromTwice32(r, 2 * modulus32, vl);
			r = rvv::ReduceFromTwice32(r, modulus32, vl);
		}

		rvv::Store32<V>(result + c * vl, r, vl);
	}
}

/// Inverse small-span stages t = 1 .. VL/2, from operand into result (they
/// may alias).
template <typename Word, class V>
void InverseFusedSlidesRVV32(Word* result,
		const Word* operand, uint64_t n, uint64_t modulus,
		const uint32_t* w_inv, const uint32_t* w_inv_precon,
		const NttSmallSpanTables<rvv::ntt::FusedSlides, uint32_t>& tables) {
	const register uint32_t modulus32 = static_cast<uint32_t>(modulus);
	const register uint64_t vl = tables.vl;
	const register uint32_t* tw  = tables.inv.data();
	const register uint32_t* twp = tables.inv_precon.data();

	V lane; //get id of each lane
	if constexpr (std::is_same_v<V, vuint32m1_t>) {
		lane = __riscv_vid_v_u32m1(vl);
	} else if constexpr (std::is_same_v<V, vuint32m2_t>) {
		lane = __riscv_vid_v_u32m2(vl);
	} else if constexpr (std::is_same_v<V, vuint32m4_t>) {
		lane = __riscv_vid_v_u32m4(vl);
	} else if constexpr (std::is_same_v<V, vuint32m8_t>) {
		lane = __riscv_vid_v_u32m8(vl);
	}

	for (uint64_t c = 0; c < n/vl; ++c){
		V r = rvv::Load32<V>(operand + c * vl, vl);

		//stages t = 1 .. vl/4 (inverse order) always require a merge
		for (uint64_t t = 1; t <= vl/4; t <<= 1, tw += vl, twp += vl){
			V W = rvv::Load32<V>(tw, vl);
			V Wp = rvv::Load32<V>(twp, vl);
			auto mask = __riscv_vmseq(__riscv_vand(lane, t, vl), 0, vl); //mask based on lane id and current stage

			V x = r;
			V y = __riscv_vslidedown(r, t, vl);
			rvv::InvButterfly32(x, y, W, Wp, modulus32, vl);
			r = __riscv_vslideup(x, y, t, vl);
			r = __riscv_vmerge(r, x, mask, vl);
		}
		//perform last stage t = vl/2 outside the loop as it doesn't require merging
		V W = rvv::Load32<V>(tw, vl); tw += vl;
		V Wp = rvv::Load32<V>(twp, vl); twp += vl;
		V x = r;
		V y = __riscv_vslidedown(r, vl/2, vl);
		rvv::InvButterfly32(x, y, W, Wp, modulus32, vl);
		r = __riscv_vslideup(x, y, vl/2, vl);

		rvv::Store32<V>(result + c * vl, r, vl);
	}
}

/// Forward small-span stages, 64-bit lanes.
template <class V>
void ForwardFusedSlidesRVV64(uint64_t* result, uint64_t n,
		uint64_t modulus, const uint64_t* root_of_unity_powers,
		const uint64_t* precon_root_of_unity_powers,
		const NttSmallSpanTables<rvv::ntt::FusedSlides, uint64_t>& tables,
		uint64_t output_mod_factor) {
	const register uint64_t vl = tables.vl;
	const register uint64_t* tw  = tables.fwd.data();
	const register uint64_t* twp = tables.fwd_precon.data();

	V lane; //get id of each lane
	if constexpr (std::is_same_v<V, vuint64m1_t>) {
		lane = __riscv_vid_v_u64m1(vl);
	} else if constexpr (std::is_same_v<V, vuint64m2_t>) {
		lane = __riscv_vid_v_u64m2(vl);
	} else if constexpr (std::is_same_v<V, vuint64m4_t>) {
		lane = __riscv_vid_v_u64m4(vl);
	} else if constexpr (std::is_same_v<V, vuint64m8_t>) {
		lane = __riscv_vid_v_u64m8(vl);
	}

	for (uint64_t c = 0; c < n/vl; ++c){
		V r = rvv::Load<V>(result + c * vl, vl);

		//perform first stage outside inner loop as its doesn't require merging
		V W = rvv::Load<V>(tw, vl); tw += vl;
		V Wp = rvv::Load<V>(twp, vl); twp += vl;
		V x = r;
		V y = __riscv_vslidedown(r, vl/2, vl);
		rvv::FwdButterfly(x, y, W, Wp, modulus, vl);
		r = __riscv_vslideup(x, y, vl/2, vl);
		//rest of the stages will always require a merge
		for (uint64_t t = vl/4; t >= 1; t >>= 1, tw += vl, twp += vl){
			V W = rvv::Load<V>(tw, vl);
			V Wp = rvv::Load<V>(twp, vl);
			auto mask = __riscv_vmseq(__riscv_vand(lane, t, vl), 0, vl); //mask based on lane id and current stage

			V x = r;
			V y = __riscv_vslidedown(r, t, vl);
			rvv::FwdButterfly(x, y, W, Wp, modulus, vl);
			r = __riscv_vslideup(x, y, t, vl);
			r = __riscv_vmerge(r, x, mask, vl);
		}

		if (output_mod_factor == 1) {
			r = rvv::ReduceFromTwice(r, 2 * modulus, vl);
			r = rvv::ReduceFromTwice(r, modulus, vl);
		}

		rvv::Store(result + c * vl, r, vl);
	}
}

/// Inverse small-span stages, 64-bit lanes.
template <class V>
void InverseFusedSlidesRVV64(uint64_t* result,
		const uint64_t* operand, uint64_t n, uint64_t modulus,
		const uint64_t* inv_root_of_unity_powers,
		const uint64_t* precon_inv_root_of_unity_powers,
		const NttSmallSpanTables<rvv::ntt::FusedSlides, uint64_t>& tables) {
	const register uint64_t vl = tables.vl;
	const register uint64_t* tw  = tables.inv.data();
	const register uint64_t* twp = tables.inv_precon.data();

	V lane; //get id of each lane
	if constexpr (std::is_same_v<V, vuint64m1_t>) {
		lane = __riscv_vid_v_u64m1(vl);
	} else if constexpr (std::is_same_v<V, vuint64m2_t>) {
		lane = __riscv_vid_v_u64m2(vl);
	} else if constexpr (std::is_same_v<V, vuint64m4_t>) {
		lane = __riscv_vid_v_u64m4(vl);
	} else if constexpr (std::is_same_v<V, vuint64m8_t>) {
		lane = __riscv_vid_v_u64m8(vl);
	}

	for (uint64_t c = 0; c < n/vl; ++c){
		V r = rvv::Load<V>(operand + c * vl, vl);

		//stages t = 1 .. vl/4 (inverse order) always require a merge
		for (uint64_t t = 1; t <= vl/4; t <<= 1, tw += vl, twp += vl){
			V W = rvv::Load<V>(tw, vl);
			V Wp = rvv::Load<V>(twp, vl);
			auto mask = __riscv_vmseq(__riscv_vand(lane, t, vl), 0, vl); //mask based on lane id and current stage

			V x = r;
			V y = __riscv_vslidedown(r, t, vl);
			rvv::InvButterfly(x, y, W, Wp, modulus, vl);
			r = __riscv_vslideup(x, y, t, vl);
			r = __riscv_vmerge(r, x, mask, vl);
		}
		//perform last stage t = vl/2 outside the loop as it doesn't require merging
		V W = rvv::Load<V>(tw, vl); tw += vl;
		V Wp = rvv::Load<V>(twp, vl); twp += vl;
		V x = r;
		V y = __riscv_vslidedown(r, vl/2, vl);
		rvv::InvButterfly(x, y, W, Wp, modulus, vl);
		r = __riscv_vslideup(x, y, vl/2, vl);

		rvv::Store(result + c * vl, r, vl);
	}
}

}  // namespace hexl
}  // namespace intel

#pragma GCC diagnostic pop

#endif  // HEXL_HAS_RVV
