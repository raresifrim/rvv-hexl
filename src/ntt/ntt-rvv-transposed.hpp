// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
//
// Small-span stages, strategy S = rvv::ntt::Transposed<Rows, Lmul>. Contract,
// terms and the twiddle layout: ntt-rvv-small-span.hpp. The kernels call these
// only when tables.vl == L (VLMAX of TransposedLane<S, V>) and N >= Rows * L;
// the kernel has already run the stages t >= Rows (tables.t_top) per block.
//
// Plan, with VT = TransposedLane<S, V> (V's element width at LMUL Lmul,
// whatever V's own LMUL), vl = L and stride = n / L, per tile c (the stages
// t < Rows in registers, nothing in between):
//   load:   row f < Rows holds elements c*Rows + f + k*stride (lanes k):
//           Rows strided loads (stride words) from x + c*Rows + f into Rows
//           variables of type VT. RVV32 on uint64_t storage: vlse64 at twice
//           Lmul + vncvt per row.
//   stages: t = Rows/2 .. 1 (forward) or 1 .. Rows/2 (inverse); for each row
//           group g < Rows/(2t): load its twiddle vector and Shoup factors
//           (the next L entries of fwd / fwd_precon, or inv / inv_precon:
//           walk a pointer, the table is in this order), butterfly rows r and
//           r + t for r in [2gt, 2gt + t): plain vector ops, every lane useful.
//           Per tile: Rows 4: 4 butterflies, 3 twiddle pairs; Rows 8: 12, 7;
//           Rows 16: 32, 15.
//   store:  the same way back (vsse).
// Intrinsic vector types cannot live in arrays, so each row count is its own
// unrolled helper, written once for any VT: a 4-, 8- and 16-row tile helper
// (rows passed by reference), dispatched on Rows with if constexpr; the tile
// loop, loads and stores around them. Every shape in
// HEXL_RVV_NTT_TRANSPOSED_SHAPES then works.
// Forward: the tile arrives in [0, 4q) after the per-block stages; reduce to
// [0, q) before the store when output_mod_factor == 1. Inverse: load from
// operand, store to result (they may alias: the copy pass disappears).
// Measured on the X100 (2026-10-15, 2 KB stride): strided loads and stores
// pipeline at ~1 cycle per element at any LMUL (vlsseg8e32 1.02, 8 x vlse32
// 0.95, vlsseg4e64 0.65, vsse 1.0); the layout study's scratch version
// (2026-10-13, e32 m4, N = 1024) did 2.04 cycles per butterfly against 2.64
// for fused slides. The A100 runs it correctly, but serializes strided access
// at 17 cycles per element: fused slides there.

#pragma once

#include "ntt/ntt-rvv-small-span.hpp"
#include "ntt/ntt-rvv-stages.hpp"

#ifdef HEXL_HAS_RVV

// The tile functions declare their constants `register` to say they are meant
// to stay in registers. C++17 gives the keyword no meaning (GCC keeps them in
// registers anyway at -O2/-O3) and warns: -Wregister is silenced for this file.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wregister"

// GCC's scheduling before register allocation hoists twiddle loads ahead of the
// butterflies and spills tile rows (16 rows at m1; 4 rows at m4 on uint64_t
// storage, where every strided access goes through e64 at m8). Without that
// pass the rows stay in registers; the butterflies are still inlined.
#define HEXL_RVV_TILE_FN __attribute__((optimize("no-schedule-insns")))

namespace intel {
namespace hexl {

/// Forward stages t = Rows/2 .. 1 on result (in place), then the output range
/// of output_mod_factor.
template <typename Word, class V, class S>
HEXL_RVV_TILE_FN void ForwardTransposedRVV32(Word* result, uint64_t n,
		uint64_t modulus, const uint32_t* w,
		const uint32_t* w_precon,
		const NttSmallSpanTables<S, uint32_t>& tables,
		uint64_t output_mod_factor) {
	static_assert(rvv::ntt::kIsTransposed<S>, "S must be a rvv::ntt::Transposed shape");
	using VT = TransposedLane<S, V>;
	const register uint32_t modulus32 = static_cast<uint32_t>(modulus);
	const register uint64_t vl = tables.vl;
	const register uint64_t stride = n/vl;
	const register uint64_t Rows = S::kRows;
	const register uint64_t tile = vl * Rows;
	const register uint32_t* tw  = tables.fwd.data();
	const register uint32_t* twp = tables.fwd_precon.data();
	if constexpr (S::kRows == 4){
		for (uint64_t c = 0; c < n/tile; ++c){
			VT r0 = rvv::LoadStrided<VT>(result + c * Rows + 0, stride, vl);
			VT r1 = rvv::LoadStrided<VT>(result + c * Rows + 1, stride, vl);
			VT r2 = rvv::LoadStrided<VT>(result + c * Rows + 2, stride, vl);
			VT r3 = rvv::LoadStrided<VT>(result + c * Rows + 3, stride, vl);

			//stage t=2
			VT W, Wp;
			W = rvv::Load<VT>(tw, vl); Wp = rvv::Load<VT>(twp, vl); tw += vl; twp += vl; //advance twiddle tables
			rvv::FwdButterfly32(r0, r2, W, Wp, modulus32, vl);
			rvv::FwdButterfly32(r1, r3, W, Wp, modulus32, vl);

			// stage t = 1
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly32(r0, r1, W, Wp, modulus32, vl);   // group 0
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly32(r2, r3, W, Wp, modulus32, vl);   // group 1

			if (output_mod_factor == 1) {
				r0 = rvv::ReduceFromTwice32(r0, 2 * modulus32, vl);
				r1 = rvv::ReduceFromTwice32(r1, 2 * modulus32, vl);
				r2 = rvv::ReduceFromTwice32(r2, 2 * modulus32, vl);
				r3 = rvv::ReduceFromTwice32(r3, 2 * modulus32, vl);
				r0 = rvv::ReduceFromTwice32(r0, modulus32, vl);
				r1 = rvv::ReduceFromTwice32(r1, modulus32, vl);
				r2 = rvv::ReduceFromTwice32(r2, modulus32, vl);
				r3 = rvv::ReduceFromTwice32(r3, modulus32, vl);
			}

			rvv::StoreStrided(result + c * Rows + 0, stride, r0, vl);
			rvv::StoreStrided(result + c * Rows + 1, stride, r1, vl);
			rvv::StoreStrided(result + c * Rows + 2, stride, r2, vl);
			rvv::StoreStrided(result + c * Rows + 3, stride, r3, vl);
		}
	} else if constexpr (S::kRows == 8){
		for (uint64_t c = 0; c < n/tile; ++c){
			VT r0 = rvv::LoadStrided<VT>(result + c * Rows + 0, stride, vl);
			VT r1 = rvv::LoadStrided<VT>(result + c * Rows + 1, stride, vl);
			VT r2 = rvv::LoadStrided<VT>(result + c * Rows + 2, stride, vl);
			VT r3 = rvv::LoadStrided<VT>(result + c * Rows + 3, stride, vl);
			VT r4 = rvv::LoadStrided<VT>(result + c * Rows + 4, stride, vl);
			VT r5 = rvv::LoadStrided<VT>(result + c * Rows + 5, stride, vl);
			VT r6 = rvv::LoadStrided<VT>(result + c * Rows + 6, stride, vl);
			VT r7 = rvv::LoadStrided<VT>(result + c * Rows + 7, stride, vl);

			//stage t=4
			VT W, Wp;
			W = rvv::Load<VT>(tw, vl); Wp = rvv::Load<VT>(twp, vl); tw += vl; twp += vl; //advance twiddle tables
			rvv::FwdButterfly32(r0, r4, W, Wp, modulus32, vl);
			rvv::FwdButterfly32(r1, r5, W, Wp, modulus32, vl);
			rvv::FwdButterfly32(r2, r6, W, Wp, modulus32, vl);
			rvv::FwdButterfly32(r3, r7, W, Wp, modulus32, vl);

			//stage t=2
			W = rvv::Load<VT>(tw, vl); Wp = rvv::Load<VT>(twp, vl); tw += vl; twp += vl; //advance twiddle tables
			rvv::FwdButterfly32(r0, r2, W, Wp, modulus32, vl);
			rvv::FwdButterfly32(r1, r3, W, Wp, modulus32, vl);
			W = rvv::Load<VT>(tw, vl); Wp = rvv::Load<VT>(twp, vl); tw += vl; twp += vl; //advance twiddle tables
			rvv::FwdButterfly32(r4, r6, W, Wp, modulus32, vl);
			rvv::FwdButterfly32(r5, r7, W, Wp, modulus32, vl);

			// stage t = 1
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly32(r0, r1, W, Wp, modulus32, vl);   // group 0
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly32(r2, r3, W, Wp, modulus32, vl);   // group 1
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly32(r4, r5, W, Wp, modulus32, vl);   // group 2
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly32(r6, r7, W, Wp, modulus32, vl);   // group 3

			if (output_mod_factor == 1) {
				r0 = rvv::ReduceFromTwice32(r0, 2 * modulus32, vl);
				r1 = rvv::ReduceFromTwice32(r1, 2 * modulus32, vl);
				r2 = rvv::ReduceFromTwice32(r2, 2 * modulus32, vl);
				r3 = rvv::ReduceFromTwice32(r3, 2 * modulus32, vl);
				r4 = rvv::ReduceFromTwice32(r4, 2 * modulus32, vl);
				r5 = rvv::ReduceFromTwice32(r5, 2 * modulus32, vl);
				r6 = rvv::ReduceFromTwice32(r6, 2 * modulus32, vl);
				r7 = rvv::ReduceFromTwice32(r7, 2 * modulus32, vl);
				r0 = rvv::ReduceFromTwice32(r0, modulus32, vl);
				r1 = rvv::ReduceFromTwice32(r1, modulus32, vl);
				r2 = rvv::ReduceFromTwice32(r2, modulus32, vl);
				r3 = rvv::ReduceFromTwice32(r3, modulus32, vl);
				r4 = rvv::ReduceFromTwice32(r4, modulus32, vl);
				r5 = rvv::ReduceFromTwice32(r5, modulus32, vl);
				r6 = rvv::ReduceFromTwice32(r6, modulus32, vl);
				r7 = rvv::ReduceFromTwice32(r7, modulus32, vl);
			}

			rvv::StoreStrided(result + c * Rows + 0, stride, r0, vl);
			rvv::StoreStrided(result + c * Rows + 1, stride, r1, vl);
			rvv::StoreStrided(result + c * Rows + 2, stride, r2, vl);
			rvv::StoreStrided(result + c * Rows + 3, stride, r3, vl);
			rvv::StoreStrided(result + c * Rows + 4, stride, r4, vl);
			rvv::StoreStrided(result + c * Rows + 5, stride, r5, vl);
			rvv::StoreStrided(result + c * Rows + 6, stride, r6, vl);
			rvv::StoreStrided(result + c * Rows + 7, stride, r7, vl);
		}

	} else if constexpr (S::kRows == 16){
		for (uint64_t c = 0; c < n/tile; ++c){
			VT r0 = rvv::LoadStrided<VT>(result + c * Rows + 0, stride, vl);
			VT r1 = rvv::LoadStrided<VT>(result + c * Rows + 1, stride, vl);
			VT r2 = rvv::LoadStrided<VT>(result + c * Rows + 2, stride, vl);
			VT r3 = rvv::LoadStrided<VT>(result + c * Rows + 3, stride, vl);
			VT r4 = rvv::LoadStrided<VT>(result + c * Rows + 4, stride, vl);
			VT r5 = rvv::LoadStrided<VT>(result + c * Rows + 5, stride, vl);
			VT r6 = rvv::LoadStrided<VT>(result + c * Rows + 6, stride, vl);
			VT r7 = rvv::LoadStrided<VT>(result + c * Rows + 7, stride, vl);
			VT r8 = rvv::LoadStrided<VT>(result + c * Rows + 8, stride, vl);
			VT r9 = rvv::LoadStrided<VT>(result + c * Rows + 9, stride, vl);
			VT r10 = rvv::LoadStrided<VT>(result + c * Rows + 10, stride, vl);
			VT r11 = rvv::LoadStrided<VT>(result + c * Rows + 11, stride, vl);
			VT r12 = rvv::LoadStrided<VT>(result + c * Rows + 12, stride, vl);
			VT r13 = rvv::LoadStrided<VT>(result + c * Rows + 13, stride, vl);
			VT r14 = rvv::LoadStrided<VT>(result + c * Rows + 14, stride, vl);
			VT r15 = rvv::LoadStrided<VT>(result + c * Rows + 15, stride, vl);

			VT W, Wp;
			// stage t = 8
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly32(r0, r8, W, Wp, modulus32, vl);
			rvv::FwdButterfly32(r1, r9, W, Wp, modulus32, vl);
			rvv::FwdButterfly32(r2, r10, W, Wp, modulus32, vl);
			rvv::FwdButterfly32(r3, r11, W, Wp, modulus32, vl);
			rvv::FwdButterfly32(r4, r12, W, Wp, modulus32, vl);
			rvv::FwdButterfly32(r5, r13, W, Wp, modulus32, vl);
			rvv::FwdButterfly32(r6, r14, W, Wp, modulus32, vl);
			rvv::FwdButterfly32(r7, r15, W, Wp, modulus32, vl);

			// stage t = 4, 2 row groups
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly32(r0, r4, W, Wp, modulus32, vl);   // group 0
			rvv::FwdButterfly32(r1, r5, W, Wp, modulus32, vl);
			rvv::FwdButterfly32(r2, r6, W, Wp, modulus32, vl);
			rvv::FwdButterfly32(r3, r7, W, Wp, modulus32, vl);
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly32(r8, r12, W, Wp, modulus32, vl);   // group 1
			rvv::FwdButterfly32(r9, r13, W, Wp, modulus32, vl);
			rvv::FwdButterfly32(r10, r14, W, Wp, modulus32, vl);
			rvv::FwdButterfly32(r11, r15, W, Wp, modulus32, vl);

			// stage t = 2, 4 row groups
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly32(r0, r2, W, Wp, modulus32, vl);   // group 0
			rvv::FwdButterfly32(r1, r3, W, Wp, modulus32, vl);
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly32(r4, r6, W, Wp, modulus32, vl);   // group 1
			rvv::FwdButterfly32(r5, r7, W, Wp, modulus32, vl);
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly32(r8, r10, W, Wp, modulus32, vl);   // group 2
			rvv::FwdButterfly32(r9, r11, W, Wp, modulus32, vl);
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly32(r12, r14, W, Wp, modulus32, vl);   // group 3
			rvv::FwdButterfly32(r13, r15, W, Wp, modulus32, vl);

			// stage t = 1, 8 row groups
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly32(r0, r1, W, Wp, modulus32, vl);   // group 0
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly32(r2, r3, W, Wp, modulus32, vl);   // group 1
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly32(r4, r5, W, Wp, modulus32, vl);   // group 2
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly32(r6, r7, W, Wp, modulus32, vl);   // group 3
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly32(r8, r9, W, Wp, modulus32, vl);   // group 4
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly32(r10, r11, W, Wp, modulus32, vl);   // group 5
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly32(r12, r13, W, Wp, modulus32, vl);   // group 6
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly32(r14, r15, W, Wp, modulus32, vl);   // group 7

			if (output_mod_factor == 1) {
				r0 = rvv::ReduceFromTwice32(r0, 2 * modulus32, vl);
				r1 = rvv::ReduceFromTwice32(r1, 2 * modulus32, vl);
				r2 = rvv::ReduceFromTwice32(r2, 2 * modulus32, vl);
				r3 = rvv::ReduceFromTwice32(r3, 2 * modulus32, vl);
				r4 = rvv::ReduceFromTwice32(r4, 2 * modulus32, vl);
				r5 = rvv::ReduceFromTwice32(r5, 2 * modulus32, vl);
				r6 = rvv::ReduceFromTwice32(r6, 2 * modulus32, vl);
				r7 = rvv::ReduceFromTwice32(r7, 2 * modulus32, vl);
				r8 = rvv::ReduceFromTwice32(r8, 2 * modulus32, vl);
				r9 = rvv::ReduceFromTwice32(r9, 2 * modulus32, vl);
				r10 = rvv::ReduceFromTwice32(r10, 2 * modulus32, vl);
				r11 = rvv::ReduceFromTwice32(r11, 2 * modulus32, vl);
				r12 = rvv::ReduceFromTwice32(r12, 2 * modulus32, vl);
				r13 = rvv::ReduceFromTwice32(r13, 2 * modulus32, vl);
				r14 = rvv::ReduceFromTwice32(r14, 2 * modulus32, vl);
				r15 = rvv::ReduceFromTwice32(r15, 2 * modulus32, vl);
				r0 = rvv::ReduceFromTwice32(r0, modulus32, vl);
				r1 = rvv::ReduceFromTwice32(r1, modulus32, vl);
				r2 = rvv::ReduceFromTwice32(r2, modulus32, vl);
				r3 = rvv::ReduceFromTwice32(r3, modulus32, vl);
				r4 = rvv::ReduceFromTwice32(r4, modulus32, vl);
				r5 = rvv::ReduceFromTwice32(r5, modulus32, vl);
				r6 = rvv::ReduceFromTwice32(r6, modulus32, vl);
				r7 = rvv::ReduceFromTwice32(r7, modulus32, vl);
				r8 = rvv::ReduceFromTwice32(r8, modulus32, vl);
				r9 = rvv::ReduceFromTwice32(r9, modulus32, vl);
				r10 = rvv::ReduceFromTwice32(r10, modulus32, vl);
				r11 = rvv::ReduceFromTwice32(r11, modulus32, vl);
				r12 = rvv::ReduceFromTwice32(r12, modulus32, vl);
				r13 = rvv::ReduceFromTwice32(r13, modulus32, vl);
				r14 = rvv::ReduceFromTwice32(r14, modulus32, vl);
				r15 = rvv::ReduceFromTwice32(r15, modulus32, vl);
			}

			rvv::StoreStrided(result + c * Rows + 0, stride, r0, vl);
			rvv::StoreStrided(result + c * Rows + 1, stride, r1, vl);
			rvv::StoreStrided(result + c * Rows + 2, stride, r2, vl);
			rvv::StoreStrided(result + c * Rows + 3, stride, r3, vl);
			rvv::StoreStrided(result + c * Rows + 4, stride, r4, vl);
			rvv::StoreStrided(result + c * Rows + 5, stride, r5, vl);
			rvv::StoreStrided(result + c * Rows + 6, stride, r6, vl);
			rvv::StoreStrided(result + c * Rows + 7, stride, r7, vl);
			rvv::StoreStrided(result + c * Rows + 8, stride, r8, vl);
			rvv::StoreStrided(result + c * Rows + 9, stride, r9, vl);
			rvv::StoreStrided(result + c * Rows + 10, stride, r10, vl);
			rvv::StoreStrided(result + c * Rows + 11, stride, r11, vl);
			rvv::StoreStrided(result + c * Rows + 12, stride, r12, vl);
			rvv::StoreStrided(result + c * Rows + 13, stride, r13, vl);
			rvv::StoreStrided(result + c * Rows + 14, stride, r14, vl);
			rvv::StoreStrided(result + c * Rows + 15, stride, r15, vl);
		}
	}

}

/// Inverse stages t = 1 .. Rows/2, from operand into result (they may alias).
template <typename Word, class V, class S>
HEXL_RVV_TILE_FN void InverseTransposedRVV32(Word* result,
		const Word* operand, uint64_t n, uint64_t modulus,
		const uint32_t* w_inv, const uint32_t* w_inv_precon,
		const NttSmallSpanTables<S, uint32_t>& tables) {
	static_assert(rvv::ntt::kIsTransposed<S>, "S must be a rvv::ntt::Transposed shape");
	using VT = TransposedLane<S, V>;
	const register uint32_t modulus32 = static_cast<uint32_t>(modulus);
	const register uint64_t vl = tables.vl;
	const register uint64_t stride = n/vl;
	const register uint64_t Rows = S::kRows;
	const register uint64_t tile = vl * Rows;
	const register uint32_t* tw  = tables.inv.data();
	const register uint32_t* twp = tables.inv_precon.data();
	if constexpr (S::kRows == 4) {
		for (uint64_t c = 0; c < n/tile; ++c){
			VT r0 = rvv::LoadStrided<VT>(operand + c * Rows + 0, stride, vl);
			VT r1 = rvv::LoadStrided<VT>(operand + c * Rows + 1, stride, vl);
			VT r2 = rvv::LoadStrided<VT>(operand + c * Rows + 2, stride, vl);
			VT r3 = rvv::LoadStrided<VT>(operand + c * Rows + 3, stride, vl);

			// stage t = 1
			VT W, Wp;
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r0, r1, W, Wp, modulus32, vl);   // group 0
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r2, r3, W, Wp, modulus32, vl);   // group 1

			//stage t=2
			W = rvv::Load<VT>(tw, vl); Wp = rvv::Load<VT>(twp, vl); tw += vl; twp += vl; //advance twiddle tables
			rvv::InvButterfly32(r0, r2, W, Wp, modulus32, vl);
			rvv::InvButterfly32(r1, r3, W, Wp, modulus32, vl);

			rvv::StoreStrided(result + c * Rows + 0, stride, r0, vl);
			rvv::StoreStrided(result + c * Rows + 1, stride, r1, vl);
			rvv::StoreStrided(result + c * Rows + 2, stride, r2, vl);
			rvv::StoreStrided(result + c * Rows + 3, stride, r3, vl);
		}
	} else if constexpr (S::kRows == 8) {
		for (uint64_t c = 0; c < n/tile; ++c){
			VT r0 = rvv::LoadStrided<VT>(operand + c * Rows + 0, stride, vl);
			VT r1 = rvv::LoadStrided<VT>(operand + c * Rows + 1, stride, vl);
			VT r2 = rvv::LoadStrided<VT>(operand + c * Rows + 2, stride, vl);
			VT r3 = rvv::LoadStrided<VT>(operand + c * Rows + 3, stride, vl);
			VT r4 = rvv::LoadStrided<VT>(operand + c * Rows + 4, stride, vl);
			VT r5 = rvv::LoadStrided<VT>(operand + c * Rows + 5, stride, vl);
			VT r6 = rvv::LoadStrided<VT>(operand + c * Rows + 6, stride, vl);
			VT r7 = rvv::LoadStrided<VT>(operand + c * Rows + 7, stride, vl);

			VT W, Wp;
			// stage t = 1, 4 row groups
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r0, r1, W, Wp, modulus32, vl);   // group 0
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r2, r3, W, Wp, modulus32, vl);   // group 1
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r4, r5, W, Wp, modulus32, vl);   // group 2
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r6, r7, W, Wp, modulus32, vl);   // group 3

			// stage t = 2, 2 row groups
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r0, r2, W, Wp, modulus32, vl);   // group 0
			rvv::InvButterfly32(r1, r3, W, Wp, modulus32, vl);
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r4, r6, W, Wp, modulus32, vl);   // group 1
			rvv::InvButterfly32(r5, r7, W, Wp, modulus32, vl);

			// stage t = 4
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r0, r4, W, Wp, modulus32, vl);
			rvv::InvButterfly32(r1, r5, W, Wp, modulus32, vl);
			rvv::InvButterfly32(r2, r6, W, Wp, modulus32, vl);
			rvv::InvButterfly32(r3, r7, W, Wp, modulus32, vl);

			rvv::StoreStrided(result + c * Rows + 0, stride, r0, vl);
			rvv::StoreStrided(result + c * Rows + 1, stride, r1, vl);
			rvv::StoreStrided(result + c * Rows + 2, stride, r2, vl);
			rvv::StoreStrided(result + c * Rows + 3, stride, r3, vl);
			rvv::StoreStrided(result + c * Rows + 4, stride, r4, vl);
			rvv::StoreStrided(result + c * Rows + 5, stride, r5, vl);
			rvv::StoreStrided(result + c * Rows + 6, stride, r6, vl);
			rvv::StoreStrided(result + c * Rows + 7, stride, r7, vl);
		}
	} else if constexpr (S::kRows == 16) {
		for (uint64_t c = 0; c < n/tile; ++c){
			VT r0 = rvv::LoadStrided<VT>(operand + c * Rows + 0, stride, vl);
			VT r1 = rvv::LoadStrided<VT>(operand + c * Rows + 1, stride, vl);
			VT r2 = rvv::LoadStrided<VT>(operand + c * Rows + 2, stride, vl);
			VT r3 = rvv::LoadStrided<VT>(operand + c * Rows + 3, stride, vl);
			VT r4 = rvv::LoadStrided<VT>(operand + c * Rows + 4, stride, vl);
			VT r5 = rvv::LoadStrided<VT>(operand + c * Rows + 5, stride, vl);
			VT r6 = rvv::LoadStrided<VT>(operand + c * Rows + 6, stride, vl);
			VT r7 = rvv::LoadStrided<VT>(operand + c * Rows + 7, stride, vl);
			VT r8 = rvv::LoadStrided<VT>(operand + c * Rows + 8, stride, vl);
			VT r9 = rvv::LoadStrided<VT>(operand + c * Rows + 9, stride, vl);
			VT r10 = rvv::LoadStrided<VT>(operand + c * Rows + 10, stride, vl);
			VT r11 = rvv::LoadStrided<VT>(operand + c * Rows + 11, stride, vl);
			VT r12 = rvv::LoadStrided<VT>(operand + c * Rows + 12, stride, vl);
			VT r13 = rvv::LoadStrided<VT>(operand + c * Rows + 13, stride, vl);
			VT r14 = rvv::LoadStrided<VT>(operand + c * Rows + 14, stride, vl);
			VT r15 = rvv::LoadStrided<VT>(operand + c * Rows + 15, stride, vl);

			VT W, Wp;
			// stage t = 1, 8 row groups
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r0, r1, W, Wp, modulus32, vl);   // group 0
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r2, r3, W, Wp, modulus32, vl);   // group 1
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r4, r5, W, Wp, modulus32, vl);   // group 2
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r6, r7, W, Wp, modulus32, vl);   // group 3
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r8, r9, W, Wp, modulus32, vl);   // group 4
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r10, r11, W, Wp, modulus32, vl);   // group 5
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r12, r13, W, Wp, modulus32, vl);   // group 6
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r14, r15, W, Wp, modulus32, vl);   // group 7

			// stage t = 2, 4 row groups
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r0, r2, W, Wp, modulus32, vl);   // group 0
			rvv::InvButterfly32(r1, r3, W, Wp, modulus32, vl);
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r4, r6, W, Wp, modulus32, vl);   // group 1
			rvv::InvButterfly32(r5, r7, W, Wp, modulus32, vl);
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r8, r10, W, Wp, modulus32, vl);   // group 2
			rvv::InvButterfly32(r9, r11, W, Wp, modulus32, vl);
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r12, r14, W, Wp, modulus32, vl);   // group 3
			rvv::InvButterfly32(r13, r15, W, Wp, modulus32, vl);

			// stage t = 4, 2 row groups
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r0, r4, W, Wp, modulus32, vl);   // group 0
			rvv::InvButterfly32(r1, r5, W, Wp, modulus32, vl);
			rvv::InvButterfly32(r2, r6, W, Wp, modulus32, vl);
			rvv::InvButterfly32(r3, r7, W, Wp, modulus32, vl);
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r8, r12, W, Wp, modulus32, vl);   // group 1
			rvv::InvButterfly32(r9, r13, W, Wp, modulus32, vl);
			rvv::InvButterfly32(r10, r14, W, Wp, modulus32, vl);
			rvv::InvButterfly32(r11, r15, W, Wp, modulus32, vl);

			// stage t = 8
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly32(r0, r8, W, Wp, modulus32, vl);
			rvv::InvButterfly32(r1, r9, W, Wp, modulus32, vl);
			rvv::InvButterfly32(r2, r10, W, Wp, modulus32, vl);
			rvv::InvButterfly32(r3, r11, W, Wp, modulus32, vl);
			rvv::InvButterfly32(r4, r12, W, Wp, modulus32, vl);
			rvv::InvButterfly32(r5, r13, W, Wp, modulus32, vl);
			rvv::InvButterfly32(r6, r14, W, Wp, modulus32, vl);
			rvv::InvButterfly32(r7, r15, W, Wp, modulus32, vl);

			rvv::StoreStrided(result + c * Rows + 0, stride, r0, vl);
			rvv::StoreStrided(result + c * Rows + 1, stride, r1, vl);
			rvv::StoreStrided(result + c * Rows + 2, stride, r2, vl);
			rvv::StoreStrided(result + c * Rows + 3, stride, r3, vl);
			rvv::StoreStrided(result + c * Rows + 4, stride, r4, vl);
			rvv::StoreStrided(result + c * Rows + 5, stride, r5, vl);
			rvv::StoreStrided(result + c * Rows + 6, stride, r6, vl);
			rvv::StoreStrided(result + c * Rows + 7, stride, r7, vl);
			rvv::StoreStrided(result + c * Rows + 8, stride, r8, vl);
			rvv::StoreStrided(result + c * Rows + 9, stride, r9, vl);
			rvv::StoreStrided(result + c * Rows + 10, stride, r10, vl);
			rvv::StoreStrided(result + c * Rows + 11, stride, r11, vl);
			rvv::StoreStrided(result + c * Rows + 12, stride, r12, vl);
			rvv::StoreStrided(result + c * Rows + 13, stride, r13, vl);
			rvv::StoreStrided(result + c * Rows + 14, stride, r14, vl);
			rvv::StoreStrided(result + c * Rows + 15, stride, r15, vl);
		}
	}
}

/// Forward stages t = Rows/2 .. 1, 64-bit lanes.
template <class V, class S>
HEXL_RVV_TILE_FN void ForwardTransposedRVV64(uint64_t* result, uint64_t n,
		uint64_t modulus, const uint64_t* root_of_unity_powers,
		const uint64_t* precon_root_of_unity_powers,
		const NttSmallSpanTables<S, uint64_t>& tables,
		uint64_t output_mod_factor) {
	static_assert(rvv::ntt::kIsTransposed<S>, "S must be a rvv::ntt::Transposed shape");
	using VT = TransposedLane<S, V>;
	const register uint64_t vl = tables.vl;
	const register uint64_t stride = n/vl;
	const register uint64_t Rows = S::kRows;
	const register uint64_t tile = vl * Rows;
	const register uint64_t* tw  = tables.fwd.data();
	const register uint64_t* twp = tables.fwd_precon.data();
	if constexpr (S::kRows == 4){
		for (uint64_t c = 0; c < n/tile; ++c){
			VT r0 = rvv::LoadStrided<VT>(result + c * Rows + 0, stride, vl);
			VT r1 = rvv::LoadStrided<VT>(result + c * Rows + 1, stride, vl);
			VT r2 = rvv::LoadStrided<VT>(result + c * Rows + 2, stride, vl);
			VT r3 = rvv::LoadStrided<VT>(result + c * Rows + 3, stride, vl);

			VT W, Wp;
			// stage t = 2
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r0, r2, W, Wp, modulus, vl);
			rvv::FwdButterfly(r1, r3, W, Wp, modulus, vl);

			// stage t = 1, 2 row groups
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r0, r1, W, Wp, modulus, vl);   // group 0
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r2, r3, W, Wp, modulus, vl);   // group 1

			if (output_mod_factor == 1) {
				r0 = rvv::ReduceFromTwice(r0, 2 * modulus, vl);
				r1 = rvv::ReduceFromTwice(r1, 2 * modulus, vl);
				r2 = rvv::ReduceFromTwice(r2, 2 * modulus, vl);
				r3 = rvv::ReduceFromTwice(r3, 2 * modulus, vl);
				r0 = rvv::ReduceFromTwice(r0, modulus, vl);
				r1 = rvv::ReduceFromTwice(r1, modulus, vl);
				r2 = rvv::ReduceFromTwice(r2, modulus, vl);
				r3 = rvv::ReduceFromTwice(r3, modulus, vl);
			}

			rvv::StoreStrided(result + c * Rows + 0, stride, r0, vl);
			rvv::StoreStrided(result + c * Rows + 1, stride, r1, vl);
			rvv::StoreStrided(result + c * Rows + 2, stride, r2, vl);
			rvv::StoreStrided(result + c * Rows + 3, stride, r3, vl);
		}
	} else if constexpr (S::kRows == 8){
		for (uint64_t c = 0; c < n/tile; ++c){
			VT r0 = rvv::LoadStrided<VT>(result + c * Rows + 0, stride, vl);
			VT r1 = rvv::LoadStrided<VT>(result + c * Rows + 1, stride, vl);
			VT r2 = rvv::LoadStrided<VT>(result + c * Rows + 2, stride, vl);
			VT r3 = rvv::LoadStrided<VT>(result + c * Rows + 3, stride, vl);
			VT r4 = rvv::LoadStrided<VT>(result + c * Rows + 4, stride, vl);
			VT r5 = rvv::LoadStrided<VT>(result + c * Rows + 5, stride, vl);
			VT r6 = rvv::LoadStrided<VT>(result + c * Rows + 6, stride, vl);
			VT r7 = rvv::LoadStrided<VT>(result + c * Rows + 7, stride, vl);

			VT W, Wp;
			// stage t = 4
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r0, r4, W, Wp, modulus, vl);
			rvv::FwdButterfly(r1, r5, W, Wp, modulus, vl);
			rvv::FwdButterfly(r2, r6, W, Wp, modulus, vl);
			rvv::FwdButterfly(r3, r7, W, Wp, modulus, vl);

			// stage t = 2, 2 row groups
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r0, r2, W, Wp, modulus, vl);   // group 0
			rvv::FwdButterfly(r1, r3, W, Wp, modulus, vl);
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r4, r6, W, Wp, modulus, vl);   // group 1
			rvv::FwdButterfly(r5, r7, W, Wp, modulus, vl);

			// stage t = 1, 4 row groups
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r0, r1, W, Wp, modulus, vl);   // group 0
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r2, r3, W, Wp, modulus, vl);   // group 1
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r4, r5, W, Wp, modulus, vl);   // group 2
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r6, r7, W, Wp, modulus, vl);   // group 3

			if (output_mod_factor == 1) {
				r0 = rvv::ReduceFromTwice(r0, 2 * modulus, vl);
				r1 = rvv::ReduceFromTwice(r1, 2 * modulus, vl);
				r2 = rvv::ReduceFromTwice(r2, 2 * modulus, vl);
				r3 = rvv::ReduceFromTwice(r3, 2 * modulus, vl);
				r4 = rvv::ReduceFromTwice(r4, 2 * modulus, vl);
				r5 = rvv::ReduceFromTwice(r5, 2 * modulus, vl);
				r6 = rvv::ReduceFromTwice(r6, 2 * modulus, vl);
				r7 = rvv::ReduceFromTwice(r7, 2 * modulus, vl);
				r0 = rvv::ReduceFromTwice(r0, modulus, vl);
				r1 = rvv::ReduceFromTwice(r1, modulus, vl);
				r2 = rvv::ReduceFromTwice(r2, modulus, vl);
				r3 = rvv::ReduceFromTwice(r3, modulus, vl);
				r4 = rvv::ReduceFromTwice(r4, modulus, vl);
				r5 = rvv::ReduceFromTwice(r5, modulus, vl);
				r6 = rvv::ReduceFromTwice(r6, modulus, vl);
				r7 = rvv::ReduceFromTwice(r7, modulus, vl);
			}

			rvv::StoreStrided(result + c * Rows + 0, stride, r0, vl);
			rvv::StoreStrided(result + c * Rows + 1, stride, r1, vl);
			rvv::StoreStrided(result + c * Rows + 2, stride, r2, vl);
			rvv::StoreStrided(result + c * Rows + 3, stride, r3, vl);
			rvv::StoreStrided(result + c * Rows + 4, stride, r4, vl);
			rvv::StoreStrided(result + c * Rows + 5, stride, r5, vl);
			rvv::StoreStrided(result + c * Rows + 6, stride, r6, vl);
			rvv::StoreStrided(result + c * Rows + 7, stride, r7, vl);
		}
	} else if constexpr (S::kRows == 16){
		for (uint64_t c = 0; c < n/tile; ++c){
			VT r0 = rvv::LoadStrided<VT>(result + c * Rows + 0, stride, vl);
			VT r1 = rvv::LoadStrided<VT>(result + c * Rows + 1, stride, vl);
			VT r2 = rvv::LoadStrided<VT>(result + c * Rows + 2, stride, vl);
			VT r3 = rvv::LoadStrided<VT>(result + c * Rows + 3, stride, vl);
			VT r4 = rvv::LoadStrided<VT>(result + c * Rows + 4, stride, vl);
			VT r5 = rvv::LoadStrided<VT>(result + c * Rows + 5, stride, vl);
			VT r6 = rvv::LoadStrided<VT>(result + c * Rows + 6, stride, vl);
			VT r7 = rvv::LoadStrided<VT>(result + c * Rows + 7, stride, vl);
			VT r8 = rvv::LoadStrided<VT>(result + c * Rows + 8, stride, vl);
			VT r9 = rvv::LoadStrided<VT>(result + c * Rows + 9, stride, vl);
			VT r10 = rvv::LoadStrided<VT>(result + c * Rows + 10, stride, vl);
			VT r11 = rvv::LoadStrided<VT>(result + c * Rows + 11, stride, vl);
			VT r12 = rvv::LoadStrided<VT>(result + c * Rows + 12, stride, vl);
			VT r13 = rvv::LoadStrided<VT>(result + c * Rows + 13, stride, vl);
			VT r14 = rvv::LoadStrided<VT>(result + c * Rows + 14, stride, vl);
			VT r15 = rvv::LoadStrided<VT>(result + c * Rows + 15, stride, vl);

			VT W, Wp;
			// stage t = 8
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r0, r8, W, Wp, modulus, vl);
			rvv::FwdButterfly(r1, r9, W, Wp, modulus, vl);
			rvv::FwdButterfly(r2, r10, W, Wp, modulus, vl);
			rvv::FwdButterfly(r3, r11, W, Wp, modulus, vl);
			rvv::FwdButterfly(r4, r12, W, Wp, modulus, vl);
			rvv::FwdButterfly(r5, r13, W, Wp, modulus, vl);
			rvv::FwdButterfly(r6, r14, W, Wp, modulus, vl);
			rvv::FwdButterfly(r7, r15, W, Wp, modulus, vl);

			// stage t = 4, 2 row groups
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r0, r4, W, Wp, modulus, vl);   // group 0
			rvv::FwdButterfly(r1, r5, W, Wp, modulus, vl);
			rvv::FwdButterfly(r2, r6, W, Wp, modulus, vl);
			rvv::FwdButterfly(r3, r7, W, Wp, modulus, vl);
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r8, r12, W, Wp, modulus, vl);   // group 1
			rvv::FwdButterfly(r9, r13, W, Wp, modulus, vl);
			rvv::FwdButterfly(r10, r14, W, Wp, modulus, vl);
			rvv::FwdButterfly(r11, r15, W, Wp, modulus, vl);

			// stage t = 2, 4 row groups
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r0, r2, W, Wp, modulus, vl);   // group 0
			rvv::FwdButterfly(r1, r3, W, Wp, modulus, vl);
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r4, r6, W, Wp, modulus, vl);   // group 1
			rvv::FwdButterfly(r5, r7, W, Wp, modulus, vl);
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r8, r10, W, Wp, modulus, vl);   // group 2
			rvv::FwdButterfly(r9, r11, W, Wp, modulus, vl);
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r12, r14, W, Wp, modulus, vl);   // group 3
			rvv::FwdButterfly(r13, r15, W, Wp, modulus, vl);

			// stage t = 1, 8 row groups
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r0, r1, W, Wp, modulus, vl);   // group 0
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r2, r3, W, Wp, modulus, vl);   // group 1
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r4, r5, W, Wp, modulus, vl);   // group 2
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r6, r7, W, Wp, modulus, vl);   // group 3
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r8, r9, W, Wp, modulus, vl);   // group 4
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r10, r11, W, Wp, modulus, vl);   // group 5
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r12, r13, W, Wp, modulus, vl);   // group 6
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::FwdButterfly(r14, r15, W, Wp, modulus, vl);   // group 7

			if (output_mod_factor == 1) {
				r0 = rvv::ReduceFromTwice(r0, 2 * modulus, vl);
				r1 = rvv::ReduceFromTwice(r1, 2 * modulus, vl);
				r2 = rvv::ReduceFromTwice(r2, 2 * modulus, vl);
				r3 = rvv::ReduceFromTwice(r3, 2 * modulus, vl);
				r4 = rvv::ReduceFromTwice(r4, 2 * modulus, vl);
				r5 = rvv::ReduceFromTwice(r5, 2 * modulus, vl);
				r6 = rvv::ReduceFromTwice(r6, 2 * modulus, vl);
				r7 = rvv::ReduceFromTwice(r7, 2 * modulus, vl);
				r8 = rvv::ReduceFromTwice(r8, 2 * modulus, vl);
				r9 = rvv::ReduceFromTwice(r9, 2 * modulus, vl);
				r10 = rvv::ReduceFromTwice(r10, 2 * modulus, vl);
				r11 = rvv::ReduceFromTwice(r11, 2 * modulus, vl);
				r12 = rvv::ReduceFromTwice(r12, 2 * modulus, vl);
				r13 = rvv::ReduceFromTwice(r13, 2 * modulus, vl);
				r14 = rvv::ReduceFromTwice(r14, 2 * modulus, vl);
				r15 = rvv::ReduceFromTwice(r15, 2 * modulus, vl);
				r0 = rvv::ReduceFromTwice(r0, modulus, vl);
				r1 = rvv::ReduceFromTwice(r1, modulus, vl);
				r2 = rvv::ReduceFromTwice(r2, modulus, vl);
				r3 = rvv::ReduceFromTwice(r3, modulus, vl);
				r4 = rvv::ReduceFromTwice(r4, modulus, vl);
				r5 = rvv::ReduceFromTwice(r5, modulus, vl);
				r6 = rvv::ReduceFromTwice(r6, modulus, vl);
				r7 = rvv::ReduceFromTwice(r7, modulus, vl);
				r8 = rvv::ReduceFromTwice(r8, modulus, vl);
				r9 = rvv::ReduceFromTwice(r9, modulus, vl);
				r10 = rvv::ReduceFromTwice(r10, modulus, vl);
				r11 = rvv::ReduceFromTwice(r11, modulus, vl);
				r12 = rvv::ReduceFromTwice(r12, modulus, vl);
				r13 = rvv::ReduceFromTwice(r13, modulus, vl);
				r14 = rvv::ReduceFromTwice(r14, modulus, vl);
				r15 = rvv::ReduceFromTwice(r15, modulus, vl);
			}

			rvv::StoreStrided(result + c * Rows + 0, stride, r0, vl);
			rvv::StoreStrided(result + c * Rows + 1, stride, r1, vl);
			rvv::StoreStrided(result + c * Rows + 2, stride, r2, vl);
			rvv::StoreStrided(result + c * Rows + 3, stride, r3, vl);
			rvv::StoreStrided(result + c * Rows + 4, stride, r4, vl);
			rvv::StoreStrided(result + c * Rows + 5, stride, r5, vl);
			rvv::StoreStrided(result + c * Rows + 6, stride, r6, vl);
			rvv::StoreStrided(result + c * Rows + 7, stride, r7, vl);
			rvv::StoreStrided(result + c * Rows + 8, stride, r8, vl);
			rvv::StoreStrided(result + c * Rows + 9, stride, r9, vl);
			rvv::StoreStrided(result + c * Rows + 10, stride, r10, vl);
			rvv::StoreStrided(result + c * Rows + 11, stride, r11, vl);
			rvv::StoreStrided(result + c * Rows + 12, stride, r12, vl);
			rvv::StoreStrided(result + c * Rows + 13, stride, r13, vl);
			rvv::StoreStrided(result + c * Rows + 14, stride, r14, vl);
			rvv::StoreStrided(result + c * Rows + 15, stride, r15, vl);
		}
	}
}

/// Inverse stages t = 1 .. Rows/2, 64-bit lanes.
template <class V, class S>
HEXL_RVV_TILE_FN void InverseTransposedRVV64(uint64_t* result,
		const uint64_t* operand, uint64_t n, uint64_t modulus,
		const uint64_t* inv_root_of_unity_powers,
		const uint64_t* precon_inv_root_of_unity_powers,
		const NttSmallSpanTables<S, uint64_t>& tables) {
	static_assert(rvv::ntt::kIsTransposed<S>, "S must be a rvv::ntt::Transposed shape");
	using VT = TransposedLane<S, V>;
	const register uint64_t vl = tables.vl;
	const register uint64_t stride = n/vl;
	const register uint64_t Rows = S::kRows;
	const register uint64_t tile = vl * Rows;
	const register uint64_t* tw  = tables.inv.data();
	const register uint64_t* twp = tables.inv_precon.data();
	if constexpr (S::kRows == 4) {
		for (uint64_t c = 0; c < n/tile; ++c){
			VT r0 = rvv::LoadStrided<VT>(operand + c * Rows + 0, stride, vl);
			VT r1 = rvv::LoadStrided<VT>(operand + c * Rows + 1, stride, vl);
			VT r2 = rvv::LoadStrided<VT>(operand + c * Rows + 2, stride, vl);
			VT r3 = rvv::LoadStrided<VT>(operand + c * Rows + 3, stride, vl);

			VT W, Wp;
			// stage t = 1, 2 row groups
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r0, r1, W, Wp, modulus, vl);   // group 0
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r2, r3, W, Wp, modulus, vl);   // group 1

			// stage t = 2
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r0, r2, W, Wp, modulus, vl);
			rvv::InvButterfly(r1, r3, W, Wp, modulus, vl);

			rvv::StoreStrided(result + c * Rows + 0, stride, r0, vl);
			rvv::StoreStrided(result + c * Rows + 1, stride, r1, vl);
			rvv::StoreStrided(result + c * Rows + 2, stride, r2, vl);
			rvv::StoreStrided(result + c * Rows + 3, stride, r3, vl);
		}
	} else if constexpr (S::kRows == 8) {
		for (uint64_t c = 0; c < n/tile; ++c){
			VT r0 = rvv::LoadStrided<VT>(operand + c * Rows + 0, stride, vl);
			VT r1 = rvv::LoadStrided<VT>(operand + c * Rows + 1, stride, vl);
			VT r2 = rvv::LoadStrided<VT>(operand + c * Rows + 2, stride, vl);
			VT r3 = rvv::LoadStrided<VT>(operand + c * Rows + 3, stride, vl);
			VT r4 = rvv::LoadStrided<VT>(operand + c * Rows + 4, stride, vl);
			VT r5 = rvv::LoadStrided<VT>(operand + c * Rows + 5, stride, vl);
			VT r6 = rvv::LoadStrided<VT>(operand + c * Rows + 6, stride, vl);
			VT r7 = rvv::LoadStrided<VT>(operand + c * Rows + 7, stride, vl);

			VT W, Wp;
			// stage t = 1, 4 row groups
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r0, r1, W, Wp, modulus, vl);   // group 0
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r2, r3, W, Wp, modulus, vl);   // group 1
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r4, r5, W, Wp, modulus, vl);   // group 2
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r6, r7, W, Wp, modulus, vl);   // group 3

			// stage t = 2, 2 row groups
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r0, r2, W, Wp, modulus, vl);   // group 0
			rvv::InvButterfly(r1, r3, W, Wp, modulus, vl);
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r4, r6, W, Wp, modulus, vl);   // group 1
			rvv::InvButterfly(r5, r7, W, Wp, modulus, vl);

			// stage t = 4
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r0, r4, W, Wp, modulus, vl);
			rvv::InvButterfly(r1, r5, W, Wp, modulus, vl);
			rvv::InvButterfly(r2, r6, W, Wp, modulus, vl);
			rvv::InvButterfly(r3, r7, W, Wp, modulus, vl);

			rvv::StoreStrided(result + c * Rows + 0, stride, r0, vl);
			rvv::StoreStrided(result + c * Rows + 1, stride, r1, vl);
			rvv::StoreStrided(result + c * Rows + 2, stride, r2, vl);
			rvv::StoreStrided(result + c * Rows + 3, stride, r3, vl);
			rvv::StoreStrided(result + c * Rows + 4, stride, r4, vl);
			rvv::StoreStrided(result + c * Rows + 5, stride, r5, vl);
			rvv::StoreStrided(result + c * Rows + 6, stride, r6, vl);
			rvv::StoreStrided(result + c * Rows + 7, stride, r7, vl);
		}
	} else if constexpr (S::kRows == 16) {
		for (uint64_t c = 0; c < n/tile; ++c){
			VT r0 = rvv::LoadStrided<VT>(operand + c * Rows + 0, stride, vl);
			VT r1 = rvv::LoadStrided<VT>(operand + c * Rows + 1, stride, vl);
			VT r2 = rvv::LoadStrided<VT>(operand + c * Rows + 2, stride, vl);
			VT r3 = rvv::LoadStrided<VT>(operand + c * Rows + 3, stride, vl);
			VT r4 = rvv::LoadStrided<VT>(operand + c * Rows + 4, stride, vl);
			VT r5 = rvv::LoadStrided<VT>(operand + c * Rows + 5, stride, vl);
			VT r6 = rvv::LoadStrided<VT>(operand + c * Rows + 6, stride, vl);
			VT r7 = rvv::LoadStrided<VT>(operand + c * Rows + 7, stride, vl);
			VT r8 = rvv::LoadStrided<VT>(operand + c * Rows + 8, stride, vl);
			VT r9 = rvv::LoadStrided<VT>(operand + c * Rows + 9, stride, vl);
			VT r10 = rvv::LoadStrided<VT>(operand + c * Rows + 10, stride, vl);
			VT r11 = rvv::LoadStrided<VT>(operand + c * Rows + 11, stride, vl);
			VT r12 = rvv::LoadStrided<VT>(operand + c * Rows + 12, stride, vl);
			VT r13 = rvv::LoadStrided<VT>(operand + c * Rows + 13, stride, vl);
			VT r14 = rvv::LoadStrided<VT>(operand + c * Rows + 14, stride, vl);
			VT r15 = rvv::LoadStrided<VT>(operand + c * Rows + 15, stride, vl);

			VT W, Wp;
			// stage t = 1, 8 row groups
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r0, r1, W, Wp, modulus, vl);   // group 0
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r2, r3, W, Wp, modulus, vl);   // group 1
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r4, r5, W, Wp, modulus, vl);   // group 2
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r6, r7, W, Wp, modulus, vl);   // group 3
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r8, r9, W, Wp, modulus, vl);   // group 4
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r10, r11, W, Wp, modulus, vl);   // group 5
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r12, r13, W, Wp, modulus, vl);   // group 6
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r14, r15, W, Wp, modulus, vl);   // group 7

			// stage t = 2, 4 row groups
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r0, r2, W, Wp, modulus, vl);   // group 0
			rvv::InvButterfly(r1, r3, W, Wp, modulus, vl);
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r4, r6, W, Wp, modulus, vl);   // group 1
			rvv::InvButterfly(r5, r7, W, Wp, modulus, vl);
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r8, r10, W, Wp, modulus, vl);   // group 2
			rvv::InvButterfly(r9, r11, W, Wp, modulus, vl);
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r12, r14, W, Wp, modulus, vl);   // group 3
			rvv::InvButterfly(r13, r15, W, Wp, modulus, vl);

			// stage t = 4, 2 row groups
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r0, r4, W, Wp, modulus, vl);   // group 0
			rvv::InvButterfly(r1, r5, W, Wp, modulus, vl);
			rvv::InvButterfly(r2, r6, W, Wp, modulus, vl);
			rvv::InvButterfly(r3, r7, W, Wp, modulus, vl);
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r8, r12, W, Wp, modulus, vl);   // group 1
			rvv::InvButterfly(r9, r13, W, Wp, modulus, vl);
			rvv::InvButterfly(r10, r14, W, Wp, modulus, vl);
			rvv::InvButterfly(r11, r15, W, Wp, modulus, vl);

			// stage t = 8
			W = rvv::Load<VT>(tw, vl);  Wp = rvv::Load<VT>(twp, vl);  tw += vl; twp += vl;
			rvv::InvButterfly(r0, r8, W, Wp, modulus, vl);
			rvv::InvButterfly(r1, r9, W, Wp, modulus, vl);
			rvv::InvButterfly(r2, r10, W, Wp, modulus, vl);
			rvv::InvButterfly(r3, r11, W, Wp, modulus, vl);
			rvv::InvButterfly(r4, r12, W, Wp, modulus, vl);
			rvv::InvButterfly(r5, r13, W, Wp, modulus, vl);
			rvv::InvButterfly(r6, r14, W, Wp, modulus, vl);
			rvv::InvButterfly(r7, r15, W, Wp, modulus, vl);

			rvv::StoreStrided(result + c * Rows + 0, stride, r0, vl);
			rvv::StoreStrided(result + c * Rows + 1, stride, r1, vl);
			rvv::StoreStrided(result + c * Rows + 2, stride, r2, vl);
			rvv::StoreStrided(result + c * Rows + 3, stride, r3, vl);
			rvv::StoreStrided(result + c * Rows + 4, stride, r4, vl);
			rvv::StoreStrided(result + c * Rows + 5, stride, r5, vl);
			rvv::StoreStrided(result + c * Rows + 6, stride, r6, vl);
			rvv::StoreStrided(result + c * Rows + 7, stride, r7, vl);
			rvv::StoreStrided(result + c * Rows + 8, stride, r8, vl);
			rvv::StoreStrided(result + c * Rows + 9, stride, r9, vl);
			rvv::StoreStrided(result + c * Rows + 10, stride, r10, vl);
			rvv::StoreStrided(result + c * Rows + 11, stride, r11, vl);
			rvv::StoreStrided(result + c * Rows + 12, stride, r12, vl);
			rvv::StoreStrided(result + c * Rows + 13, stride, r13, vl);
			rvv::StoreStrided(result + c * Rows + 14, stride, r14, vl);
			rvv::StoreStrided(result + c * Rows + 15, stride, r15, vl);
		}
	}
}

}  // namespace hexl
}  // namespace intel

#undef HEXL_RVV_TILE_FN
#pragma GCC diagnostic pop

#endif  // HEXL_HAS_RVV
