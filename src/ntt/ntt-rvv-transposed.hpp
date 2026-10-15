// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
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

namespace intel {
namespace hexl {

/// Forward stages t = Rows/2 .. 1 on result (in place), then the output range
/// of output_mod_factor.
template <typename Word, class V, class S>
void ForwardTransposedRVV32(Word* result, uint64_t n,
		uint64_t modulus, const uint32_t* w,
		const uint32_t* w_precon,
		const NttSmallSpanTables<S, uint32_t>& tables,
		uint64_t output_mod_factor) {
	static_assert(rvv::ntt::kIsTransposed<S>, "S must be a rvv::ntt::Transposed shape");
	// Stub: the per-block loops until the tile body is written.
	const uint32_t modulus32 = static_cast<uint32_t>(modulus);
	ForwardStagesRVV32<Word, V>(result, n, modulus32, w, w_precon, tables.t_top/2, 1);
	if (output_mod_factor == 1) ReduceFromFourRVV32<Word, V>(result, n, modulus32);
}

/// Inverse stages t = 1 .. Rows/2, from operand into result (they may alias).
template <typename Word, class V, class S>
void InverseTransposedRVV32(Word* result,
		const Word* operand, uint64_t n, uint64_t modulus,
		const uint32_t* w_inv, const uint32_t* w_inv_precon,
		const NttSmallSpanTables<S, uint32_t>& tables) {
	static_assert(rvv::ntt::kIsTransposed<S>, "S must be a rvv::ntt::Transposed shape");
	// Stub: copy, then the per-block loops until the tile body is written.
	CopyRVV32<Word, V>(result, operand, n);
	InverseStagesRVV32<Word, V>(result, n, static_cast<uint32_t>(modulus), w_inv, w_inv_precon, 1, tables.t_top/2);
}

/// Forward stages t = Rows/2 .. 1, 64-bit lanes.
template <class V, class S>
void ForwardTransposedRVV64(uint64_t* result, uint64_t n,
		uint64_t modulus, const uint64_t* root_of_unity_powers,
		const uint64_t* precon_root_of_unity_powers,
		const NttSmallSpanTables<S, uint64_t>& tables,
		uint64_t output_mod_factor) {
	static_assert(rvv::ntt::kIsTransposed<S>, "S must be a rvv::ntt::Transposed shape");
	// Stub: the per-block loops until the tile body is written.
	ForwardStagesRVV64<V>(result, n, modulus, root_of_unity_powers, precon_root_of_unity_powers, tables.t_top/2, 1);
	if (output_mod_factor == 1) ReduceFromFourRVV64<V>(result, n, modulus);
}

/// Inverse stages t = 1 .. Rows/2, 64-bit lanes.
template <class V, class S>
void InverseTransposedRVV64(uint64_t* result,
		const uint64_t* operand, uint64_t n, uint64_t modulus,
		const uint64_t* inv_root_of_unity_powers,
		const uint64_t* precon_inv_root_of_unity_powers,
		const NttSmallSpanTables<S, uint64_t>& tables) {
	static_assert(rvv::ntt::kIsTransposed<S>, "S must be a rvv::ntt::Transposed shape");
	// Stub: copy, then the per-block loops until the tile body is written.
	CopyRVV64<V>(result, operand, n);
	InverseStagesRVV64<V>(result, n, modulus, inv_root_of_unity_powers, precon_inv_root_of_unity_powers, 1, tables.t_top/2);
}

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
