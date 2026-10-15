// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Small-span stages, strategy rvv::ntt::Transposed. Contract, terms and the
// twiddle layout: ntt-rvv-small-span.hpp. The kernels call these only when
// tables.vl == VLMAX of V and N >= VL^2, so there are R = N / VL >= VL rows.
//
// Plan (B = SmallSpanScratch<E>(n), E = uint32_t for RVV32, uint64_t for RVV64):
//   transpose in:  for each row r < R: vlse (stride R words) from x + r, vse
//                  to B + r*VL. Row r lane k = element r + k*R. RVV32 on
//                  uint64_t storage: vlse64 at twice the LMUL + vncvt (there
//                  is no strided Load32 helper yet).
//   small stages:  for each stage t (forward VL/2 .. 1, inverse 1 .. VL/2),
//                  for each row group g < R/(2t): load the group's per-lane
//                  twiddles once (next VL entries), then for each row r in
//                  [2gt, 2gt + t): butterfly rows r and r + t with plain
//                  vle / vse, every lane useful.
//   transpose out: for each row r: vle from B + r*VL, vsse (stride R words)
//                  to x + r.
// Forward: the rows arrive in [0, 4q) after the large stages; reduce to [0, q)
// before the transpose out when output_mod_factor == 1. Inverse: transpose in
// from operand, out to result (the copy pass disappears).
// Measured in the layout study (whole forward NTT, cycles per butterfly,
// N = 1024): X100 e32 m4 2.04 (best; fused 2.64); on the A100 strided access
// costs ~17 cycles per element, so fused slides win there.

#pragma once

#include "ntt/ntt-rvv-small-span.hpp"
#include "ntt/ntt-rvv-stages.hpp"

#ifdef HEXL_HAS_RVV

namespace intel {
namespace hexl {

/// Forward small-span stages t = VL/2 .. 1 on result (in place), then the
/// output range of output_mod_factor.
template <typename Word, class V>
void ForwardSmallSpanRVV32(rvv::ntt::Transposed, Word* result, uint64_t n,
		uint64_t modulus, const uint32_t* w,
		const uint32_t* w_precon,
		const NttSmallSpanTables<rvv::ntt::Transposed, uint32_t>& tables,
		uint64_t output_mod_factor) {
	// Stub: the per-block loops until the transposed body is written.
	const uint32_t modulus32 = static_cast<uint32_t>(modulus);
	ForwardStagesRVV32<Word, V>(result, n, modulus32, w, w_precon, tables.vl/2, 1);
	if (output_mod_factor == 1) ReduceFromFourRVV32<Word, V>(result, n, modulus32);
}

/// Inverse small-span stages t = 1 .. VL/2, from operand into result (they
/// may alias).
template <typename Word, class V>
void InverseSmallSpanRVV32(rvv::ntt::Transposed, Word* result,
		const Word* operand, uint64_t n, uint64_t modulus,
		const uint32_t* w_inv, const uint32_t* w_inv_precon,
		const NttSmallSpanTables<rvv::ntt::Transposed, uint32_t>& tables) {
	// Stub: copy, then the per-block loops until the transposed body is written.
	CopyRVV32<Word, V>(result, operand, n);
	InverseStagesRVV32<Word, V>(result, n, static_cast<uint32_t>(modulus), w_inv, w_inv_precon, 1, tables.vl/2);
}

/// Forward small-span stages, 64-bit lanes.
template <class V>
void ForwardSmallSpanRVV64(rvv::ntt::Transposed, uint64_t* result, uint64_t n,
		uint64_t modulus, const uint64_t* root_of_unity_powers,
		const uint64_t* precon_root_of_unity_powers,
		const NttSmallSpanTables<rvv::ntt::Transposed, uint64_t>& tables,
		uint64_t output_mod_factor) {
	// Stub: the per-block loops until the transposed body is written.
	ForwardStagesRVV64<V>(result, n, modulus, root_of_unity_powers, precon_root_of_unity_powers, tables.vl/2, 1);
	if (output_mod_factor == 1) ReduceFromFourRVV64<V>(result, n, modulus);
}

/// Inverse small-span stages, 64-bit lanes.
template <class V>
void InverseSmallSpanRVV64(rvv::ntt::Transposed, uint64_t* result,
		const uint64_t* operand, uint64_t n, uint64_t modulus,
		const uint64_t* inv_root_of_unity_powers,
		const uint64_t* precon_inv_root_of_unity_powers,
		const NttSmallSpanTables<rvv::ntt::Transposed, uint64_t>& tables) {
	// Stub: copy, then the per-block loops until the transposed body is written.
	CopyRVV64<V>(result, operand, n);
	InverseStagesRVV64<V>(result, n, modulus, inv_root_of_unity_powers, precon_inv_root_of_unity_powers, 1, tables.vl/2);
}

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
