// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// RVV NTT kernels: the RISC-V counterpart of upstream's fwd-ntt-avx512.cpp /
// inv-ntt-avx512.cpp.
//
// Each kernel is a template on its lane type V (SEW + LMUL) and on the
// small-span strategy (how the stages with span t < VLMAX are vectorized),
// both with defaults in src/util/rvv-config.hpp. Structure:
//   forward: copy, large stages t = n/2 .. VL per block (ntt-rvv-stages.hpp),
//            small stages by the strategy (ntt-rvv-fused-slides.hpp,
//            ntt-rvv-transposed.hpp), which also produces the output range;
//   inverse: small stages by the strategy (operand -> result), large stages
//            per block, last stage with N^{-1} folded in.
// Without tables for this n and VL (and always for rvv::ntt::PerBlock) every
// stage runs per block. Contract and twiddle layouts: ntt-rvv-small-span.hpp.
// Next steps after the small spans: a 32-bit scratch pass for RVV32 on
// uint64_t storage, and radix-4 for the large spans (two stages per pass).

#include "ntt/ntt-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include <type_traits>

#include "ntt/ntt-rvv-fused-slides.hpp"
#include "ntt/ntt-rvv-small-span.hpp"
#include "ntt/ntt-rvv-stages.hpp"
#include "ntt/ntt-rvv-transposed.hpp"
#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

template <typename Word, class V, class SmallSpan>
void ForwardTransformToBitReverseRVV32(Word* result,
		const Word* operand, uint64_t n,
		uint64_t modulus, const uint32_t* w,
		const uint32_t* w_precon,
		uint64_t input_mod_factor,
		uint64_t output_mod_factor,
		const NttSmallSpanTables<SmallSpan, uint32_t>* small_span) {
	const uint32_t modulus32 = static_cast<uint32_t>(modulus);
	CopyRVV32<Word, V>(result, operand, n);   // then in place
	if constexpr (!std::is_same_v<SmallSpan, rvv::ntt::PerBlock>) {
		const uint64_t vl = rvv::VlMax<V>();
		if (UseSmallSpan(small_span, n, vl)) {
			ForwardStagesRVV32<Word, V>(result, n, modulus32, w, w_precon, n/2, vl);   // large spans
			ForwardSmallSpanRVV32<Word, V>(SmallSpan{}, result, n, modulus, w, w_precon, *small_span, output_mod_factor);
			return;
		}
	}
	ForwardStagesRVV32<Word, V>(result, n, modulus32, w, w_precon, n/2, 1);       // every stage
	if (output_mod_factor == 1) ReduceFromFourRVV32<Word, V>(result, n, modulus32);
}

template <typename Word, class V, class SmallSpan>
void InverseTransformFromBitReverseRVV32(Word* result,
		const Word* operand, uint64_t n,
		uint64_t modulus, const uint32_t* w_inv,
		const uint32_t* w_inv_precon,
		uint64_t input_mod_factor,
		uint64_t output_mod_factor,
		const NttSmallSpanTables<SmallSpan, uint32_t>* small_span) {
	// Mirror of the RVV32 forward: Gentleman-Sande in 32-bit lanes, 32-bit
	// twiddle tables in stage order, Harvey lazy butterflies in [0, 2q), and
	// the N^{-1} scaling folded into the last stage.
	const uint32_t modulus32 = static_cast<uint32_t>(modulus);
	uint64_t t = 1;                                                    // first stage left to run
	if constexpr (!std::is_same_v<SmallSpan, rvv::ntt::PerBlock>) {
		const uint64_t vl = rvv::VlMax<V>();
		if (UseSmallSpan(small_span, n, vl)) {
			InverseSmallSpanRVV32<Word, V>(SmallSpan{}, result, operand, n, modulus, w_inv, w_inv_precon, *small_span);
			t = vl;
		}
	}
	if (t == 1) CopyRVV32<Word, V>(result, operand, n);
	InverseStagesRVV32<Word, V>(result, n, modulus32, w_inv, w_inv_precon, t, n/4);
	InverseLastStageRVV32<Word, V>(result, n, modulus, w_inv, output_mod_factor);
}

template <class V, class SmallSpan>
void ForwardTransformToBitReverseRVV64(
		uint64_t* result, const uint64_t* operand, uint64_t n, uint64_t modulus,
		const uint64_t* root_of_unity_powers,
		const uint64_t* precon_root_of_unity_powers, uint64_t input_mod_factor,
		uint64_t output_mod_factor,
		const NttSmallSpanTables<SmallSpan, uint64_t>* small_span) {
	CopyRVV64<V>(result, operand, n);
	if constexpr (!std::is_same_v<SmallSpan, rvv::ntt::PerBlock>) {
		const uint64_t vl = rvv::VlMax<V>();
		if (UseSmallSpan(small_span, n, vl)) {
			ForwardStagesRVV64<V>(result, n, modulus, root_of_unity_powers, precon_root_of_unity_powers, n/2, vl);
			ForwardSmallSpanRVV64<V>(SmallSpan{}, result, n, modulus, root_of_unity_powers, precon_root_of_unity_powers, *small_span, output_mod_factor);
			return;
		}
	}
	ForwardStagesRVV64<V>(result, n, modulus, root_of_unity_powers, precon_root_of_unity_powers, n/2, 1);
	if (output_mod_factor == 1) ReduceFromFourRVV64<V>(result, n, modulus);
}

template <class V, class SmallSpan>
void InverseTransformFromBitReverseRVV64(
		uint64_t* result, const uint64_t* operand, uint64_t n, uint64_t modulus,
		const uint64_t* inv_root_of_unity_powers,
		const uint64_t* precon_inv_root_of_unity_powers, uint64_t input_mod_factor,
		uint64_t output_mod_factor,
		const NttSmallSpanTables<SmallSpan, uint64_t>* small_span) {
	uint64_t t = 1;                                                    // first stage left to run
	if constexpr (!std::is_same_v<SmallSpan, rvv::ntt::PerBlock>) {
		const uint64_t vl = rvv::VlMax<V>();
		if (UseSmallSpan(small_span, n, vl)) {
			InverseSmallSpanRVV64<V>(SmallSpan{}, result, operand, n, modulus, inv_root_of_unity_powers, precon_inv_root_of_unity_powers, *small_span);
			t = vl;
		}
	}
	if (t == 1) CopyRVV64<V>(result, operand, n);
	InverseStagesRVV64<V>(result, n, modulus, inv_root_of_unity_powers, precon_inv_root_of_unity_powers, t, n/4);
	InverseLastStageRVV64<V>(result, n, modulus, inv_root_of_unity_powers, output_mod_factor);
}

// Every lane type and strategy, so tests and benchmarks can compare them; the
// public NTT uses the defaults from src/util/rvv-config.hpp. RVV32 on uint64_t
// storage stops at u32m4 (the narrowing load needs e64 at twice the LMUL).
#define HEXL_NTT_RVV32(Word, V, S)                                                                  \
	template void ForwardTransformToBitReverseRVV32<Word, V, S>(                                  \
			Word*, const Word*, uint64_t, uint64_t, const uint32_t*, const uint32_t*, uint64_t,   \
			uint64_t, const NttSmallSpanTables<S, uint32_t>*);                                    \
			template void InverseTransformFromBitReverseRVV32<Word, V, S>(                                \
					Word*, const Word*, uint64_t, uint64_t, const uint32_t*, const uint32_t*, uint64_t,   \
					uint64_t, const NttSmallSpanTables<S, uint32_t>*)
#define HEXL_NTT_RVV64(V, S)                                                                        \
	template void ForwardTransformToBitReverseRVV64<V, S>(                                        \
			uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint64_t*, const uint64_t*,     \
			uint64_t, uint64_t, const NttSmallSpanTables<S, uint64_t>*);                          \
			template void InverseTransformFromBitReverseRVV64<V, S>(                                      \
					uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint64_t*, const uint64_t*,     \
					uint64_t, uint64_t, const NttSmallSpanTables<S, uint64_t>*)
#define HEXL_NTT_RVV32_LANES(S)                     \
	HEXL_NTT_RVV32(uint64_t, vuint32m1_t, S);     \
	HEXL_NTT_RVV32(uint64_t, vuint32m2_t, S);     \
	HEXL_NTT_RVV32(uint64_t, vuint32m4_t, S);     \
	HEXL_NTT_RVV32(uint32_t, vuint32m1_t, S);     \
	HEXL_NTT_RVV32(uint32_t, vuint32m2_t, S);     \
	HEXL_NTT_RVV32(uint32_t, vuint32m4_t, S);     \
	HEXL_NTT_RVV32(uint32_t, vuint32m8_t, S)
#define HEXL_NTT_RVV64_LANES(S)                     \
	HEXL_NTT_RVV64(vuint64m1_t, S);               \
	HEXL_NTT_RVV64(vuint64m2_t, S);               \
	HEXL_NTT_RVV64(vuint64m4_t, S);               \
	HEXL_NTT_RVV64(vuint64m8_t, S)

HEXL_NTT_RVV32_LANES(rvv::ntt::PerBlock);
HEXL_NTT_RVV32_LANES(rvv::ntt::FusedSlides);
HEXL_NTT_RVV32_LANES(rvv::ntt::Transposed);
HEXL_NTT_RVV64_LANES(rvv::ntt::PerBlock);
HEXL_NTT_RVV64_LANES(rvv::ntt::FusedSlides);
HEXL_NTT_RVV64_LANES(rvv::ntt::Transposed);

#undef HEXL_NTT_RVV64_LANES
#undef HEXL_NTT_RVV32_LANES
#undef HEXL_NTT_RVV64
#undef HEXL_NTT_RVV32

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
