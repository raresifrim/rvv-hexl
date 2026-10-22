// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
//
// RVV NTT kernels: the RISC-V counterpart of upstream's fwd-ntt-avx512.cpp /
// inv-ntt-avx512.cpp.
//
// Each kernel is a template on its lane type V (SEW + LMUL) and on the
// small-span strategy (how the stages with span t < VLMAX are vectorized),
// both with defaults in src/util/rvv-config.hpp. Structure:
//   forward: copy, large stages t = n/2 .. VL per block (ntt-rvv-stages.hpp),
//            small stages by the strategy (Forward{FusedSlides,Transposed}RVV*
//            in ntt-rvv-fused-slides.hpp / ntt-rvv-transposed.hpp, picked by
//            the dispatch below), which also produces the output range;
//   inverse: small stages by the strategy (operand -> result), large stages
//            per block, last stage with N^{-1} folded in.
// Without tables for this n and VL (and always for rvv::ntt::PerBlock) every
// stage runs per block. Contract and twiddle layouts: ntt-rvv-small-span.hpp.
// RVV32 on uint64_t storage (OpenFHE NATIVE_SIZE=64 with a modulus < 2^30)
// runs on a 32-bit scratch: narrowed once on the way in, widened once on the
// way out, every stage on uint32_t (see NttScratch32 below).
// Next step: radix-4 for the large spans (two stages per pass).

#include "ntt/ntt-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include <type_traits>

#include "ntt/ntt-rvv-fused-slides.hpp"
#include "ntt/ntt-rvv-small-span.hpp"
#include "ntt/ntt-rvv-stages.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "ntt/ntt-rvv-transposed.hpp"
#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

// Small-span stages of strategy SmallSpan: one function per strategy and
// direction (ntt-rvv-fused-slides.hpp, ntt-rvv-transposed.hpp), chosen at
// compile time. PerBlock has none; the kernels below never call these for it.
template <class SmallSpan>
inline constexpr bool kHasSmallSpanStages = std::is_same_v<SmallSpan, rvv::ntt::FusedSlides> || rvv::ntt::kIsTransposed<SmallSpan>;

template <typename Word, class V, class SmallSpan>
inline void ForwardSmallSpanRVV32(Word* result, uint64_t n, uint64_t modulus,
		const uint32_t* w, const uint32_t* w_precon,
		const NttSmallSpanTables<SmallSpan, uint32_t>& tables,
		uint64_t output_mod_factor) {
	static_assert(kHasSmallSpanStages<SmallSpan>, "no small-span stages for this strategy");
	if constexpr (std::is_same_v<SmallSpan, rvv::ntt::FusedSlides>) {
		ForwardFusedSlidesRVV32<Word, V>(result, n, modulus, w, w_precon, tables, output_mod_factor);
	} else {
		ForwardTransposedRVV32<Word, V, SmallSpan>(result, n, modulus, w, w_precon, tables, output_mod_factor);
	}
}

template <typename Word, class V, class SmallSpan>
inline void InverseSmallSpanRVV32(Word* result, const Word* operand, uint64_t n,
		uint64_t modulus, const uint32_t* w_inv,
		const uint32_t* w_inv_precon,
		const NttSmallSpanTables<SmallSpan, uint32_t>& tables) {
	static_assert(kHasSmallSpanStages<SmallSpan>, "no small-span stages for this strategy");
	if constexpr (std::is_same_v<SmallSpan, rvv::ntt::FusedSlides>) {
		InverseFusedSlidesRVV32<Word, V>(result, operand, n, modulus, w_inv, w_inv_precon, tables);
	} else {
		InverseTransposedRVV32<Word, V, SmallSpan>(result, operand, n, modulus, w_inv, w_inv_precon, tables);
	}
}

template <class V, class SmallSpan>
inline void ForwardSmallSpanRVV64(uint64_t* result, uint64_t n, uint64_t modulus,
		const uint64_t* root_of_unity_powers,
		const uint64_t* precon_root_of_unity_powers,
		const NttSmallSpanTables<SmallSpan, uint64_t>& tables,
		uint64_t output_mod_factor) {
	static_assert(kHasSmallSpanStages<SmallSpan>, "no small-span stages for this strategy");
	if constexpr (std::is_same_v<SmallSpan, rvv::ntt::FusedSlides>) {
		ForwardFusedSlidesRVV64<V>(result, n, modulus, root_of_unity_powers, precon_root_of_unity_powers, tables, output_mod_factor);
	} else {
		ForwardTransposedRVV64<V, SmallSpan>(result, n, modulus, root_of_unity_powers, precon_root_of_unity_powers, tables, output_mod_factor);
	}
}

template <class V, class SmallSpan>
inline void InverseSmallSpanRVV64(uint64_t* result, const uint64_t* operand,
		uint64_t n, uint64_t modulus,
		const uint64_t* inv_root_of_unity_powers,
		const uint64_t* precon_inv_root_of_unity_powers,
		const NttSmallSpanTables<SmallSpan, uint64_t>& tables) {
	static_assert(kHasSmallSpanStages<SmallSpan>, "no small-span stages for this strategy");
	if constexpr (std::is_same_v<SmallSpan, rvv::ntt::FusedSlides>) {
		InverseFusedSlidesRVV64<V>(result, operand, n, modulus, inv_root_of_unity_powers, precon_inv_root_of_unity_powers, tables);
	} else {
		InverseTransposedRVV64<V, SmallSpan>(result, operand, n, modulus, inv_root_of_unity_powers, precon_inv_root_of_unity_powers, tables);
	}
}

// The 32-bit scratch of the RVV32 transforms on uint64_t storage. In place on
// uint64_t words every stage narrows its loads and widens its stores (e64
// loads and stores at twice the LMUL, vnsrl and vzext, 4 vsetvli per strip,
// twice the memory traffic); on the scratch the stages are plain e32, and the
// conversion costs two passes per transform instead of one per stage.
// Measured per large stage, N = 1024 (asm review, 2026-10-19): 2.40x on the
// X100, 1.57x on the A100. One buffer per thread (OpenFHE may transform from
// OpenMP threads), grown to the largest n seen, never shrunk.
inline uint32_t* NttScratch32(uint64_t n) {
	thread_local AlignedVector64<uint32_t> scratch;
	if (scratch.size() < n) scratch.resize(n);
	return scratch.data();
}

template <typename Word, class V, class SmallSpan>
void ForwardTransformToBitReverseRVV32(Word* result,
		const Word* operand, uint64_t n,
		uint64_t modulus, const uint32_t* w,
		const uint32_t* w_precon,
		uint64_t input_mod_factor,
		uint64_t output_mod_factor,
		const NttSmallSpanTables<SmallSpan, uint32_t>* small_span) {
	if constexpr (std::is_same_v<Word, uint64_t>) {   // on the 32-bit scratch
		uint32_t* scratch = NttScratch32(n);
		CopyRVV32<uint32_t, V>(scratch, operand, n);   // narrow
		ForwardTransformToBitReverseRVV32<uint32_t, V, SmallSpan>(scratch, scratch, n, modulus, w, w_precon,
				input_mod_factor, output_mod_factor, small_span);
		CopyRVV32<uint64_t, V>(result, scratch, n);    // widen
		return;
	}
	const uint32_t modulus32 = static_cast<uint32_t>(modulus);
	CopyRVV32<Word, V>(result, operand, n);   // then in place
	if constexpr (!std::is_same_v<SmallSpan, rvv::ntt::PerBlock>) {
		if (UseSmallSpan(small_span, n, SmallSpanLanes<SmallSpan, V>())) {
			ForwardStagesRVV32<Word, V>(result, n, modulus32, w, w_precon, n/2, small_span->t_top);   // per block down to t_top
			ForwardSmallSpanRVV32<Word, V, SmallSpan>(result, n, modulus, w, w_precon, *small_span, output_mod_factor);
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
	if constexpr (std::is_same_v<Word, uint64_t>) {   // on the 32-bit scratch
		uint32_t* scratch = NttScratch32(n);
		CopyRVV32<uint32_t, V>(scratch, operand, n);   // narrow
		InverseTransformFromBitReverseRVV32<uint32_t, V, SmallSpan>(scratch, scratch, n, modulus, w_inv, w_inv_precon,
				input_mod_factor, output_mod_factor, small_span);
		CopyRVV32<uint64_t, V>(result, scratch, n);    // widen
		return;
	}
	const uint32_t modulus32 = static_cast<uint32_t>(modulus);
	if constexpr (!std::is_same_v<SmallSpan, rvv::ntt::PerBlock>) {
		if (UseSmallSpan(small_span, n, SmallSpanLanes<SmallSpan, V>())) {
			// small spans first, operand -> result (no copy pass)
			InverseSmallSpanRVV32<Word, V, SmallSpan>(result, operand, n, modulus, w_inv, w_inv_precon, *small_span);
			InverseStagesRVV32<Word, V>(result, n, modulus32, w_inv, w_inv_precon, small_span->t_top, n/4);   // per block from t_top
			InverseLastStageRVV32<Word, V>(result, n, modulus, w_inv, output_mod_factor);
			return;
		}
	}
	CopyRVV32<Word, V>(result, operand, n);                                           // then in place
	InverseStagesRVV32<Word, V>(result, n, modulus32, w_inv, w_inv_precon, 1, n/4);   // every stage but the last
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
		if (UseSmallSpan(small_span, n, SmallSpanLanes<SmallSpan, V>())) {
			ForwardStagesRVV64<V>(result, n, modulus, root_of_unity_powers, precon_root_of_unity_powers, n/2, small_span->t_top);
			ForwardSmallSpanRVV64<V, SmallSpan>(result, n, modulus, root_of_unity_powers, precon_root_of_unity_powers, *small_span, output_mod_factor);
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
	if constexpr (!std::is_same_v<SmallSpan, rvv::ntt::PerBlock>) {
		if (UseSmallSpan(small_span, n, SmallSpanLanes<SmallSpan, V>())) {
			// small spans first, operand -> result (no copy pass)
			InverseSmallSpanRVV64<V, SmallSpan>(result, operand, n, modulus, inv_root_of_unity_powers, precon_inv_root_of_unity_powers, *small_span);
			InverseStagesRVV64<V>(result, n, modulus, inv_root_of_unity_powers, precon_inv_root_of_unity_powers, small_span->t_top, n/4);   // per block from t_top
			InverseLastStageRVV64<V>(result, n, modulus, inv_root_of_unity_powers, output_mod_factor);
			return;
		}
	}
	CopyRVV64<V>(result, operand, n);                                                                           // then in place
	InverseStagesRVV64<V>(result, n, modulus, inv_root_of_unity_powers, precon_inv_root_of_unity_powers, 1, n/4);   // every stage but the last
	InverseLastStageRVV64<V>(result, n, modulus, inv_root_of_unity_powers, output_mod_factor);
}

// Every lane type and strategy, so tests and benchmarks can compare them; the
// public NTT uses the defaults from src/util/rvv-config.hpp. RVV32 on uint64_t
// storage stops at u32m4 (the narrowing copy needs e64 at twice the LMUL).
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

HEXL_NTT_RVV64_LANES(rvv::ntt::PerBlock);
HEXL_NTT_RVV64_LANES(rvv::ntt::FusedSlides);
#define HEXL_NTT_TRANSPOSED(S)              \
	HEXL_NTT_RVV32_LANES(rvv::ntt::S); \
HEXL_NTT_RVV64_LANES(rvv::ntt::S);
HEXL_RVV_NTT_TRANSPOSED_SHAPES(HEXL_NTT_TRANSPOSED);
#undef HEXL_NTT_TRANSPOSED

#undef HEXL_NTT_RVV64_LANES
#undef HEXL_NTT_RVV32_LANES
#undef HEXL_NTT_RVV64
#undef HEXL_NTT_RVV32

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
