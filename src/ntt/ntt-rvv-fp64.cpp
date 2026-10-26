// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
//
// FP64 NTT path (rvv-hexl extension): the tables and the RVV kernels
// (algorithm and conditions: ntt-rvv-fp64.hpp). The inverse is a STUB until
// written: it calls the e64 integer kernel, so the dispatch, the tests and the
// benchmarks work from the start.

#include "ntt/ntt-rvv-fp64.hpp"

#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "ntt/ntt-internal.hpp"

#ifdef HEXL_HAS_RVV
#include <riscv_vector.h>

#include <type_traits>

#include "ntt/ntt-rvv-fused-slides.hpp"
#include "ntt/ntt-rvv-small-span.hpp"
#include "ntt/ntt-rvv-stages.hpp"
#include "ntt/ntt-rvv-transposed.hpp"
#include "util/rvv-util.hpp"
#endif

namespace intel {
namespace hexl {

// The bounds of the path, checked at compile time in every build.
static_assert(!NttFp64Applicable(kFp64MinModulus - 1, 1024), "below 2^30: the e32 lanes");
static_assert(NttFp64Applicable(kFp64MinModulus, 1024) && NttFp64Applicable(kFp64MaxModulus - 1, 1024));
static_assert(!NttFp64Applicable(kFp64MaxModulus, 1024), "2^50 and above: the e64 integer lanes");
static_assert(4 * (kFp64MaxModulus - 1) < (1ULL << 52), "lazy values must be exact doubles");
static_assert(!NttFp64Applicable(kFp64MinModulus, 1) && !NttFp64Applicable(kFp64MinModulus, 768));

std::shared_ptr<const NttFp64Tables> BuildNttFp64Tables(const NTT& ntt) {
	const uint64_t n = ntt.GetDegree(), q = ntt.GetModulus();
	HEXL_CHECK(NttFp64Applicable(q, n), "BuildNttFp64Tables: modulus " << q << " outside [2^30, 2^50)");
	auto t = std::make_shared<NttFp64Tables>();
	const auto& w = ntt.GetRootOfUnityPowers();
	const auto& wi = ntt.GetInvRootOfUnityPowers();
	const double qd = static_cast<double>(q);
	t->w.resize(w.size());
	t->w_over_q.resize(w.size());
	t->w_inv.resize(wi.size());
	t->w_inv_over_q.resize(wi.size());
	for (size_t i = 0; i < w.size(); ++i) {
		t->w[i] = static_cast<double>(w[i]);  // < q < 2^50: exact
		t->w_over_q[i] = t->w[i] / qd;        // one correctly rounded division of exact values
	}
	for (size_t i = 0; i < wi.size(); ++i) {
		t->w_inv[i] = static_cast<double>(wi[i]);
		t->w_inv_over_q[i] = t->w_inv[i] / qd;
	}
	t->q = qd;
	t->q_inv = 1.0 / qd;
	const uint64_t n_inv = InverseMod(n, q);
	const uint64_t wn_inv = MultiplyMod(wi[n - 1], n_inv, q);  // as InverseLastStageRVV64
	t->n_inv = static_cast<double>(n_inv);
	t->n_inv_over_q = t->n_inv / qd;
	t->wn_inv = static_cast<double>(wn_inv);
	t->wn_inv_over_q = t->wn_inv / qd;
#ifdef HEXL_HAS_RVV
	using S = rvv::cfg::NttSmallSpanF64;
	if (has_rvv) {  // the layout reads VLMAX
		const uint64_t lanes = SmallSpanLanes<S, rvv::UintOf<rvv::cfg::NttF64>>();
		if (SmallSpanRuns<S>(n, lanes))
			t->small_span = std::make_shared<const NttSmallSpanTables<S, double>>(BuildNttSmallSpanTablesF64<S>(*t, lanes));
	}
#endif
	return t;
}

#ifdef HEXL_HAS_RVV

// Small-span stages of strategy SmallSpan, FP64 (ntt-rvv-fused-slides.hpp,
// ntt-rvv-transposed.hpp), chosen at compile time; PerBlock has none. Forward:
// scratch f -> result (the output range); inverse: operand -> scratch f.
template <class VF, class SmallSpan>
inline void ForwardSmallSpanRVVF64(uint64_t* result, const double* f, uint64_t n, uint64_t modulus,
		const NttSmallSpanTables<SmallSpan, double>& tables, uint64_t output_mod_factor) {
	if constexpr (std::is_same_v<SmallSpan, rvv::ntt::FusedSlides>) {
		ForwardFusedSlidesRVVF64<VF>(result, f, n, modulus, tables, output_mod_factor);
	} else {
		static_assert(rvv::ntt::kIsTransposed<SmallSpan>, "no small-span stages for this strategy");
		ForwardTransposedRVVF64<VF, SmallSpan>(result, f, n, modulus, tables, output_mod_factor);
	}
}

template <class VF, class SmallSpan>
inline void InverseSmallSpanRVVF64(double* f, const uint64_t* operand, uint64_t n, uint64_t modulus,
		const NttSmallSpanTables<SmallSpan, double>& tables) {
	if constexpr (std::is_same_v<SmallSpan, rvv::ntt::FusedSlides>) {
		InverseFusedSlidesRVVF64<VF>(f, operand, n, modulus, tables);
	} else {
		static_assert(rvv::ntt::kIsTransposed<SmallSpan>, "no small-span stages for this strategy");
		InverseTransposedRVVF64<VF, SmallSpan>(f, operand, n, modulus, tables);
	}
}

// The double scratch of the FP64 transforms (the values between the input and
// the output conversion). One buffer per thread (OpenFHE may transform from
// OpenMP threads), grown to the largest n seen, never shrunk; result may then
// alias operand.
inline double* NttScratchF64(uint64_t n) {
	thread_local AlignedVector64<double> scratch;
	if (scratch.size() < n) scratch.resize(n);
	return scratch.data();
}

template <class VF, class SmallSpan>
void ForwardTransformToBitReverseRVVF64(uint64_t* result, const uint64_t* operand, uint64_t n,
		uint64_t modulus, const NttFp64Tables& fp, uint64_t input_mod_factor, uint64_t output_mod_factor,
		const NttSmallSpanTables<SmallSpan, double>* small_span) {
	double* f = NttScratchF64(n);
	ForwardInputRVVF64<VF>(f, operand, n, modulus, input_mod_factor);                                   // [0, 2q)
	if constexpr (!std::is_same_v<SmallSpan, rvv::ntt::PerBlock>) {
		if (UseSmallSpan(small_span, n, SmallSpanLanes<SmallSpan, rvv::UintOf<VF>>())) {
			ForwardStagesRVVF64<VF>(f, n, modulus, fp.w.data(), fp.w_over_q.data(), n/2, small_span->t_top);  // per block down to t_top
			ForwardSmallSpanRVVF64<VF, SmallSpan>(result, f, n, modulus, *small_span, output_mod_factor);
			return;
		}
	}
	ForwardStagesRVVF64<VF>(f, n, modulus, fp.w.data(), fp.w_over_q.data(), n/2, 1);                   // every stage
	ForwardOutputRVVF64<VF>(result, f, n, modulus, output_mod_factor);
}

template <class VF>
void InverseTransformFromBitReverseRVVF64(uint64_t* result, const uint64_t* operand, uint64_t n,
		uint64_t modulus, const NttFp64Tables& fp, const uint64_t* inv_root_of_unity_powers,
		const uint64_t* precon_inv_root_of_unity_powers, uint64_t input_mod_factor, uint64_t output_mod_factor,
		const NttFp64IntSmallSpan* small_span) {
	// STUB: replace with the FP64 kernel (ntt-rvv-fp64.hpp)
	(void)fp;
	InverseTransformFromBitReverseRVV64(result, operand, n, modulus, inv_root_of_unity_powers,
			precon_inv_root_of_unity_powers, input_mod_factor, output_mod_factor, small_span);
}

// Every FP64 lane type (and, forward, every strategy), so tests and benchmarks
// can compare them; the dispatch uses rvv::cfg::NttF64 and NttSmallSpanF64.
#define HEXL_NTT_RVVF64_FWD(VF, S)                                                                          \
	template void ForwardTransformToBitReverseRVVF64<VF, S>(uint64_t*, const uint64_t*, uint64_t, uint64_t, \
			const NttFp64Tables&, uint64_t, uint64_t, const NttSmallSpanTables<S, double>*)
#define HEXL_NTT_RVVF64_INV(VF)                                                                             \
	template void InverseTransformFromBitReverseRVVF64<VF>(uint64_t*, const uint64_t*, uint64_t, uint64_t, \
			const NttFp64Tables&, const uint64_t*, const uint64_t*, uint64_t, uint64_t,                   \
			const NttFp64IntSmallSpan*)
#define HEXL_NTT_RVVF64_LANES(S)                  \
	HEXL_NTT_RVVF64_FWD(vfloat64m1_t, S);       \
	HEXL_NTT_RVVF64_FWD(vfloat64m2_t, S);       \
	HEXL_NTT_RVVF64_FWD(vfloat64m4_t, S);       \
	HEXL_NTT_RVVF64_FWD(vfloat64m8_t, S)
HEXL_NTT_RVVF64_LANES(rvv::ntt::PerBlock);
HEXL_NTT_RVVF64_LANES(rvv::ntt::FusedSlides);
#define HEXL_NTT_TRANSPOSED(S) HEXL_NTT_RVVF64_LANES(rvv::ntt::S);
HEXL_RVV_NTT_TRANSPOSED_SHAPES(HEXL_NTT_TRANSPOSED);
#undef HEXL_NTT_TRANSPOSED
HEXL_NTT_RVVF64_INV(vfloat64m1_t);
HEXL_NTT_RVVF64_INV(vfloat64m2_t);
HEXL_NTT_RVVF64_INV(vfloat64m4_t);
HEXL_NTT_RVVF64_INV(vfloat64m8_t);
#undef HEXL_NTT_RVVF64_LANES
#undef HEXL_NTT_RVVF64_INV
#undef HEXL_NTT_RVVF64_FWD

#endif  // HEXL_HAS_RVV

}  // namespace hexl
}  // namespace intel
