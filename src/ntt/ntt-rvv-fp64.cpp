// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
//
// FP64 NTT path (rvv-hexl extension): the tables, and the RVV kernels, which
// are STUBS until written (algorithm and conditions: ntt-rvv-fp64.hpp). The
// stubs call the e64 integer kernels, so the dispatch, the tests and the
// benchmarks work from the start.

#include "ntt/ntt-rvv-fp64.hpp"

#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "ntt/ntt-internal.hpp"

#ifdef HEXL_HAS_RVV
#include <riscv_vector.h>

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
	return t;
}

#ifdef HEXL_HAS_RVV

template <class VF>
void ForwardTransformToBitReverseRVVF64(uint64_t* result, const uint64_t* operand, uint64_t n,
		uint64_t modulus, const NttFp64Tables& fp, const uint64_t* root_of_unity_powers,
		const uint64_t* precon_root_of_unity_powers, uint64_t input_mod_factor, uint64_t output_mod_factor,
		const NttFp64IntSmallSpan* small_span) {
	// STUB: replace with the FP64 kernel (ntt-rvv-fp64.hpp)
	(void)fp;
	ForwardTransformToBitReverseRVV64(result, operand, n, modulus, root_of_unity_powers, precon_root_of_unity_powers,
			input_mod_factor, output_mod_factor, small_span);
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

// Every FP64 lane type, so tests and benchmarks can compare them; the dispatch
// uses rvv::cfg::NttF64.
#define HEXL_NTT_RVVF64(VF)                                                                             \
	template void ForwardTransformToBitReverseRVVF64<VF>(uint64_t*, const uint64_t*, uint64_t, uint64_t, \
			const NttFp64Tables&, const uint64_t*, const uint64_t*, uint64_t, uint64_t,                   \
			const NttFp64IntSmallSpan*);                                                                  \
	template void InverseTransformFromBitReverseRVVF64<VF>(uint64_t*, const uint64_t*, uint64_t, uint64_t, \
			const NttFp64Tables&, const uint64_t*, const uint64_t*, uint64_t, uint64_t,                   \
			const NttFp64IntSmallSpan*)
HEXL_NTT_RVVF64(vfloat64m1_t);
HEXL_NTT_RVVF64(vfloat64m2_t);
HEXL_NTT_RVVF64(vfloat64m4_t);
HEXL_NTT_RVVF64(vfloat64m8_t);
#undef HEXL_NTT_RVVF64

#endif  // HEXL_HAS_RVV

}  // namespace hexl
}  // namespace intel
