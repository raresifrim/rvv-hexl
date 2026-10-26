// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
//
// FP64 NTT path (rvv-hexl extension). For moduli between
// 2^30 and 2^50 the butterfly's modular multiply can run on the FP64 FMA; the
// method and the modulus conditions are in util/fp64-mod.hpp. Here: with a
// twiddle w < q precompute w / q (one multiply fewer). Inside a transform the
// values stay doubles, in a per-thread scratch (convert once on the way in and
// out); the butterflies (rvv::FwdButterflyF64 / InvButterflyF64) keep signed
// lazy ranges, forward (-2q, 2q), inverse [-q, q], exact because 2q < 2^51.
//
// Structure, as the integer kernels (ntt-rvv.cpp), one small-span strategy at
// compile time (rvv::cfg::NttSmallSpanF64, or the SmallSpan parameter):
//   forward: first stage t = n/2 with the input conversion fused in (operand
//            -> scratch, ForwardFirstStageRVVF64), per-block stages t = n/4 ..
//            t_top (ForwardStagesRVVF64), small spans by the
//            strategy, scratch -> result in the output range (FusedSlides /
//            Transposed: Forward*RVVF64); PerBlock: every stage per block,
//            then ForwardOutputRVVF64.
//   inverse: small spans by the strategy, operand -> scratch
//            (Inverse*RVVF64; PerBlock: the input pass InverseInputRVVF64),
//            per-block stages t_top .. n/4 (InverseStagesRVVF64), the last
//            stage with N^{-1} scratch -> result (InverseLastStageRVVF64).
// Small-span tables: NttSmallSpanTables<S, double>, the integer layouts with
// the double twiddles and their w / q (BuildNttSmallSpanTablesF64).
//
// Conditions (NttFp64Applicable): ModMulFp64Applicable(q) and N a power of
// two >= 2; at run time an RVV unit with FP64 (RVA23: V includes D) and the
// switch rvv::cfg::kNttFp64 (UseNttFp64). Moduli >= 2^50 keep the e64 path.

#pragma once

#include <stdint.h>

#include <memory>

#include "hexl/ntt/ntt.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "ntt/ntt-rvv-small-span.hpp"
#include "util/cpu-features.hpp"
#include "util/fp64-mod.hpp"
#include "util/rvv-config.hpp"  // lane types (empty without RVV)

namespace intel {
namespace hexl {

/// The arithmetic conditions of the FP64 NTT alone (no hardware, no switch).
constexpr bool NttFp64Applicable(uint64_t modulus, uint64_t n) {
	return ModMulFp64Applicable(modulus) && n >= 2 && (n & (n - 1)) == 0;
}

/// FP64 twiddle tables of one NTT object: its root-of-unity tables (same orders
/// as NTT::GetRootOfUnityPowers / GetInvRootOfUnityPowers) as doubles, q and 1/q.
/// Each twiddle w comes with its quotient factor w / q, the FP counterpart of a
/// Shoup factor: the butterfly's quotient is round(y * (w / q)), which does not
/// wait for the product y * w. The last inverse stage has N^{-1} folded in, as
/// in the integer kernels (InverseLastStageRVV64).
struct NttFp64Tables {
	AlignedVector64<double> w, w_inv;                // w < q < 2^50: exact
	AlignedVector64<double> w_over_q, w_inv_over_q;  // fl(w / q): relative error <= 2^-53
	double q = 0, q_inv = 0;                         // q, fl(1 / q)
	double n_inv = 0, n_inv_over_q = 0;              // N^{-1} mod q, fl(N^{-1} / q)
	double wn_inv = 0, wn_inv_over_q = 0;            // w_inv[n - 1] N^{-1} mod q (last inverse twiddle), / q
#ifdef HEXL_HAS_RVV
	/// The layout of rvv::cfg::NttSmallSpanF64 for the lanes it runs at under
	/// rvv::cfg::NttF64; null when it does not run (PerBlock, N too small, no V).
	std::shared_ptr<const NttSmallSpanTables<rvv::cfg::NttSmallSpanF64, double>> small_span;
#endif
};

/// Builds the FP64 tables of `ntt` (its modulus must satisfy NttFp64Applicable).
std::shared_ptr<const NttFp64Tables> BuildNttFp64Tables(const NTT& ntt);

#ifdef HEXL_HAS_RVV
/// Whether NTT::ComputeForward / ComputeInverse (64-bit storage) run the FP64
/// kernels: the switch rvv::cfg::kNttFp64, an RVV unit, and the conditions.
inline bool UseNttFp64(uint64_t modulus, uint64_t n) {
	return rvv::cfg::kNttFp64 && has_rvv && NttFp64Applicable(modulus, n);
}

/// The small-span layout of Strategy for the FP64 kernels at `lanes`
/// (SmallSpanLanes<Strategy, rvv::UintOf<VF>>()), from the FP64 tables.
template <class Strategy>
NttSmallSpanTables<Strategy, double> BuildNttSmallSpanTablesF64(const NttFp64Tables& fp, uint64_t lanes) {
	return BuildNttSmallSpanTables<Strategy, double>(fp.w.size(), lanes, fp.w.data(), fp.w_over_q.data(),
			fp.w_inv.data(), fp.w_inv_over_q.data());
}

/// FP64 forward NTT. Same contract as the e64 kernel (input_mod_factor 1, 2,
/// 4; output 1 or 4; result may alias operand); `fp` from BuildNttFp64Tables,
/// `small_span` from BuildNttSmallSpanTablesF64<SmallSpan> at the lanes
/// SmallSpan runs at under VF (null, or built for other lanes: every stage per
/// block). VF: vfloat64m1_t .. vfloat64m8_t (the per-block stages; Transposed
/// rows have their own LMUL).
template <class VF = rvv::cfg::NttF64, class SmallSpan = rvv::cfg::NttSmallSpanF64>
void ForwardTransformToBitReverseRVVF64(uint64_t* result, const uint64_t* operand, uint64_t n,
		uint64_t modulus, const NttFp64Tables& fp, uint64_t input_mod_factor, uint64_t output_mod_factor,
		const NttSmallSpanTables<SmallSpan, double>* small_span = nullptr);

/// FP64 inverse NTT, N^{-1} scaling included. Same contract as the e64 kernel
/// (input_mod_factor 1 or 2; output 1 or 2, always [0, q) here); tables and
/// lanes as the forward.
template <class VF = rvv::cfg::NttF64, class SmallSpan = rvv::cfg::NttSmallSpanF64>
void InverseTransformFromBitReverseRVVF64(uint64_t* result, const uint64_t* operand, uint64_t n,
		uint64_t modulus, const NttFp64Tables& fp, uint64_t input_mod_factor, uint64_t output_mod_factor,
		const NttSmallSpanTables<SmallSpan, double>* small_span = nullptr);
#endif  // HEXL_HAS_RVV

}  // namespace hexl
}  // namespace intel
