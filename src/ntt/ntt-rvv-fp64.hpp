// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// SPDX-License-Identifier: Apache-2.0
//
// FP64 NTT path (rvv-hexl extension, SCAFFOLDING: the kernels are stubs that
// call the e64 integer kernels). For moduli between 2^30 and 2^50 the
// butterfly's modular multiply can run on the FP64 FMA; the method and the
// modulus conditions are in util/fp64-mod.hpp. Here: with a twiddle w < q
// precompute w * qi (one multiply fewer). Inside a transform the values stay
// doubles (convert once on the way in and out, vfcvt.f.xu / vfcvt.xu.f); the
// butterfly's additions are vfadd / vfsub with the same lazy ranges as the
// integer kernels (forward [0, 4q), inverse [0, 2q)), exact because 4q < 2^52.
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
struct NttFp64Tables {
	AlignedVector64<double> w, w_inv;
	double q = 0, q_inv = 0;
};

/// Builds the FP64 tables of `ntt` (its modulus must satisfy NttFp64Applicable).
std::shared_ptr<const NttFp64Tables> BuildNttFp64Tables(const NTT& ntt);

#ifdef HEXL_HAS_RVV
/// Whether NTT::ComputeForward / ComputeInverse (64-bit storage) run the FP64
/// kernels: the switch rvv::cfg::kNttFp64, an RVV unit, and the conditions.
inline bool UseNttFp64(uint64_t modulus, uint64_t n) {
	return rvv::cfg::kNttFp64 && has_rvv && NttFp64Applicable(modulus, n);
}

/// The e64 kernel's small-span tables (NttSmallSpanSet::rvv64), which the stubs
/// pass on; the FP64 kernels may ignore them.
using NttFp64IntSmallSpan = NttSmallSpanTables<rvv::cfg::NttSmallSpan64, uint64_t>;

/// FP64 forward NTT (STUB: calls ForwardTransformToBitReverseRVV64). Same
/// contract as the e64 kernel (input_mod_factor 1, 2, 4; output 1 or 4); `fp`
/// from BuildNttFp64Tables. The integer tables and `small_span` are the stub's;
/// the FP64 body needs only `fp`. VF: vfloat64m1_t .. vfloat64m8_t.
template <class VF = rvv::cfg::NttF64>
void ForwardTransformToBitReverseRVVF64(uint64_t* result, const uint64_t* operand, uint64_t n,
		uint64_t modulus, const NttFp64Tables& fp, const uint64_t* root_of_unity_powers,
		const uint64_t* precon_root_of_unity_powers, uint64_t input_mod_factor, uint64_t output_mod_factor,
		const NttFp64IntSmallSpan* small_span = nullptr);

/// FP64 inverse NTT, N^{-1} scaling included (STUB: calls
/// InverseTransformFromBitReverseRVV64). input_mod_factor 1 or 2; output 1 or 2.
template <class VF = rvv::cfg::NttF64>
void InverseTransformFromBitReverseRVVF64(uint64_t* result, const uint64_t* operand, uint64_t n,
		uint64_t modulus, const NttFp64Tables& fp, const uint64_t* inv_root_of_unity_powers,
		const uint64_t* precon_inv_root_of_unity_powers, uint64_t input_mod_factor, uint64_t output_mod_factor,
		const NttFp64IntSmallSpan* small_span = nullptr);
#endif  // HEXL_HAS_RVV

}  // namespace hexl
}  // namespace intel
