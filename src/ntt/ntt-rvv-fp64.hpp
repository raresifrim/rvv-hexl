// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// SPDX-License-Identifier: Apache-2.0
//
// FP64 NTT path (rvv-hexl extension, SCAFFOLDING: the kernels are stubs that
// call the e64 integer kernels). For moduli between 2^30 and 2^50, the
// butterfly's modular multiply can run on the FP64 FMA instead of the e64
// integer multiplier: on the K3 the FMA retires 4 elements per cycle, the e64
// vmul / vmulhu 1 (both clusters, rvv-hexl/results/ntt-width-2026-10-20).
//
// Modular multiply with doubles (Harvey, "Faster arithmetic for
// number-theoretic transforms", J. Symb. Comput. 2014), a, b < 4q < 2^52 as
// exact doubles, qd = q, qi = 1/q:
//   h = a * b                      the rounded product (vfmul)
//   l = fma(a, b, -h)              its exact rounding error: h + l = a * b (vfmsub)
//   t = round(h * qi)              quotient estimate (vfmul + vfcvt with the
//                                  rounding mode, or the 2^52 + 2^51 shift trick)
//   r = fma(-t, qd, h) + l         a * b - t q, exact while |r| < 2^53 (vfnmsac, vfadd)
//   r in (-q, q): + q when negative (vmflt + masked vfadd)
// With a twiddle w < q the Shoup-style variant precomputes w * qi (one FMA
// fewer). Inside a transform the values stay doubles (convert once on the way
// in and out, vfcvt.f.xu / vfcvt.xu.f); the butterfly's additions are vfadd /
// vfsub with the same lazy ranges as the integer kernels (forward [0, 4q),
// inverse [0, 2q)), exact because 4q < 2^52.
//
// Conditions (NttFp64Applicable):
//   q >= 2^30   below it the RVV32 kernels run e32 integer lanes, 8 elements
//               per cycle and one multiply per product: faster than FP64
//   q <  2^50   the lazy values (< 4q < 2^52) and the products' error terms
//               are exact in a double's 53-bit significand
//   N a power of two >= 2 (as every NTT)
// and, at run time, an RVV unit with FP64 (RVA23: V includes D) and the switch
// rvv::cfg::kNttFp64 (UseNttFp64). Moduli >= 2^50 keep the e64 integer path.
//
// No FP32 path: the same scheme on floats needs q < 2^22 (4q < 2^24), below
// every modulus binfhe and BFV use, and costs about twice the instructions of
// the e32 Shoup multiply, which runs at the full integer rate. On the K3 FP32
// runs at the e32 integer rate on the X100 and twice it on the A100
// (results/ntt-width-2026-10-20/fma-throughput32.txt): even at best a tie.

#pragma once

#include <stdint.h>

#include <memory>

#include "hexl/ntt/ntt.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "ntt/ntt-rvv-small-span.hpp"
#include "util/cpu-features.hpp"
#include "util/rvv-config.hpp"  // lane types (empty without RVV)

namespace intel {
namespace hexl {

/// Smallest modulus the FP64 path takes: below it, the e32 integer lanes win.
constexpr uint64_t kNttFp64MinModulus = 1ULL << 30;
/// Bound on the modulus: the lazy values below 4q must be exact doubles.
constexpr uint64_t kNttFp64MaxModulus = 1ULL << 50;

/// The arithmetic conditions of the FP64 path alone (no hardware, no switch).
constexpr bool NttFp64Applicable(uint64_t modulus, uint64_t n) {
	return modulus >= kNttFp64MinModulus && modulus < kNttFp64MaxModulus && n >= 2 && (n & (n - 1)) == 0;
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
