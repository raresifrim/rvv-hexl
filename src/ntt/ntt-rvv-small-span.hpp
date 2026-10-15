// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Small-span NTT stages: what every strategy (rvv::ntt::PerBlock /
// FusedSlides / Transposed, chosen in src/util/rvv-config.hpp) agrees on,
// and the twiddle layouts of the two that need their own tables.
//
// Terms (n = N, VL = VLMAX of the kernel's lane type, both powers of two):
//   * stage t: butterfly span t; partners are elements e and e + t, with
//     e & t == 0. Forward stages run t = n/2 .. 1, inverse t = 1 .. n/2.
//   * block: the 2t elements one twiddle serves; element e is in block
//     e / (2t) of the stage, and the stage has m = n / (2t) blocks.
//   * small span: t < VL. Large spans run the generic per-block loops
//     (ntt-rvv-stages.hpp), with every strip full.
//
// Kernel structure (src/ntt/ntt-rvv.cpp):
//   forward: large stages t = n/2 .. VL, then ForwardSmallSpanRVV32/64(
//            Strategy{}, ...) for t = VL/2 .. 1. In place on result, values in
//            [0, 4q) on entry; the function also produces the output range
//            ([0, q) for output_mod_factor 1, [0, 4q) for 4), so the final
//            reduction can be fused into its last stage.
//   inverse: InverseSmallSpanRVV32/64(Strategy{}, ...) for t = 1 .. VL/2,
//            reading operand ([0, 2q)) and writing result ([0, 2q)); they may
//            alias, so the copy pass can be fused into the first stage. Then
//            the large stages, then the last stage with N^-1 folded in.
//   The strategy runs only when the kernel gets tables built for this n and
//   VL (UseSmallSpan below); otherwise every stage runs per block. That covers
//   N too small for the strategy (tables.vl == 0), tables built for another
//   lane type, and PerBlock, which has no small-span functions at all.
//
// Twiddle sources. The NTT tables: forward W, bit-reversed, the twiddle of
// block i at the stage with m blocks is W[m + i]; inverse Winv in stage order,
// the twiddle of block i at stage t is Winv[n - 2m + 1 + i]. Shoup factors
// alongside, 32-bit for e32 lanes, 64-bit for e64 lanes. Each small-span
// layout below holds VL-entry vectors of per-lane twiddles copied from these,
// one load per vector (plus one for the Shoup factors).
//
// FusedSlides layout: chunk c = elements c*VL .. c*VL + VL - 1 (one register).
//   Per chunk, per small stage, lane l holds the twiddle of element c*VL + l
//   (both partners of a pair hold the same one, so either half can be used):
//     fwd[(c * S + s) * VL + l]   stages in forward order:  t = VL/2 >> s
//     inv[(c * S + s) * VL + l]   stages in inverse order:  t = 1 << s
//   S = log2(VL) small stages, n * S entries per table. Needs N >= 2 VL.
//   Memory: 4 tables of n * S words (twiddles and Shoup factors, both
//   directions), e.g. N = 65536 e64 at VL = 16: 8 MB per NTT object. A compact
//   alternative is one VL/(2t)-entry run per chunk and stage, expanded with
//   vrgather (slow on the A100: 64 cycles at m1).
//
// Transposed layout: R = n / VL rows of VL lanes, row r lane k = element
//   r + k*R (the transposed copy B[r * VL + k], built with one strided load
//   per row). At a small stage t (t < VL <= R) the partners of row r are row
//   r + t, lane by lane, for r & t == 0. Lane k of rows r .. r + 2t - 1 uses
//   the twiddle of block k*R/(2t) + g, with g = r / (2t), so G = R / (2t)
//   row groups share one vector:
//     fwd: stages t = VL/2 .. 1, groups g = 0 .. G-1, lanes k: fwd[off + g * VL + k]
//     inv: stages t = 1 .. VL/2, same order within a stage
//   (off = the entries of the earlier stages; a stage has G * VL = n / (2t)
//   entries, the table fewer than n.) Needs N >= VL^2 and n words of
//   scratch per thread (SmallSpanScratch).

#pragma once

#include <stdint.h>

#include <memory>
#include <type_traits>
#include <vector>

#include "hexl/ntt/ntt.hpp"
#include "util/cpu-features.hpp"

#ifdef HEXL_HAS_RVV

#include "util/rvv-config.hpp"

namespace intel {
namespace hexl {

/// @brief Small-span twiddles of Strategy for one (N, q) and one VL.
/// E = uint32_t for the RVV32 kernels, uint64_t for RVV64.
template <class Strategy, typename E>
struct NttSmallSpanTables {
	uint64_t n = 0;
	/// VLMAX the layout was built for; 0 when Strategy does not run at this n
	/// (or is PerBlock): the kernels then run every stage per block.
	uint64_t vl = 0;
	std::vector<E> fwd, fwd_precon;  // forward per-lane twiddles, Shoup factors
	std::vector<E> inv, inv_precon;  // inverse
};

/// @brief Whether Strategy runs its small-span functions at this n and VL.
template <class Strategy>
constexpr bool SmallSpanRuns(uint64_t n, uint64_t vl) {
	if constexpr (std::is_same_v<Strategy, rvv::ntt::FusedSlides>) {
		return vl >= 2 && n >= 2 * vl;
	} else if constexpr (std::is_same_v<Strategy, rvv::ntt::Transposed>) {
		return vl >= 2 && n >= vl * vl;
	} else {
		static_assert(std::is_same_v<Strategy, rvv::ntt::PerBlock>, "unknown small-span strategy");
		return false;
	}
}

/// @brief Kernel-side check: tables present and built for this n and VL.
template <class Strategy, typename E>
inline bool UseSmallSpan(const NttSmallSpanTables<Strategy, E>* tables, uint64_t n, uint64_t vl) {
	return tables != nullptr && tables->vl == vl && tables->n == n;
}

/// @brief Builds Strategy's layout (see the top of this file) for lanes of
/// VLMAX vl, from the NTT tables:
///   w, w_precon:          forward table (bit-reversed) and its Shoup factors
///   w_inv, w_inv_precon:  inverse table (stage order) and its Shoup factors
/// The Shoup factors must be the ones for E: 32-bit (Precon32) for uint32_t,
/// 64-bit (Precon64) for uint64_t. Scalar, runs once per NTT object.
template <class Strategy, typename E>
NttSmallSpanTables<Strategy, E> BuildNttSmallSpanTables(
		uint64_t n, uint64_t vl, const uint64_t* w, const uint64_t* w_precon,
		const uint64_t* w_inv, const uint64_t* w_inv_precon);

/// @brief Same, from an NTT object's tables (tests, benchmarks).
template <class Strategy, typename E>
NttSmallSpanTables<Strategy, E> BuildNttSmallSpanTables(const NTT& ntt, uint64_t vl) {
	constexpr bool k32 = std::is_same_v<E, uint32_t>;
	return BuildNttSmallSpanTables<Strategy, E>(
			ntt.GetDegree(), vl, ntt.GetRootOfUnityPowers().data(),
			(k32 ? ntt.GetPrecon32RootOfUnityPowers() : ntt.GetPrecon64RootOfUnityPowers()).data(),
			ntt.GetInvRootOfUnityPowers().data(),
			(k32 ? ntt.GetPrecon32InvRootOfUnityPowers() : ntt.GetPrecon64InvRootOfUnityPowers()).data());
}

/// @brief What an NTT object keeps for the configured strategies
/// (NTT::m_rvv_small_span): one table set per lane type it dispatches to.
struct NttSmallSpanSet {
	using T32 = NttSmallSpanTables<rvv::cfg::NttSmallSpan32, uint32_t>;
	using T64 = NttSmallSpanTables<rvv::cfg::NttSmallSpan64, uint64_t>;
	std::shared_ptr<const T32> rvv32;      // lanes rvv::cfg::Ntt32 (uint32_t storage)
	std::shared_ptr<const T32> rvv32_u64;  // lanes rvv::cfg::Ntt32U64 (uint64_t storage); rvv32 when the VLMAX match
	std::shared_ptr<const T64> rvv64;      // lanes rvv::cfg::Ntt64
};

/// @brief The tables of the configured strategies for this NTT (RVV32 tables
/// for q < 2^30, RVV64 otherwise); null when no strategy runs (PerBlock, or N
/// too small). Needs V at run time (has_rvv): it reads VLMAX.
std::shared_ptr<const NttSmallSpanSet> BuildNttSmallSpanSet(const NTT& ntt);

/// @brief Per-thread scratch of at least n elements, for the Transposed
/// strategy (an NTT object is used from many OpenMP threads at once, so the
/// scratch cannot live in it). Grows, never shrinks.
template <typename E>
inline E* SmallSpanScratch(uint64_t n) {
	thread_local std::vector<E> buf;
	if (buf.size() < n) buf.resize(n);
	return buf.data();
}

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
