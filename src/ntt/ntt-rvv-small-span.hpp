// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Small-span NTT stages: what every strategy (rvv::ntt::PerBlock /
// FusedSlides / Transposed<Rows, Lmul>, chosen in src/util/rvv-config.hpp)
// agrees on, and the twiddle layouts of the ones that need their own tables.
//
// Terms (n = N, all sizes powers of two):
//   * Word: the storage type of one NTT element, x[0 .. n-1] (uint64_t or
//     uint32_t). SEW: the element width inside the vectors (e32 for the RVV32
//     kernels, also on uint64_t storage, narrowed on load; e64 for RVV64).
//   * vl: the running vector length, the number of NTT elements one vector
//     instruction processes; lane k: the k-th of them.
//   * stage t: butterfly span t; partners are elements e and e + t, with
//     e & t == 0. Forward stages run t = n/2 .. 1, inverse t = 1 .. n/2.
//   * block: the 2t elements one twiddle serves; element e is in block
//     e / (2t) of the stage, and the stage has m = n / (2t) blocks.
//   * L: the vl a strategy runs at (SmallSpanLanes, stored as tables.vl):
//     VLMAX of the kernel's lane type V for FusedSlides, VLMAX of
//     TransposedLane<S, V> for Transposed.
//   * t_top: the strategy runs the stages t < t_top (L for FusedSlides, Rows
//     for Transposed<Rows, Lmul>); the kernel runs t >= t_top with the generic
//     per-block loops (ntt-rvv-stages.hpp), underfilled only where t < VLMAX.
//
// Kernel structure (src/ntt/ntt-rvv.cpp):
//   forward: per-block stages t = n/2 .. t_top, then ForwardSmallSpanRVV32/64
//            <..., Strategy>, which calls the strategy's function (if
//            constexpr), for t = t_top/2 .. 1. In place on result, values in
//            [0, 4q) on entry; the function also produces the output range
//            ([0, q) for output_mod_factor 1, [0, 4q) for 4), so the final
//            reduction can be fused into its last stage.
//   inverse: InverseSmallSpanRVV32/64<..., Strategy> for t = 1 .. t_top/2,
//            reading operand ([0, 2q)) and writing result ([0, 2q)); they may
//            alias, so the copy pass can be fused into the first stage. Then
//            the per-block stages from t_top, then the last stage with N^-1.
//   The strategy runs only when the kernel gets tables built for this n and
//   L (UseSmallSpan below); otherwise every stage runs per block. That covers
//   N too small for the strategy (tables.vl == 0), tables built for another
//   lane type, and PerBlock, which has no small-span functions at all.
//
// Twiddle sources. The NTT tables: forward W, bit-reversed, the twiddle of
// block i at the stage with m blocks is W[m + i]; inverse Winv in stage order,
// the twiddle of block i at stage t is Winv[n - 2m + 1 + i]. Shoup factors
// alongside, 32-bit for e32 lanes, 64-bit for e64 lanes. Each layout below
// holds L-entry vectors of per-lane twiddles copied from these, one load per
// vector (plus one for the Shoup factors).
//
// FusedSlides layout (VL = L = VLMAX of V): chunk c = elements c*VL ..
//   c*VL + VL - 1 (one register). Per chunk, per small stage, lane l holds the
//   twiddle of element c*VL + l (both partners of a pair hold the same one, so
//   either half can be used):
//     fwd[(c * S + s) * VL + l]   stages in forward order:  t = VL/2 >> s
//     inv[(c * S + s) * VL + l]   stages in inverse order:  t = 1 << s
//   S = log2(VL) small stages, n * S entries per table. Needs N >= 2 VL.
//   Memory: 4 tables of n * S words (twiddles and Shoup factors, both
//   directions), e.g. N = 65536 e64 at VL = 16: 8 MB per NTT object. A compact
//   alternative is one VL/(2t)-entry run per chunk and stage, expanded with
//   vrgather (slow on the A100: 64 cycles at m1).
//
// Transposed<T, M> layout (T rows per tile, rows of LMUL M; T = 4, 8 or 16,
//   M = 1, 2 or 4, T * M <= 16 registers):
//   * row: one register group of TransposedLane<S, V> (SEW at LMUL M,
//     whatever V's LMUL), L = VLMAX of that type lanes, stride = n / L.
//     Row r (0 <= r < stride) = x[r], x[r + stride], .., x[r + (L-1)*stride]:
//     lane k of row r is x[r + k*stride]. One strided load (vl = L, stride
//     words) reads it.
//   * For t < stride the partners of row r are row r + t, lane by lane
//     (r & t == 0), so an aligned group of T rows is closed under the stages
//     t < T. Tile c = rows c*T .. c*T+T-1: T * L elements in T * M registers.
//     Lane k of a tile, down its T rows, is the contiguous chunk
//     x[k*stride + c*T .. k*stride + c*T + T-1].
//   The body loads a tile straight into registers, runs the stages
//   t = T/2 .. 1 there and stores it back the same way: one strided pass each
//   way, no scratch, on any VLEN.
//   Inside a tile, stage t has T/(2t) row groups g (rows 2gt .. 2gt+2t-1);
//   lane k of group g uses the twiddle of block c*T/(2t) + g + k*stride/(2t),
//   one L-entry vector per group. Tables are tile-major, T - 1 vectors per tile:
//     fwd[(c*(T-1) + T/(2t) - 1 + g) * L + k]   stages t = T/2 .. 1
//     inv[(c*(T-1) + T - T/t + g) * L + k]      stages t = 1 .. T/2
//   (the heap order of the NTT tables, per tile: n*(T-1)/T entries each.)
//   Needs N >= T * L (stride >= T).
//   Shape trade-off at a fixed budget T * M = 16 registers (the same T * L
//   elements per tile): more rows merge more stages per pass over memory
//   (log2 T of them; the per-block stages before the tile are the rest),
//   more lanes run them in fewer, longer instructions. Strided access costs
//   the same per element at any LMUL (X100 ~1 cycle, A100 17). The
//   instantiated shapes are HEXL_RVV_NTT_TRANSPOSED_SHAPES in rvv-config.hpp;
//   on the K3, rows x lanes at e32 (e64: half the lanes):
//     Transposed16 <16,1>: X100 16 x 8,  A100 16 x 32
//     Transposed8  <8,1>:  X100  8 x 8,  A100  8 x 32
//     Transposed8x2 <8,2>: X100  8 x 16, A100  8 x 64
//     Transposed4x4 <4,4>: X100  4 x 32, A100  4 x 128 (rows + twiddles +
//                          temporaries fill the 32 registers: may spill)

#pragma once

#include <stdint.h>

#include <memory>
#include <type_traits>
#include <vector>

#include "hexl/ntt/ntt.hpp"
#include "util/cpu-features.hpp"

#ifdef HEXL_HAS_RVV

#include "util/rvv-config.hpp"
#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

/// @brief Small-span twiddles of Strategy for one (N, q) and one lane count.
/// E = uint32_t for the RVV32 kernels, uint64_t for RVV64.
template <class Strategy, typename E>
struct NttSmallSpanTables {
	uint64_t n = 0;
	/// Lanes L the layout was built for (the vl the strategy runs at); 0 when
	/// Strategy does not run at this n (or is PerBlock): the kernels then run
	/// every stage per block.
	uint64_t vl = 0;
	/// The strategy runs the stages t < t_top, the kernel the rest per block.
	uint64_t t_top = 0;
	std::vector<E> fwd, fwd_precon;  // forward per-lane twiddles, Shoup factors
	std::vector<E> inv, inv_precon;  // inverse
};

namespace detail {
template <bool E32, unsigned Lmul>
struct TransposedLaneOf;
template <> struct TransposedLaneOf<true, 1> { using type = vuint32m1_t; };
template <> struct TransposedLaneOf<true, 2> { using type = vuint32m2_t; };
template <> struct TransposedLaneOf<true, 4> { using type = vuint32m4_t; };
template <> struct TransposedLaneOf<false, 1> { using type = vuint64m1_t; };
template <> struct TransposedLaneOf<false, 2> { using type = vuint64m2_t; };
template <> struct TransposedLaneOf<false, 4> { using type = vuint64m4_t; };
}  // namespace detail

/// @brief The lane type of one row of a Transposed<Rows, Lmul> tile S: V's
/// element width at LMUL Lmul, whatever V's own LMUL.
template <class S, class V>
using TransposedLane = typename detail::TransposedLaneOf<rvv::IsE32<V>, S::kLmul>::type;

/// @brief The lanes L Strategy runs at under kernel lane type V.
template <class Strategy, class V>
inline uint64_t SmallSpanLanes() {
	if constexpr (rvv::ntt::kIsTransposed<Strategy>) {
		return rvv::VlMax<TransposedLane<Strategy, V>>();
	} else {
		return rvv::VlMax<V>();
	}
}

/// @brief Whether Strategy runs its small-span functions at this n and L.
template <class Strategy>
constexpr bool SmallSpanRuns(uint64_t n, uint64_t lanes) {
	if constexpr (std::is_same_v<Strategy, rvv::ntt::FusedSlides>) {
		return lanes >= 2 && n >= 2 * lanes;
	} else if constexpr (rvv::ntt::kIsTransposed<Strategy>) {
		return lanes >= 2 && n >= Strategy::kRows * lanes;
	} else {
		static_assert(std::is_same_v<Strategy, rvv::ntt::PerBlock>, "unknown small-span strategy");
		return false;
	}
}

/// @brief Kernel-side check: tables present and built for this n and L.
template <class Strategy, typename E>
inline bool UseSmallSpan(const NttSmallSpanTables<Strategy, E>* tables, uint64_t n, uint64_t lanes) {
	return tables != nullptr && tables->vl == lanes && tables->n == n;
}

/// @brief Builds Strategy's layout (see the top of this file) for L = lanes,
/// from the NTT tables:
///   w, w_precon:          forward table (bit-reversed) and its Shoup factors
///   w_inv, w_inv_precon:  inverse table (stage order) and its Shoup factors
/// The Shoup factors must be the ones for E: 32-bit (Precon32) for uint32_t,
/// 64-bit (Precon64) for uint64_t. Scalar, runs once per NTT object.
template <class Strategy, typename E>
NttSmallSpanTables<Strategy, E> BuildNttSmallSpanTables(
		uint64_t n, uint64_t lanes, const uint64_t* w, const uint64_t* w_precon,
		const uint64_t* w_inv, const uint64_t* w_inv_precon);

/// @brief Same, from an NTT object's tables (tests, benchmarks).
template <class Strategy, typename E>
NttSmallSpanTables<Strategy, E> BuildNttSmallSpanTables(const NTT& ntt, uint64_t lanes) {
	constexpr bool k32 = std::is_same_v<E, uint32_t>;
	return BuildNttSmallSpanTables<Strategy, E>(
			ntt.GetDegree(), lanes, ntt.GetRootOfUnityPowers().data(),
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
	std::shared_ptr<const T32> rvv32_u64;  // lanes rvv::cfg::Ntt32U64 (uint64_t storage); rvv32 when L matches
	std::shared_ptr<const T64> rvv64;      // lanes rvv::cfg::Ntt64
};

/// @brief The tables of the configured strategies for this NTT (RVV32 tables
/// for q < 2^30, RVV64 otherwise); null when no strategy runs (PerBlock, or N
/// too small). Needs V at run time (has_rvv): it reads VLMAX.
std::shared_ptr<const NttSmallSpanSet> BuildNttSmallSpanSet(const NTT& ntt);

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
