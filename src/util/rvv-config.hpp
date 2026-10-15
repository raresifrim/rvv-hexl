// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// The lane type (SEW + LMUL) of every RVV kernel, in one place.
//
// Each kernel is a template on its lane type V and takes its default from
// here; a different LMUL is a one-line change. Tests (test/test-rvv-lanes.cpp)
// and benchmarks (bench-hexl BM_Lanes_*) instantiate the same kernels at
// m1/m2/m4/m8 to compare them.
//
// Defaults: m4, except MultMod at m8. Measured on the K3 (n = 4096), m4 is the
// best LMUL on the X100 for every kernel and helper measured so far; the A100
// is 5-20% faster at m8 for some, but m8 leaves only 4 register groups, so
// kernels that keep many values live can spill. MultMod does not (checked in
// the disassembly): at m8 it ties m4 on the X100 (6.40 vs 6.37 cycles/elem
// e64, 1.67 vs 1.64 e32) and is 6-10% faster on the A100 (5.37 vs 5.72 e64,
// 1.13 vs 1.25 e32), so it runs at m8. With input_mod_factor 4 the e64
// kernel is ~4% slower at m8 on the X100 (8.6 vs 8.25) and 6% faster on the
// A100 (6.34 vs 6.77).
// Constraints:
//   * e64 lanes: vuint64m1_t..m8_t; e32 lanes: vuint32mf2_t..m8_t.
//   * e32 lanes on uint64_t storage (Load32/Store32): at most vuint32m4_t.
// The kernel's lane SEW follows its storage word (LaneFor below), except the
// RVV32 paths of MultMod/FMA, which compute in e32 for both storage words.

#pragma once

#include "util/cpu-features.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>
#include <stdint.h>

#include <type_traits>

namespace intel {
namespace hexl {
namespace rvv {

// Small-span NTT strategies: how the stages with span t < VLMAX (butterfly
// partners closer than one register) are vectorized. Large-span stages are
// the same for all three. Layouts and contracts: src/ntt/ntt-rvv-small-span.hpp.
namespace ntt {
/// Each block's t butterflies as one strip with vl = t (the first-version
/// kernels): no extra tables, but the register is underfilled.
struct PerBlock {};
/// Load VL contiguous elements once, run every small stage in the register
/// with vslidedown / vslideup / vmerge and per-lane twiddles, store once.
/// Contiguous access only; needs N >= 2 VL. Measured best on the A100.
struct FusedSlides {};
/// Tiles of Rows rows, each row one register group of LMUL Lmul (L = VLMAX
/// lanes of V's element width at that LMUL), loaded with strided loads
/// straight into registers, in which butterfly partners share a lane across
/// rows: the stages t < Rows as plain row butterflies with per-lane twiddles,
/// stored back with strided stores. Rows * Lmul <= 16 registers (the
/// twiddles and the temporaries take the rest of the 32). More rows merge
/// more stages per pass over memory, more lanes run them in fewer, longer
/// instructions. Needs N >= Rows * L. Runs on any VLEN; fast where strided
/// access is cheap (X100), correct but slow on the A100.
template <unsigned Rows, unsigned Lmul = 1>
struct Transposed {
	static_assert(Rows == 4 || Rows == 8 || Rows == 16, "Transposed tiles have 4, 8 or 16 rows");
	static_assert(Lmul == 1 || Lmul == 2 || Lmul == 4, "Transposed rows are m1, m2 or m4");
	static_assert(Rows * Lmul <= 16, "Transposed tile rows must fit in 16 registers");
	static constexpr unsigned kRows = Rows;
	static constexpr unsigned kLmul = Lmul;
};
template <class S>
inline constexpr bool kIsTransposed = false;
template <unsigned Rows, unsigned Lmul>
inline constexpr bool kIsTransposed<Transposed<Rows, Lmul>> = true;

// The Transposed shapes the library instantiates (kernels, twiddle tables,
// tests, benchmarks), rows x LMUL. To add one: an alias here and an entry in
// the list. cfg::NttSmallSpan64/32 must name one of them.
using Transposed16 = Transposed<16, 1>;
using Transposed8 = Transposed<8, 1>;
using Transposed8x2 = Transposed<8, 2>;
using Transposed4x4 = Transposed<4, 4>;
#define HEXL_RVV_NTT_TRANSPOSED_SHAPES(X) X(Transposed16) X(Transposed8) X(Transposed8x2) X(Transposed4x4)
}  // namespace ntt

namespace cfg {

using AddMod64 = vuint64m4_t;
using AddMod32 = vuint32m4_t;
using SubMod64 = vuint64m4_t;
using SubMod32 = vuint32m4_t;
using CmpAdd64 = vuint64m4_t;
using CmpAdd32 = vuint32m4_t;
using CmpSubMod64 = vuint64m4_t;
using CmpSubMod32 = vuint32m4_t;
using ReduceMod64 = vuint64m4_t;
using ReduceMod32 = vuint32m4_t;
using MultMod64 = vuint64m8_t;     // RVV64: 64-bit storage, q >= 2^30
using MultMod32 = vuint32m8_t;     // RVV32 on uint32_t storage, q < 2^30
using MultMod32U64 = vuint32m4_t;  // RVV32 on uint64_t storage, q < 2^30 (<= m4)
using FMAMod64 = vuint64m4_t;
using FMAMod32 = vuint32m4_t;   // <= m4
				// EltwiseMultAddMod (rvv-hexl extension): m4. Three input streams plus the
				// Barrett temporaries keep more groups live than MultMod; BM_Lanes_MultAddMod
				// (2026-10-14): m4 best or within 2% on both clusters, m8 6-14% slower on the X100.
using MultAddMod64 = vuint64m4_t;
using MultAddMod32 = vuint32m4_t;     // RVV32 on uint32_t storage
using MultAddMod32U64 = vuint32m4_t;  // RVV32 on uint64_t storage (<= m4)
				      // NTT at m1 for now: with the per-block small-span strategy (below), stages
				      // with span t < VLMAX run with vl = t, and a vector instruction costs about
				      // the same however small vl is (more at higher LMUL). Measured (forward,
				      // N=4096 e64 / N=1024 e32, X100): RVV64 m1 125 us vs m4 210 us; RVV32 m1
				      // 16.5 us vs m4 26.4 us. Revisit with the fused-slides / transposed
				      // strategies: the layout study measured both best at m4.
using Ntt64 = vuint64m1_t;
using Ntt32 = vuint32m1_t;     // RVV32 NTT on uint32_t storage
using Ntt32U64 = vuint32m1_t;  // RVV32 NTT on uint64_t storage (<= m4: narrowing load)

// How the small-span NTT stages (span t < VLMAX of the lane type) run; see
// rvv::ntt above and src/ntt/ntt-rvv-small-span.hpp. One for the RVV64
// kernels, one for both RVV32 kernels (both storage words).
// NTT::ComputeRootOfUnityPowers builds the twiddle layout of the strategy
// chosen here, for the lane types above (Ntt32 and Ntt32U64 get one layout
// each when their VLMAX differs). Tests and BM_Lanes_Ntt* run all three.
// PerBlock until the other two have bodies (their stubs run PerBlock).
using NttSmallSpan64 = ntt::PerBlock;
using NttSmallSpan32 = ntt::PerBlock;

}  // namespace cfg

/// Lane type of a kernel whose SEW follows its storage word.
template <class Word, class L64, class L32>
using LaneFor = std::conditional_t<std::is_same_v<Word, uint64_t>, L64, L32>;

}  // namespace rvv
}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
