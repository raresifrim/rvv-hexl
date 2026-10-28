// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
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
// SignedDigitDecompose (rvv-hexl extension, binfhe), BM_Lanes_SignedDigitDecompose (2026-10-19,
// ns per call, X100 / A100): e64 m4 8422 / 6249, m8 8472 / 5399; e32 on uint32_t m4 5796 / 4002,
// m8 5713 / 3521; e32 on uint64_t (narrowing, <= m4) m2 11002 / 6802, m4 10012 / 5717. m8 is
// even on the X100 and 12-14% faster on the A100: m8, and u32m4 on uint64_t storage.
using SignedDigitDecompose64 = vuint64m8_t;
using SignedDigitDecompose32 = vuint32m8_t;     // RVV32 on uint32_t storage
using SignedDigitDecompose32U64 = vuint32m4_t;  // RVV32 on uint64_t storage (<= m4)
// BaseConvert (rvv-hexl extension, OpenFHE key switching): one accumulator
// and the Shoup temporaries live, so every LMUL fits. BM_Lanes_BaseConvert,
// N = 8192, median of 5 (rvv-hexl results/baseconv-rvv-2026-10-26), gain over
// BaseConvertNative, shape 1 -> 4 (the L = 1 path, CKKS ModUp / ModDown) and
// 3 -> 5 (the L > 1 path), X100 / A100:
//          1 -> 4         3 -> 5        geomean
//   m1   2.06 / 6.73   1.06 / 4.66     2.88
//   m2   2.54 / 7.33   1.48 / 5.47     3.50
//   m4   2.65 / 7.74   1.59 / 5.95     3.73
//   m8   2.51 / 7.96   1.65 / 6.24     3.79
// The default is the best geomean (m4 is best on the X100 for L = 1).
using BaseConvert64 = vuint64m8_t;
// EltwiseExternalProductMod (rvv-hexl extension, binfhe blind rotation):
// placeholders until BM_Lanes_ExternalProduct has been measured on both
// clusters. The kernel keeps k digit strips and four sums live (k = 6 at
// STD128), so the useful LMUL is small.
using ExternalProduct32 = vuint32m2_t;     // RVV32 on uint32_t storage
using ExternalProduct32U64 = vuint32m2_t;  // RVV32 on uint64_t storage (<= m4)
using ExternalProduct64 = vuint64m2_t;

// FP64 eltwise path (src/util/fp64-mod.hpp), for 2^30 <= q < 2^50, 64-bit
// storage: EltwiseMultMod / FMAMod / MultAddMod on the FP64 FMA (4 elements per
// cycle on the K3) instead of the e64 integer multiply (1). On by default: on
// the blindloop protocol (K3, 2026-10-27) the FP64 eltwise + NTT paths over the
// integer ones never lose on the X100 (1.00-1.09x central) and gain up to 1.33x
// central on the A100 (SEAL BFV); 60-bit primes keep the integer path. BM_Lanes_*F64 (n = 4096, 49-bit q, 2026-10-22, median of 5;
// gain over the integer kernel at its default lanes, X100 / A100 / geomean;
// rvv-hexl results/eltwise-fp64-2026-10-22):
//                 m1                 m2                 m4                 m8
//   MultMod     0.56 / 0.64 / 0.60  1.07 / 0.92 / 1.00  1.64 / 1.23 / 1.42  1.60 / 1.39 / 1.49
//   FMAMod      0.41 / 0.63 / 0.51  0.76 / 0.91 / 0.83  1.17 / 1.19 / 1.18  1.16 / 1.31 / 1.23
//   MultAddMod  0.54 / 0.70 / 0.61  0.98 / 1.04 / 1.01  1.37 / 1.28 / 1.32  1.14 / 1.39 / 1.26
// The X100 peaks at m4, the A100 at m8; the defaults are the best geomean.
inline constexpr bool kEltwiseFp64 = true;
using MultModF64 = vfloat64m8_t;
using FMAModF64 = vfloat64m8_t;
using MultAddModF64 = vfloat64m4_t;

// ---- NTT -------------------------------------------------------------------
// How the small-span NTT stages (span t < VLMAX) run, one strategy for the RVV64
// kernels and one for both RVV32 kernels (see rvv::ntt above and
// src/ntt/ntt-rvv-small-span.hpp). Defaults: the best overall measured strategy,
// i.e. the highest geomean over both K3 clusters (X100 VLEN 256, A100 VLEN 1024),
// N = 1024 .. 32768, forward and inverse, against PerBlock at m1
// (rvv-hexl results/ntt-fused-2026-10-17, 2026-10-17):
//                     RVV64   RVV32/uint32_t   RVV32/uint64_t
//   PerBlock           1.00x       1.00x            1.00x
//   FusedSlides        1.97x       4.14x            3.98x
//   Transposed16       2.05x       2.84x            2.51x
//   Transposed8        1.95x       2.53x            2.19x
//   Transposed8x2      1.94x       2.63x            2.20x
//   Transposed4x4      1.60x       2.01x            1.84x
// (each at its best lane type, below). Per cluster the order can differ: on the
// X100 alone Transposed16 is best for RVV32 too (2.96x vs FusedSlides 2.49x), on
// the A100 FusedSlides is best for both (RVV32 6.90x). The library stays
// board-agnostic: change these two lines to pick another strategy.
using NttSmallSpan64 = ntt::Transposed16;
using NttSmallSpan32 = ntt::FusedSlides;

// The NTT kernel lane types that measured best with each strategy (same sweep):
// changing a strategy above picks its lanes here. Ntt32U64 is RVV32 on uint64_t
// storage (at most u32m4, the narrowing load). To force a lane type, replace the
// Ntt64 / Ntt32 / Ntt32U64 aliases below.
template <class Strategy>
struct NttBestLanes;
template <>
struct NttBestLanes<ntt::PerBlock> {  // per-block small spans run with vl = t: m1
	using Ntt64 = vuint64m1_t;
	using Ntt32 = vuint32m1_t;
	using Ntt32U64 = vuint32m1_t;
};
template <>
struct NttBestLanes<ntt::FusedSlides> {
	using Ntt64 = vuint64m2_t;
	using Ntt32 = vuint32m2_t;
	using Ntt32U64 = vuint32m2_t;
};
template <>
struct NttBestLanes<ntt::Transposed16> {
	using Ntt64 = vuint64m2_t;
	using Ntt32 = vuint32m2_t;
	using Ntt32U64 = vuint32m2_t;
};
template <>
struct NttBestLanes<ntt::Transposed8> {
	using Ntt64 = vuint64m2_t;
	using Ntt32 = vuint32m2_t;
	using Ntt32U64 = vuint32m1_t;
};
template <>
struct NttBestLanes<ntt::Transposed8x2> {
	using Ntt64 = vuint64m2_t;
	using Ntt32 = vuint32m2_t;
	using Ntt32U64 = vuint32m1_t;
};
template <>
struct NttBestLanes<ntt::Transposed4x4> {
	using Ntt64 = vuint64m1_t;
	using Ntt32 = vuint32m2_t;
	using Ntt32U64 = vuint32m1_t;
};

using Ntt64 = NttBestLanes<NttSmallSpan64>::Ntt64;        // RVV64 NTT
using Ntt32 = NttBestLanes<NttSmallSpan32>::Ntt32;        // RVV32 NTT on uint32_t storage
using Ntt32U64 = NttBestLanes<NttSmallSpan32>::Ntt32U64;  // RVV32 NTT on uint64_t storage

// FP64 NTT path (src/ntt/ntt-rvv-fp64.hpp), for 2^30 <= q < 2^50: the K3 runs
// FP64 FMA at 4 elements per cycle against 1 for the e64 integer multiply
// (vfmadd.vf / vmul.vx at m1, 16 chains, both clusters, 2026-10-20). On by
// default (protocol results with kEltwiseFp64 above). Strategy (any rvv::ntt strategy, as
// NttSmallSpan64) and lanes: the best overall geomean over both clusters and
// both directions, N = 1024 .. 16384, 49-bit q, against the integer default
// (Ntt64 + NttSmallSpan64), X100 / A100 / overall
// (rvv-hexl results/ntt-fp64-small-span-2026-10-26, 2026-10-26). Forward
// (input conversion fused into the first stage: +1-3% over a separate pass):
//                   m1                 m2                 m4                 m8
//   PerBlock      0.50 / 0.52 / 0.51  0.55 / 0.40 / 0.47  0.44 / 0.24 / 0.33  0.20 / 0.10 / 0.14
//   FusedSlides   0.60 / 1.46 / 0.93  0.87 / 1.67 / 1.20  0.95 / 1.67 / 1.26  0.57 / 0.98 / 0.75
//   Transposed16  0.75 / 1.13 / 0.92  0.96 / 1.17 / 1.06  1.05 / 1.10 / 1.07  0.87 / 0.79 / 0.83
//   Transposed8   0.75 / 1.07 / 0.89  0.98 / 1.05 / 1.02  1.00 / 0.91 / 0.95  0.74 / 0.55 / 0.63
//   Transposed8x2 0.78 / 1.10 / 0.93  1.05 / 1.09 / 1.07  1.07 / 0.93 / 1.00  0.78 / 0.55 / 0.66
//   Transposed4x4 0.71 / 0.95 / 0.82  0.89 / 0.87 / 0.88  0.81 / 0.67 / 0.74  0.52 / 0.34 / 0.42
// Inverse (the small spans read the integer operand):
//   PerBlock      0.52 / 0.49 / 0.51  0.53 / 0.38 / 0.45  0.43 / 0.25 / 0.33  0.19 / 0.09 / 0.13
//   FusedSlides   0.62 / 1.45 / 0.95  0.86 / 1.64 / 1.19  1.01 / 1.64 / 1.29  0.57 / 0.91 / 0.72
//   Transposed16  0.81 / 1.15 / 0.97  1.02 / 1.20 / 1.10  1.19 / 1.13 / 1.16  0.96 / 0.78 / 0.86
//   Transposed8   0.80 / 1.09 / 0.94  1.03 / 1.07 / 1.05  1.14 / 0.93 / 1.03  0.80 / 0.54 / 0.66
//   Transposed8x2 0.79 / 1.13 / 0.95  1.02 / 1.11 / 1.06  1.13 / 0.95 / 1.04  0.79 / 0.55 / 0.66
//   Transposed4x4 0.74 / 0.95 / 0.84  0.89 / 0.87 / 0.88  0.86 / 0.68 / 0.76  0.55 / 0.33 / 0.43
// Both directions: FusedSlides m4 0.98 / 1.66 / 1.28 (the default, best on the
// A100), Transposed16 m4 1.12 / 1.11 / 1.12 (best on the X100).
inline constexpr bool kNttFp64 = true;
using NttF64 = vfloat64m4_t;
using NttSmallSpanF64 = ntt::FusedSlides;

}  // namespace cfg

/// Lane type of a kernel whose SEW follows its storage word.
template <class Word, class L64, class L32>
using LaneFor = std::conditional_t<std::is_same_v<Word, uint64_t>, L64, L32>;

}  // namespace rvv
}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
