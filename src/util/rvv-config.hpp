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
using Ntt64 = vuint64m4_t;
using Ntt32 = vuint32m4_t;      // <= m4

}  // namespace cfg

/// Lane type of a kernel whose SEW follows its storage word.
template <class Word, class L64, class L32>
using LaneFor = std::conditional_t<std::is_same_v<Word, uint64_t>, L64, L32>;

}  // namespace rvv
}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
