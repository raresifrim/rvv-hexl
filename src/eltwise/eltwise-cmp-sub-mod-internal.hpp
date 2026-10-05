// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Kernels behind EltwiseCmpSubMod.
// Kernels are templates on the storage word:
//   Word = uint64_t  OpenFHE NATIVE_SIZE=64 (upstream HEXL API)
//   Word = uint32_t  OpenFHE NATIVE_SIZE=32 (rvv-hexl extension)
// Write one body and specialise where it pays with
//   if constexpr (std::is_same_v<Word, uint32_t>) { ... } else { ... }

#pragma once

#include <stdint.h>

#include "hexl/util/util.hpp"
#include "util/cpu-features.hpp"
#include "util/rvv-config.hpp"  // lane types (empty without RVV)

namespace intel {
namespace hexl {

/// @brief Native: r = operand1[i] mod modulus;
///   result[i] = cmp(operand1[i], bound) ? (r - diff) mod modulus : r
///   (the comparison uses the ORIGINAL value, before reduction)
template <typename Word>
void EltwiseCmpSubModNative(Word* result, const Word* operand1, uint64_t n, uint64_t modulus, CMPINT cmp, uint64_t bound, uint64_t diff);

#ifdef HEXL_HAS_RVV
/// @brief RVV variant, same contract. Word = uint32_t only reaches it with
/// modulus < 2^30 (where a modulus applies).
template <typename Word, class V = rvv::LaneFor<Word, rvv::cfg::CmpSubMod64, rvv::cfg::CmpSubMod32>>
void EltwiseCmpSubModRVV(Word* result, const Word* operand1, uint64_t n, uint64_t modulus, CMPINT cmp, uint64_t bound, uint64_t diff);
#endif

}  // namespace hexl
}  // namespace intel
