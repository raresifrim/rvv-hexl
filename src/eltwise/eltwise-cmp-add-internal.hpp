// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Kernels behind EltwiseCmpAdd.
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

/// @brief Native: result[i] = cmp(operand1[i], bound) ? operand1[i] + diff : operand1[i]
///   (plain add, wrapping modulo 2^(8*sizeof(Word)); no reduction)
template <typename Word>
	void EltwiseCmpAddNative(Word* result, const Word* operand1, uint64_t n, CMPINT cmp, uint64_t bound, uint64_t diff);

#ifdef HEXL_HAS_RVV
/// @brief RVV variant, same contract. Word = uint32_t only reaches it with
/// modulus < 2^30 (where a modulus applies).
template <typename Word, class V = rvv::LaneFor<Word, rvv::cfg::CmpAdd64, rvv::cfg::CmpAdd32>>
	void EltwiseCmpAddRVV(Word* result, const Word* operand1, uint64_t n, CMPINT cmp, uint64_t bound, uint64_t diff);
#endif

}  // namespace hexl
}  // namespace intel
