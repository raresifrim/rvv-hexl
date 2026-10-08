// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Kernels behind EltwiseReduceMod.
// Kernels are templates on the storage word:
//   Word = uint64_t  OpenFHE NATIVE_SIZE=64 (upstream HEXL API)
//   Word = uint32_t  OpenFHE NATIVE_SIZE=32 (rvv-hexl extension)
// Write one body and specialise where it pays with
//   if constexpr (std::is_same_v<Word, uint32_t>) { ... } else { ... }

#pragma once

#include <stdint.h>

#include "util/cpu-features.hpp"
#include "util/rvv-config.hpp"  // lane types (empty without RVV)

namespace intel {
namespace hexl {

/// @brief Native: result[i] = operand[i] reduced into [0, output_mod_factor*q).
/// input_mod_factor is modulus (any Word value), 2 or 4; output_mod_factor is
/// 1 or 2; never equal here (the public entry point handles the "copy" case).
template <typename Word>
	void EltwiseReduceModNative(Word* result, const Word* operand, uint64_t n, uint64_t modulus, uint64_t input_mod_factor, uint64_t output_mod_factor);

#ifdef HEXL_HAS_RVV
/// @brief RVV variant, same contract. Word = uint32_t only reaches it with
/// modulus < 2^30 (where a modulus applies).
template <typename Word, class V = rvv::LaneFor<Word, rvv::cfg::ReduceMod64, rvv::cfg::ReduceMod32>>
	void EltwiseReduceModRVV(Word* result, const Word* operand, uint64_t n, uint64_t modulus, uint64_t input_mod_factor, uint64_t output_mod_factor);
#endif

}  // namespace hexl
}  // namespace intel
