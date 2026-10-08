// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Kernels behind EltwiseFMAMod.
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

/// @brief Native: result[i] = (arg1[i] * arg2 + arg3[i]) mod modulus
/// (arg3 == nullptr: no addition). All inputs in [0, InputModFactor * q),
/// output in [0, q). InputModFactor is 1, 2, 4 or 8.
template <typename Word, int InputModFactor>
	void EltwiseFMAModNative(Word* result, const Word* arg1, uint64_t arg2, const Word* arg3, uint64_t n, uint64_t modulus);

#ifdef HEXL_HAS_RVV
/// @brief RVV, 32-bit lanes: modulus < kMaxModulusRVV32 and every input fits
/// in 32 bits. Word = uint64_t or uint32_t storage (rvv::Load32 / Store32).
template <typename Word, int InputModFactor, class V = rvv::cfg::FMAMod32>
	void EltwiseFMAModRVV32(Word* result, const Word* arg1, uint64_t arg2, const Word* arg3, uint64_t n, uint64_t modulus);

/// @brief RVV, 64-bit lanes, 64-bit storage only.
template <int InputModFactor, class V = rvv::cfg::FMAMod64>
	void EltwiseFMAModRVV64(uint64_t* result, const uint64_t* arg1, uint64_t arg2,
			const uint64_t* arg3, uint64_t n, uint64_t modulus);
#endif

}  // namespace hexl
}  // namespace intel
