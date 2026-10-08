// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Kernels behind EltwiseSubMod. Native = portable scalar C++ (also the "no RVV"
// reference path). RVV = RVV 1.0 kernels in eltwise-sub-mod-rvv.cpp.
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

/// @brief Native vector-vector: result[i] = (operand1[i] - operand2[i]) mod modulus
template <typename Word>
	void EltwiseSubModNative(Word* result, const Word* operand1, const Word* operand2,
			uint64_t n, uint64_t modulus);

/// @brief Native vector-scalar: result[i] = (operand1[i] - operand2) mod modulus
template <typename Word>
	void EltwiseSubModNative(Word* result, const Word* operand1, uint64_t operand2,
			uint64_t n, uint64_t modulus);

#ifdef HEXL_HAS_RVV
/// @brief RVV vector-vector. Word = uint64_t: any modulus < 2^63.
/// Word = uint32_t: modulus < 2^30 (guaranteed by the dispatcher).
template <typename Word, class V = rvv::LaneFor<Word, rvv::cfg::SubMod64, rvv::cfg::SubMod32>>
	void EltwiseSubModRVV(Word* result, const Word* operand1, const Word* operand2,
			uint64_t n, uint64_t modulus);

/// @brief RVV vector-scalar, same constraints.
template <typename Word, class V = rvv::LaneFor<Word, rvv::cfg::SubMod64, rvv::cfg::SubMod32>>
	void EltwiseSubModRVV(Word* result, const Word* operand1, uint64_t operand2,
			uint64_t n, uint64_t modulus);
#endif

}  // namespace hexl
}  // namespace intel
