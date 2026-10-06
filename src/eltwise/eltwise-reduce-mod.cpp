// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#include "hexl/eltwise/eltwise-reduce-mod.hpp"
#include <algorithm>
#include "eltwise/eltwise-reduce-mod-internal.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "util/cpu-features.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

void EltwiseReduceMod(uint64_t* result, const uint64_t* operand, uint64_t n,
                      uint64_t modulus, uint64_t input_mod_factor,
                      uint64_t output_mod_factor) {
  HEXL_CHECK(operand != nullptr, "Require operand != nullptr");
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(input_mod_factor == modulus || input_mod_factor == 2 ||
                 input_mod_factor == 4,
             "input_mod_factor must be modulus or 2 or 4" << input_mod_factor);
  HEXL_CHECK(output_mod_factor == 1 || output_mod_factor == 2,
             "output_mod_factor must be 1 or 2 " << output_mod_factor);

  // Same short-cut as upstream: nothing to reduce, just copy.
  if (input_mod_factor == output_mod_factor) {
    if (operand != result) {
      for (size_t i = 0; i < n; ++i) {
        result[i] = operand[i];
      }
    }
    return;
  }

#ifdef HEXL_HAS_RVV
  if (has_rvv) {
    HEXL_VLOG(3, "Calling EltwiseReduceModRVV<uint64_t>");
    EltwiseReduceModRVV<uint64_t>(result, operand, n, modulus, input_mod_factor, output_mod_factor);
    return;
  }
#endif

  HEXL_VLOG(3, "Calling EltwiseReduceModNative<uint64_t>");
  EltwiseReduceModNative<uint64_t>(result, operand, n, modulus, input_mod_factor, output_mod_factor);
}

// ---- rvv-hexl extension: 32-bit storage ----------------------------------

void EltwiseReduceMod(uint32_t* result, const uint32_t* operand, uint64_t n,
                      uint64_t modulus, uint64_t input_mod_factor,
                      uint64_t output_mod_factor) {
  HEXL_CHECK(operand != nullptr, "Require operand != nullptr");
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(input_mod_factor == modulus || input_mod_factor == 2 ||
                 input_mod_factor == 4,
             "input_mod_factor must be modulus or 2 or 4" << input_mod_factor);
  HEXL_CHECK(output_mod_factor == 1 || output_mod_factor == 2,
             "output_mod_factor must be 1 or 2 " << output_mod_factor);

  // Same short-cut as upstream: nothing to reduce, just copy.
  if (input_mod_factor == output_mod_factor) {
    if (operand != result) {
      for (size_t i = 0; i < n; ++i) {
        result[i] = operand[i];
      }
    }
    return;
  }
  HEXL_CHECK(modulus <= (1ULL << 32), "Require modulus <= 2**32");
  HEXL_CHECK(input_mod_factor == modulus || input_mod_factor * modulus <= (1ULL << 32),
             "Require input_mod_factor * modulus <= 2**32");

#ifdef HEXL_HAS_RVV
  if (has_rvv && modulus < kMaxModulusRVV32) {
    HEXL_VLOG(3, "Calling EltwiseReduceModRVV<uint32_t>");
    EltwiseReduceModRVV<uint32_t>(result, operand, n, modulus, input_mod_factor, output_mod_factor);
    return;
  }
#endif

  HEXL_VLOG(3, "Calling EltwiseReduceModNative<uint32_t>");
  EltwiseReduceModNative<uint32_t>(result, operand, n, modulus, input_mod_factor, output_mod_factor);
}

// ---------------------------------------------------------------------------
// Native (scalar) kernel. 
// ---------------------------------------------------------------------------

template <typename Word>
void EltwiseReduceModNative(Word* result, const Word* operand, uint64_t n, uint64_t modulus, uint64_t input_mod_factor, uint64_t output_mod_factor) {
    
  if (input_mod_factor == modulus && output_mod_factor == 1){
      const uint64_t q_barr = MultiplyFactor(1, 64, modulus).BarrettFactor();
      #pragma GCC novector
      for(uint64_t i=0; i<n; ++i)
	  result[i] = static_cast<Word>(BarrettReduce64<1>(operand[i], modulus, q_barr)); 
  }
  else if (input_mod_factor == modulus && output_mod_factor == 2){
      const uint64_t q_barr = MultiplyFactor(1, 64, modulus).BarrettFactor();
      #pragma GCC novector
      for(uint64_t i=0; i<n; ++i)
	  result[i] = static_cast<Word>(BarrettReduce64<2>(operand[i], modulus, q_barr)); 
  }
  else if (input_mod_factor == 4 && output_mod_factor == 1) { 
      #pragma GCC novector
      for(uint64_t i=0; i<n; ++i){
            uint64_t a = std::min(static_cast<uint64_t>(operand[i]), static_cast<uint64_t>(operand[i]) - 2*modulus);
	    result[i] = static_cast<Word>(std::min(a, a - modulus));
      }
  }
  else if (input_mod_factor == 4 && output_mod_factor == 2) { 
      #pragma GCC novector
      for(uint64_t i=0; i<n; ++i)
	    result[i] = static_cast<Word>(std::min(static_cast<uint64_t>(operand[i]), static_cast<uint64_t>(operand[i]) - 2*modulus)); 
  }
  else if (input_mod_factor == 2 && output_mod_factor == 1) { 
      #pragma GCC novector
      for(uint64_t i=0; i<n; ++i)
	    result[i] = static_cast<Word>(std::min(static_cast<uint64_t>(operand[i]), static_cast<uint64_t>(operand[i]) - modulus)); 
  } 
}

}  // namespace hexl
}  // namespace intel
