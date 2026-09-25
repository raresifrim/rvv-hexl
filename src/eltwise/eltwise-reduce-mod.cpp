// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#include "hexl/eltwise/eltwise-reduce-mod.hpp"

#include "eltwise/eltwise-reduce-mod-internal.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "util/cpu-features.hpp"
#include "util/not-implemented.hpp"

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
    HEXL_VLOG(3, "Calling EltwiseReduceModRVV");
    EltwiseReduceModRVV(result, operand, n, modulus, input_mod_factor,
                        output_mod_factor);
    return;
  }
#endif

  HEXL_VLOG(3, "Calling EltwiseReduceModNative");
  EltwiseReduceModNative(result, operand, n, modulus, input_mod_factor,
                         output_mod_factor);
}

// ---------------------------------------------------------------------------
// Native (scalar) kernel.  TODO(port)
// ---------------------------------------------------------------------------

void EltwiseReduceModNative(uint64_t* result, const uint64_t* operand,
                            uint64_t n, uint64_t modulus,
                            uint64_t input_mod_factor,
                            uint64_t output_mod_factor) {
  // TODO(port): three cases.
  //   input_mod_factor == modulus: arbitrary 64-bit x -> Barrett with
  //     q_barr = MultiplyFactor(1, 64, q).BarrettFactor();
  //     BarrettReduce64<1> (output [0,q)) or BarrettReduce64<2> ([0,2q)).
  //   input_mod_factor == 2: x in [0,2q) -> ReduceMod<2>.
  //   input_mod_factor == 4: x in [0,4q) -> ReduceMod<4> (to [0,q)) or one
  //     conditional subtract of 2q (to [0,2q)).
  //   Any output congruent to x mod q inside [0, output_mod_factor * q) is
  //   accepted by the tests (and by OpenFHE).
  HEXL_NOT_IMPLEMENTED();
}

}  // namespace hexl
}  // namespace intel
