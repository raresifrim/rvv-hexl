// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#include "hexl/eltwise/eltwise-cmp-sub-mod.hpp"

#include "eltwise/eltwise-cmp-sub-mod-internal.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "util/cpu-features.hpp"
#include "util/not-implemented.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

void EltwiseCmpSubMod(uint64_t* result, const uint64_t* operand1,
                        uint64_t n, uint64_t modulus, CMPINT cmp,
                        uint64_t bound, uint64_t diff) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(diff != 0, "Require diff != 0");
  HEXL_CHECK(diff < modulus, "Diff " << diff << " >= modulus " << modulus);

#ifdef HEXL_HAS_RVV
  if (has_rvv) {
    HEXL_VLOG(3, "Calling EltwiseCmpSubModRVV");
    EltwiseCmpSubModRVV(result, operand1, n, modulus, cmp, bound, diff);
    return;
  }
#endif

  HEXL_VLOG(3, "Calling EltwiseCmpSubModNative");
  EltwiseCmpSubModNative(result, operand1, n, modulus, cmp, bound, diff);
}

// ---------------------------------------------------------------------------
// Native (scalar) kernel.  TODO(port)
// ---------------------------------------------------------------------------

void EltwiseCmpSubModNative(uint64_t* result, const uint64_t* operand1,
                        uint64_t n, uint64_t modulus, CMPINT cmp,
                        uint64_t bound, uint64_t diff) {
  // TODO(port): for each i:
  //     bool c = Compare(cmp, operand1[i], bound);   // on the unreduced value
  //     uint64_t r = operand1[i] % modulus;          // any 64-bit input!
  //     result[i] = c ? SubUIntMod(r, diff, modulus) : r;
  //   "% modulus" on RV64 is a remu (fine but slow, ~20-40 cycles): Barrett
  //   with MultiplyFactor(1, 64, q) is the fast alternative.
  HEXL_NOT_IMPLEMENTED();
}

}  // namespace hexl
}  // namespace intel
