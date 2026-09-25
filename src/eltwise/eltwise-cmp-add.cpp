// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#include "hexl/eltwise/eltwise-cmp-add.hpp"

#include "eltwise/eltwise-cmp-add-internal.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "util/cpu-features.hpp"
#include "util/not-implemented.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

void EltwiseCmpAdd(uint64_t* result, const uint64_t* operand1, uint64_t n,
                   CMPINT cmp, uint64_t bound, uint64_t diff) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(diff != 0, "Require diff != 0");

#ifdef HEXL_HAS_RVV
  if (has_rvv) {
    HEXL_VLOG(3, "Calling EltwiseCmpAddRVV<uint64_t>");
    EltwiseCmpAddRVV<uint64_t>(result, operand1, n, cmp, bound, diff);
    return;
  }
#endif

  HEXL_VLOG(3, "Calling EltwiseCmpAddNative<uint64_t>");
  EltwiseCmpAddNative<uint64_t>(result, operand1, n, cmp, bound, diff);
}

// ---- rvv-hexl extension: 32-bit storage ----------------------------------

void EltwiseCmpAdd(uint32_t* result, const uint32_t* operand1, uint64_t n,
                   CMPINT cmp, uint64_t bound, uint64_t diff) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(diff != 0, "Require diff != 0");
  HEXL_CHECK(bound < (1ULL << 32), "Require bound < 2**32");
  HEXL_CHECK(diff < (1ULL << 32), "Require diff < 2**32");

#ifdef HEXL_HAS_RVV
  if (has_rvv) {
    HEXL_VLOG(3, "Calling EltwiseCmpAddRVV<uint32_t>");
    EltwiseCmpAddRVV<uint32_t>(result, operand1, n, cmp, bound, diff);
    return;
  }
#endif

  HEXL_VLOG(3, "Calling EltwiseCmpAddNative<uint32_t>");
  EltwiseCmpAddNative<uint32_t>(result, operand1, n, cmp, bound, diff);
}

// ---------------------------------------------------------------------------
// Native (scalar) kernel.  TODO(port)
// ---------------------------------------------------------------------------

template <typename Word>
void EltwiseCmpAddNative(Word* result, const Word* operand1, uint64_t n, CMPINT cmp, uint64_t bound, uint64_t diff) {
  // TODO(port): result[i] = cmp(operand1[i], bound) ? operand1[i] + diff
  //                                             : operand1[i]
  //   Handle CMPINT::TRUE / FALSE up front (copy or add everywhere). Keep the
  //   switch on cmp OUTSIDE the element loop (one loop per predicate, or a
  //   template on the predicate) so the loop body stays branch-free. The add
  //   wraps in Word (cast the result back to Word).
  HEXL_NOT_IMPLEMENTED();
}

}  // namespace hexl
}  // namespace intel
