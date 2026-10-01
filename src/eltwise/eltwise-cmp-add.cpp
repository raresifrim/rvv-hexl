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
// Native (scalar) kernel.
// ---------------------------------------------------------------------------

// Per-element work, shared by every comparison: only the condition differs.
// Select the VALUE to add (diff or 0) rather than one of two results, the same
// form as CmpSubModOne. The add wraps in Word (the cast), as the API specifies.
template <typename Word>
static inline Word CmpAddOne(uint64_t x, bool hit, uint64_t diff) {
  return (Word)(x + (hit ? diff : 0U)); //this style forces a czero instruction instead of using a branch
}

template <typename Word>
void EltwiseCmpAddNative(Word* result, const Word* operand1, uint64_t n, CMPINT cmp,
                         uint64_t bound, uint64_t diff) {
  switch (cmp) {
    case CMPINT::EQ:
      for (uint64_t i = 0; i < n; ++i) {
        result[i] = CmpAddOne<Word>(operand1[i], operand1[i] == bound, diff);
      }
      break;
    case CMPINT::LT:
      for (uint64_t i = 0; i < n; ++i) {
        result[i] = CmpAddOne<Word>(operand1[i], operand1[i] < bound, diff);
      }
      break;
    case CMPINT::LE:
      for (uint64_t i = 0; i < n; ++i) {
        result[i] = CmpAddOne<Word>(operand1[i], operand1[i] <= bound, diff);
      }
      break;
    case CMPINT::NE:
      for (uint64_t i = 0; i < n; ++i) {
        result[i] = CmpAddOne<Word>(operand1[i], operand1[i] != bound, diff);
      }
      break;
    case CMPINT::NLT:
      for (uint64_t i = 0; i < n; ++i) {
        result[i] = CmpAddOne<Word>(operand1[i], operand1[i] >= bound, diff);
      }
      break;
    case CMPINT::NLE:
      for (uint64_t i = 0; i < n; ++i) {
        result[i] = CmpAddOne<Word>(operand1[i], operand1[i] > bound, diff);
      }
      break;
    case CMPINT::TRUE:
      for (uint64_t i = 0; i < n; ++i) {
        result[i] = CmpAddOne<Word>(operand1[i], true, diff);
      }
      break;
    case CMPINT::FALSE:
      // no element changes: copy only when not in place (a pointer compare,
      // one instruction; in place the call does no memory traffic at all)
      if (result != operand1) {
        for (uint64_t i = 0; i < n; ++i) {
          result[i] = operand1[i];
        }
      }
      break;
    default:
      // not a valid CMPINT; the scalar Compare() treats unknown values as TRUE
      for (uint64_t i = 0; i < n; ++i) {
        result[i] = CmpAddOne<Word>(operand1[i], true, diff);
      }
      break;
  }
}

}  // namespace hexl
}  // namespace intel
