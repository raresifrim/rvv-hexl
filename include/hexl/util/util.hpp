// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace intel {
namespace hexl {

#undef TRUE   // MSVC defines TRUE
#undef FALSE  // MSVC defines FALSE

/// @enum CMPINT
/// @brief Comparison predicate used by EltwiseCmpAdd / EltwiseCmpSubMod.
/// Values match the AVX-512 _MM_CMPINT_ENUM encoding upstream; keep them as-is
/// for ABI/source compatibility even though RISC-V does not need the encoding.
enum class CMPINT {
  EQ = 0,     ///< Equal
  LT = 1,     ///< Less than
  LE = 2,     ///< Less than or equal
  FALSE = 3,  ///< False
  NE = 4,     ///< Not equal
  NLT = 5,    ///< Not less than
  NLE = 6,    ///< Not less than or equal
  TRUE = 7    ///< True
};

/// @brief Returns the logical negation of a binary comparison
inline CMPINT Not(CMPINT cmp) {
  switch (cmp) {
    case CMPINT::EQ:
      return CMPINT::NE;
    case CMPINT::LT:
      return CMPINT::NLT;
    case CMPINT::LE:
      return CMPINT::NLE;
    case CMPINT::FALSE:
      return CMPINT::TRUE;
    case CMPINT::NE:
      return CMPINT::EQ;
    case CMPINT::NLT:
      return CMPINT::LT;
    case CMPINT::NLE:
      return CMPINT::LE;
    case CMPINT::TRUE:
      return CMPINT::FALSE;
    default:
      return CMPINT::FALSE;
  }
}

}  // namespace hexl
}  // namespace intel
