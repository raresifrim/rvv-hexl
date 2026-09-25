// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#include "hexl/eltwise/eltwise-add-mod.hpp"

#include "eltwise/eltwise-add-mod-internal.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/util/check.hpp"
#include "util/cpu-features.hpp"
#include "util/not-implemented.hpp"

namespace intel {
namespace hexl {

// ---------------------------------------------------------------------------
// Public entry points: argument checks + dispatch. Boilerplate, done.
// ---------------------------------------------------------------------------

void EltwiseAddMod(uint64_t* result, const uint64_t* operand1,
                   const uint64_t* operand2, uint64_t n, uint64_t modulus) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(operand2 != nullptr, "Require operand2 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(modulus < (1ULL << 63), "Require modulus < 2**63");
  HEXL_CHECK_BOUNDS(operand1, n, modulus,
                    "operand1 value exceeds bound " << modulus);
  HEXL_CHECK_BOUNDS(operand2, n, modulus,
                    "operand2 value exceeds bound " << modulus);

#ifdef HEXL_HAS_RVV
  if (has_rvv) {
    HEXL_VLOG(3, "Calling EltwiseAddModRVV");
    EltwiseAddModRVV(result, operand1, operand2, n, modulus);
    return;
  }
#endif

  HEXL_VLOG(3, "Calling EltwiseAddModNative");
  EltwiseAddModNative(result, operand1, operand2, n, modulus);
}

void EltwiseAddMod(uint64_t* result, const uint64_t* operand1,
                   uint64_t operand2, uint64_t n, uint64_t modulus) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(modulus < (1ULL << 63), "Require modulus < 2**63");
  HEXL_CHECK_BOUNDS(operand1, n, modulus,
                    "operand1 value exceeds bound " << modulus);
  HEXL_CHECK(operand2 < modulus, "Require operand2 < modulus");

#ifdef HEXL_HAS_RVV
  if (has_rvv) {
    HEXL_VLOG(3, "Calling EltwiseAddModRVV");
    EltwiseAddModRVV(result, operand1, operand2, n, modulus);
    return;
  }
#endif

  HEXL_VLOG(3, "Calling EltwiseAddModNative");
  EltwiseAddModNative(result, operand1, operand2, n, modulus);
}

// ---------------------------------------------------------------------------
// Native (scalar) kernels.  TODO(port)
// ---------------------------------------------------------------------------

void EltwiseAddModNative(uint64_t* result, const uint64_t* operand1,
                          const uint64_t* operand2, uint64_t n,
                          uint64_t modulus) {
  // TODO(port): for i in [0, n): result[i] = (operand1[i] + operand2[i]) mod modulus
  //   Inputs are < modulus < 2^63, so the sum cannot overflow: one add and one conditional subtract per element.
  //   result may alias operand1 (OpenFHE calls it in place).
  HEXL_NOT_IMPLEMENTED();
}

void EltwiseAddModNative(uint64_t* result, const uint64_t* operand1,
                          uint64_t operand2, uint64_t n, uint64_t modulus) {
  // TODO(port): for i in [0, n): result[i] = (operand1[i] + operand2) mod modulus
  HEXL_NOT_IMPLEMENTED();
}

}  // namespace hexl
}  // namespace intel
