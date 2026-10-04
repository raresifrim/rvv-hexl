// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#include "hexl/eltwise/eltwise-mult-mod.hpp"

#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "util/cpu-features.hpp"
#include "util/not-implemented.hpp"

namespace intel {
namespace hexl {

void EltwiseMultMod(uint64_t* result, const uint64_t* operand1,
                    const uint64_t* operand2, uint64_t n, uint64_t modulus,
                    uint64_t input_mod_factor) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(operand2 != nullptr, "Require operand2 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  // The pre-shift Barrett estimate is at most one short only for moduli of up
  // to 61 bits (OpenFHE <= 60, SEAL <= 61); same bound as EltwiseFMAMod.
  HEXL_CHECK(modulus < (1ULL << 61), "Require modulus < (1ULL << 61)");
  HEXL_CHECK(input_mod_factor * modulus < (1ULL << 63),
             "Require input_mod_factor * modulus < (1ULL << 63)");
  HEXL_CHECK(
      input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4,
      "Require input_mod_factor = 1, 2, or 4")
  HEXL_CHECK_BOUNDS(operand1, n, input_mod_factor * modulus,
                    "operand1 exceeds bound " << (input_mod_factor * modulus))
  HEXL_CHECK_BOUNDS(operand2, n, input_mod_factor * modulus,
                    "operand2 exceeds bound " << (input_mod_factor * modulus))

#ifdef HEXL_HAS_RVV
  if (has_rvv) {
    if (modulus < kMaxModulusRVV32) {
      HEXL_VLOG(3, "Calling EltwiseMultModRVV32<uint64_t>");
      switch (input_mod_factor) {
        case 1: EltwiseMultModRVV32<uint64_t, 1>(result, operand1, operand2, n, modulus); break;
        case 2: EltwiseMultModRVV32<uint64_t, 2>(result, operand1, operand2, n, modulus); break;
        case 4: EltwiseMultModRVV32<uint64_t, 4>(result, operand1, operand2, n, modulus); break;
      }
    } else {
      HEXL_VLOG(3, "Calling EltwiseMultModRVV64");
      switch (input_mod_factor) {
        case 1: EltwiseMultModRVV64<1>(result, operand1, operand2, n, modulus); break;
        case 2: EltwiseMultModRVV64<2>(result, operand1, operand2, n, modulus); break;
        case 4: EltwiseMultModRVV64<4>(result, operand1, operand2, n, modulus); break;
      }
    }
    return;
  }
#endif

  HEXL_VLOG(3, "Calling EltwiseMultModNative<uint64_t>");
  switch (input_mod_factor) {
    case 1: EltwiseMultModNative<uint64_t, 1>(result, operand1, operand2, n, modulus); break;
    case 2: EltwiseMultModNative<uint64_t, 2>(result, operand1, operand2, n, modulus); break;
    case 4: EltwiseMultModNative<uint64_t, 4>(result, operand1, operand2, n, modulus); break;
  }
}

// ---- rvv-hexl extension: 32-bit storage ----------------------------------

void EltwiseMultMod(uint32_t* result, const uint32_t* operand1,
                    const uint32_t* operand2, uint64_t n, uint64_t modulus,
                    uint64_t input_mod_factor) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(operand2 != nullptr, "Require operand2 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(
      input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4,
      "Require input_mod_factor = 1, 2, or 4")
  HEXL_CHECK(input_mod_factor * modulus <= (1ULL << 32),
             "Require input_mod_factor * modulus <= 2**32");
  HEXL_CHECK_BOUNDS(operand1, n, input_mod_factor * modulus,
                    "operand1 exceeds bound " << (input_mod_factor * modulus))
  HEXL_CHECK_BOUNDS(operand2, n, input_mod_factor * modulus,
                    "operand2 exceeds bound " << (input_mod_factor * modulus))

#ifdef HEXL_HAS_RVV
  if (has_rvv && modulus < kMaxModulusRVV32) {
    HEXL_VLOG(3, "Calling EltwiseMultModRVV32<uint32_t>");
    switch (input_mod_factor) {
      case 1: EltwiseMultModRVV32<uint32_t, 1>(result, operand1, operand2, n, modulus); break;
      case 2: EltwiseMultModRVV32<uint32_t, 2>(result, operand1, operand2, n, modulus); break;
      case 4: EltwiseMultModRVV32<uint32_t, 4>(result, operand1, operand2, n, modulus); break;
    }
    return;
  }
#endif

  HEXL_VLOG(3, "Calling EltwiseMultModNative<uint32_t>");
  switch (input_mod_factor) {
    case 1: EltwiseMultModNative<uint32_t, 1>(result, operand1, operand2, n, modulus); break;
    case 2: EltwiseMultModNative<uint32_t, 2>(result, operand1, operand2, n, modulus); break;
    case 4: EltwiseMultModNative<uint32_t, 4>(result, operand1, operand2, n, modulus); break;
  }
}

// ---------------------------------------------------------------------------
// Native (scalar) kernel.  TODO(port)
// ---------------------------------------------------------------------------

template <typename Word, int InputModFactor>
void EltwiseMultModNative(Word* result, const Word* operand1, const Word* operand2, uint64_t n, uint64_t modulus) {
  // TODO(port): result[i] = (operand1[i] * operand2[i]) mod modulus, in [0, q).
  //   Inputs are NOT reduced: first bring each into [0, q) with
  //   ReduceMod<InputModFactor>(x, q, &twice_q) (number-theory.hpp), or fold
  //   the input range into the Barrett bound instead.
  //   Barrett with a precomputed floor(2^(2k)/q) (k = bit length of the
  //   largest input) avoids the 128-bit division of a naive (a*b) % q, which
  //   on RV64 is a libgcc __umodti3 call: ~100x slower than mul+mulhu.
  //   Word = uint32_t: the product fits in uint64_t, so a 64-bit Barrett (or
  //   even a single 64-bit % for a first version) is enough.
  //   Upstream's EltwiseMultModNative is the reference algorithm.
  HEXL_NOT_IMPLEMENTED();
}

}  // namespace hexl
}  // namespace intel
