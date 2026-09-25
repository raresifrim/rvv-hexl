// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// RISC-V counterpart of upstream's util/cpu-features.hpp (which asks
// google/cpu_features for AVX512DQ/IFMA/VBMI2).
//
// Two independent gates decide whether an RVV kernel runs:
//
//   compile time  HEXL_HAS_RVV   the TU was compiled with V enabled
//                               (-march=rva23u64 / rv64gcv). With ISA=scalar
//                               (-march=rv64gc...) every *-rvv.cpp compiles to
//                               nothing and the library is pure scalar: that
//                               is the "base RISC-V, no RVV" configuration.
//   run time      has_rvv       the kernel reports V (AT_HWCAP) and the user
//                               did not set HEXL_DISABLE_RVV. Lets ONE binary
//                               measure both paths:
//                                 ./bench-hexl                    -> RVV
//                                 HEXL_DISABLE_RVV=1 ./bench-hexl -> native
//                               (mirrors upstream's HEXL_DISABLE_AVX512DQ).
//
// Every public entry point dispatches with the same pattern:
//
//   #ifdef HEXL_HAS_RVV
//     if (has_rvv) { XxxRVV(...); return; }
//   #endif
//     XxxNative(...);

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <cstdlib>

#if defined(__riscv) && defined(__riscv_vector) && defined(__riscv_v_intrinsic)
#define HEXL_HAS_RVV 1
#endif

namespace intel {
namespace hexl {

/// @brief true iff the CPU/OS supports V (always false without HEXL_HAS_RVV)
bool DetectRVV();

/// @brief VLEN in bits of the CURRENT hart, 0 without RVV.
/// @warning Never cache this in a table that outlives the call. On the K3 the
/// X100 (VLEN=256) and A100 (VLEN=1024) clusters differ; kernels must stay
/// VLEN-agnostic (vsetvl in every loop) so one binary is correct on both.
size_t CurrentVLenBits();

/// @brief Moduli below this take the RVV e32 (32-bit lane) kernels. Same
/// bound as NTT::s_max_fwd_32_modulus: 2 bits of headroom for lazy [0, 4q).
static constexpr uint64_t kMaxModulusRVV32 = 1ULL << 30;

static const bool disable_rvv = (std::getenv("HEXL_DISABLE_RVV") != nullptr);
static const bool has_rvv = !disable_rvv && DetectRVV();

}  // namespace hexl
}  // namespace intel
