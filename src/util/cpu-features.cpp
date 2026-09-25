// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#include "util/cpu-features.hpp"

#if defined(__linux__) && defined(__riscv)
#include <sys/auxv.h>
#endif

#ifdef HEXL_HAS_RVV
#include <riscv_vector.h>
#endif

#include "hexl/rvv/port-info.hpp"

#ifndef HEXL_BUILD_FLAGS
#define HEXL_BUILD_FLAGS "unknown"
#endif

namespace intel {
namespace hexl {

bool DetectRVV() {
#if !defined(HEXL_HAS_RVV)
  return false;  // RVV kernels were not compiled in
#elif defined(__linux__)
  // Single-letter extensions are reported as bit ('X' - 'A') of AT_HWCAP.
  // (For multi-letter extensions such as Zvbb use the riscv_hwprobe syscall.)
  const unsigned long hwcap = getauxval(AT_HWCAP);
  return (hwcap & (1UL << ('V' - 'A'))) != 0;
#else
  // Bare metal / proxy kernel (spike + pk): no auxv. Trust the -march the
  // library was compiled with.
  return true;
#endif
}

size_t CurrentVLenBits() {
#ifdef HEXL_HAS_RVV
  if (!DetectRVV()) return 0;
  // VLMAX for SEW=8, LMUL=1 is VLEN/8 = vlenb.
  return __riscv_vsetvlmax_e8m1() * 8;
#else
  return 0;
#endif
}

namespace rvv {

PortInfo GetPortInfo() {
  PortInfo info;
#ifdef HEXL_HAS_RVV
  info.compiled_with_rvv = true;
#else
  info.compiled_with_rvv = false;
#endif
  info.rvv_available = DetectRVV();
  info.rvv_enabled = has_rvv;
  info.vlen_bits = CurrentVLenBits();
  info.build_flags = HEXL_BUILD_FLAGS;
#if defined(__clang__)
  info.compiler = "clang " __clang_version__;
#elif defined(__GNUC__)
  info.compiler = "gcc " __VERSION__;
#else
  info.compiler = "unknown";
#endif
  return info;
}

}  // namespace rvv
}  // namespace hexl
}  // namespace intel
