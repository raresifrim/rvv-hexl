// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// rvv-hexl extension (no upstream equivalent). Reports how the library was
// built and which code path it will take at runtime, so every benchmark result
// can record it. Same motivation as the "manipulation check" in the IPCEI
// build scripts: a run that silently fell back to the scalar path must be
// detectable from its own output, not assumed.

#pragma once

#include <stddef.h>

namespace intel {
namespace hexl {
namespace rvv {

struct PortInfo {
  /// Library objects were compiled with the V extension (RVV kernels exist)
  bool compiled_with_rvv;
  /// The CPU/kernel reports V at runtime (Linux: AT_HWCAP)
  bool rvv_available;
  /// RVV kernels will actually be used: compiled && available && the
  /// HEXL_DISABLE_RVV environment variable is not set
  bool rvv_enabled;
  /// VLEN in bits of the hart this was queried on (0 if no RVV). On the
  /// SpaceMiT K3 this is 256 on the X100 cluster and 1024 on the A100 cluster.
  size_t vlen_bits;
  /// Compiler flags the library was built with (e.g. "-O3 -march=rva23u64")
  const char* build_flags;
  /// Compiler identification string
  const char* compiler;
};

/// @brief Returns build/runtime information about this rvv-hexl library.
PortInfo GetPortInfo();

}  // namespace rvv
}  // namespace hexl
}  // namespace intel
