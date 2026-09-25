// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Upstream wires easylogging++ in debug builds. The port keeps the same macro
// names but prints to stderr, gated by the HEXL_VLOG environment variable
// (HEXL_VLOG=3 prints every "Calling EltwiseXxxRVV" style message at level <= 3).
// Release builds compile all of it away.

#pragma once

#include <algorithm>
#include <vector>

#include "hexl/util/defines.hpp"

#ifdef HEXL_DEBUG
#include <cstdlib>
#include <iostream>

namespace intel {
namespace hexl {
namespace internal {
inline int VlogLevel() {
  static const int level = [] {
    const char* s = std::getenv("HEXL_VLOG");
    return s ? std::atoi(s) : 0;
  }();
  return level;
}
}  // namespace internal
}  // namespace hexl
}  // namespace intel

#define HEXL_VLOG(N, rest)                                   \
  do {                                                       \
    if (::intel::hexl::internal::VlogLevel() >= (N)) {       \
      std::cerr << "[hexl] " << rest << std::endl;           \
    }                                                        \
  } while (0);

#else

#define HEXL_VLOG(N, rest) \
  {}

#endif  // HEXL_DEBUG

// Upstream test/bench mains call this; it is a no-op here.
#define START_EASYLOGGINGPP(X, Y) \
  {}
