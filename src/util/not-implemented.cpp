// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#include "util/not-implemented.hpp"

#include <stdexcept>
#include <string>

namespace intel {
namespace hexl {
namespace internal {

void ThrowNotImplemented(const char* function, const char* file, int line) {
  throw std::logic_error(std::string("[rvv-hexl TODO] ") + function + " (" +
                         file + ":" + std::to_string(line) + ")");
}

}  // namespace internal
}  // namespace hexl
}  // namespace intel
