// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#include "hexl/util/aligned-allocator.hpp"

namespace intel {
namespace hexl {

// Default strategy used by every AlignedAllocator constructed without one.
// (Upstream defines it in ntt-internal.cpp; moved here because it is not NTT
// specific.)
AllocatorStrategyPtr mallocStrategy = AllocatorStrategyPtr(new MallocStrategy);

}  // namespace hexl
}  // namespace intel
