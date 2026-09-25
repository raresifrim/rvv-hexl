// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hexl/util/defines.hpp"

#if defined(HEXL_USE_CLANG)
#include "hexl/util/clang.hpp"
#elif defined(HEXL_USE_GNU)
#include "hexl/util/gcc.hpp"
#endif
