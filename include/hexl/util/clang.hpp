// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hexl/util/gcc.hpp"  // the 128-bit helpers are identical for Clang

#if defined(HEXL_USE_CLANG)
#define HEXL_LOOP_UNROLL_4 _Pragma("clang loop unroll_count(4)")
#define HEXL_LOOP_UNROLL_8 _Pragma("clang loop unroll_count(8)")
#endif
