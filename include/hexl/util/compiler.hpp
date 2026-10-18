// Copyright (C) 2020 Intel Corporation
// Modifications Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hexl/util/defines.hpp"

#if defined(HEXL_USE_CLANG)
#include "hexl/util/clang.hpp"
#elif defined(HEXL_USE_GNU)
#include "hexl/util/gcc.hpp"
#endif
