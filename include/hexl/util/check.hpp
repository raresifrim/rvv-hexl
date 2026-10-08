// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Argument-validation macros. Compiled out in release builds, exactly like
// upstream. Upstream routes the debug message through easylogging++; the port
// builds it with an ostringstream instead so there is no third-party dependency.

#pragma once

#include <stdint.h>

#include "hexl/util/types.hpp"

#ifdef HEXL_DEBUG
#include <sstream>
#include <stdexcept>

/// @brief If cond is false, throws std::runtime_error with message expr
/// (expr may use operator<< chaining, e.g. "n " << n << " too big").
#define HEXL_CHECK(cond, expr)                                           \
	if (!(cond)) {                                                         \
		std::ostringstream hexl_check_oss_;                                  \
		hexl_check_oss_ << expr << " in function: " << __FUNCTION__          \
		<< " in file: " __FILE__ << ":" << __LINE__;         \
		throw std::runtime_error(hexl_check_oss_.str());                     \
	}

/// @brief Checks that arg[0..n) are all < bound
#define HEXL_CHECK_BOUNDS(arg, n, bound, expr)                            \
	for (size_t hexl_check_idx = 0; hexl_check_idx < n; ++hexl_check_idx) { \
		HEXL_CHECK((arg)[hexl_check_idx] < bound, expr);                      \
	}

#else  // HEXL_DEBUG=OFF

#define HEXL_CHECK(cond, expr) \
{}
#define HEXL_CHECK_BOUNDS(...) \
{}

#endif  // HEXL_DEBUG
