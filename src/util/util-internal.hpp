// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "hexl/util/util.hpp"

namespace intel {
namespace hexl {

/// @brief Returns whether or not the comparison is true (scalar helper for the
/// native EltwiseCmpAdd / EltwiseCmpSubMod kernels).
inline bool Compare(CMPINT cmp, uint64_t lhs, uint64_t rhs) {
	switch (cmp) {
		case CMPINT::EQ:
			return lhs == rhs;
		case CMPINT::LT:
			return lhs < rhs;
		case CMPINT::LE:
			return lhs <= rhs;
		case CMPINT::FALSE:
			return false;
		case CMPINT::NE:
			return lhs != rhs;
		case CMPINT::NLT:
			return lhs >= rhs;
		case CMPINT::NLE:
			return lhs > rhs;
		case CMPINT::TRUE:
			return true;
		default:
			return true;
	}
}

}  // namespace hexl
}  // namespace intel
