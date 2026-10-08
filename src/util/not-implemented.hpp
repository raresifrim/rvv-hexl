// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Marker for the functions you still have to write.
//
// Every stub calls HEXL_NOT_IMPLEMENTED(), which throws std::logic_error with a
// message starting "[rvv-hexl TODO]". The test runner (test/main.cpp)
// recognises that prefix and reports the test as TODO instead of FAIL, so
// `make test` doubles as a progress report. `make todo` lists the remaining
// stubs by grepping for this macro.
//
// It is deliberately a throw, not an abort: OpenFHE unit tests and the
// benchmarks also surface the exact function name instead of crashing silently.

#pragma once

namespace intel {
namespace hexl {
namespace internal {

[[noreturn]] void ThrowNotImplemented(const char* function, const char* file,
		int line);

}  // namespace internal
}  // namespace hexl
}  // namespace intel

#define HEXL_NOT_IMPLEMENTED() \
	::intel::hexl::internal::ThrowNotImplemented(__func__, __FILE__, __LINE__)
