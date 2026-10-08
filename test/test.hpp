// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Minimal self-contained test harness (no GoogleTest: builds with plain make,
// runs under spike + pk, and needs nothing installed on the K3).
//
//   TEST(Suite_Name) { CHECK(...); CHECK_EQ(a, b); CHECK_VEC_EQ(exp, got); }
//
// A test that reaches a stub (an exception whose message starts with
// "[rvv-hexl TODO]") is reported as TODO, not FAIL, together with the name of
// the stub it hit. So the test binary is also the port's progress report.
//
// Tests only use the PUBLIC HEXL API (the same API upstream Intel HEXL exposes;
// running them against upstream is how the oracles in oracle.hpp were validated).

#pragma once

#include <stdint.h>

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace hexltest {

struct Case {
	const char* name;
	void (*fn)();
};

inline std::vector<Case>& Registry() {
	static std::vector<Case> cases;
	return cases;
}

struct Registrar {
	Registrar(const char* name, void (*fn)()) { Registry().push_back({name, fn}); }
};

/// Thrown by a failing CHECK; carries the formatted message.
struct Failure {
	std::string message;
};

[[noreturn]] inline void Fail(const char* file, int line,
		const std::string& what) {
	std::ostringstream oss;
	oss << file << ":" << line << ": " << what;
	throw Failure{oss.str()};
}

}  // namespace hexltest

#define TEST(name)                                                   \
	static void name();                                                \
	static ::hexltest::Registrar name##_registrar_(#name, &name);      \
	static void name()

#define CHECK(cond)                                                  \
	do {                                                               \
		if (!(cond)) ::hexltest::Fail(__FILE__, __LINE__, "CHECK(" #cond ")"); \
	} while (0)

/// Extra context is streamed after the values, e.g.
///   CHECK_EQ(got, want, "n=" << n << " q=" << q);
#define CHECK_EQ(a, b, ...)                                          \
	do {                                                               \
		const auto hexltest_a_ = (a);                                    \
		const auto hexltest_b_ = (b);                                    \
		if (!(hexltest_a_ == hexltest_b_)) {                             \
			std::ostringstream hexltest_oss_;                              \
			hexltest_oss_ << #a " == " #b ": " << hexltest_a_ << " vs "    \
			<< hexltest_b_ << " " __VA_ARGS__;               \
			::hexltest::Fail(__FILE__, __LINE__, hexltest_oss_.str());     \
		}                                                                \
	} while (0)

/// Compares two sequences and reports the FIRST differing index.
#define CHECK_VEC_EQ(expected, actual, ...)                          \
	do {                                                               \
		const auto& hexltest_e_ = (expected);                            \
		const auto& hexltest_g_ = (actual);                              \
		if (hexltest_e_.size() != hexltest_g_.size()) {                  \
			std::ostringstream hexltest_oss_;                              \
			hexltest_oss_ << "size " << hexltest_e_.size() << " vs "       \
			<< hexltest_g_.size() << " " __VA_ARGS__;        \
			::hexltest::Fail(__FILE__, __LINE__, hexltest_oss_.str());     \
		}                                                                \
		for (size_t hexltest_i_ = 0; hexltest_i_ < hexltest_e_.size(); ++hexltest_i_) { \
			if (!(hexltest_e_[hexltest_i_] == hexltest_g_[hexltest_i_])) { \
				std::ostringstream hexltest_oss_;                            \
				hexltest_oss_ << "first mismatch at [" << hexltest_i_        \
				<< "]: expected " << hexltest_e_[hexltest_i_]  \
				<< ", got " << hexltest_g_[hexltest_i_] << " " \
				__VA_ARGS__;                                   \
				::hexltest::Fail(__FILE__, __LINE__, hexltest_oss_.str());   \
			}                                                              \
		}                                                                \
	} while (0)
