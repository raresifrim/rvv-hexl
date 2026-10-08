// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// bench-hexl entry point. Standard Google Benchmark flags apply, e.g.
//   bench-hexl --benchmark_filter=NTT --benchmark_format=json --benchmark_out=r.json
//   HEXL_DISABLE_RVV=1 bench-hexl      # same binary, native (scalar) kernels
//
// Every result file records WHICH code path produced it (library, RVV
// compiled/enabled, VLEN of the hart it started on) in its "context" block:
// a run that silently fell back to scalar must be identifiable from its own
// output, never assumed.

#include <cstdlib>
#include <string>

#include "bench-common.hpp"

int main(int argc, char** argv) {
	const auto info = intel::hexl::rvv::GetPortInfo();
	benchmark::AddCustomContext("hexl_impl", "rvv-hexl");
	benchmark::AddCustomContext("hexl_rvv_compiled", info.compiled_with_rvv ? "1" : "0");
	benchmark::AddCustomContext("hexl_rvv_available", info.rvv_available ? "1" : "0");
	benchmark::AddCustomContext("hexl_rvv_enabled", info.rvv_enabled ? "1" : "0");
	benchmark::AddCustomContext("hexl_vlen_bits", std::to_string(info.vlen_bits));
	benchmark::AddCustomContext("hexl_build_flags", info.build_flags);
	benchmark::AddCustomContext("hexl_compiler", info.compiler);
	if (const char* v = std::getenv("HEXL_DISABLE_RVV")) {
		benchmark::AddCustomContext("HEXL_DISABLE_RVV", v);
	}
	if (const char* v = std::getenv("BENCH_CLUSTER")) {
		benchmark::AddCustomContext("cluster", v);
	}

	benchmark::Initialize(&argc, argv);
	if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
	benchmark::RunSpecifiedBenchmarks();
	benchmark::Shutdown();
	return 0;
}
