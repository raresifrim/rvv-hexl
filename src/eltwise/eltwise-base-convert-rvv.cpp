// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
//
// RVV kernel for BaseConvert (rvv-hexl extension): OpenFHE's approximate RNS
// base conversion. STUB until written: calls BaseConvertNative, so the
// dispatch, the tests and the benchmarks work from the start. The arithmetic
// to vectorise (one strip of coefficients, e64 lanes) is BaseConvertNative's,
// documented in eltwise-base-convert-internal.hpp.

#include <stdint.h>

#include "eltwise/eltwise-base-convert-internal.hpp"

#ifdef HEXL_HAS_RVV
#include <riscv_vector.h>

#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

template <class V>
void BaseConvertRVV64(uint64_t* const* result, const uint64_t* const* operand,
		uint64_t n, const BaseConvertArgs& a) {
	// STUB: replace with the RVV kernel (eltwise-base-convert-internal.hpp)
	BaseConvertNative(result, operand, n, a);
}

template void BaseConvertRVV64<vuint64m1_t>(uint64_t* const*, const uint64_t* const*, uint64_t, const BaseConvertArgs&);
template void BaseConvertRVV64<vuint64m2_t>(uint64_t* const*, const uint64_t* const*, uint64_t, const BaseConvertArgs&);
template void BaseConvertRVV64<vuint64m4_t>(uint64_t* const*, const uint64_t* const*, uint64_t, const BaseConvertArgs&);
template void BaseConvertRVV64<vuint64m8_t>(uint64_t* const*, const uint64_t* const*, uint64_t, const BaseConvertArgs&);

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
