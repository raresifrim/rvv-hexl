// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// SPDX-License-Identifier: Apache-2.0
//
// RVV kernels for SignedDigitDecompose (rvv-hexl extension). Body plan, per
// strip of vl elements, everything in registers across the digits:
//   load operand (rvv::Load32 narrows uint64_t storage), reinterpret as signed
//   (__riscv_vreinterpret_i32 / _i64);
//   centre: mask = x >= floor(q/2) (unsigned compare), c = x - q under the mask;
//   dropped digit: r = vsra(vsll(c, W - g), W - g); c = vsra(c - r, g);
//   for d in 0 .. num_digits - 1:
//     r = vsra(vsll(c, W - g), W - g);  c = vsra(c - r, g);
//     r += q where r < 0 (vmslt, masked vadd);
//     store r to result[d] + offset (rvv::Store32 widens for uint64_t storage).
// W is the lane width (32 or 64); log_base is a runtime shift (.vx forms).

#include "eltwise/eltwise-signed-digit-decompose-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

template <typename Word, class V>
void SignedDigitDecomposeRVV32(Word* const* result, const Word* operand,
		uint64_t n, uint64_t modulus, uint64_t log_base, uint64_t num_digits) {
	// STUB: replace with the RVV32 kernel (body plan above)
	SignedDigitDecomposeStub<Word>(result, operand, n, modulus, log_base, num_digits);
}

template <class V>
void SignedDigitDecomposeRVV64(uint64_t* const* result, const uint64_t* operand,
		uint64_t n, uint64_t modulus, uint64_t log_base, uint64_t num_digits) {
	// STUB: replace with the RVV64 kernel (body plan above)
	SignedDigitDecomposeStub<uint64_t>(result, operand, n, modulus, log_base, num_digits);
}

// Every lane type, so tests and benchmarks can compare them; the public
// function uses the defaults from src/util/rvv-config.hpp. On uint64_t storage
// the e32 lanes stop at u32m4 (the narrowing load).
#define HEXL_SDD_RVV32(Word, V)                                                        \
	template void SignedDigitDecomposeRVV32<Word, V>(Word* const*, const Word*,      \
			uint64_t, uint64_t, uint64_t, uint64_t)
#define HEXL_SDD_RVV64(V)                                                              \
	template void SignedDigitDecomposeRVV64<V>(uint64_t* const*, const uint64_t*,    \
			uint64_t, uint64_t, uint64_t, uint64_t)

HEXL_SDD_RVV32(uint64_t, vuint32m1_t);
HEXL_SDD_RVV32(uint64_t, vuint32m2_t);
HEXL_SDD_RVV32(uint64_t, vuint32m4_t);
HEXL_SDD_RVV32(uint32_t, vuint32m1_t);
HEXL_SDD_RVV32(uint32_t, vuint32m2_t);
HEXL_SDD_RVV32(uint32_t, vuint32m4_t);
HEXL_SDD_RVV32(uint32_t, vuint32m8_t);
HEXL_SDD_RVV64(vuint64m1_t);
HEXL_SDD_RVV64(vuint64m2_t);
HEXL_SDD_RVV64(vuint64m4_t);
HEXL_SDD_RVV64(vuint64m8_t);

#undef HEXL_SDD_RVV64
#undef HEXL_SDD_RVV32

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
