// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
//
// RVV kernels for SignedDigitDecompose (rvv-hexl extension). Per strip of vl
// elements, everything in registers across the digits (W = lane width, g =
// log_base, h = 2^(g-1)):
//   load (rvv::Load32 narrows uint64_t storage);
//   centre: c = x - q where x > floor(q/2) - 1 (vmsgtu.vx + masked vsub);
//   next value: (c - r) >> g with r the signed low digit is floor((c + h) / 2^g),
//     the rounding arithmetic shift vssra with vxrm = RNU: one instruction for
//     OpenFHE's shift-left, shift-right, subtract, shift-right. The dropped
//     digit costs only this.
//   each kept digit: r = vsra(vsll(c, W - g), W - g), independent of the next
//     c, so the two chains overlap; r mod q = vminu(r, r + q) read unsigned (a
//     negative r is huge unsigned, r + q wraps into [0, q)), no mask register;
//   store (rvv::Store32 widens for uint64_t storage).
// 5 vector instructions per kept digit and a store, against OpenFHE's 6 scalar
// operations, a compare and a conditional add.

#include "eltwise/eltwise-signed-digit-decompose-internal.hpp"

#ifdef HEXL_HAS_RVV

#include <riscv_vector.h>

#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

namespace {

/// The kernel for any lane type V (e32 or e64) and storage word: Elem is V's
/// element type, Word the storage word (the e32 lanes narrow / widen uint64_t
/// storage through Load32 / Store32).
template <typename Word, class V, typename Elem>
inline void SignedDigitDecomposeStrips(Word* const* result, const Word* operand,
		uint64_t n, uint64_t modulus, uint64_t log_base, uint64_t num_digits) {
	constexpr size_t kBits = sizeof(Elem) * 8;
	const Elem q = static_cast<Elem>(modulus);
	const Elem q_half = static_cast<Elem>(modulus >> 1);
	const size_t g = static_cast<size_t>(log_base);
	const size_t up = kBits - g;
	for (size_t i = 0, vl; i < n; i += vl) {
		vl = rvv::SetVl<V>(n - i);
		V x;
		if constexpr (sizeof(Elem) == 4) {
			x = rvv::Load32<V>(operand + i, vl);
		} else {
			x = rvv::Load<V>(operand + i, vl);
		}
		const auto upper = __riscv_vmsgtu(x, q_half - 1, vl);  // x >= floor(q/2): vmsgtu.vx (no vmsgeu.vx)
		auto c = rvv::AsSigned(__riscv_vsub_mu(upper, x, x, q, vl));     // centred, in (-q/2, q/2]
		c = __riscv_vssra(c, g, __RISCV_VXRM_RNU, vl);                   // the dropped lowest digit
		for (uint64_t d = 0; d < num_digits; ++d) {
			const V r = rvv::AsUnsigned(__riscv_vsra(__riscv_vsll(c, up, vl), up, vl));  // signed low digit
			if (d + 1 < num_digits) c = __riscv_vssra(c, g, __RISCV_VXRM_RNU, vl);       // (c - r) >> g
			const V out = __riscv_vminu(r, __riscv_vadd(r, q, vl), vl);                  // r < 0 ? r + q : r
			if constexpr (sizeof(Elem) == 4) {
				rvv::Store32(result[d] + i, out, vl);
			} else {
				rvv::Store(result[d] + i, out, vl);
			}
		}
	}
}

}  // namespace

template <typename Word, class V>
void SignedDigitDecomposeRVV32(Word* const* result, const Word* operand,
		uint64_t n, uint64_t modulus, uint64_t log_base, uint64_t num_digits) {
	static_assert(rvv::IsE32<V>, "SignedDigitDecomposeRVV32 takes e32 lanes");
	SignedDigitDecomposeStrips<Word, V, uint32_t>(result, operand, n, modulus, log_base, num_digits);
}

template <class V>
void SignedDigitDecomposeRVV64(uint64_t* const* result, const uint64_t* operand,
		uint64_t n, uint64_t modulus, uint64_t log_base, uint64_t num_digits) {
	static_assert(rvv::IsE64<V>, "SignedDigitDecomposeRVV64 takes e64 lanes");
	SignedDigitDecomposeStrips<uint64_t, V, uint64_t>(result, operand, n, modulus, log_base, num_digits);
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
