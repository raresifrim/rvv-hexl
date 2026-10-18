// Copyright (C) 2020 Intel Corporation
// Modifications Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// SPDX-License-Identifier: Apache-2.0
//
// The NTT class: construction (twiddle precomputation) and dispatch.

#include "hexl/ntt/ntt.hpp"

#include <vector>

#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "hexl/util/check.hpp"
#include "ntt/ntt-internal.hpp"
#include "ntt/ntt-rvv-small-span.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

NTT::NTT(uint64_t degree, uint64_t q, uint64_t root_of_unity,
		std::shared_ptr<AllocatorBase> alloc_ptr)
	: m_degree(degree),
	m_q(q),
	m_w(root_of_unity),
	m_alloc(alloc_ptr),
	m_aligned_alloc(AlignedAllocator<uint64_t, 64>(m_alloc)),
	m_root_of_unity_powers(m_aligned_alloc),
	m_precon32_root_of_unity_powers(m_aligned_alloc),
	m_precon64_root_of_unity_powers(m_aligned_alloc),
	m_avx512_root_of_unity_powers(m_aligned_alloc),
	m_avx512_precon32_root_of_unity_powers(m_aligned_alloc),
	m_avx512_precon52_root_of_unity_powers(m_aligned_alloc),
	m_avx512_precon64_root_of_unity_powers(m_aligned_alloc),
	m_precon32_inv_root_of_unity_powers(m_aligned_alloc),
	m_precon52_inv_root_of_unity_powers(m_aligned_alloc),
	m_precon64_inv_root_of_unity_powers(m_aligned_alloc),
	m_inv_root_of_unity_powers(m_aligned_alloc),
	m_rvv32_root_of_unity_powers(
			AlignedAllocator<uint32_t, 64>(m_aligned_alloc)),
	m_rvv32_precon_root_of_unity_powers(
			AlignedAllocator<uint32_t, 64>(m_aligned_alloc)),
	m_rvv32_inv_root_of_unity_powers(
			AlignedAllocator<uint32_t, 64>(m_aligned_alloc)),
	m_rvv32_precon_inv_root_of_unity_powers(
			AlignedAllocator<uint32_t, 64>(m_aligned_alloc)) {
		HEXL_CHECK(CheckArguments(degree, q), "");
		HEXL_CHECK(IsPrimitiveRoot(m_w, 2 * degree, q),
				m_w << " is not a primitive 2*" << degree << "'th root of unity");

		m_degree_bits = Log2(m_degree);
		m_w_inv = InverseMod(m_w, m_q);
		ComputeRootOfUnityPowers();
	}

NTT::NTT(uint64_t degree, uint64_t q, std::shared_ptr<AllocatorBase> alloc_ptr)
	: NTT(degree, q, MinimalPrimitiveRoot(2 * degree, q), alloc_ptr) {}

NTT::~NTT() = default;

void NTT::ComputeRootOfUnityPowers() {
	// Runs once per (N, q), when the NTT object is built, so it favours clarity:
	// scalar loops with exact arithmetic (Shoup MultiplyMod, exact for q < 2^63).
	//
	// Forward table, bit-reversed index:   W[ReverseBits(i, log2 N)] = w^i, i < N.
	// Inverse table, stage order (the Gentleman-Sande loops read it with one
	// counter): entry 0 = 1, then for m = N/2, N/4, ..., 1 the m inverses of
	// the forward twiddles W[m .. 2m) that stage undoes.
	// The inverse twiddles are powers of w^-1 (m_w_inv, one InverseMod in the
	// constructor) instead of an InverseMod per entry.
	//
	// Only the uint32_t copies for RVV32 can vectorize: the power chain is a
	// dependency chain, the bit-reversed writes are a scatter, and every Shoup
	// factor is a 128-by-64-bit division. Those copies carry #pragma GCC ivdep
	// (no overlap between source and destination), so GCC vectorizes them
	// without a runtime alias check where a vector unit exists, and they stay
	// plain scalar loops everywhere else.
	const uint64_t n = m_degree;

	// powers of w and w^-1, stored at bit-reversed positions
	m_root_of_unity_powers.assign(n, 0);
	std::vector<uint64_t> inv_bitrev(n);
	const uint64_t w_p = MultiplyFactor(m_w, 64, m_q).BarrettFactor();
	const uint64_t w_inv_p = MultiplyFactor(m_w_inv, 64, m_q).BarrettFactor();
	uint64_t pw = 1, pw_inv = 1;
	for (uint64_t i = 0; i < n; i++) {
		const uint64_t r = ReverseBits(i, m_degree_bits);
		m_root_of_unity_powers[r] = pw;
		inv_bitrev[r] = pw_inv;
		pw = MultiplyMod(pw, m_w, w_p, m_q);
		pw_inv = MultiplyMod(pw_inv, m_w_inv, w_inv_p, m_q);
	}

	// inverse table in the order the inverse kernels consume it
	m_inv_root_of_unity_powers.assign(n, 0);
	m_inv_root_of_unity_powers[0] = inv_bitrev[0];
	uint64_t idx = 1;
	for (uint64_t m = n / 2; m > 0; m >>= 1)
		for (uint64_t i = 0; i < m; i++)
			m_inv_root_of_unity_powers[idx++] = inv_bitrev[m + i];

	// Shoup factors floor(W * 2^64 / q) and floor(W * 2^32 / q) for both tables
	m_precon64_root_of_unity_powers.assign(n, 0);
	m_precon32_root_of_unity_powers.assign(n, 0);
	m_precon64_inv_root_of_unity_powers.assign(n, 0);
	m_precon32_inv_root_of_unity_powers.assign(n, 0);
	for (uint64_t k = 0; k < n; k++) {
		m_precon64_root_of_unity_powers[k] = MultiplyFactor(m_root_of_unity_powers[k], 64, m_q).BarrettFactor();
		m_precon32_root_of_unity_powers[k] = MultiplyFactor(m_root_of_unity_powers[k], 32, m_q).BarrettFactor();
		m_precon64_inv_root_of_unity_powers[k] = MultiplyFactor(m_inv_root_of_unity_powers[k], 64, m_q).BarrettFactor();
		m_precon32_inv_root_of_unity_powers[k] = MultiplyFactor(m_inv_root_of_unity_powers[k], 32, m_q).BarrettFactor();
	}

	// RVV e32 tables (q < 2^30): uint32_t copies in the same layouts as the
	// 64-bit tables, with the 32-bit Shoup factors.
	if (m_q < s_max_fwd_32_modulus) {
		m_rvv32_root_of_unity_powers.resize(n);
		m_rvv32_precon_root_of_unity_powers.resize(n);
		m_rvv32_inv_root_of_unity_powers.resize(n);
		m_rvv32_precon_inv_root_of_unity_powers.resize(n);
		uint32_t* w32 = m_rvv32_root_of_unity_powers.data();
		uint32_t* wp32 = m_rvv32_precon_root_of_unity_powers.data();
		uint32_t* wi32 = m_rvv32_inv_root_of_unity_powers.data();
		uint32_t* wip32 = m_rvv32_precon_inv_root_of_unity_powers.data();
		const uint64_t* w64 = m_root_of_unity_powers.data();
		const uint64_t* wp64 = m_precon32_root_of_unity_powers.data();
		const uint64_t* wi64 = m_inv_root_of_unity_powers.data();
		const uint64_t* wip64 = m_precon32_inv_root_of_unity_powers.data();
#pragma GCC ivdep
		for (uint64_t k = 0; k < n; k++) {
			w32[k] = static_cast<uint32_t>(w64[k]);
			wp32[k] = static_cast<uint32_t>(wp64[k]);
		}
#pragma GCC ivdep
		for (uint64_t k = 0; k < n; k++) {
			wi32[k] = static_cast<uint32_t>(wi64[k]);
			wip32[k] = static_cast<uint32_t>(wip64[k]);
		}
	}

#ifdef HEXL_HAS_RVV
	// The VL-dependent layouts of the configured small-span strategies, for the
	// configured lane types (reads VLMAX, so only with V at run time).
	m_rvv_small_span.reset();
	if (has_rvv) m_rvv_small_span = BuildNttSmallSpanSet(*this);
#endif
	// m_avx512_* and m_precon52_* stay empty: x86 AVX-512 only.
}

#ifdef HEXL_HAS_RVV
namespace {
/// NTT::m_rvv_small_span as what it holds (null when no strategy runs).
const NttSmallSpanSet* SmallSpanSet(const std::shared_ptr<const void>& p) {
	return static_cast<const NttSmallSpanSet*>(p.get());
}
}  // namespace
#endif

bool NTT::CheckArguments(uint64_t degree, uint64_t modulus) {
	HEXL_UNUSED(degree);
	HEXL_UNUSED(modulus);
	// Upper bound on degree is used for security reasons, not correctness
	HEXL_CHECK(IsPowerOfTwo(degree),
			"degree " << degree << " is not a power of 2");
	HEXL_CHECK(degree <= (1ULL << NTT::MaxDegreeBits()),
			"degree should be less than 2^" << NTT::MaxDegreeBits() << " got "
			<< degree);
	HEXL_CHECK(modulus <= (1ULL << NTT::MaxModulusBits()),
			"modulus should be less than 2^" << NTT::MaxModulusBits()
			<< " got " << modulus);
	HEXL_CHECK(modulus % (2 * degree) == 1, "modulus mod 2n != 1");
	HEXL_CHECK(IsPrime(modulus), "modulus is not prime");

	return true;
}

void NTT::ComputeForward(uint64_t* result, const uint64_t* operand,
		uint64_t input_mod_factor,
		uint64_t output_mod_factor) {
	HEXL_CHECK(result != nullptr, "result == nullptr");
	HEXL_CHECK(operand != nullptr, "operand == nullptr");
	HEXL_CHECK(
			input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4,
			"input_mod_factor must be 1, 2 or 4; got " << input_mod_factor);
	HEXL_CHECK(output_mod_factor == 1 || output_mod_factor == 4,
			"output_mod_factor must be 1 or 4; got " << output_mod_factor);
	HEXL_CHECK_BOUNDS(
			operand, m_degree, m_q * input_mod_factor,
			"value in operand exceeds bound " << m_q * input_mod_factor);

#ifdef HEXL_HAS_RVV
	if (has_rvv) {
		const NttSmallSpanSet* ss = SmallSpanSet(m_rvv_small_span);
		if (m_q < s_max_fwd_32_modulus) {
			HEXL_VLOG(3, "Calling ForwardTransformToBitReverseRVV32");
			ForwardTransformToBitReverseRVV32<uint64_t>(
					result, operand, m_degree, m_q, m_rvv32_root_of_unity_powers.data(),
					m_rvv32_precon_root_of_unity_powers.data(), input_mod_factor,
					output_mod_factor, ss ? ss->rvv32_u64.get() : nullptr);
		} else {
			HEXL_VLOG(3, "Calling ForwardTransformToBitReverseRVV64");
			ForwardTransformToBitReverseRVV64(
					result, operand, m_degree, m_q, m_root_of_unity_powers.data(),
					m_precon64_root_of_unity_powers.data(), input_mod_factor,
					output_mod_factor, ss ? ss->rvv64.get() : nullptr);
		}
		return;
	}
#endif

	HEXL_VLOG(3, "Calling ForwardTransformToBitReverseRadix2");
	ForwardTransformToBitReverseRadix2<uint64_t>(
			result, operand, m_degree, m_q, m_root_of_unity_powers.data(),
			m_precon64_root_of_unity_powers.data(), input_mod_factor,
			output_mod_factor);
}

void NTT::ComputeInverse(uint64_t* result, const uint64_t* operand,
		uint64_t input_mod_factor,
		uint64_t output_mod_factor) {
	HEXL_CHECK(result != nullptr, "result == nullptr");
	HEXL_CHECK(operand != nullptr, "operand == nullptr");
	HEXL_CHECK(input_mod_factor == 1 || input_mod_factor == 2,
			"input_mod_factor must be 1 or 2; got " << input_mod_factor);
	HEXL_CHECK(output_mod_factor == 1 || output_mod_factor == 2,
			"output_mod_factor must be 1 or 2; got " << output_mod_factor);
	HEXL_CHECK_BOUNDS(operand, m_degree, m_q * input_mod_factor,
			"operand exceeds bound " << m_q * input_mod_factor);

#ifdef HEXL_HAS_RVV
	if (has_rvv) {
		const NttSmallSpanSet* ss = SmallSpanSet(m_rvv_small_span);
		if (m_q < s_max_inv_32_modulus) {
			HEXL_VLOG(3, "Calling InverseTransformFromBitReverseRVV32");
			InverseTransformFromBitReverseRVV32<uint64_t>(
					result, operand, m_degree, m_q,
					m_rvv32_inv_root_of_unity_powers.data(),
					m_rvv32_precon_inv_root_of_unity_powers.data(), input_mod_factor,
					output_mod_factor, ss ? ss->rvv32_u64.get() : nullptr);
		} else {
			HEXL_VLOG(3, "Calling InverseTransformFromBitReverseRVV64");
			InverseTransformFromBitReverseRVV64(
					result, operand, m_degree, m_q, m_inv_root_of_unity_powers.data(),
					m_precon64_inv_root_of_unity_powers.data(), input_mod_factor,
					output_mod_factor, ss ? ss->rvv64.get() : nullptr);
		}
		return;
	}
#endif

	HEXL_VLOG(3, "Calling InverseTransformFromBitReverseRadix2");
	InverseTransformFromBitReverseRadix2<uint64_t>(
			result, operand, m_degree, m_q, m_inv_root_of_unity_powers.data(),
			m_precon64_inv_root_of_unity_powers.data(), input_mod_factor,
			output_mod_factor);
}

// ---- rvv-hexl extension: 32-bit storage (OpenFHE NATIVE_SIZE=32) ---------

void NTT::ComputeForward(uint32_t* result, const uint32_t* operand,
		uint64_t input_mod_factor,
		uint64_t output_mod_factor) {
	HEXL_CHECK(result != nullptr, "result == nullptr");
	HEXL_CHECK(operand != nullptr, "operand == nullptr");
	HEXL_CHECK(
			input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4,
			"input_mod_factor must be 1, 2 or 4; got " << input_mod_factor);
	HEXL_CHECK(output_mod_factor == 1 || output_mod_factor == 4,
			"output_mod_factor must be 1 or 4; got " << output_mod_factor);
	HEXL_CHECK(m_q < (1ULL << 30),
			"32-bit storage needs q < 2^30 (the lazy [0, 4q) values must fit in 32 bits), q = "
			<< m_q);
	HEXL_CHECK_BOUNDS(
			operand, m_degree, m_q * input_mod_factor,
			"value in operand exceeds bound " << m_q * input_mod_factor);

#ifdef HEXL_HAS_RVV
	if (has_rvv && m_q < s_max_fwd_32_modulus) {
		HEXL_VLOG(3, "Calling ForwardTransformToBitReverseRVV32<uint32_t>");
		const NttSmallSpanSet* ss = SmallSpanSet(m_rvv_small_span);
		ForwardTransformToBitReverseRVV32<uint32_t>(
				result, operand, m_degree, m_q, m_rvv32_root_of_unity_powers.data(),
				m_rvv32_precon_root_of_unity_powers.data(), input_mod_factor,
				output_mod_factor, ss ? ss->rvv32.get() : nullptr);
		return;
	}
#endif

	HEXL_VLOG(3, "Calling ForwardTransformToBitReverseRadix2<uint32_t>");
	ForwardTransformToBitReverseRadix2<uint32_t>(
			result, operand, m_degree, m_q, m_root_of_unity_powers.data(),
			m_precon64_root_of_unity_powers.data(), input_mod_factor,
			output_mod_factor);
}

void NTT::ComputeInverse(uint32_t* result, const uint32_t* operand,
		uint64_t input_mod_factor,
		uint64_t output_mod_factor) {
	HEXL_CHECK(result != nullptr, "result == nullptr");
	HEXL_CHECK(operand != nullptr, "operand == nullptr");
	HEXL_CHECK(input_mod_factor == 1 || input_mod_factor == 2,
			"input_mod_factor must be 1 or 2; got " << input_mod_factor);
	HEXL_CHECK(output_mod_factor == 1 || output_mod_factor == 2,
			"output_mod_factor must be 1 or 2; got " << output_mod_factor);
	HEXL_CHECK(m_q < (1ULL << 30),
			"32-bit storage needs q < 2^30 (the lazy [0, 4q) values must fit in 32 bits), q = "
			<< m_q);
	HEXL_CHECK_BOUNDS(operand, m_degree, m_q * input_mod_factor,
			"operand exceeds bound " << m_q * input_mod_factor);

#ifdef HEXL_HAS_RVV
	if (has_rvv && m_q < s_max_inv_32_modulus) {
		HEXL_VLOG(3, "Calling InverseTransformFromBitReverseRVV32<uint32_t>");
		const NttSmallSpanSet* ss = SmallSpanSet(m_rvv_small_span);
		InverseTransformFromBitReverseRVV32<uint32_t>(
				result, operand, m_degree, m_q,
				m_rvv32_inv_root_of_unity_powers.data(),
				m_rvv32_precon_inv_root_of_unity_powers.data(), input_mod_factor,
				output_mod_factor, ss ? ss->rvv32.get() : nullptr);
		return;
	}
#endif

	HEXL_VLOG(3, "Calling InverseTransformFromBitReverseRadix2<uint32_t>");
	InverseTransformFromBitReverseRadix2<uint32_t>(
			result, operand, m_degree, m_q, m_inv_root_of_unity_powers.data(),
			m_precon64_inv_root_of_unity_powers.data(), input_mod_factor,
			output_mod_factor);
}

}  // namespace hexl
}  // namespace intel
