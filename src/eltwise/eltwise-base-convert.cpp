// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.

#include "hexl/eltwise/eltwise-base-convert.hpp"

#include <algorithm>
#include <vector>

#include "eltwise/eltwise-base-convert-internal.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

BaseConvertArgs MakeBaseConvertArgs(const uint64_t* in_moduli, uint64_t in_count,
		const uint64_t* out_moduli, uint64_t out_count,
		const uint64_t* q_hat_inv_mod_q, const uint64_t* q_hat_mod_p) {
	BaseConvertArgs a;
	a.in_count = in_count;
	a.out_count = out_count;
	a.in_moduli = in_moduli;
	a.out_moduli = out_moduli;
	a.q_hat_inv = q_hat_inv_mod_q;
	a.q_hat_mod_p = q_hat_mod_p;
	a.q_hat_inv_precon.resize(in_count);
	a.q_hat_mod_p_precon.resize(in_count * out_count);
	for (uint64_t i = 0; i < in_count; ++i) {
		a.q_hat_inv_precon[i] = MultiplyFactor(q_hat_inv_mod_q[i], 64, in_moduli[i]).BarrettFactor();
		for (uint64_t j = 0; j < out_count; ++j)
			a.q_hat_mod_p_precon[i * out_count + j] =
				MultiplyFactor(q_hat_mod_p[i * out_count + j], 64, out_moduli[j]).BarrettFactor();
	}
	return a;
}

namespace {

void CheckBaseConvertArgs(uint64_t* const* result, const uint64_t* const* operand,
		uint64_t n, const uint64_t* in_moduli, uint64_t in_count,
		const uint64_t* out_moduli, uint64_t out_count,
		const uint64_t* q_hat_inv_mod_q, const uint64_t* q_hat_mod_p) {
	HEXL_CHECK(result != nullptr && operand != nullptr, "Require result, operand != nullptr");
	HEXL_CHECK(n != 0, "Require n != 0");
	HEXL_CHECK(in_count >= 1 && in_count <= 128, "Require 1 <= in_count <= 128");
	HEXL_CHECK(out_count >= 1, "Require out_count >= 1");
	HEXL_CHECK(in_moduli != nullptr && out_moduli != nullptr, "Require moduli != nullptr");
	HEXL_CHECK(q_hat_inv_mod_q != nullptr && q_hat_mod_p != nullptr, "Require the constants != nullptr");
	for (uint64_t i = 0; i < in_count; ++i) {
		HEXL_CHECK(in_moduli[i] > 1 && in_moduli[i] < (1ULL << 60), "Require 2 <= in_moduli[" << i << "] < 2**60");
		HEXL_CHECK(q_hat_inv_mod_q[i] < in_moduli[i], "Require q_hat_inv_mod_q[" << i << "] < q_" << i);
		HEXL_CHECK(operand[i] != nullptr, "Require operand[" << i << "] != nullptr");
		HEXL_CHECK_BOUNDS(operand[i], n, in_moduli[i], "operand[" << i << "] exceeds bound " << in_moduli[i]);
		for (uint64_t j = 0; j < out_count; ++j)
			HEXL_CHECK(q_hat_mod_p[i * out_count + j] < out_moduli[j],
					"Require q_hat_mod_p[" << i << " * K + " << j << "] < p_" << j);
	}
	for (uint64_t j = 0; j < out_count; ++j) {
		HEXL_CHECK(out_moduli[j] > 1 && out_moduli[j] < (1ULL << 60), "Require 2 <= out_moduli[" << j << "] < 2**60");
		HEXL_CHECK(result[j] != nullptr, "Require result[" << j << "] != nullptr");
		for (uint64_t i = 0; i < in_count; ++i)
			HEXL_CHECK(result[j] != operand[i], "Require result[" << j << "] != operand[" << i << "]");
	}
	(void)result; (void)operand; (void)n; (void)in_moduli; (void)in_count; (void)out_moduli; (void)out_count;
	(void)q_hat_inv_mod_q; (void)q_hat_mod_p;
}

}  // namespace

void BaseConvert(uint64_t* const* result, const uint64_t* const* operand,
		uint64_t n, const uint64_t* in_moduli, uint64_t in_count,
		const uint64_t* out_moduli, uint64_t out_count,
		const uint64_t* q_hat_inv_mod_q, const uint64_t* q_hat_mod_p) {
	CheckBaseConvertArgs(result, operand, n, in_moduli, in_count, out_moduli, out_count, q_hat_inv_mod_q, q_hat_mod_p);

	const BaseConvertArgs a = MakeBaseConvertArgs(in_moduli, in_count, out_moduli, out_count,
			q_hat_inv_mod_q, q_hat_mod_p);

#ifdef HEXL_HAS_RVV
	if (has_rvv) {
		HEXL_VLOG(3, "Calling BaseConvertRVV64");
		BaseConvertRVV64(result, operand, n, a);
		return;
	}
#endif

	HEXL_VLOG(3, "Calling BaseConvertNative");
	BaseConvertNative(result, operand, n, a);
}

void BaseConvertExact(uint64_t* const* result, const uint64_t* const* operand,
		uint64_t n, const uint64_t* in_moduli, uint64_t in_count,
		const uint64_t* out_moduli, uint64_t out_count,
		const uint64_t* q_hat_inv_mod_q, const uint64_t* q_hat_mod_p,
		const double* q_inv, const uint64_t* alpha_q_mod_p) {
	CheckBaseConvertArgs(result, operand, n, in_moduli, in_count, out_moduli, out_count, q_hat_inv_mod_q, q_hat_mod_p);
	HEXL_CHECK(q_inv != nullptr && alpha_q_mod_p != nullptr, "Require q_inv, alpha_q_mod_p != nullptr");
	for (uint64_t a = 0; a <= in_count; ++a)
		for (uint64_t j = 0; j < out_count; ++j)
			HEXL_CHECK(alpha_q_mod_p[a * out_count + j] < out_moduli[j],
					"Require alpha_q_mod_p[" << a << " * K + " << j << "] < p_" << j);

	BaseConvertArgs a = MakeBaseConvertArgs(in_moduli, in_count, out_moduli, out_count,
			q_hat_inv_mod_q, q_hat_mod_p);
	a.q_inv = q_inv;
	a.alpha_q_mod_p = alpha_q_mod_p;

#ifdef HEXL_HAS_RVV
	if (has_rvv) {
		HEXL_VLOG(3, "Calling BaseConvertExactRVV64");
		BaseConvertExactRVV64(result, operand, n, a);
		return;
	}
#endif

	HEXL_VLOG(3, "Calling BaseConvertExactNative");
	BaseConvertExactNative(result, operand, n, a);
}

// ---------------------------------------------------------------------------
// Native (scalar) kernel.
// ---------------------------------------------------------------------------

void BaseConvertNative(uint64_t* const* result, const uint64_t* const* operand,
		uint64_t n, const BaseConvertArgs& a) {
	// The corrections are written min(s, s - m) (s - m wraps when s < m): one
	// minu (Zbb) after the sub, where the ternary compiles to sltu + czero
	// (Zicond) + sub. result[j] never aliases an operand (checked by
	// BaseConvert), so each output is stored as soon as it is known.
	const uint64_t L = a.in_count, K = a.out_count;
	std::vector<uint64_t> y(L);
	for (uint64_t k = 0; k < n; ++k) {
		for (uint64_t i = 0; i < L; ++i)  // exact, [0, q_i)
			y[i] = MultiplyMod(operand[i][k], a.q_hat_inv[i], a.q_hat_inv_precon[i], a.in_moduli[i]);
		for (uint64_t j = 0; j < K; ++j) {
			const uint64_t p = a.out_moduli[j], two_p = 2 * p;
			uint64_t s = 0;  // in [0, 2p)
			for (uint64_t i = 0; i < L; ++i) {
				s += MultiplyModLazy<64>(y[i], a.q_hat_mod_p[i * K + j], a.q_hat_mod_p_precon[i * K + j], p);
				s = std::min(s, s - two_p);  // s < 4p before: one subtraction
			}
			result[j][k] = std::min(s, s - p);
		}
	}
}

// BaseConvertNative's Shoup scheme plus the double nu and the alpha row: y_i exact
// (alpha must stay in [0, L]), lazy terms with one correction each, then the alpha
// row subtracted as min(d, d + p) (d = s - c wraps when s < c; s, c < p < 2^63).
void BaseConvertExactNative(uint64_t* const* result, const uint64_t* const* operand,
		uint64_t n, const BaseConvertArgs& a) {
	const uint64_t L = a.in_count, K = a.out_count;
	std::vector<uint64_t> y(L);
	for (uint64_t k = 0; k < n; ++k) {
		double nu = 0.5;
		for (uint64_t i = 0; i < L; ++i) {  // exact, [0, q_i)
			y[i] = MultiplyMod(operand[i][k], a.q_hat_inv[i], a.q_hat_inv_precon[i], a.in_moduli[i]);
			nu += static_cast<double>(y[i]) * a.q_inv[i];
		}
		const uint64_t alpha = static_cast<uint64_t>(nu);
		for (uint64_t j = 0; j < K; ++j) {
			const uint64_t p = a.out_moduli[j], two_p = 2 * p;
			uint64_t s = 0;  // in [0, 2p)
			for (uint64_t i = 0; i < L; ++i) {
				s += MultiplyModLazy<64>(y[i], a.q_hat_mod_p[i * K + j], a.q_hat_mod_p_precon[i * K + j], p);
				s = std::min(s, s - two_p);  // s < 4p before: one subtraction
			}
			s = std::min(s, s - p);  // [0, p)
			const uint64_t d = s - a.alpha_q_mod_p[alpha * K + j];
			result[j][k] = std::min(d, d + p);
		}
	}
}

}  // namespace hexl
}  // namespace intel
