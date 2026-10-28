// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
//
// Kernels behind BaseConvert (rvv-hexl extension): OpenFHE's approximate RNS
// base conversion (see the public header). 64-bit storage only (OpenFHE's CKKS
// and BFV key switching run at NATIVE_SIZE=64).
//
// Every constant is fixed per call: q_hat_inv_mod_q[i] (one per input limb)
// and q_hat_mod_p[i][j] (one per input and output limb), so every product is a
// Shoup multiply with a precomputed factor (BaseConvertArgs, built once per
// call by the dispatch). Per strip of coefficients:
//   y_i   = x_i * q_hat_inv_i mod q_i                  exact Shoup, in [0, q_i)
//   acc_j = sum_i (y_i * q_hat_mod_p[i][j] mod p_j)    lazy Shoup, each term in
//           [0, 2 p_j); acc_j kept below 2 p_j by one conditional subtraction
//           per added term (p_j < 2^60: no overflow, whatever L)
//   out_j = acc_j mod p_j                               one final subtraction
// y_i is read once and used for all K outputs: the RVV body can keep the K
// accumulators in registers when K is small (OpenFHE CKKS, one limb per digit:
// ModUp L = 1 -> K = all other limbs and P, ModDown L = |P| -> K = |Q|).
//
// BaseConvertExact (OpenFHE's SwitchCRTBasis, HPS) adds, per coefficient, the
// double nu = 0.5 + sum_i y_i * (1 / q_i), alpha = floor(nu) in [0, L], and
// subtracts alpha_q_mod_p[alpha][j] from each output: one more accumulator, in
// FP64 lanes, beside the K integer ones, and a gather of the alpha row per
// coefficient (L + 1 rows: a compare-and-select chain is enough for small L).

#pragma once

#include <stdint.h>

#include <vector>

#include "util/cpu-features.hpp"
#include "util/rvv-config.hpp"  // lane types (empty without RVV)

namespace intel {
namespace hexl {

/// @brief The arguments of one conversion with the Shoup factors of its
/// constants (floor(c * 2^64 / m), MultiplyFactor(c, 64, m).BarrettFactor()).
/// Built by the dispatch (BaseConvert), once per call.
struct BaseConvertArgs {
	uint64_t in_count = 0, out_count = 0;    // L, K
	const uint64_t* in_moduli = nullptr;     // q_i, L values
	const uint64_t* out_moduli = nullptr;    // p_j, K values
	const uint64_t* q_hat_inv = nullptr;     // Q_i^{-1} mod q_i, L values
	const uint64_t* q_hat_mod_p = nullptr;   // Q_i mod p_j, row-major [i * K + j]
	std::vector<uint64_t> q_hat_inv_precon;  // Shoup factors of q_hat_inv (mod q_i)
	std::vector<uint64_t> q_hat_mod_p_precon;  // Shoup factors of q_hat_mod_p (mod p_j), [i * K + j]
	// BaseConvertExact only (nullptr for BaseConvert):
	const double* q_inv = nullptr;            // 1 / q_i, L values
	const uint64_t* alpha_q_mod_p = nullptr;  // a Q mod p_j, (L + 1) x K, row-major [a * K + j]
};

/// @brief Builds the Shoup factors of the conversion's constants.
BaseConvertArgs MakeBaseConvertArgs(const uint64_t* in_moduli, uint64_t in_count,
		const uint64_t* out_moduli, uint64_t out_count,
		const uint64_t* q_hat_inv_mod_q, const uint64_t* q_hat_mod_p);

/// @brief Reference: OpenFHE's own arithmetic (dcrtpoly-impl.h, v1.5.1),
/// unsigned __int128 sums reduced with %, one coefficient at a time. Slow; the
/// tests compare the kernels against it.
inline void BaseConvertReference(uint64_t* const* result, const uint64_t* const* operand,
		uint64_t n, const BaseConvertArgs& a) {
	using u128 = unsigned __int128;
	for (uint64_t k = 0; k < n; ++k) {
		for (uint64_t j = 0; j < a.out_count; ++j) {
			u128 sum = 0;
			for (uint64_t i = 0; i < a.in_count; ++i) {
				const uint64_t y = static_cast<uint64_t>(static_cast<u128>(operand[i][k]) * a.q_hat_inv[i] %
						a.in_moduli[i]);
				sum += static_cast<u128>(y) * a.q_hat_mod_p[i * a.out_count + j];
			}
			result[j][k] = static_cast<uint64_t>(sum % a.out_moduli[j]);
		}
	}
}

/// @brief Reference of BaseConvertExact: OpenFHE's SwitchCRTBasis (hexldcrtpoly-impl.h,
/// v1.5.1, the HAVE_INT128 branch): unsigned __int128 sums reduced with %, nu summed in
/// double in OpenFHE's order, the alpha row subtracted. One coefficient at a time.
inline void BaseConvertExactReference(uint64_t* const* result, const uint64_t* const* operand,
		uint64_t n, const BaseConvertArgs& a) {
	using u128 = unsigned __int128;
	const uint64_t L = a.in_count, K = a.out_count;
	std::vector<uint64_t> y(L);
	for (uint64_t k = 0; k < n; ++k) {
		double nu = 0.5;
		for (uint64_t i = 0; i < L; ++i) {
			y[i] = static_cast<uint64_t>(static_cast<u128>(operand[i][k]) * a.q_hat_inv[i] % a.in_moduli[i]);
			nu += static_cast<double>(y[i]) * a.q_inv[i];
		}
		const uint64_t alpha = static_cast<uint64_t>(nu);
		for (uint64_t j = 0; j < K; ++j) {
			const uint64_t p = a.out_moduli[j];
			u128 sum = 0;
			for (uint64_t i = 0; i < L; ++i) sum += static_cast<u128>(y[i]) * a.q_hat_mod_p[i * K + j];
			const uint64_t s = static_cast<uint64_t>(sum % p), c = a.alpha_q_mod_p[alpha * K + j];
			result[j][k] = s >= c ? s - c : s + p - c;
		}
	}
}

/// @brief Native (scalar) kernel: the Shoup scheme above, one coefficient at
/// a time (the RVV kernel's arithmetic, without vectors).
void BaseConvertNative(uint64_t* const* result, const uint64_t* const* operand,
		uint64_t n, const BaseConvertArgs& a);

#ifdef HEXL_HAS_RVV
/// @brief RVV, 64-bit lanes (eltwise-base-convert-rvv.cpp): BaseConvertNative's
/// arithmetic on strips of coefficients. Same contract as BaseConvertNative.
/// Default lane: rvv::cfg::BaseConvert64.
template <class V = rvv::cfg::BaseConvert64>
void BaseConvertRVV64(uint64_t* const* result, const uint64_t* const* operand,
		uint64_t n, const BaseConvertArgs& a);
#endif

/// @brief Native (scalar) kernel of BaseConvertExact: a.q_inv and a.alpha_q_mod_p set.
void BaseConvertExactNative(uint64_t* const* result, const uint64_t* const* operand,
		uint64_t n, const BaseConvertArgs& a);

#ifdef HEXL_HAS_RVV
/// @brief RVV kernel of BaseConvertExact, 64-bit integer lanes for the sums
/// (eltwise-base-convert-rvv.cpp). Default lane: rvv::cfg::BaseConvertExact64.
template <class V = rvv::cfg::BaseConvertExact64>
void BaseConvertExactRVV64(uint64_t* const* result, const uint64_t* const* operand,
		uint64_t n, const BaseConvertArgs& a);
#endif

}  // namespace hexl
}  // namespace intel
