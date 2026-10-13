// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Native (portable scalar) NTT. This is simultaneously
//   * the fallback whenever RVV is absent or HEXL_DISABLE_RVV=1,
//   * the "base RISC-V, no RVV" data point (make ISA=scalar),
//   * the correctness reference the RVV kernels are diffed against.
// Write it first and make every NTT test pass with it before touching RVV.

#include <cstring>

#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "ntt/ntt-internal.hpp"

namespace intel {
namespace hexl {

template <typename Word>
void ForwardTransformToBitReverseRadix2(
		Word* result, const Word* operand, uint64_t n, uint64_t modulus,
		const uint64_t* root_of_unity_powers,
		const uint64_t* precon_root_of_unity_powers, uint64_t input_mod_factor,
		uint64_t output_mod_factor) {
	// Cooley-Tukey, decimation in time, natural -> bit-reversed.

	if (result != operand) //copy operand into result, then work in place.
		for (uint64_t i=0; i<n; ++i)
			result[i] = operand[i];

	for (uint64_t t = n/2, m = 1; m < n; m <<= 1, t >>= 1){      // log2(n) stages
		for (uint64_t i = 0; i < m; i++){                          // m blocks
			uint64_t W = root_of_unity_powers[m + i];
			uint64_t Wp = precon_root_of_unity_powers[m + i];
			for (uint64_t j = 2*i*t; j < 2*i*t + t; j++){       // t butterflies
				uint64_t X = static_cast<uint64_t>(result[j]);
				uint64_t Y = static_cast<uint64_t>(result[j + t]);
				X = std::min(X - 2*modulus, X);
				uint64_t T  = MultiplyModLazy<64>(Y, W, Wp, modulus);  // in [0, 2q)
				result[j] = static_cast<Word>(X + T);     // [0, 4q)
				result[j + t] = static_cast<Word>(X - T + 2*modulus); // [0, 4q)	
			}
		}
	}

	if (output_mod_factor == 1) 
		for (uint64_t i=0; i<n; ++i){
			result[i] = static_cast<Word>(std::min(static_cast<uint64_t>(result[i]), static_cast<uint64_t>(result[i]) - 2*modulus)); //reduce from 4q to 2q
			result[i] = static_cast<Word>(std::min(static_cast<uint64_t>(result[i]), static_cast<uint64_t>(result[i]) - modulus)); //reduce from 2q to q
		}	
}

template <typename Word>
void InverseTransformFromBitReverseRadix2(
		Word* result, const Word* operand, uint64_t n, uint64_t modulus,
		const uint64_t* inv_root_of_unity_powers,
		const uint64_t* precon_inv_root_of_unity_powers, uint64_t input_mod_factor,
		uint64_t output_mod_factor) {
	// Gentleman-Sande, decimation in frequency, bit-reversed -> natural.
	// inv_root_of_unity_powers is in stage order (NTT::ComputeRootOfUnityPowers):
	// entry k is the inverse twiddle of the k-th block this loop visits, so a
	// single counter walks the table. Harvey lazy butterflies keep every value
	// in [0, 2q); the N^{-1} scaling is folded into the last stage.
	if (result != operand) //copy operand into result, then work in place.
		for (uint64_t i=0; i<n; ++i)
			result[i] = operand[i];

	const uint64_t twice_mod = 2*modulus;
	uint64_t k = 1;                                                    // next inverse twiddle
	for (uint64_t t = 1, m = n/2; m > 1; m >>= 1, t <<= 1){            // all stages but the last
		for (uint64_t i = 0; i < m; i++, k++){                          // m blocks
			uint64_t W = inv_root_of_unity_powers[k];
			uint64_t Wp = precon_inv_root_of_unity_powers[k];
			for (uint64_t j = 2*i*t; j < 2*i*t + t; j++){                  // t butterflies
				uint64_t X = static_cast<uint64_t>(result[j]);              // [0, 2q)
				uint64_t Y = static_cast<uint64_t>(result[j + t]);          // [0, 2q)
				uint64_t S = X + Y;                                          // [0, 4q)
				result[j] = static_cast<Word>(std::min(S, S - twice_mod));  // [0, 2q)
				result[j + t] = static_cast<Word>(MultiplyModLazy<64>(X - Y + twice_mod, W, Wp, modulus)); // [0, 2q)
			}
		}
	}

	// Last stage (one block, t = n/2) with the N^{-1} scaling folded in:
	//   X' = (X + Y) * N^{-1},   Y' = (X - Y) * (W * N^{-1})
	const uint64_t t = n/2;
	const uint64_t n_inv = InverseMod(n, modulus);
	const uint64_t n_inv_p = MultiplyFactor(n_inv, 64, modulus).BarrettFactor();
	const uint64_t W = inv_root_of_unity_powers[k];                    // k == n - 1 here
	const uint64_t Wn = MultiplyMod(W, n_inv, n_inv_p, modulus);       // W * N^{-1} mod q
	const uint64_t Wn_p = MultiplyFactor(Wn, 64, modulus).BarrettFactor();
	const uint64_t final_red = (output_mod_factor == 1) ? modulus : 0; // min(v, v - 0) = v leaves [0, 2q)
	for (uint64_t j = 0; j < t; j++){
		uint64_t X = static_cast<uint64_t>(result[j]);
		uint64_t Y = static_cast<uint64_t>(result[j + t]);
		uint64_t Xr = MultiplyModLazy<64>(X + Y, n_inv, n_inv_p, modulus);          // [0, 2q)
		uint64_t Yr = MultiplyModLazy<64>(X - Y + twice_mod, Wn, Wn_p, modulus);    // [0, 2q)
		result[j] = static_cast<Word>(std::min(Xr, Xr - final_red));               // [0, q) if output_mod_factor == 1
		result[j + t] = static_cast<Word>(std::min(Yr, Yr - final_red));
	}
}

void ReferenceForwardTransformToBitReverse(
		uint64_t* operand, uint64_t n, uint64_t modulus,
		const uint64_t* root_of_unity_powers) {
	// same loop nest as the radix-2 forward, but with fully reduced
	// arithmetic (MultiplyMod / AddUIntMod / SubUIntMod) and no lazy tricks.
	for (uint64_t t = n/2, m = 1; m < n; m <<= 1, t >>= 1){      // log2(n) stages
		for (uint64_t i = 0; i < m; i++){                          // m blocks
			uint64_t W = root_of_unity_powers[m + i];
			uint64_t Wp = MultiplyFactor(W, 64, modulus).BarrettFactor();
			for (uint64_t j = 2*i*t; j < 2*i*t + t; j++){            // t butterflies
				uint64_t X = operand[j]; 
				uint64_t Y = operand[j + t];
				uint64_t T = MultiplyMod(Y, W, Wp, modulus);
				operand[j] = AddUIntMod(X, T, modulus);                      
				operand[j + t] = SubUIntMod(X, T, modulus);
			}
		}
	}
}

void ReferenceInverseTransformFromBitReverse(
		uint64_t* operand, uint64_t n, uint64_t modulus,
		const uint64_t* inv_root_of_unity_powers) {
	// textbook Gentleman-Sande, fully reduced (AddUIntMod / SubUIntMod / Shoup
	// MultiplyMod), stages in reverse order of the forward, then a separate
	// N^{-1} scaling pass. Same table order as the fast kernel.
	uint64_t k = 1;                                                    // next inverse twiddle
	for (uint64_t t = 1, m = n/2; m >= 1; m >>= 1, t <<= 1){           // log2(n) stages
		for (uint64_t i = 0; i < m; i++, k++){                          // m blocks
			uint64_t W = inv_root_of_unity_powers[k];
			uint64_t Wp = MultiplyFactor(W, 64, modulus).BarrettFactor();
			for (uint64_t j = 2*i*t; j < 2*i*t + t; j++){                  // t butterflies
				uint64_t X = operand[j];
				uint64_t Y = operand[j + t];
				operand[j] = AddUIntMod(X, Y, modulus);
				operand[j + t] = MultiplyMod(SubUIntMod(X, Y, modulus), W, Wp, modulus);
			}
		}
	}

	const uint64_t n_inv = InverseMod(n, modulus);
	const uint64_t n_inv_p = MultiplyFactor(n_inv, 64, modulus).BarrettFactor();
	for (uint64_t i = 0; i < n; i++)
		operand[i] = MultiplyMod(operand[i], n_inv, n_inv_p, modulus);
}

// Both storage types are dispatched from ntt.cpp.
template void ForwardTransformToBitReverseRadix2<uint64_t>(
		uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint64_t*,
		const uint64_t*, uint64_t, uint64_t);
template void ForwardTransformToBitReverseRadix2<uint32_t>(
		uint32_t*, const uint32_t*, uint64_t, uint64_t, const uint64_t*,
		const uint64_t*, uint64_t, uint64_t);
template void InverseTransformFromBitReverseRadix2<uint64_t>(
		uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint64_t*,
		const uint64_t*, uint64_t, uint64_t);
template void InverseTransformFromBitReverseRadix2<uint32_t>(
		uint32_t*, const uint32_t*, uint64_t, uint64_t, const uint64_t*,
		const uint64_t*, uint64_t, uint64_t);

}  // namespace hexl
}  // namespace intel
