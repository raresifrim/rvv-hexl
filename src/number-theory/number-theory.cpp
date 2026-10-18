// Copyright (C) 2020 Intel Corporation
// Modifications Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// SPDX-License-Identifier: Apache-2.0
//
// Scalar number theory. Nothing here is RISC-V specific and nothing is on a
// hot path (it runs once per NTT object / per parameter set), so write it for
// clarity and exactness, not speed. Start here: the NTT constructor and most
// tests depend on these functions.
//
// Contracts are documented in include/hexl/number-theory/number-theory.hpp.
// The tests (test/test-number-theory.cpp) check every function against an
// independent __int128 oracle in test/oracle.hpp.

#include "hexl/number-theory/number-theory.hpp"
#include <random>
#include "hexl/logging/logging.hpp"
#include "hexl/util/check.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

uint64_t InverseMod(uint64_t input, uint64_t modulus) {
	// extended Euclid on (input mod modulus, modulus).
	int64_t t_prev=0, t_cur=1;
	uint64_t r_prev=modulus, r_cur=input%modulus;
	while(r_cur != 0){
		int64_t k = r_prev / r_cur;
		uint64_t r_new = r_prev - k * r_cur;
		int64_t t_new = t_prev - k * t_cur;
		r_prev = r_cur;
		t_prev = t_cur;
		r_cur = r_new; 
		t_cur = t_new; 
	}
	HEXL_CHECK(r_prev == 1, "Previous remainder must be 1!");
	return static_cast<uint64_t>(t_prev + (t_prev < 0 ? modulus : 0U)); 
}

uint64_t AddUIntMod(uint64_t x, uint64_t y, uint64_t modulus) {
	//(x + y) mod modulus for x, y < modulus.
	uint64_t sum = x+y;
	return std::min(sum, sum-modulus);
}

uint64_t SubUIntMod(uint64_t x, uint64_t y, uint64_t modulus) {
	// (x - y) mod modulus for x, y < modulus.
	uint64_t diff = x-y;
	return std::min(diff, diff + modulus);
}

static inline uint64_t MontgomeryReduce(uint64_t x_hi, uint64_t x_lo, uint64_t n, uint64_t modulus){
	uint64_t q = x_lo * n;
	uint64_t m;
	asm("mulhu %0, %1, %2" : "=r"(m) : "r"(q), "r"(modulus));
	uint64_t val = x_hi - m;
	return val + (x_hi < m ? modulus : 0U);
}

static inline uint64_t ComputeMontgomeryN(uint64_t n){
	// Step 1: Base approximation (correct modulo 8)
	uint64_t nr = n; 

	// Step 2: Scale up to 64 bits via Newton-Raphson iterations
	nr = nr * (2 - n * nr); // Correct to 6 bits
	nr = nr * (2 - n * nr); // Correct to 12 bits
	nr = nr * (2 - n * nr); // Correct to 24 bits
	nr = nr * (2 - n * nr); // Correct to 48 bits
	nr = nr * (2 - n * nr); // Correct to 96 bits (covers full 64-bit space)

	return nr; 
}

static inline uint64_t ToMontgomery(uint64_t x, uint64_t modulus){
	return static_cast<uint64_t>((static_cast<unsigned __int128>(x) << 64) % modulus);
}

static inline uint64_t FromMontgomery(uint64_t x, uint64_t n, uint64_t modulus) { 
	return MontgomeryReduce(0, x, n, modulus); 
}

static inline uint64_t MontgomeryMultiply(uint64_t a, uint64_t b, uint64_t n, uint64_t modulus){
	// a * b * R^-1 mod modulus: Montgomery form in, Montgomery form out
	uint64_t hi, lo;
	asm("mulhu %0, %1, %2" : "=r"(hi) : "r"(a), "r"(b));
	asm("mul %0, %1, %2" : "=r"(lo) : "r"(a), "r"(b));
	return MontgomeryReduce(hi, lo, n, modulus);
}

uint64_t PowMod(uint64_t base, uint64_t exp, uint64_t modulus) {
	//square-and-multiply based on Montgomery; modulus has to be odd
	if (exp == 0) return 1;
	HEXL_CHECK(modulus & 1,"Modulus must be odd!");
	base = ToMontgomery(base, modulus);
	uint64_t n = ComputeMontgomeryN(modulus);	
	uint64_t result = 1;
	for(; exp!=0; exp>>=1){
		uint64_t t_hi, t_lo;
		if((exp & 0x1) == 1){
			asm("mulhu %0, %1, %2" : "=r"(t_hi) : "r"(result), "r"(base));
			asm("mul %0, %1, %2" : "=r"(t_lo) : "r"(result), "r"(base));
			result = MontgomeryReduce(t_hi, t_lo, n, modulus);
		}
		asm("mulhu %0, %1, %2" : "=r"(t_hi) : "r"(base), "r"(base));
		asm("mul %0, %1, %2" : "=r"(t_lo) : "r"(base), "r"(base));
		base = MontgomeryReduce(t_hi, t_lo, n, modulus);

	}
	return result;
}

bool IsPrimitiveRoot(uint64_t root, uint64_t degree, uint64_t modulus) {
	// root == 0 -> false; otherwise, for power-of-two degree,
	// root is a primitive degree-th root iff root^(degree/2) == modulus - 1.
	if(root == 0U) return false;
	HEXL_CHECK(((degree-1) & degree) == 0U, "Degree must be a power of two!");
	return PowMod(root, degree >> 1, modulus) == (modulus - 1);
}

uint64_t GeneratePrimitiveRoot(uint64_t degree, uint64_t modulus) {
	// pick random x in [0, q), set r = x^((q-1)/degree); r is a
	// degree-th root of unity; return it if IsPrimitiveRoot(r, degree, q).	
	std::mt19937_64 mt(420);
	std::uniform_int_distribution<uint64_t> dist(1, modulus-1);
	for (int i=0; i<200; ++i){
		uint64_t x = dist(mt);
		uint64_t root = PowMod(x, (modulus - 1) / degree, modulus);   // root^degree = x^(q-1) = 1
		if (IsPrimitiveRoot(root, degree, modulus)) return root;	
	}
	HEXL_CHECK(false, "Could not find a primitive root for the provided degree");
	return 0;
}

uint64_t MinimalPrimitiveRoot(uint64_t degree, uint64_t modulus) {
	// r = GeneratePrimitiveRoot(degree, q). All primitive
	// degree-th roots are r^k for odd k, so walk r, r^3, r^5, ... (multiply by
	// r^2 each step, degree/2 steps) and return the smallest value seen.
	uint64_t root = GeneratePrimitiveRoot(degree, modulus);
	const uint64_t root_sqr = PowMod(root, 2U, modulus);
	const uint64_t root_sqr_precon = MultiplyFactor(root_sqr, 64, modulus).BarrettFactor();
	uint64_t root_min = root;
	for(uint64_t i=1; i<degree/2; ++i){
		root = MultiplyMod(root, root_sqr, root_sqr_precon, modulus);
		root_min = std::min(root, root_min);
	}
	return root_min;
}

uint64_t ReverseBits(uint64_t x, uint64_t bit_width) {
	// reverse the low bit_width bits of x; bit_width == 0 -> 0.
	if (bit_width == 0) return 0;
	x = ((x >> 1) & 0x5555555555555555ULL) | ((x & 0x5555555555555555ULL) << 1);   // swap adjacent bits
	x = ((x >> 2) & 0x3333333333333333ULL) | ((x & 0x3333333333333333ULL) << 2);   // swap bit pairs
	x = ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((x & 0x0F0F0F0F0F0F0F0FULL) << 4);   // swap nibbles
	x = __builtin_bswap64(x);                                                       // reverse the bytes
	return x >> (64 - bit_width);                                                   // keep the low bit_width bits
}

bool IsPrime(uint64_t n) {
	// deterministic Miller-Rabin. 
	// Handle small baseline cases
	if (n < 2) return false;
	if (n == 2 || n == 3) return true;
	if (n % 2 == 0 || n % 3 == 0) return false;

	// 1. Factor out powers of 2 from n - 1: write n - 1 as d * 2^s
	uint64_t d = n - 1;
	int s = 0;
	while ((d & 1) == 0) {
		d >>= 1;
		s++;
	}

	// 2. Montgomery constants for n (odd here), shared by every witness. The
	// whole test runs in Montgomery form, so there is no division per squaring
	// (Montgomery form preserves equality, so 1 and n-1 are compared as forms).
	const uint64_t n_inv = ComputeMontgomeryN(n);
	const uint64_t one_m = ToMontgomery(1, n);   // Montgomery form of 1
	const uint64_t minus_one_m = n - one_m;      // Montgomery form of n - 1

	// 3. Minimal 7-witness set proven valid for all n < 2^64 by Jim Sinclair
	// (We add 2 specifically, the rest are composite/prime magic numbers)
	const uint64_t witnesses[] = {2, 325, 9375, 28178, 450775, 9780504, 1795265022};

	// 4. Evaluate each witness
	for (uint64_t a : witnesses) {
		if (a >= n) {
			a %= n;
			if (a == 0) continue; // Skip multiples of n
		}

		// x = a^d mod n, in Montgomery form
		uint64_t base = ToMontgomery(a, n);
		uint64_t x = one_m;
		for (uint64_t e = d; e != 0; e >>= 1) {
			if (e & 1) x = MontgomeryMultiply(x, base, n_inv, n);
			base = MontgomeryMultiply(base, base, n_inv, n);
		}

		// If x is 1 or n-1, this witness fails to prove n is composite
		if (x == one_m || x == minus_one_m) continue;

		// Repeatedly square x up to s-1 times
		bool composite = true;
		for (int r = 0; r < s - 1; r++) {
			x = MontgomeryMultiply(x, x, n_inv, n);
			if (x == minus_one_m) {
				composite = false; // Found a matching sequence; might be prime
				break;
			}
		}

		// If no step yielded 1 or n-1, 'a' is a definitive witness of compositeness
		if (composite) return false;
	}

	return true; // Passed all witnesses; definitively prime
}

std::vector<uint64_t> GeneratePrimes(size_t num_primes, size_t bit_size,
		bool prefer_small_primes,
		size_t ntt_size) {
	// Candidates are q = 1 mod 2N strictly inside (2^bit_size, 2^(bit_size+1)),
	// tested with IsPrime in walk order: upward from 2^bit_size + 1 when
	// prefer_small_primes, otherwise downward from the largest such q below
	// 2^(bit_size+1). Unsigned, so bit_size may go up to 63 (upstream's signed
	// walk overflows at 63); the bounds checks keep the walk from wrapping.
	HEXL_CHECK(num_primes > 0, "num_primes == 0");
	HEXL_CHECK(IsPowerOfTwo(ntt_size), "ntt_size " << ntt_size << " is not a power of two");
	HEXL_CHECK(bit_size < 64, "bit_size " << bit_size << " must be < 64");
	HEXL_CHECK(Log2(ntt_size) < bit_size,
			"log2(ntt_size) " << Log2(ntt_size) << " must be less than bit_size " << bit_size);

	const uint64_t step = 2 * static_cast<uint64_t>(ntt_size);
	const uint64_t lower = (uint64_t{1} << bit_size) + 1;  // = 1 mod 2N, since 2N divides 2^bit_size
	const uint64_t upper = (bit_size + 1 >= 64) ? ~uint64_t{0} : (uint64_t{1} << (bit_size + 1)) - 1;

	std::vector<uint64_t> primes;
	primes.reserve(num_primes);
	if (prefer_small_primes) {
		for (uint64_t q = lower; q < upper; q += step) {
			if (IsPrime(q)) {
				primes.push_back(q);
				if (primes.size() == num_primes) return primes;
			}
			if (upper - q <= step) break;  // q + step would reach upper (or wrap)
		}
	} else {
		// upper is odd and step even, so upper - upper % step + 1 <= upper
		for (uint64_t q = upper - upper % step + 1; q > lower; q -= step) {
			if (IsPrime(q)) {
				primes.push_back(q);
				if (primes.size() == num_primes) return primes;
			}
		}
	}
	HEXL_CHECK(false, "Found only " << primes.size() << " of " << num_primes << " primes of "
			<< bit_size << " bits with q = 1 mod " << step);
	return primes;
}

}  // namespace hexl
}  // namespace intel
