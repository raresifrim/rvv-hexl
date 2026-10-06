// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Scalar modular arithmetic. The inline functions below are part of the
// upstream *header* API and are provided as-is. The non-inline ones are
// declared here and are YOURS to implement in src/number-theory/number-theory.cpp.

#pragma once

#include <stdint.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "hexl/util/check.hpp"
#include "hexl/util/compiler.hpp"

namespace intel {
namespace hexl {

/// @brief Pre-computes a Barrett factor with which modular multiplication can
/// be performed more efficiently
class MultiplyFactor {
 public:
  MultiplyFactor() = default;

  /// @brief Computes and stores the Barrett factor floor((operand << bit_shift)
  /// / modulus). This is useful when modular multiplication of the form
  /// (x * operand) mod modulus is performed with same modulus and operand
  /// several times (Shoup's trick: NTT twiddles, FMA scalars). Passing
  /// operand=1 pre-computes floor(2^bit_shift / modulus) for plain Barrett.
  /// @details bit_shift 32 is the natural choice for the RVV e32 path
  /// (SEW=32, modulus < 2^30); bit_shift 64 for the e64 path. 52 exists only
  /// for upstream's AVX512-IFMA path and is kept for compatibility.
  MultiplyFactor(uint64_t operand, uint64_t bit_shift, uint64_t modulus)
      : m_operand(operand) {
    HEXL_CHECK(operand <= modulus, "operand " << operand
                                              << " must be less than modulus "
                                              << modulus);
    HEXL_CHECK(bit_shift == 32 || bit_shift == 52 || bit_shift == 64,
               "Unsupported BitShift " << bit_shift);
    uint64_t op_hi = operand >> (64 - bit_shift);
    uint64_t op_lo = (bit_shift == 64) ? 0 : (operand << bit_shift);

    // RV64 has no 128-by-64 divide: the generic path below is a libgcc
    // __udivti3 call (~50-70 cycles on the K3). The two common cases need
    // only one 64-bit divu (4-8x faster, measured on the X100 and A100):
    if (op_hi == 0) {
      // The dividend fits in 64 bits: bit_shift 32/52 with operand 1, and the
      // e32 Shoup factors floor(y * 2^32 / q) for y < q < 2^30.
      m_barrett_factor = op_lo / modulus;
    } else if ((op_hi == 1) && (op_lo == 0)) {
      // 2^64 / modulus (operand 1, bit_shift 64: the plain Barrett factor):
      // floor(2^64 / q) = floor((2^64 - 1) / q) + 1 if q divides 2^64, else + 0.
      // q divides 2^64 exactly when (2^64 - 1) mod q == q - 1 (q a power of two).
      const uint64_t all_ones = ~uint64_t{0};
      m_barrett_factor =
          (all_ones / modulus) + (((all_ones % modulus) == (modulus - 1)) ? 1 : 0);
    } else {
      m_barrett_factor = DivideUInt128UInt64Lo(op_hi, op_lo, modulus);
    }
  }

  /// @brief Returns the pre-computed Barrett factor
  inline uint64_t BarrettFactor() const { return m_barrett_factor; }

  /// @brief Returns the operand corresponding to the Barrett factor
  inline uint64_t Operand() const { return m_operand; }

 private:
  uint64_t m_operand;
  uint64_t m_barrett_factor;
};

/// @brief Returns whether or not num is a power of two
inline bool IsPowerOfTwo(uint64_t num) { return num && !(num & (num - 1)); }

/// @brief Returns floor(log2(x))
inline uint64_t Log2(uint64_t x) { return MSB(x); }

inline bool IsPowerOfFour(uint64_t num) {
  return IsPowerOfTwo(num) && (Log2(num) % 2 == 0);
}

/// @brief Returns the maximum value that can be represented using \p bits bits
inline uint64_t MaximumValue(uint64_t bits) {
  HEXL_CHECK(bits <= 64, "MaximumValue requires bits <= 64; got " << bits);
  if (bits == 64) {
    return (std::numeric_limits<uint64_t>::max)();
  }
  return (1ULL << bits) - 1;
}

// ===========================================================================
// Implemented in src/number-theory/number-theory.cpp
// ===========================================================================

/// @brief Reverses the low \p bit_width bits of \p x.
/// @return e.g. ReverseBits(0b0011, 4) == 0b1100. bit_width == 0 returns 0.
uint64_t ReverseBits(uint64_t x, uint64_t bit_width);

/// @brief Returns x^{-1} mod modulus. Requires x % modulus != 0 and
/// gcd(x, modulus) == 1 (modulus need not be prime: use extended Euclid).
uint64_t InverseMod(uint64_t x, uint64_t modulus);

/// @brief Returns (x * y) mod modulus. Assumes x, y < modulus < 2^61.
/// (rvv-hexl computes it with upstream HEXL's pre-shift Barrett, exact for
/// moduli of up to 61 bits; OpenFHE uses <= 60 bits, SEAL <= 61.)
inline uint64_t MultiplyMod(uint64_t x, uint64_t y, uint64_t modulus) {
  // 128-bit product (MultiplyUInt64 -> hi, lo) then
  // BarrettReduce128(hi, lo, modulus). Assumes x, y < modulus.
  //SEAL and OpenFHE guarantee moduli under 2^61 otherwise we might need to perform 2 final subtractions to get the putput in [0,modulus) 
  HEXL_CHECK(modulus < (1ULL << 61), "Require modulus < (1ULL << 61)"); 
  const uint8_t n = MSB(modulus) + 1;
  
  //compute raw x*y
  uint64_t hi, lo;
  asm("mulhu %0, %1, %2" : "=r"(hi) : "r"(x), "r"(y));
  asm("mul %0, %1, %2" : "=r"(lo) : "r"(x), "r"(y));

  //compute the multiplication factor optimized for less multiplication
  uint64_t m = MultiplyFactor(uint64_t(1U) << (n-2), 64, modulus).BarrettFactor();
  uint64_t c = lo >> (n-2) | hi << (64-(n-2));
  uint64_t q;
  asm("mulhu %0, %1, %2" : "=r"(q) : "r"(c), "r"(m));

  //get final result
  uint64_t r = lo - (q * modulus);
  return std::min(r, r - modulus);
}

/// @brief Returns (x * y) mod modulus, Shoup style.
/// @param[in] y_precon floor(y * 2^64 / modulus), i.e.
/// MultiplyFactor(y, 64, modulus).BarrettFactor(). (The upstream doxygen says
/// "floor(2**64 / modulus)", which is wrong; the implementation and every
/// caller use the y-dependent factor.)
inline uint64_t MultiplyMod(uint64_t x, uint64_t y, uint64_t y_precon,
                            uint64_t modulus) {
  // Shoup's trick. y_precon = floor(y * 2^64 / q):
  // Q = MultiplyUInt64Hi<64>(x, y_precon);   // approx. floor(x*y/q)
  // r = x * y - Q * q;                       // wrapping 64-bit, in [0, 2q)
  // return r >= q ? r - q : r; 
  uint64_t Q;
  asm("mulhu %0, %1, %2" : "=r"(Q) : "r"(x), "r"(y_precon));
  uint64_t r = x * y - Q * modulus;
  return std::min(r, r - modulus); 
}

/// @brief Returns (x + y) mod modulus. Assumes x, y < modulus <= 2^63.
/// (The rvv-hexl implementation is the branch-free min(s, s - q), which needs
/// x + y < 2^64. OpenFHE and SEAL moduli are at most 61 bits.)
uint64_t AddUIntMod(uint64_t x, uint64_t y, uint64_t modulus);

/// @brief Returns (x - y) mod modulus. Assumes x, y < modulus <= 2^63.
/// (The rvv-hexl implementation is the branch-free min(d, d + q), which needs
/// d + q < 2^64 for every d < q.)
uint64_t SubUIntMod(uint64_t x, uint64_t y, uint64_t modulus);

/// @brief Returns base^exp mod modulus
uint64_t PowMod(uint64_t base, uint64_t exp, uint64_t modulus);

/// @brief Returns whether or not root is a primitive degree-th root of unity
/// mod modulus. root == 0 returns false.
/// @param[in] degree Degree of root of unity; must be a power of two
/// @details For a power-of-two degree this is exactly root^(degree/2) == -1.
bool IsPrimitiveRoot(uint64_t root, uint64_t degree, uint64_t modulus);

/// @brief Tries to return a primitive degree-th root of unity (randomised).
/// @details Returns 0 (or throws in debug builds) if no root is found.
uint64_t GeneratePrimitiveRoot(uint64_t degree, uint64_t modulus);

/// @brief Returns the numerically smallest primitive degree-th root of unity.
/// @details The NTT(degree, q) constructor uses this with degree = 2N, so the
/// result must be deterministic (all primitive 2N-th roots are psi^k, k odd).
uint64_t MinimalPrimitiveRoot(uint64_t degree, uint64_t modulus);

// ===========================================================================

/// @brief Computes (x * y) mod modulus, except that the output is in [0, 2 *
/// modulus]
/// @param[in] x
/// @param[in] y_operand also denoted y
/// @param[in] modulus
/// @param[in] y_barrett_factor Pre-computed Barrett reduction factor floor((y
/// << BitShift) / modulus)
template <int BitShift>
inline uint64_t MultiplyModLazy(uint64_t x, uint64_t y_operand,
                                uint64_t y_barrett_factor, uint64_t modulus) {
  HEXL_CHECK(y_operand < modulus, "y_operand " << y_operand
                                               << " must be less than modulus "
                                               << modulus);
  HEXL_CHECK(
      modulus <= MaximumValue(BitShift),
      "Modulus " << modulus << " exceeds bound " << MaximumValue(BitShift));
  HEXL_CHECK(x <= MaximumValue(BitShift),
             "Operand " << x << " exceeds bound " << MaximumValue(BitShift));

  uint64_t Q = MultiplyUInt64Hi<BitShift>(x, y_barrett_factor);
  return y_operand * x - Q * modulus;
}

/// @brief Computes (x * y) mod modulus, except that the output is in [0, 2 *
/// modulus]
template <int BitShift>
inline uint64_t MultiplyModLazy(uint64_t x, uint64_t y, uint64_t modulus) {
  HEXL_CHECK(BitShift == 64 || BitShift == 52,
             "Unsupported BitShift " << BitShift);
  HEXL_CHECK(x <= MaximumValue(BitShift),
             "Operand " << x << " exceeds bound " << MaximumValue(BitShift));
  HEXL_CHECK(y < modulus,
             "y " << y << " must be less than modulus " << modulus);
  HEXL_CHECK(
      modulus <= MaximumValue(BitShift),
      "Modulus " << modulus << " exceeds bound " << MaximumValue(BitShift));

  uint64_t y_barrett = MultiplyFactor(y, BitShift, modulus).BarrettFactor();
  return MultiplyModLazy<BitShift>(x, y, y_barrett, modulus);
}

/// @brief Fused Shoup multiply-add: returns (x * w + y) mod modulus in
/// [0, modulus), the EltwiseFMAMod element (w = arg2, y = arg3).
/// @param[in] x any 64-bit word (Shoup's quotient estimate is off by less than
/// 2 for any x when w < modulus, so x needs no reduction)
/// @param[in] w fixed multiplier, w < modulus
/// @param[in] w_precon floor(w * 2^64 / modulus), i.e.
/// MultiplyFactor(w, 64, modulus).BarrettFactor(); also for 32-bit storage
/// (the scalar registers and mulhu are 64-bit)
/// @param[in] y addend in [0, InputModFactor * modulus)
/// @param[in] modulus < 2^61, so 8 * modulus fits in a word
/// @details y is reduced to [0, 2q) (InputModFactor 4: one step, 8: two), then
/// x * w + y - Q * q lies in [0, 2q + y) < 4q (exact in wrapping 64-bit
/// arithmetic) and two conditional subtracts (2q, q) finish. Compared with
/// MultiplyMod(x, w, w_precon, q) followed by an add, y needs one reduction
/// step less, and for InputModFactor 2 none.
template <int InputModFactor>
inline uint64_t MultiplyAddMod(uint64_t x, uint64_t w, uint64_t w_precon,
                               uint64_t y, uint64_t modulus) {
  static_assert(InputModFactor == 1 || InputModFactor == 2 ||
                    InputModFactor == 4 || InputModFactor == 8,
                "InputModFactor must be 1, 2, 4 or 8");
  HEXL_CHECK(modulus < (1ULL << 61), "Require modulus < 2^61");
  HEXL_CHECK(w < modulus, "w " << w << " must be less than modulus " << modulus);
  HEXL_CHECK(y < InputModFactor * modulus,
             "y " << y << " exceeds bound " << InputModFactor * modulus);
  if constexpr (InputModFactor == 8) y = std::min(y, y - 4 * modulus);  // [0, 8q) -> [0, 4q)
  if constexpr (InputModFactor >= 4) y = std::min(y, y - 2 * modulus);  // [0, 4q) -> [0, 2q)
  uint64_t Q;
  asm("mulhu %0, %1, %2" : "=r"(Q) : "r"(x), "r"(w_precon));  // ~floor(x * w / q)
  uint64_t r = x * w + y - Q * modulus;  // in [0, 2q + y) < 4q
  r = std::min(r, r - 2 * modulus);      // [0, 4q) -> [0, 2q)
  return std::min(r, r - modulus);       // [0, 2q) -> [0, q)
}

/// @brief Adds two unsigned 64-bit integers
/// @return The carry bit
inline unsigned char AddUInt64(uint64_t operand1, uint64_t operand2,
                               uint64_t* result) {
  *result = operand1 + operand2;
  return static_cast<unsigned char>(*result < operand1);
}

// ===========================================================================
// Implemented in src/number-theory/number-theory.cpp (stubs left: make todo)
// ===========================================================================

/// @brief Returns whether or not the input is prime (must be deterministic and
/// exact for every 64-bit input: e.g. Miller-Rabin with a fixed witness set)
bool IsPrime(uint64_t n);

/// @brief Generates a list of num_primes primes in the range [2^(bit_size),
/// 2^(bit_size+1)]. Ensures each prime q satisfies q % (2*N) == 1, where N is
/// ntt_size.
/// @param[in] prefer_small_primes When true, walks upward from 2^(bit_size);
/// when false, walks downward from 2^(bit_size+1). The order of the returned
/// list is the walk order (ascending resp. descending).
/// @param[in] ntt_size N; power of two, log2(N) < bit_size.
std::vector<uint64_t> GeneratePrimes(size_t num_primes, size_t bit_size,
                                     bool prefer_small_primes,
                                     size_t ntt_size = 1);

// ===========================================================================

/// @brief Returns input mod modulus, computed via 64-bit Barrett reduction
/// @param[in] q_barr floor(2^64 / modulus)
template <int OutputModFactor = 1>
uint64_t BarrettReduce64(uint64_t input, uint64_t modulus, uint64_t q_barr) {
  HEXL_CHECK(modulus != 0, "modulus == 0");
  uint64_t q = MultiplyUInt64Hi<64>(input, q_barr);
  uint64_t q_times_input = input - q * modulus;
  if (OutputModFactor == 2) {
    return q_times_input;
  } else {
    return (q_times_input >= modulus) ? q_times_input - modulus : q_times_input;
  }
}

/// @brief Returns x mod modulus, assuming x < InputModFactor * modulus
/// @param[in] twice_modulus 2 * q; must not be nullptr if InputModFactor == 4
/// or 8
/// @param[in] four_times_modulus 4 * q; must not be nullptr if InputModFactor
/// == 8
template <int InputModFactor>
uint64_t ReduceMod(uint64_t x, uint64_t modulus,
                   const uint64_t* twice_modulus = nullptr,
                   const uint64_t* four_times_modulus = nullptr) {
  HEXL_CHECK(InputModFactor == 1 || InputModFactor == 2 ||
                 InputModFactor == 4 || InputModFactor == 8,
             "InputModFactor should be 1, 2, 4, or 8");
  if (InputModFactor == 1) {
    return x;
  }
  if (InputModFactor == 2) {
    if (x >= modulus) {
      x -= modulus;
    }
    return x;
  }
  if (InputModFactor == 4) {
    HEXL_CHECK(twice_modulus != nullptr, "twice_modulus should not be nullptr");
    if (x >= *twice_modulus) {
      x -= *twice_modulus;
    }
    if (x >= modulus) {
      x -= modulus;
    }
    return x;
  }
  if (InputModFactor == 8) {
    HEXL_CHECK(twice_modulus != nullptr, "twice_modulus should not be nullptr");
    HEXL_CHECK(four_times_modulus != nullptr,
               "four_times_modulus should not be nullptr");

    if (x >= *four_times_modulus) {
      x -= *four_times_modulus;
    }
    if (x >= *twice_modulus) {
      x -= *twice_modulus;
    }
    if (x >= modulus) {
      x -= modulus;
    }
    return x;
  }
  HEXL_CHECK(false, "Should be unreachable");
  return x;
}

/// @brief Returns Montgomery form of ab mod q, computed via the REDC algorithm,
/// also known as Montgomery reduction.
/// @param[in] r with R = 2^r such that gcd(R, q) = 1. R > q.
/// @param[in] mod_R_msk take r last bits to apply mod R.
/// @param[in] inv_mod q * inv_mod = -1 mod R (Hensel's lemma, see below).
template <int BitShift>
inline uint64_t MontgomeryReduce(uint64_t T_hi, uint64_t T_lo, uint64_t q,
                                 int r, uint64_t mod_R_msk, uint64_t inv_mod) {
  HEXL_CHECK(BitShift == 64 || BitShift == 52,
             "Unsupported BitShift " << BitShift);
  HEXL_CHECK((1ULL << r) > static_cast<uint64_t>(q),
             "R value should be greater than q = " << static_cast<uint64_t>(q));

  uint64_t mq_hi;
  uint64_t mq_lo;

  uint64_t m = ((T_lo & mod_R_msk) * inv_mod) & mod_R_msk;
  MultiplyUInt64(m, q, &mq_hi, &mq_lo);

  if (BitShift == 52) {
    mq_hi = (mq_hi << 12) | (mq_lo >> 52);
    mq_lo &= (1ULL << 52) - 1;
  }

  uint64_t t_hi;
  uint64_t t_lo;

  // first 64bit block
  t_lo = T_lo + mq_lo;
  unsigned int carry = static_cast<unsigned int>(t_lo < T_lo);
  t_hi = T_hi + mq_hi + carry;

  t_hi = t_hi << (BitShift - r);
  t_lo = t_lo >> r;
  t_lo = t_hi + t_lo;

  return (t_lo >= q) ? (t_lo - q) : t_lo;
}

/// @brief Hensel's Lemma for 2-adic numbers: find x such that q * x + 1 = 0
/// mod 2^r. Used to precompute the Montgomery constant.
inline uint64_t HenselLemma2adicRoot(uint32_t r, uint64_t q) {
  uint64_t a_prev = 1;
  uint64_t c = 2;
  uint64_t mod_mask = 3;

  // Root:
  //    f(x) = qX + 1 and a_(0) = 1 then f(1) ≡ 0 mod 2
  //    General Case:
  //    - a_(n) ≡ a_(n-1) mod 2^(n)
  //      => a_(n) = a_(n-1) + 2^(n)*t
  //    - Find 't' such that f(a_(n)) = 0 mod  2^(n+1)
  // First case in for:
  //    - a_(1) ≡ 1 mod 2 or a_(1) = 1 + 2t
  //    - Find 't' so f(a_(1)) ≡ 0 mod 4  => q(1 + 2t) + 1 ≡ 0 mod 4
  for (uint64_t k = 2; k <= r; k++) {
    uint64_t f = 0;
    uint64_t t = 0;
    uint64_t a = 0;

    do {
      a = a_prev + c * t++;
      f = q * a + 1ULL;
    } while (f & mod_mask);  // f(a) ≡ 0 mod 2^(k)

    // Update vars
    mod_mask = mod_mask * 2 + 1ULL;
    c *= 2;
    a_prev = a;
  }

  return a_prev;
}

}  // namespace hexl
}  // namespace intel
