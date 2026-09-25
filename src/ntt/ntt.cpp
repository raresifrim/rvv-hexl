// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// The NTT class: construction (twiddle precomputation) and dispatch.
// Constructors, CheckArguments and the Compute* dispatch are boilerplate and
// done. ComputeRootOfUnityPowers() is the part you write.

#include "hexl/ntt/ntt.hpp"

#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "hexl/util/check.hpp"
#include "ntt/ntt-internal.hpp"
#include "util/cpu-features.hpp"
#include "util/not-implemented.hpp"

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
  // TODO(port): fill the twiddle tables. Runs once per (N, q), so clarity
  // beats speed. Inputs: m_degree (N), m_degree_bits, m_q, m_w, m_w_inv.
  //
  // REQUIRED by the native path (and by GetRootOfUnityPowers() users):
  //   m_root_of_unity_powers[ReverseBits(i, m_degree_bits)] = w^i mod q,
  //     for i in [0, N)  (so [0] = 1, [1] = w^(N/2), ...)
  //   m_precon64_root_of_unity_powers[k] = MultiplyFactor(W_k, 64, q).BarrettFactor()
  //   m_precon32_root_of_unity_powers[k] = MultiplyFactor(W_k, 32, q).BarrettFactor()
  //     (W_k = m_root_of_unity_powers[k]; the 32-bit one only matters for
  //      q < 2^32, compute it anyway for API compatibility when it fits)
  //
  //   m_inv_root_of_unity_powers: inverse twiddles in the order YOUR inverse
  //     kernel consumes them. Upstream: let inv[k] = InverseMod(W_k, q) in the
  //     same bit-reversed indexing, then store them stage by stage,
  //     for (m = N/2; m > 0; m >>= 1) for (i = 0; i < m; i++) push inv[m + i]
  //     (index 0 holds inv[0] = 1), so the Gentleman-Sande loop reads the
  //     table sequentially.
  //   m_precon64_inv_root_of_unity_powers / m_precon32_...: Shoup factors of
  //     the table above, same order.
  //
  // RVV e32 path, only when m_q < s_max_fwd_32_modulus:
  //   m_rvv32_root_of_unity_powers / m_rvv32_precon_root_of_unity_powers
  //   m_rvv32_inv_root_of_unity_powers / m_rvv32_precon_inv_root_of_unity_powers
  //   uint32_t copies of the tables above IN THE LAYOUT YOUR RVV KERNEL WANTS.
  //   The layout is the main design lever for the last log2(VL) stages,
  //   where the butterfly span t is smaller than the vector length: e.g.
  //   replicate each twiddle t times so one unit-stride vle32 feeds a whole
  //   register of butterflies (upstream AVX512 does exactly this for t = 1, 2:
  //   see its "W2_roots / W4_roots" trick). Leave them empty for q >= 2^30.
  //
  // Leave the m_avx512_* and m_precon52_* tables empty.
  HEXL_NOT_IMPLEMENTED();
}

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
    if (m_q < s_max_fwd_32_modulus) {
      HEXL_VLOG(3, "Calling ForwardTransformToBitReverseRVV32");
      ForwardTransformToBitReverseRVV32<uint64_t>(
          result, operand, m_degree, m_q, m_rvv32_root_of_unity_powers.data(),
          m_rvv32_precon_root_of_unity_powers.data(), input_mod_factor,
          output_mod_factor);
    } else {
      HEXL_VLOG(3, "Calling ForwardTransformToBitReverseRVV64");
      ForwardTransformToBitReverseRVV64(
          result, operand, m_degree, m_q, m_root_of_unity_powers.data(),
          m_precon64_root_of_unity_powers.data(), input_mod_factor,
          output_mod_factor);
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
    if (m_q < s_max_inv_32_modulus) {
      HEXL_VLOG(3, "Calling InverseTransformFromBitReverseRVV32");
      InverseTransformFromBitReverseRVV32<uint64_t>(
          result, operand, m_degree, m_q,
          m_rvv32_inv_root_of_unity_powers.data(),
          m_rvv32_precon_inv_root_of_unity_powers.data(), input_mod_factor,
          output_mod_factor);
    } else {
      HEXL_VLOG(3, "Calling InverseTransformFromBitReverseRVV64");
      InverseTransformFromBitReverseRVV64(
          result, operand, m_degree, m_q, m_inv_root_of_unity_powers.data(),
          m_precon64_inv_root_of_unity_powers.data(), input_mod_factor,
          output_mod_factor);
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
  HEXL_CHECK(input_mod_factor * m_q <= (1ULL << 32) &&
                 output_mod_factor * m_q <= (1ULL << 32),
             "32-bit storage needs mod_factor * q <= 2^32, q = " << m_q);
  HEXL_CHECK_BOUNDS(
      operand, m_degree, m_q * input_mod_factor,
      "value in operand exceeds bound " << m_q * input_mod_factor);

#ifdef HEXL_HAS_RVV
  if (has_rvv && m_q < s_max_fwd_32_modulus) {
    HEXL_VLOG(3, "Calling ForwardTransformToBitReverseRVV32<uint32_t>");
    ForwardTransformToBitReverseRVV32<uint32_t>(
        result, operand, m_degree, m_q, m_rvv32_root_of_unity_powers.data(),
        m_rvv32_precon_root_of_unity_powers.data(), input_mod_factor,
        output_mod_factor);
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
  HEXL_CHECK(input_mod_factor * m_q <= (1ULL << 32) &&
                 output_mod_factor * m_q <= (1ULL << 32),
             "32-bit storage needs mod_factor * q <= 2^32, q = " << m_q);
  HEXL_CHECK_BOUNDS(operand, m_degree, m_q * input_mod_factor,
                    "operand exceeds bound " << m_q * input_mod_factor);

#ifdef HEXL_HAS_RVV
  if (has_rvv && m_q < s_max_inv_32_modulus) {
    HEXL_VLOG(3, "Calling InverseTransformFromBitReverseRVV32<uint32_t>");
    InverseTransformFromBitReverseRVV32<uint32_t>(
        result, operand, m_degree, m_q,
        m_rvv32_inv_root_of_unity_powers.data(),
        m_rvv32_precon_inv_root_of_unity_powers.data(), input_mod_factor,
        output_mod_factor);
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
