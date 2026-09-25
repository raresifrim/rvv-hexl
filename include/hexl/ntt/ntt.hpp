// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Public API: identical to Intel HEXL v1.2.6, plus an explicitly marked block
// of RVV-specific twiddle tables/getters (a pure superset: nothing upstream
// removed or changed). Implementation: src/ntt/.
//
// This is THE class of the port. OpenFHE's HEXL backend
// (math/hal/intnat-hexl/transformnathexl-impl.h) uses exactly this surface:
//
//   static std::unordered_map<std::pair<uint64_t,uint64_t>, intel::hexl::NTT, ..> m_IntelNtt;
//   intel::hexl::NTT ntt(N, q, psi);            // psi = primitive 2N-th root
//   m_IntelNtt[key] = std::move(ntt);           // needs NTT() + move-assign
//   p_ntt->ComputeForward(data, data, 1, 1);    // in-place, fully reduced
//   p_ntt->ComputeInverse(data, data, 1, 1);
//
// and it calls ComputeForward/ComputeInverse on the SAME object from many
// OpenMP threads at once (the pointer is used after the map lock is released).
// => both methods must be const in behaviour: no mutable scratch in the object.

#pragma once

#include <stdint.h>

#include <memory>
#include <utility>
#include <vector>

#include "hexl/util/aligned-allocator.hpp"
#include "hexl/util/allocator.hpp"
#include "hexl/util/check.hpp"

namespace intel {
namespace hexl {

/// @brief Performs negacyclic forward and inverse number-theoretic transform
/// (NTT), commonly used in RLWE cryptography.
/// @details The number-theoretic transform (NTT) specializes the discrete
/// Fourier transform (DFT) to the finite field \f$ \mathbb{Z}_q[X] / (X^N + 1)
/// \f$.
class NTT {
 public:
  /// @brief Helper class for custom memory allocation: wraps any object with
  /// allocate(size_t) / deallocate(void*, size_t) members into an AllocatorBase.
  template <class Adaptee, class... Args>
  struct AllocatorAdapter
      : public AllocatorInterface<AllocatorAdapter<Adaptee, Args...>> {
    explicit AllocatorAdapter(Adaptee&& _a, Args&&... args)
        : alloc(std::move(_a)) {
      HEXL_UNUSED(sizeof...(args));
    }
    AllocatorAdapter(const Adaptee& _a, Args&... args) : alloc(_a) {
      HEXL_UNUSED(sizeof...(args));
    }

    // interface implementation
    void* allocate_impl(size_t bytes_count) {
      return alloc.allocate(bytes_count);
    }
    void deallocate_impl(void* p, size_t n) { alloc.deallocate(p, n); }

   private:
    Adaptee alloc;
  };

  /// @brief Initializes an empty NTT object (OpenFHE needs this for
  /// std::unordered_map::operator[])
  NTT() = default;

  /// @brief Destructs the NTT object
  ~NTT();

  /// @brief Initializes an NTT object with degree \p degree and modulus \p q,
  /// using the minimal primitive 2N-th root of unity.
  /// @param[in] degree also known as N. Size of the NTT transform. Must be a
  /// power of 2
  /// @param[in] q Prime modulus. Must satisfy \f$ q == 1 \mod 2N \f$
  /// @param[in] alloc_ptr Custom memory allocator used for intermediate
  /// calculations
  /// @details Performs pre-computation necessary for forward and inverse
  /// transforms
  NTT(uint64_t degree, uint64_t q,
      std::shared_ptr<AllocatorBase> alloc_ptr = {});

  template <class Allocator, class... AllocatorArgs>
  NTT(uint64_t degree, uint64_t q, Allocator&& a, AllocatorArgs&&... args)
      : NTT(degree, q,
            std::static_pointer_cast<AllocatorBase>(
                std::make_shared<AllocatorAdapter<Allocator, AllocatorArgs...>>(
                    std::move(a), std::forward<AllocatorArgs>(args)...))) {}

  /// @brief Initializes an NTT object with degree \p degree and modulus
  /// \p q.
  /// @param[in] degree also known as N. Size of the NTT transform. Must be a
  /// power of 2
  /// @param[in] q Prime modulus. Must satisfy \f$ q == 1 \mod 2N \f$
  /// @param[in] root_of_unity 2N'th root of unity in \f$ \mathbb{Z_q} \f$.
  /// (This is the constructor OpenFHE uses; it passes its own root.)
  /// @param[in] alloc_ptr Custom memory allocator used for intermediate
  /// calculations
  /// @details  Performs pre-computation necessary for forward and inverse
  /// transforms
  NTT(uint64_t degree, uint64_t q, uint64_t root_of_unity,
      std::shared_ptr<AllocatorBase> alloc_ptr = {});

  template <class Allocator, class... AllocatorArgs>
  NTT(uint64_t degree, uint64_t q, uint64_t root_of_unity, Allocator&& a,
      AllocatorArgs&&... args)
      : NTT(degree, q, root_of_unity,
            std::static_pointer_cast<AllocatorBase>(
                std::make_shared<AllocatorAdapter<Allocator, AllocatorArgs...>>(
                    std::move(a), std::forward<AllocatorArgs>(args)...))) {}

  /// @brief Returns true if arguments satisfy constraints for negacyclic NTT
  /// @param[in] degree N. Size of the transform, i.e. the polynomial degree.
  /// Must be a power of two.
  /// @param[in] modulus Prime modulus q. Must satisfy q mod 2N = 1
  static bool CheckArguments(uint64_t degree, uint64_t modulus);

  /// @brief Compute forward NTT. Results are bit-reversed.
  /// @param[out] result Stores the result. May equal operand (in-place).
  /// @param[in] operand Data on which to compute the NTT
  /// @param[in] input_mod_factor Assume input \p operand are in [0,
  /// input_mod_factor * q). Must be 1, 2 or 4.
  /// @param[in] output_mod_factor Returns output \p result in [0,
  /// output_mod_factor * q). Must be 1 or 4.
  /// @details result[i] = sum_j operand[j] * w^((2*rev(i)+1)*j) mod q, where
  /// w = GetMinimalRootOfUnity() and rev() reverses log2(N) bits.
  void ComputeForward(uint64_t* result, const uint64_t* operand,
                      uint64_t input_mod_factor, uint64_t output_mod_factor);

  /// Compute inverse NTT. Input is bit-reversed, output in natural order and
  /// already scaled by N^{-1}, so ComputeInverse(ComputeForward(x)) == x.
  /// @param[out] result Stores the result. May equal operand (in-place).
  /// @param[in] operand Data on which to compute the NTT
  /// @param[in] input_mod_factor Assume input \p operand are in [0,
  /// input_mod_factor * q). Must be 1 or 2.
  /// @param[in] output_mod_factor Returns output \p result in [0,
  /// output_mod_factor * q). Must be 1 or 2.
  void ComputeInverse(uint64_t* result, const uint64_t* operand,
                      uint64_t input_mod_factor, uint64_t output_mod_factor);

  /// @brief Returns the minimal 2N'th root of unity
  uint64_t GetMinimalRootOfUnity() const { return m_w; }

  /// @brief Returns the degree N
  uint64_t GetDegree() const { return m_degree; }

  /// @brief Returns the word-sized prime modulus
  uint64_t GetModulus() const { return m_q; }

  /// @brief Returns the root of unity powers in bit-reversed order:
  /// GetRootOfUnityPowers()[ReverseBits(i, log2 N)] == w^i mod q
  const AlignedVector64<uint64_t>& GetRootOfUnityPowers() const {
    return m_root_of_unity_powers;
  }

  /// @brief Returns the root of unity power at bit-reversed index i.
  uint64_t GetRootOfUnityPower(size_t i) { return GetRootOfUnityPowers()[i]; }

  /// @brief Returns 32-bit pre-conditioned root of unity powers in
  /// bit-reversed order: floor(W * 2^32 / q)
  const AlignedVector64<uint64_t>& GetPrecon32RootOfUnityPowers() const {
    return m_precon32_root_of_unity_powers;
  }

  /// @brief Returns 64-bit pre-conditioned root of unity powers in
  /// bit-reversed order: floor(W * 2^64 / q)
  const AlignedVector64<uint64_t>& GetPrecon64RootOfUnityPowers() const {
    return m_precon64_root_of_unity_powers;
  }

  // The AVX512 getters are kept only for source compatibility with upstream
  // code/tests. The port leaves these tables EMPTY; use the RVV block below.
  const AlignedVector64<uint64_t>& GetAVX512RootOfUnityPowers() const {
    return m_avx512_root_of_unity_powers;
  }
  const AlignedVector64<uint64_t>& GetAVX512Precon32RootOfUnityPowers() const {
    return m_avx512_precon32_root_of_unity_powers;
  }
  const AlignedVector64<uint64_t>& GetAVX512Precon52RootOfUnityPowers() const {
    return m_avx512_precon52_root_of_unity_powers;
  }
  const AlignedVector64<uint64_t>& GetAVX512Precon64RootOfUnityPowers() const {
    return m_avx512_precon64_root_of_unity_powers;
  }

  /// @brief Returns the inverse root of unity powers, in the order consumed by
  /// the inverse transform (upstream stores them stage by stage, see
  /// ComputeRootOfUnityPowers in src/ntt/ntt.cpp)
  const AlignedVector64<uint64_t>& GetInvRootOfUnityPowers() const {
    return m_inv_root_of_unity_powers;
  }

  /// @brief Returns the inverse root of unity power at index i.
  uint64_t GetInvRootOfUnityPower(size_t i) {
    return GetInvRootOfUnityPowers()[i];
  }

  /// @brief floor(W^-1 * 2^32 / q) for the inverse powers above
  const AlignedVector64<uint64_t>& GetPrecon32InvRootOfUnityPowers() const {
    return m_precon32_inv_root_of_unity_powers;
  }

  /// @brief floor(W^-1 * 2^52 / q). IFMA-only upstream; left empty by the port.
  const AlignedVector64<uint64_t>& GetPrecon52InvRootOfUnityPowers() const {
    return m_precon52_inv_root_of_unity_powers;
  }

  /// @brief floor(W^-1 * 2^64 / q) for the inverse powers above
  const AlignedVector64<uint64_t>& GetPrecon64InvRootOfUnityPowers() const {
    return m_precon64_inv_root_of_unity_powers;
  }

  // ==========================================================================
  // rvv-hexl additions: tables for the RVV e32 path (q < 2^30)
  // ==========================================================================
  // The K3 measurements fix SEW=e32 as the design point (the e64 multiplier
  // retires 1 element/cycle). When q < s_max_fwd_32_modulus the RVV kernels
  // compute in 32-bit lanes, so their twiddles live in uint32_t tables that a
  // plain vle32 can load, halving twiddle traffic. The LAYOUT of these tables
  // is yours to choose in ComputeRootOfUnityPowers() (e.g. replicate twiddles
  // for the last stages where t < VL, like upstream's AVX512 W2/W4 trick).
  // They are left empty when q >= 2^30.

  /// @brief Forward twiddles, 32-bit, layout chosen by the RVV kernel
  const AlignedVector64<uint32_t>& GetRVV32RootOfUnityPowers() const {
    return m_rvv32_root_of_unity_powers;
  }
  /// @brief Shoup factors floor(W * 2^32 / q) matching the table above
  const AlignedVector64<uint32_t>& GetRVV32PreconRootOfUnityPowers() const {
    return m_rvv32_precon_root_of_unity_powers;
  }
  /// @brief Inverse twiddles, 32-bit, layout chosen by the RVV kernel
  const AlignedVector64<uint32_t>& GetRVV32InvRootOfUnityPowers() const {
    return m_rvv32_inv_root_of_unity_powers;
  }
  /// @brief Shoup factors floor(W^-1 * 2^32 / q) matching the table above
  const AlignedVector64<uint32_t>& GetRVV32PreconInvRootOfUnityPowers() const {
    return m_rvv32_precon_inv_root_of_unity_powers;
  }
  // ==========================================================================

  /// @brief Maximum power of 2 in degree
  static size_t MaxDegreeBits() { return 20; }

  /// @brief Maximum number of bits in modulus;
  static size_t MaxModulusBits() { return 62; }

  /// @brief Default bit shift used in Barrett precomputation
  static const size_t s_default_shift_bits{64};

  /// @brief Bit shift used in Barrett precomputation when AVX512-IFMA
  /// acceleration is enabled (kept for compatibility)
  static const size_t s_ifma_shift_bits{52};

  /// @brief Maximum modulus for the 32-bit forward path. Upstream: AVX512-DQ
  /// 32-bit. Port: the RVV e32 path. 2^30 leaves 2 bits of headroom for the
  /// lazy [0, 4q) butterfly outputs inside a 32-bit lane.
  static const size_t s_max_fwd_32_modulus{1ULL << (32 - 2)};

  /// @brief Maximum modulus for the 32-bit inverse path
  static const size_t s_max_inv_32_modulus{1ULL << (32 - 2)};

  /// @brief Maximum modulus to use AVX512-IFMA acceleration for the forward
  /// transform (kept for compatibility)
  static const size_t s_max_fwd_ifma_modulus{1ULL << (s_ifma_shift_bits - 2)};

  /// @brief Maximum modulus to use AVX512-IFMA acceleration for the inverse
  /// transform (kept for compatibility)
  static const size_t s_max_inv_ifma_modulus{1ULL << (s_ifma_shift_bits - 2)};

  /// @brief Maximum modulus to use AVX512-DQ acceleration for the inverse
  /// transform (kept for compatibility)
  static const size_t s_max_inv_dq_modulus{1ULL << (s_default_shift_bits - 2)};

  static size_t s_max_fwd_modulus(int bit_shift) {
    if (bit_shift == 32) {
      return s_max_fwd_32_modulus;
    } else if (bit_shift == 52) {
      return s_max_fwd_ifma_modulus;
    } else if (bit_shift == 64) {
      return 1ULL << MaxModulusBits();
    }
    HEXL_CHECK(false, "Invalid bit_shift " << bit_shift);
    return 0;
  }

  static size_t s_max_inv_modulus(int bit_shift) {
    if (bit_shift == 32) {
      return s_max_inv_32_modulus;
    } else if (bit_shift == 52) {
      return s_max_inv_ifma_modulus;
    } else if (bit_shift == 64) {
      return 1ULL << MaxModulusBits();
    }
    HEXL_CHECK(false, "Invalid bit_shift " << bit_shift);
    return 0;
  }

 private:
  /// @brief Fills every twiddle table below from m_w / m_q / m_degree.
  /// TODO(port): implement in src/ntt/ntt.cpp.
  void ComputeRootOfUnityPowers();

  uint64_t m_degree;  // N: size of NTT transform, should be power of 2
  uint64_t m_q;       // prime modulus. Must satisfy q == 1 mod 2n

  uint64_t m_degree_bits;  // log_2(m_degree)

  uint64_t m_w_inv;  // Inverse of minimal root of unity
  uint64_t m_w;      // A 2N'th root of unity

  std::shared_ptr<AllocatorBase> m_alloc;

  AlignedAllocator<uint64_t, 64> m_aligned_alloc;

  // powers of the minimal root of unity, bit-reversed
  AlignedVector64<uint64_t> m_root_of_unity_powers;
  // vector of floor(W * 2**32 / m_q), with W the root of unity powers
  AlignedVector64<uint64_t> m_precon32_root_of_unity_powers;
  // vector of floor(W * 2**64 / m_q), with W the root of unity powers
  AlignedVector64<uint64_t> m_precon64_root_of_unity_powers;

  // upstream AVX512 layouts: intentionally left empty by the port
  AlignedVector64<uint64_t> m_avx512_root_of_unity_powers;
  AlignedVector64<uint64_t> m_avx512_precon32_root_of_unity_powers;
  AlignedVector64<uint64_t> m_avx512_precon52_root_of_unity_powers;
  AlignedVector64<uint64_t> m_avx512_precon64_root_of_unity_powers;

  // vector of floor(W * 2**32 / m_q), with W the inverse root of unity powers
  AlignedVector64<uint64_t> m_precon32_inv_root_of_unity_powers;
  // vector of floor(W * 2**52 / m_q): IFMA-only, left empty
  AlignedVector64<uint64_t> m_precon52_inv_root_of_unity_powers;
  // vector of floor(W * 2**64 / m_q), with W the inverse root of unity powers
  AlignedVector64<uint64_t> m_precon64_inv_root_of_unity_powers;

  AlignedVector64<uint64_t> m_inv_root_of_unity_powers;

  // rvv-hexl: RVV e32-path tables (see the public getters above)
  AlignedVector64<uint32_t> m_rvv32_root_of_unity_powers;
  AlignedVector64<uint32_t> m_rvv32_precon_root_of_unity_powers;
  AlignedVector64<uint32_t> m_rvv32_inv_root_of_unity_powers;
  AlignedVector64<uint32_t> m_rvv32_precon_inv_root_of_unity_powers;
};

}  // namespace hexl
}  // namespace intel
