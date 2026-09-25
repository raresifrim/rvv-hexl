// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Compile-time proof of drop-in compatibility. Each static_cast below only
// compiles if a function with EXACTLY the upstream Intel HEXL v1.2.6 signature
// exists, so any accidental change to the public API breaks the build here
// rather than inside OpenFHE's (much slower) build.

#include <type_traits>
#include <unordered_map>

#include "hexl/hexl.hpp"
#include "test.hpp"

using namespace intel::hexl;

TEST(Api_SignaturesMatchUpstream) {
  using VV = void (*)(uint64_t*, const uint64_t*, const uint64_t*, uint64_t, uint64_t);
  using VS = void (*)(uint64_t*, const uint64_t*, uint64_t, uint64_t, uint64_t);
  (void)static_cast<VV>(&EltwiseAddMod);
  (void)static_cast<VS>(&EltwiseAddMod);
  (void)static_cast<VV>(&EltwiseSubMod);
  (void)static_cast<VS>(&EltwiseSubMod);
  (void)static_cast<void (*)(uint64_t*, const uint64_t*, const uint64_t*, uint64_t,
                             uint64_t, uint64_t)>(&EltwiseMultMod);
  (void)static_cast<void (*)(uint64_t*, const uint64_t*, uint64_t, const uint64_t*,
                             uint64_t, uint64_t, uint64_t)>(&EltwiseFMAMod);
  (void)static_cast<void (*)(uint64_t*, const uint64_t*, uint64_t, uint64_t,
                             uint64_t, uint64_t)>(&EltwiseReduceMod);
  (void)static_cast<void (*)(uint64_t*, const uint64_t*, uint64_t, CMPINT, uint64_t,
                             uint64_t)>(&EltwiseCmpAdd);
  (void)static_cast<void (*)(uint64_t*, const uint64_t*, uint64_t, uint64_t, CMPINT,
                             uint64_t, uint64_t)>(&EltwiseCmpSubMod);

  (void)static_cast<void (NTT::*)(uint64_t*, const uint64_t*, uint64_t, uint64_t)>(
      &NTT::ComputeForward);
  (void)static_cast<void (NTT::*)(uint64_t*, const uint64_t*, uint64_t, uint64_t)>(
      &NTT::ComputeInverse);
  (void)static_cast<bool (*)(uint64_t, uint64_t)>(&NTT::CheckArguments);

  (void)static_cast<uint64_t (*)(uint64_t, uint64_t)>(&InverseMod);
  (void)static_cast<uint64_t (*)(uint64_t, uint64_t, uint64_t)>(&MultiplyMod);
  (void)static_cast<uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t)>(&MultiplyMod);
  (void)static_cast<uint64_t (*)(uint64_t, uint64_t, uint64_t)>(&PowMod);
  (void)static_cast<bool (*)(uint64_t, uint64_t, uint64_t)>(&IsPrimitiveRoot);
  (void)static_cast<uint64_t (*)(uint64_t, uint64_t)>(&MinimalPrimitiveRoot);
  (void)static_cast<std::vector<uint64_t> (*)(size_t, size_t, bool, size_t)>(
      &GeneratePrimes);

  static_assert(std::is_default_constructible<NTT>::value,
                "OpenFHE's unordered_map<.., NTT>::operator[] needs NTT()");
  static_assert(std::is_move_assignable<NTT>::value,
                "OpenFHE move-assigns NTT objects into its cache");
  static_assert(std::is_constructible<NTT, uint64_t, uint64_t, uint64_t>::value,
                "OpenFHE constructs NTT(N, q, root)");
  static_assert(std::is_same<AlignedVector64<uint64_t>::value_type, uint64_t>::value,
                "OpenFHE stores NativeVector data in AlignedVector64");
  static_assert(static_cast<int>(CMPINT::NLE) == 6, "CMPINT encoding");
  static_assert(NTT::s_max_fwd_32_modulus == (1ULL << 30), "32-bit NTT bound");

  CHECK(true);
}

#ifdef HEXL_RVV_HAS_32BIT_API
// rvv-hexl extension for OpenFHE NATIVE_SIZE=32: the same functions on uint32_t
// data (only the data pointers change; scalars stay uint64_t). The patched
// openfhe-hexl overlay reaches exactly these through BasicInteger* casts.
TEST(Api_32BitExtension) {
  using VV = void (*)(uint32_t*, const uint32_t*, const uint32_t*, uint64_t, uint64_t);
  using VS = void (*)(uint32_t*, const uint32_t*, uint64_t, uint64_t, uint64_t);
  (void)static_cast<VV>(&EltwiseAddMod);
  (void)static_cast<VS>(&EltwiseAddMod);
  (void)static_cast<VV>(&EltwiseSubMod);
  (void)static_cast<VS>(&EltwiseSubMod);
  (void)static_cast<void (*)(uint32_t*, const uint32_t*, const uint32_t*, uint64_t,
                             uint64_t, uint64_t)>(&EltwiseMultMod);
  (void)static_cast<void (*)(uint32_t*, const uint32_t*, uint64_t, const uint32_t*,
                             uint64_t, uint64_t, uint64_t)>(&EltwiseFMAMod);
  (void)static_cast<void (*)(uint32_t*, const uint32_t*, uint64_t, uint64_t,
                             uint64_t, uint64_t)>(&EltwiseReduceMod);
  (void)static_cast<void (*)(uint32_t*, const uint32_t*, uint64_t, CMPINT, uint64_t,
                             uint64_t)>(&EltwiseCmpAdd);
  (void)static_cast<void (*)(uint32_t*, const uint32_t*, uint64_t, uint64_t, CMPINT,
                             uint64_t, uint64_t)>(&EltwiseCmpSubMod);
  (void)static_cast<void (NTT::*)(uint32_t*, const uint32_t*, uint64_t, uint64_t)>(
      &NTT::ComputeForward);
  (void)static_cast<void (NTT::*)(uint32_t*, const uint32_t*, uint64_t, uint64_t)>(
      &NTT::ComputeInverse);
  CHECK(true);
}
#endif
