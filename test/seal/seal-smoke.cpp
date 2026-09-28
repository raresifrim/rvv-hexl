// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// seal-smoke: end-to-end correctness check for one SEAL build (`make seal-check`).
//
// With SEAL_USE_INTEL_HEXL, SEAL sends every call below to HEXL (i.e. to rvv-hexl),
// with no per-call fallback (see docs/PORTING_GUIDE.md, "SEAL"):
//   NTT forward (lazy 4->4, 4->1) and inverse (2->2, 2->1)   seal/util/ntt.cpp
//   EltwiseAddMod / EltwiseSubMod (vector and scalar)         seal/util/polyarithsmallmod.cpp
//   EltwiseMultMod (input factor 4), EltwiseFMAMod (nullptr addend, input factor 8),
//   EltwiseReduceMod (input factor = modulus)
// The same program runs against the stock SEAL build, so a result that differs between
// the two builds is the backend's fault. BFV is checked exactly, CKKS within 1e-3.
//
// Exit status: 0 all PASS, 1 a FAIL, 2 an rvv-hexl stub was reached (TODO; `make seal-check`
// reports it as not a failure unless STRICT=1, like `make rvv-hexl-test`).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "seal/seal.h"

#if defined(SEAL_USE_INTEL_HEXL) && __has_include("hexl/rvv/port-info.hpp")
#include "hexl/rvv/port-info.hpp"
#define SMOKE_RVV_HEXL 1
#endif

using namespace seal;

namespace {

int failures = 0;

void Report(const std::string& name, bool ok, const std::string& detail = "") {
  std::cout << (ok ? "  PASS  " : "  FAIL  ") << name << (detail.empty() ? "" : "  (" + detail + ")")
            << "\n";
  if (!ok) ++failures;
}

// BFV, batching: every Evaluator path that reaches the HEXL-backed primitives.
void BfvRoundTrip() {
  EncryptionParameters parms(scheme_type::bfv);
  const size_t n = 8192;
  parms.set_poly_modulus_degree(n);
  parms.set_coeff_modulus(CoeffModulus::BFVDefault(n));  // 5 primes, 43-44 bits: the e64 path
  parms.set_plain_modulus(PlainModulus::Batching(n, 20));
  SEALContext ctx(parms);
  const uint64_t t = parms.plain_modulus().value();

  KeyGenerator keygen(ctx);
  PublicKey pk;
  keygen.create_public_key(pk);
  RelinKeys rk;
  keygen.create_relin_keys(rk);
  GaloisKeys gk;
  keygen.create_galois_keys(gk);
  Encryptor encryptor(ctx, pk);
  Decryptor decryptor(ctx, keygen.secret_key());
  Evaluator ev(ctx);
  BatchEncoder be(ctx);
  const size_t slots = be.slot_count(), row = slots / 2;

  std::mt19937_64 rng(1234);
  std::vector<uint64_t> a(slots), b(slots), p(slots);
  for (size_t i = 0; i < slots; ++i) {
    a[i] = rng() % t;
    b[i] = rng() % t;
    p[i] = rng() % t;
  }

  // expected: rotate_rows(-( ((a*b + a - b) + p) * p ), 1), all mod t
  std::vector<uint64_t> r(slots), want(slots);
  for (size_t i = 0; i < slots; ++i) {
    unsigned __int128 v = (unsigned __int128)a[i] * b[i] % t;
    v = (v + a[i]) % t;
    v = (v + t - b[i]) % t;
    v = (v + p[i]) % t;
    v = v * p[i] % t;
    r[i] = (uint64_t)((t - v) % t);
  }
  for (size_t i = 0; i < slots; ++i) {
    size_t base = (i / row) * row, j = i % row;
    want[i] = r[base + (j + 1) % row];
  }

  Plaintext pa, pb, pp;
  be.encode(a, pa);
  be.encode(b, pb);
  be.encode(p, pp);
  Ciphertext ca, cb, c;
  encryptor.encrypt(pa, ca);  // NTT + FMA (public-key encryption)
  encryptor.encrypt(pb, cb);

  ev.multiply(ca, cb, c);  // BEHZ base conversion: NTTs, FMAMod, MultMod, Add/Sub
  ev.relinearize_inplace(c, rk);
  ev.add_inplace(c, ca);  // EltwiseAddMod
  ev.sub_inplace(c, cb);  // EltwiseSubMod
  ev.add_plain_inplace(c, pp);
  ev.multiply_plain_inplace(c, pp);
  ev.negate_inplace(c);
  ev.rotate_rows_inplace(c, 1, gk);  // key switching: NTTs, MultMod, ReduceMod
  ev.mod_switch_to_next_inplace(c);

  Plaintext out;
  decryptor.decrypt(c, out);
  std::vector<uint64_t> got;
  be.decode(out, got);
  size_t bad = 0;
  for (size_t i = 0; i < slots; ++i) bad += (got[i] != want[i]);
  Report("BFV n=8192, 5x~44-bit q: mul/relin/add/sub/plain ops/negate/rotate/mod-switch", bad == 0,
         std::to_string(bad) + "/" + std::to_string(slots) + " slots wrong, noise budget left " +
             std::to_string(decryptor.invariant_noise_budget(c)) + " bits");
}

// CKKS: rescale (NTT + multiply_poly_scalar) and vector rotation.
void CkksRoundTrip() {
  EncryptionParameters parms(scheme_type::ckks);
  const size_t n = 8192;
  parms.set_poly_modulus_degree(n);
  parms.set_coeff_modulus(CoeffModulus::Create(n, {60, 40, 40, 60}));
  SEALContext ctx(parms);
  const double scale = std::pow(2.0, 40);

  KeyGenerator keygen(ctx);
  PublicKey pk;
  keygen.create_public_key(pk);
  RelinKeys rk;
  keygen.create_relin_keys(rk);
  GaloisKeys gk;
  keygen.create_galois_keys(gk);
  Encryptor encryptor(ctx, pk);
  Decryptor decryptor(ctx, keygen.secret_key());
  Evaluator ev(ctx);
  CKKSEncoder enc(ctx);
  const size_t slots = enc.slot_count();

  std::mt19937_64 rng(99);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> x(slots), y(slots);
  for (size_t i = 0; i < slots; ++i) {
    x[i] = u(rng);
    y[i] = u(rng);
  }

  Plaintext px, py;
  enc.encode(x, scale, px);
  enc.encode(y, scale, py);
  Ciphertext cx, cy, c;
  encryptor.encrypt(px, cx);
  encryptor.encrypt(py, cy);
  ev.multiply(cx, cy, c);
  ev.relinearize_inplace(c, rk);
  ev.rescale_to_next_inplace(c);
  ev.rotate_vector_inplace(c, 1, gk);

  Plaintext out;
  decryptor.decrypt(c, out);
  std::vector<double> got;
  enc.decode(out, got);
  double err = 0;
  for (size_t i = 0; i < slots; ++i) err = std::max(err, std::fabs(got[i] - x[(i + 1) % slots] * y[(i + 1) % slots]));
  char buf[64];
  std::snprintf(buf, sizeof(buf), "max abs error %.2e", err);
  Report("CKKS n=8192, {60,40,40,60}: mul/relin/rescale/rotate", err < 1e-3, buf);
}

}  // namespace

int main() {
  std::cout << "seal-smoke  SEAL " << SEAL_VERSION << ", backend: "
#ifdef SEAL_USE_INTEL_HEXL
#ifdef SMOKE_RVV_HEXL
            << "HEXL = rvv-hexl";
  const auto info = intel::hexl::rvv::GetPortInfo();
  std::cout << " (RVV compiled " << info.compiled_with_rvv << ", enabled " << info.rvv_enabled
            << ", VLEN " << info.vlen_bits << ")";
#else
            << "HEXL (not rvv-hexl)";
#endif
#else
            << "stock (no HEXL)";
#endif
  std::cout << "\n";

  try {
    BfvRoundTrip();
    CkksRoundTrip();
  } catch (const std::exception& e) {
    if (std::strstr(e.what(), "[rvv-hexl TODO]")) {
      std::cout << "  TODO  " << e.what() << "\n";
      return 2;
    }
    std::cout << "  FAIL  exception: " << e.what() << "\n";
    return 1;
  }
  std::cout << (failures ? "FAILED" : "all PASS") << "\n";
  return failures ? 1 : 0;
}
