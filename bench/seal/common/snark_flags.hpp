// snark_flags.hpp — reusable flag-comparison Groth16 proof (BN254).
//
// Proves flag_i = (reading_i < threshold_i) for S sensors of w bits, reading_i a
// range-checked private witness (see snark_flags.cpp). This is the "enhanced
// ADSC-SNARK" flag-honesty step; it is scheme-independent, so the SAME object is
// linked into both the BFV (SEAL) and TFHE (OpenFHE) round-trip benches to fold
// its prover (end-zone) / verifier (central) cost into each whole-node total.
//
// This header deliberately exposes NO libsnark types, so the FHE benches (which
// compile at -std=c++17) can include it while snark_flags.cpp is compiled at
// -std=c++14 (libfqfft's xgcd.tcc uses std::bind1st, removed in C++17).
#pragma once
#include <cstddef>

struct FlagProofResult {
    long   constraints;
    int    public_inputs;
    double setup_ms;     // one-time Groth16 setup
    double prove_ms;     // per-round, averaged  -> END-ZONE impact
    double verify_ms;    // per-round, averaged  -> CENTRAL impact
    size_t proof_bytes;
    size_t vk_bytes;
    int    ok_count;
    int    rounds;
};

// Build the S x w-bit comparison circuit, run Groth16 setup once, then prove+verify
// `rounds` times with fresh random honest witnesses. Safe to call once per process
// (public-params init is idempotent-guarded internally).
FlagProofResult flag_proof_benchmark(int sensors, int width, int rounds);
