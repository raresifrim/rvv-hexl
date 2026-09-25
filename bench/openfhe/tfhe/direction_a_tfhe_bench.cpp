// direction_a_tfhe_bench.cpp
// ---------------------------------------------------------------------------
// End-to-end TFHE (OpenFHE binfhe, GINX method) secure-actuation round trip —
// STRESS variant, NO homomorphic MAC (integrity-of-computation deferred).
//
// This is the TFHE sibling of the BFV Direction-A benches (../BFV). It measures
// the boolean-circuit realisation of the power-window command loop when, instead
// of shipping single-bit flags (as the BFV note does), the end-zone ships the
// RAW 32-bit sensor measurements and the central node does the comparisons and
// the AND homomorphically.
//
// Flow (one round):
//   [end-zone]  encrypt S sensor measurements (w-bit each) -> per-bit LWE bundle
//               Schnorr-sign the whole uplink bundle
//               (+ prove the honest PID loop — cost carried over, see note below)
//   [central]   Schnorr-verify the bundle
//               per sensor: bit-serial (sensor < threshold) comparator -> 1 bit
//               AND the S predicate bits -> 1 command bit
//               Schnorr-sign the downlink result
//   [end-zone]  Schnorr-verify, decrypt the 1-bit command, (apply to motor)
//
// Measures: uplink/downlink bytes, bootstrapped-gate count, central eval time,
//   Schnorr sign/verify time, key material (BSK/KSK) size, and correctness vs a
//   plaintext oracle.
//
// PID-PROOF NOTE (requirement 3): the honest-PID-loop ADSC-SNARK proof and its
//   verification are INDEPENDENT of the confidentiality scheme (they prove the
//   control-law arithmetic, not the FHE ciphertext). Re-linking libsnark into
//   this OpenFHE binary is a heavy, separate build, so the PID prove/verify cost
//   is carried over as a constant from ../BFV/direction_a_adsc_snark_bench.cpp
//   (+38.6 ms prover / +4.6 ms verifier, 511 B proof, 32 B commitment) and folded
//   into the round-trip summary, clearly labelled. Swap in a live measurement once
//   the combined OpenFHE+libsnark build exists.
//
// Schnorr NOTE (requirement 2): implemented with a real Ed25519 signature
//   (OpenSSL EVP — a Schnorr-family signature). The BFV note used BN254 Schnorr to
//   share the SNARK curve; here the signature is decoupled from the (absent) SNARK,
//   so a standard curve is used. Sign/verify are measured over the actual bytes.
//
// Build: see CMakeLists.txt in this directory (links OpenFHE binfhe + OpenSSL).
// ---------------------------------------------------------------------------

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <random>
#include <cstdint>
#include <cstring>

#include "binfhe/binfhecontext.h"
#include "binfhe/binfhecontext-ser.h"

#include <openssl/evp.h>

#include "../common/snark_flags.hpp"   // enhanced-ADSC-SNARK flag proof (linked when HAVE_SNARK)

using namespace lbcrypto;
using Clock = std::chrono::high_resolution_clock;

static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// ---- serialization helpers (for on-wire size accounting) -------------------
template <class T>
static std::string ser(const T& obj) {
    std::stringstream ss;
    Serial::Serialize(obj, ss, SerType::BINARY);
    return ss.str();
}
template <class T>
static size_t serbytes(const T& obj) { return ser(obj).size(); }

// ---- Ed25519 (Schnorr-family) signature over the bundle bytes --------------
static EVP_PKEY* ed25519_gen() {
    EVP_PKEY* k = nullptr;
    EVP_PKEY_CTX* c = EVP_PKEY_CTX_new_id(EVP_PKEY_ED25519, nullptr);
    EVP_PKEY_keygen_init(c);
    EVP_PKEY_keygen(c, &k);
    EVP_PKEY_CTX_free(c);
    return k;
}
static std::string ed25519_sign(EVP_PKEY* k, const std::string& msg) {
    EVP_MD_CTX* md = EVP_MD_CTX_new();
    EVP_DigestSignInit(md, nullptr, nullptr, nullptr, k);
    size_t n = 0;
    EVP_DigestSign(md, nullptr, &n, (const unsigned char*)msg.data(), msg.size());
    std::string sig(n, 0);
    EVP_DigestSign(md, (unsigned char*)sig.data(), &n, (const unsigned char*)msg.data(), msg.size());
    sig.resize(n);
    EVP_MD_CTX_free(md);
    return sig;
}
static bool ed25519_verify(EVP_PKEY* k, const std::string& msg, const std::string& sig) {
    EVP_MD_CTX* md = EVP_MD_CTX_new();
    EVP_DigestVerifyInit(md, nullptr, nullptr, nullptr, k);
    int r = EVP_DigestVerify(md, (const unsigned char*)sig.data(), sig.size(),
                             (const unsigned char*)msg.data(), msg.size());
    EVP_MD_CTX_free(md);
    return r == 1;
}

// ---- bit-serial less-than comparator: encrypted a < PUBLIC constant c ------
// a is an LSB-first vector of encrypted bits; c is a public w-bit threshold known
// to the central (a safety limit). Returns an encrypted bit (a < c). No threshold
// ciphertext is needed, so no public key is involved.
// Recurrence (LSB->MSB, higher bits override), with c_i a public bit:
//   c_i==1:  a_i<c_i = NOT a_i ,  a_i==c_i = a_i     -> lt = OR(NOT a_i, AND(a_i, lt))
//   c_i==0:  a_i<c_i = 0        ,  a_i==c_i = NOT a_i -> lt = AND(NOT a_i, lt)
// Below the lowest set bit of c, lt is identically false, so we seed lt at that bit
// as NOT a_i (free) and skip the bits underneath — avoiding any encrypted constant.
// EvalNOT is a free negation (no bootstrap); each AND/OR is one gate bootstrap.
static LWECiphertext less_than_pub(BinFHEContext& cc,
                                   const std::vector<LWECiphertext>& a,
                                   uint32_t c, long& gate_count) {
    const size_t w = a.size();
    int low = -1;
    for (size_t i = 0; i < w; ++i) if ((c >> i) & 1u) { low = (int)i; break; }

    LWECiphertext lt;
    size_t start;
    if (low < 0) {
        // c == 0 over these w bits: a < c is always false -> encrypted 0 = (a0 AND NOT a0)
        LWECiphertext na0 = cc.EvalNOT(a[0]);
        lt = cc.EvalBinGate(AND, a[0], na0); gate_count++;
        start = 1;
    } else {
        lt = cc.EvalNOT(a[low]);   // c_low==1 => a_low<c_low = NOT a_low (free)
        start = (size_t)low + 1;
    }
    for (size_t i = start; i < w; ++i) {
        if ((c >> i) & 1u) {                                   // c_i == 1
            LWECiphertext t   = cc.EvalBinGate(AND, a[i], lt); gate_count++;
            LWECiphertext nai = cc.EvalNOT(a[i]);
            lt = cc.EvalBinGate(OR, nai, t);                   gate_count++;
        } else {                                               // c_i == 0
            LWECiphertext nai = cc.EvalNOT(a[i]);
            lt = cc.EvalBinGate(AND, nai, lt);                 gate_count++;
        }
    }
    return lt;
}

// ---- seeded-LWE compression -------------------------------------------------
// An LWE ct is (a, b) with b = <a,s> + e + encode(m). The mask a is uniform, so if
// both sides derive a from a shared PRNG seed, only b (one integer) needs to travel
// (+ a 32 B seed sent once for the whole bundle). The end-zone (holder of s) rewrites
// each body so it matches the seed-derived mask: b' = b + <a_seed - a_real, s>; noise
// and message are unchanged, so the rebuilt ct is a valid fresh ct and gates as usual.
// Regenerate `count` masks of dimension n mod q from `seed`, in a single stream.
static std::vector<NativeVector> gen_masks(uint64_t seed, size_t count, uint32_t n,
                                           const NativeInteger& q) {
    std::mt19937_64 g(seed);
    uint64_t qq = q.ConvertToInt();
    std::vector<NativeVector> out;
    out.reserve(count);
    for (size_t c = 0; c < count; ++c) {
        NativeVector a(n, q);
        for (uint32_t i = 0; i < n; ++i) a[i] = NativeInteger(g() % qq);
        out.push_back(std::move(a));
    }
    return out;
}
// End-zone side: turn full cts into bodies matched to the seed-derived masks.
static std::vector<NativeInteger> compress_bodies(const std::vector<LWECiphertext>& cts,
                                                  const std::vector<NativeVector>& masks,
                                                  const NativeVector& s, const NativeInteger& q) {
    const uint32_t n = s.GetLength();
    std::vector<NativeInteger> bodies(cts.size());
    for (size_t c = 0; c < cts.size(); ++c) {
        const NativeVector& ar = cts[c]->GetA();
        NativeInteger inner((uint64_t)0);
        for (uint32_t i = 0; i < n; ++i) {
            NativeInteger diff = masks[c][i].ModSub(ar[i], q);
            inner = inner.ModAdd(diff.ModMul(s[i], q), q);
        }
        bodies[c] = cts[c]->GetB().ModAdd(inner, q);
    }
    return bodies;
}
// Central side: rebuild full cts from the regenerated masks + received bodies.
static std::vector<LWECiphertext> reconstruct(const std::vector<NativeVector>& masks,
                                              const std::vector<NativeInteger>& bodies) {
    std::vector<LWECiphertext> out(bodies.size());
    for (size_t c = 0; c < bodies.size(); ++c)
        out[c] = std::make_shared<LWECiphertextImpl>(masks[c], bodies[c], NativeInteger(4));
    return out;
}


int main(int argc, char** argv) {
    // ---- args ----
    int    S      = 4;      // sensors
    int    w      = 32;     // measurement width in bits (stress mode)
    int    rounds = 1;
    unsigned seed = 12345;
    std::string paramset = "STD128";
    std::string mode = "stress";   // stress = raw w-bit compare | flags = 1-bit pre-booleanized
    bool   compress = false;       // seeded-LWE uplink compression
    bool   prove_flags = false;    // enhanced-ADSC-SNARK: prove flag = (reading < threshold)
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&](int& dst){ if (i + 1 < argc) dst = std::atoi(argv[++i]); };
        if      (a == "--sensors")  val(S);
        else if (a == "--width")    val(w);
        else if (a == "--rounds")   val(rounds);
        else if (a == "--seed")     { if (i+1<argc) seed = (unsigned)std::atoi(argv[++i]); }
        else if (a == "--paramset") { if (i+1<argc) paramset = argv[++i]; }
        else if (a == "--mode")     { if (i+1<argc) mode = argv[++i]; }
        else if (a == "--compress") compress = true;
        else if (a == "--prove-flags") prove_flags = true;
        else if (a == "--help") {
            std::cout << "usage: " << argv[0]
                      << " [--mode stress|flags] [--compress] [--prove-flags] [--sensors S] [--width w]"
                         " [--rounds R] [--paramset STD128|STD128_4|MEDIUM|TOY] [--seed N]\n"
                      << "  stress : ship raw w-bit sensors, compare < public threshold centrally\n"
                      << "  flags  : ship 1-bit pre-booleanized predicates (native TFHE), AND only\n"
                      << "  --prove-flags : enhanced-ADSC-SNARK flag proof (flags mode ONLY — in stress the\n"
                      << "                  central does the comparison, so there is no source flag to prove)\n"
                      << "  smoke  : --sensors 2 --width 8 --rounds 1\n";
            return 0;
        }
    }
    if (w < 1 || w > 32) { std::cerr << "width must be in 1..32\n"; return 1; }
    const bool flags = (mode == "flags");
    const int  wb = flags ? 1 : w;          // bit-cts per sensor on the wire

    std::cout << "=== Direction-A TFHE end-to-end bench (OpenFHE binfhe / GINX, no MAC) ===\n";
    std::cout << "mode=" << mode << (compress ? " +compress" : "")
              << "  sensors=" << S << "  width=" << (flags ? 1 : w)
              << "  rounds=" << rounds << "  paramset=" << paramset << "\n\n";

    // ---- FHE context (GINX = TFHE method; binary secret; LUT-capable) ----
    BINFHE_PARAMSET ps = STD128;
    if      (paramset == "TOY")      ps = TOY;
    else if (paramset == "STD128_4") ps = STD128_4;
    else if (paramset == "MEDIUM")   ps = MEDIUM;

    auto t0 = Clock::now();
    BinFHEContext cc;
    cc.GenerateBinFHEContext(ps, GINX);
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);                 // bootstrapping (refresh) key + key-switch key
    double setup_ms = ms(t0, Clock::now());

    size_t bsk_bytes = serbytes(cc.GetRefreshKey());   // BSK (central-side)
    size_t ksk_bytes = serbytes(cc.GetSwitchKey());    // KSK (central-side)

    // LWE geometry for seeded compression (n, q, secret s), read off a sample ct
    NativeVector  s_vec = sk->GetElement();
    auto          sample = cc.Encrypt(sk, 0);
    const uint32_t nlwe = sample->GetA().GetLength();
    const NativeInteger qlwe = sample->GetModulus();
    uint64_t qq = qlwe.ConvertToInt();
    int qbits = 0; while ((qbits < 64) && ((uint64_t(1) << qbits) < qq)) ++qbits;
    const int body_bytes = (qbits + 7) / 8;     // tight per-body wire size
    const int SEED_BYTES = 32;                  // one shared CSPRNG seed per bundle

    // ---- Schnorr (Ed25519): end-zone signs uplink, central signs downlink ----
    EVP_PKEY* ez_key = ed25519_gen();
    EVP_PKEY* cn_key = ed25519_gen();

    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> distw(0, (w >= 32) ? 0xFFFFFFFFu : ((1u << w) - 1));
    std::uniform_int_distribution<int>      distb(0, 1);

    double acc_enc = 0, acc_up_sign = 0, acc_up_verify = 0, acc_eval = 0,
           acc_dn_sign = 0, acc_dn_verify = 0, acc_dec = 0, acc_comp = 0, acc_decomp = 0;
    size_t up_bytes = 0, dn_bytes = 0, up_sig = 0, dn_sig = 0, up_comp_bytes = 0;
    long   gates_per_round = 0;
    int    correct = 0, fired = 0;

    // ---- oracle mix ----
    // With sensors AND thresholds both uniform, `expected` (the AND of S coin flips) is true
    // with probability 2^-S, so a stuck-at-false evaluation would still score rounds/rounds.
    // Mirror tfhe-rs-ipcei (EXCURSION_EVERY): every ORACLE_INHIBIT_EVERY-th round is forced to
    // inhibit, the others to fire. The thresholds keep their original RNG draws (they alone fix
    // the comparator's gate count, so timings stay comparable with earlier runs); only the
    // sensor readings / flags are steered, from a separate RNG stream.
    const int ORACLE_INHIBIT_EVERY = 3;
    std::mt19937 orng(seed ^ 0x5EED0ACEu);

    const double PID_PROVE_MS = 38.6, PID_VERIFY_MS = 4.6;   // carried, scheme-independent
    const int    PID_PROOF_B  = 511,  PID_COMMIT_B  = 32;

    for (int r = 0; r < rounds; ++r) {
        // ----- plaintext oracle -----
        std::vector<uint32_t> sensor(S), thresh(S);
        std::vector<int>      flag(S);
        for (int s = 0; s < S; ++s) {                       // original draw order, unchanged
            if (flags) flag[s] = distb(rng);
            else { sensor[s] = distw(rng); thresh[s] = distw(rng); }
        }
        const bool want_fire = ((r + 1) % ORACLE_INHIBIT_EVERY) != 0;
        const int  culprit   = r % S;                        // the one input that inhibits
        for (int s = 0; s < S; ++s) {
            if (flags) {
                flag[s] = (!want_fire && s == culprit) ? 0 : 1;
            } else {
                uint64_t t = thresh[s];
                if (!want_fire && s == culprit) {            // reading >= threshold
                    uint64_t span = (uint64_t(1) << w) - t;   // t .. 2^w-1
                    sensor[s] = (uint32_t)(t + (orng() % span));
                } else {                                     // reading < threshold
                    if (t == 0) { thresh[s] = t = 1; }       // prob. 2^-w; keeps 0 < 1 possible
                    sensor[s] = (uint32_t)(orng() % t);
                }
            }
        }
        bool expected = true;
        for (int s = 0; s < S; ++s)
            expected = expected && (flags ? (flag[s] != 0) : (sensor[s] < thresh[s]));
        if (expected) fired++;

        // ===== END-ZONE: encrypt (stress: w bits/sensor | flags: 1 bit/sensor) =====
        auto te = Clock::now();
        std::vector<std::vector<LWECiphertext>> enc(S, std::vector<LWECiphertext>(wb));
        for (int s = 0; s < S; ++s)
            for (int j = 0; j < wb; ++j) {
                uint32_t bit = flags ? (uint32_t)flag[s] : ((sensor[s] >> j) & 1u);
                enc[s][j] = cc.Encrypt(sk, bit);
            }
        acc_enc += ms(te, Clock::now());

        // flat uplink list
        std::vector<LWECiphertext> flat;
        for (int s = 0; s < S; ++s) for (int j = 0; j < wb; ++j) flat.push_back(enc[s][j]);
        const size_t Ncts = flat.size();

        // full (uncompressed) bundle bytes
        std::string bundle;
        for (auto& c : flat) bundle += ser(c);
        up_bytes = bundle.size();

        // ----- optional seeded-LWE compression -----
        std::string wire = bundle;      // what the Schnorr actually signs
        if (compress) {
            uint64_t bseed = 0x9E3779B97F4A7C15ull ^ (uint64_t)(seed + r);
            auto tcz = Clock::now();
            auto masks_tx = gen_masks(bseed, Ncts, nlwe, qlwe);
            auto bodies    = compress_bodies(flat, masks_tx, s_vec, qlwe);
            acc_comp += ms(tcz, Clock::now());
            // compact wire = 32 B seed + Ncts * body_bytes
            up_comp_bytes = (size_t)SEED_BYTES + Ncts * (size_t)body_bytes;
            wire.clear();
            wire.append((const char*)&bseed, sizeof(bseed));
            for (auto& b : bodies) { uint64_t v = b.ConvertToInt(); wire.append((const char*)&v, body_bytes); }
            // central regenerates masks + rebuilds full cts
            auto tdz = Clock::now();
            auto masks_rx = gen_masks(bseed, Ncts, nlwe, qlwe);
            auto rebuilt  = reconstruct(masks_rx, bodies);
            acc_decomp += ms(tdz, Clock::now());
            size_t k = 0;
            for (int s = 0; s < S; ++s) for (int j = 0; j < wb; ++j) enc[s][j] = rebuilt[k++];
        }

        auto ts = Clock::now();
        std::string up_signature = ed25519_sign(ez_key, wire);
        acc_up_sign += ms(ts, Clock::now());
        up_sig = up_signature.size();

        // ===== CENTRAL: verify bundle =====
        auto tv = Clock::now();
        bool up_ok = ed25519_verify(ez_key, wire, up_signature);
        acc_up_verify += ms(tv, Clock::now());
        if (!up_ok) { std::cerr << "round " << r << ": uplink Schnorr FAILED\n"; return 2; }

        // ===== CENTRAL: eval (stress: compare<thresh + AND | flags: AND only) =====
        long gates = 0;
        auto tc = Clock::now();
        std::vector<LWECiphertext> predicate(S);
        if (flags) for (int s = 0; s < S; ++s) predicate[s] = enc[s][0];
        else       for (int s = 0; s < S; ++s) predicate[s] = less_than_pub(cc, enc[s], thresh[s], gates);
        LWECiphertext command = predicate[0];
        for (int s = 1; s < S; ++s) { command = cc.EvalBinGate(AND, predicate[s], command); gates++; }
        acc_eval += ms(tc, Clock::now());
        gates_per_round = gates;

        // ===== downlink: one LWE ct + Schnorr =====
        std::string dn = ser(command);
        dn_bytes = dn.size();
        auto td = Clock::now();
        std::string dn_signature = ed25519_sign(cn_key, dn);
        acc_dn_sign += ms(td, Clock::now());
        dn_sig = dn_signature.size();

        // ===== END-ZONE: verify + decrypt =====
        auto tdv = Clock::now();
        bool dn_ok = ed25519_verify(cn_key, dn, dn_signature);
        acc_dn_verify += ms(tdv, Clock::now());
        if (!dn_ok) { std::cerr << "round " << r << ": downlink Schnorr FAILED\n"; return 3; }

        LWEPlaintext out;
        auto tdec = Clock::now();
        cc.Decrypt(sk, command, &out);
        acc_dec += ms(tdec, Clock::now());

        bool got = (out & 1);
        if (got == expected) correct++;
        std::cout << "round " << std::setw(2) << r << "  cmd=" << got
                  << " (expected " << expected << ")  gates=" << gates
                  << "  eval=" << std::fixed << std::setprecision(1) << ms(tc, Clock::now()) << " ms\n";
    }

    // ---- optional: enhanced-ADSC-SNARK flag proof (proves flag = reading<threshold) ----
    // Only meaningful in flags mode: there the SOURCE booleanizes, so it must prove each flag.
    // In stress mode the comparison is done homomorphically at the central, so there is NO
    // source-computed flag to certify — the flag proof is not part of that protocol.
    FlagProofResult fp{}; bool fp_done = false;
    if (prove_flags && !flags) {
        std::cerr << "note: --prove-flags is not part of the stress (raw-32-bit) protocol — the central\n"
                     "      does the comparison homomorphically, so there is no source flag to prove. Ignored.\n";
    } else if (prove_flags) {
#ifdef HAVE_SNARK
        fp = flag_proof_benchmark(S, w, rounds);   // w = real sensor width being compared at the source
        fp_done = (fp.rounds > 0 && fp.ok_count == fp.rounds);
        if (!fp_done) std::cerr << "warning: flag proof failed (" << fp.ok_count << "/" << fp.rounds << ")\n";
#else
        std::cerr << "warning: --prove-flags requires building with HAVE_SNARK (link snark_flags.o); ignored\n";
#endif
    }

    // ---- summary ----
    auto A = [&](double x){ return x / rounds; };
    double central_ms = A(acc_up_verify) + A(acc_eval) + A(acc_dn_sign) + PID_VERIFY_MS;
    double endzone_ms = A(acc_enc) + A(acc_comp) + A(acc_up_sign) + PID_PROVE_MS + A(acc_dn_verify) + A(acc_dec);
    if (compress) central_ms += A(acc_decomp);
    if (fp_done) { endzone_ms += fp.prove_ms; central_ms += fp.verify_ms; }
    const int NctsR = S * wb;

    std::cout << "\n=== summary (per round, averaged over " << rounds << ") ===\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "correctness            : " << correct << "/" << rounds << "\n";
    // Without the mix a correctness score of rounds/rounds is unfalsifiable (see ORACLE_INHIBIT_EVERY).
    std::cout << "oracle mix (fire/inhibit) : " << fired << "/" << (rounds - fired) << "\n";
    std::cout << "bootstrapped gates     : " << gates_per_round
              << (flags ? "  (S-1 AND, pre-booleanized flags)"
                        : "  (S x w-bit compare + S-1 AND)") << "\n\n";

    std::cout << "-- end-zone (per round) --\n";
    std::cout << "  encrypt " << NctsR << " bit(s)   : " << A(acc_enc) << " ms\n";
    if (compress)
    std::cout << "  seeded compress uplink: " << A(acc_comp) << " ms\n";
    std::cout << "  Schnorr sign uplink   : " << A(acc_up_sign)   << " ms\n";
    std::cout << "  PID prove (carried)   : " << PID_PROVE_MS     << " ms  [ADSC-SNARK, scheme-independent]\n";
    if (fp_done)
    std::cout << "  flag proof (Groth16)  : " << fp.prove_ms      << " ms  [" << S << "x" << w
              << "-bit compare, " << fp.constraints << " constraints]\n";
    std::cout << "  Schnorr verify dnlink : " << A(acc_dn_verify) << " ms\n";
    std::cout << "  decrypt command bit   : " << A(acc_dec)       << " ms\n";
    std::cout << "  end-zone total        : " << endzone_ms       << " ms\n\n";

    std::cout << "-- central (per round) --\n";
    if (compress)
    std::cout << "  seeded reconstruct    : " << A(acc_decomp)    << " ms  (regenerate masks + rebuild cts)\n";
    std::cout << "  Schnorr verify uplink : " << A(acc_up_verify) << " ms\n";
    std::cout << "  homomorphic eval      : " << A(acc_eval)      << " ms\n";
    std::cout << "  Schnorr sign dnlink   : " << A(acc_dn_sign)   << " ms\n";
    std::cout << "  PID verify (carried)  : " << PID_VERIFY_MS    << " ms  [ADSC-SNARK, scheme-independent]\n";
    if (fp_done)
    std::cout << "  flag proof verify     : " << fp.verify_ms     << " ms  [Groth16, proof " << fp.proof_bytes << " B]\n";
    std::cout << "  central total         : " << central_ms       << " ms\n";
    // CROSS-LIBRARY ROW: homomorphic gate work ALONE — excludes the Schnorr sign/verify, the
    // seeded-LWE reconstruct, the serialization, and the carried PID_VERIFY_MS constant (which is
    // measured on a different binary entirely). This is the only central number that means the
    // same thing here and in the tfhe-rs round-trips, which print an identically-labelled row.
    std::cout << "  central eval only     : " << A(acc_eval)      << " ms  [homomorphic gates only — cross-library comparable]\n\n";

    std::cout << "-- packet sizes --\n";
    std::cout << "  uplink full           : " << up_bytes << " B  (" << NctsR << " LWE cts)"
              << "   [BFV note packs to ~587 B]\n";
    if (compress)
    std::cout << "  uplink seed-compressed: " << up_comp_bytes << " B  (32 B seed + " << NctsR
              << " x " << body_bytes << " B body)  ->  " << std::setprecision(0)
              << ((double)up_bytes / (up_comp_bytes ? up_comp_bytes : 1)) << "x smaller"
              << std::setprecision(2) << "\n";
    std::cout << "  uplink Schnorr sig    : " << up_sig   << " B\n";
    std::cout << "  downlink command      : " << dn_bytes << " B  (1 LWE ct)\n";
    std::cout << "  downlink Schnorr sig  : " << dn_sig   << " B\n";
    std::cout << "  PID proof / commit    : " << PID_PROOF_B << " B / " << PID_COMMIT_B << " B  [carried]\n\n";

    std::cout << "-- one-time / central key material --\n";
    std::cout << "  LWE dim n / q-bits    : " << nlwe << " / " << qbits << "\n";
    std::cout << "  setup (keygen+BTKey)  : " << setup_ms  << " ms\n";
    std::cout << "  bootstrapping key BSK : " << (bsk_bytes / 1024.0 / 1024.0) << " MB\n";
    std::cout << "  key-switch key KSK    : " << (ksk_bytes / 1024.0 / 1024.0) << " MB\n";

    EVP_PKEY_free(ez_key);
    EVP_PKEY_free(cn_key);
    return (correct == rounds) ? 0 : 4;
}
