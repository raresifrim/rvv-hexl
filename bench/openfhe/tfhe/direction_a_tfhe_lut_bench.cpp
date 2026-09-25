// direction_a_tfhe_lut_bench.cpp
// ---------------------------------------------------------------------------
// §5a MIDDLE PATH — compare a w-bit sensor reading against a public threshold
// with a functional bootstrap, instead of the bit-serial 32-bit comparator
// (~44 gate-bootstraps/sensor in the stress bench).
//
// Two facts this bench establishes on OpenFHE binfhe:
//   1. a SINGLE programmable bootstrap (EvalFunc/LUT) only spans the plaintext
//      space GetMaxPlaintextSpace() = 8 (3 bits) for the 128-bit sets — a larger
//      ring buys SECURITY, not precision. So >3-bit values need decomposition.
//   2. the natural multi-bit comparison is EvalSign(reading - threshold): it
//      handles a large plaintext space p = maxP * (Q/q) via ~logQ/3 internal
//      PBS, so its cost grows with the bit width. This bench measures that for
//      8 / 16 / 32-bit at STD128 (N=2048) and STD128Q_4 (N=4096).
//
// Comparison encoding: with EvalSign returning 1 iff the message >= p/2, we form
//   m = p/2 + threshold - reading   (homomorphically: negate ct, add (p/2+thr)*Delta)
// so m >= p/2  <=>  reading < threshold  =>  flag = EvalSign(m). Requires p > 2^(w+1).
// ---------------------------------------------------------------------------
#include <cstdint>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "binfhe/binfhecontext.h"

using namespace lbcrypto;
using Clock = std::chrono::high_resolution_clock;
static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// LUT function pointer (no capture) for the single-PBS path: flag = (reading < threshold).
static uint64_t g_threshold = 0;
static NativeInteger lessThanLUT(NativeInteger m, NativeInteger p) {
    return (m.ConvertToInt() < g_threshold) ? NativeInteger(1) : NativeInteger(0);
}

// Homomorphically form  m = p/2 + threshold - reading  from ct(reading) (large-Q LWE).
static LWECiphertext shiftForCompare(const LWECiphertext& ct, uint64_t threshold,
                                     uint64_t p, uint64_t Delta) {
    NativeInteger Q = ct->GetModulus();
    NativeVector a = ct->GetA();
    for (uint32_t i = 0; i < a.GetLength(); ++i) a[i] = Q.ModSub(a[i], Q);   // negate mask
    NativeInteger b = Q.ModSub(ct->GetB(), Q);                              // negate body
    uint64_t k = p/2 + threshold;                                          // public constant
    b = b.ModAdd(NativeInteger(k).ModMul(NativeInteger(Delta), Q), Q);
    return std::make_shared<LWECiphertextImpl>(a, b, ct->GetptModulus());
}

int main(int argc, char** argv) {
    int S = 5, w = 8, rounds = 3, logQ = 0;
    std::string paramset = "STD128";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--sensors"  && i+1 < argc) S = std::atoi(argv[++i]);
        else if (a == "--width"    && i+1 < argc) w = std::atoi(argv[++i]);
        else if (a == "--rounds"   && i+1 < argc) rounds = std::atoi(argv[++i]);
        else if (a == "--logq"     && i+1 < argc) logQ = std::atoi(argv[++i]);
        else if (a == "--paramset" && i+1 < argc) paramset = argv[++i];
        else if (a == "--help") {
            std::cout << "usage: " << argv[0] << " [--paramset STD128|STD128Q_4|...]"
                         " [--width w] [--sensors S] [--rounds R] [--logq N]\n"
                         "  default logQ = w+10 (so plaintext space p > 2^(w+1) for the sign compare)\n";
            return 0;
        }
    }
    BINFHE_PARAMSET ps = STD128;
    if      (paramset == "STD128Q_4") ps = STD128Q_4;
    else if (paramset == "STD128Q_3") ps = STD128Q_3;
    else if (paramset == "STD128_3")  ps = STD128_3;
    else if (paramset == "STD128_4")  ps = STD128_4;
    if (logQ == 0) logQ = w + 10;                          // p = 2^(logQ-9) > 2^(w+1)

    std::cout << "=== TFHE §5a middle path — EvalSign w-bit comparison (functional bootstrap) ===\n";
    std::cout << "paramset=" << paramset << "  width=" << w << "-bit  sensors=" << S
              << "  logQ=" << logQ << "  rounds=" << rounds << "\n\n";

    auto t0 = Clock::now();
    BinFHEContext cc;
    // Single PBS (EvalFunc LUT) only reaches ~3-bit plaintext; wider values use EvalSign.
    const bool single = (w <= 3);
    if (single) cc.GenerateBinFHEContext(ps, true, 12);                        // arbFunc small-precision LUT
    else        cc.GenerateBinFHEContext(ps, false, (uint32_t)logQ, 0, GINX, false); // large-precision sign
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);
    double setup_ms = ms(t0, Clock::now());

    uint64_t Q      = (uint64_t)1 << logQ;
    uint64_t qsmall = 4096;
    uint64_t maxP   = cc.GetMaxPlaintextSpace().ConvertToInt();
    uint64_t factor = single ? 1 : Q / qsmall;
    uint64_t p      = single ? maxP : maxP * factor;             // effective plaintext space
    uint64_t Delta  = single ? 0 : Q / p;
    int      pbits  = (int)std::floor(std::log2((double)p));

    std::cout << "-- capacity --\n";
    std::cout << "  single-PBS plaintext (GetMaxPlaintextSpace) : " << maxP << "  (" << (int)std::log2((double)maxP) << " bits)\n";
    if (!single)
    std::cout << "  EvalSign plaintext space p                  : " << p << "  (" << pbits << " bits) via ~" << (logQ-12)/3+1 << " internal PBS\n";
    std::cout << "  mode                                        : " << (single ? "single EvalFunc PBS" : "EvalSign (multi-PBS)") << "\n";
    std::cout << "  setup (keygen+BTKey)                        : " << std::fixed << std::setprecision(1) << setup_ms << " ms\n\n";

    if (!single && ((uint64_t)1 << (w+1)) > p) {
        std::cout << "==> " << w << "-bit compare needs p>2^" << (w+1) << " = " << ((uint64_t)1<<(w+1))
                  << ", but p=" << p << ". Raise --logq (>= " << (w+10) << ", cap 29 at STD128) or quantize down.\n";
        return 0;
    }
    if (single && ((uint64_t)1 << w) > maxP) {
        std::cout << "==> single-PBS holds only " << maxP << " values; " << w << "-bit needs " << ((uint64_t)1<<w) << ".\n";
        return 0;
    }

    std::mt19937 rng(1234);
    std::uniform_int_distribution<uint32_t> dist(0, (w >= 32) ? 0xFFFFFFFFu : ((1u << w) - 1));

    double acc_cmp = 0; long cmp_count = 0; int flag_ok = 0, flag_tot = 0;
    for (int r = 0; r < rounds; ++r) {
        for (int s = 0; s < S; ++s) {
            uint64_t reading = dist(rng), threshold = dist(rng);
            LWECiphertext flag;
            double dt;
            if (single) {                                    // one EvalFunc PBS: flag = LUT(reading)
                g_threshold = threshold;
                auto lut = cc.GenerateLUTviaFunction(lessThanLUT, maxP);
                auto ct  = cc.Encrypt(sk, reading % maxP, LARGE_DIM, maxP);
                auto tc = Clock::now();
                flag = cc.EvalFunc(ct, lut);
                dt = ms(tc, Clock::now());
                LWEPlaintext fv; cc.Decrypt(sk, flag, &fv, maxP);
                flag_tot++; if (((int)fv & 1) == (int)(reading < threshold)) flag_ok++;
            } else {                                         // EvalSign(reading - threshold)
                auto ct = cc.Encrypt(sk, reading, LARGE_DIM, p, Q);
                auto m  = shiftForCompare(ct, threshold, p, Delta);
                auto tc = Clock::now();
                flag = cc.EvalSign(m);
                dt = ms(tc, Clock::now());
                LWEPlaintext fv; cc.Decrypt(sk, flag, &fv, 2);
                flag_tot++; if (((int)fv & 1) == (int)(reading < threshold)) flag_ok++;
            }
            acc_cmp += dt; cmp_count++;
        }
        std::cout << "round " << std::setw(2) << r << "  flags " << flag_ok << "/" << flag_tot << " correct so far\n";
    }

    double per = acc_cmp / cmp_count;
    std::cout << "\n=== summary (" << paramset << ", " << w << "-bit, " << S << " sensors) ===\n";
    std::cout << "flag correctness            : " << flag_ok << "/" << flag_tot << "\n";
    std::cout << "comparison / sensor         : " << std::setprecision(1) << per << " ms   ("
              << (single ? "1 EvalFunc PBS" : "EvalSign multi-PBS") << ")\n";
    std::cout << "central compares / round    : " << per*S << " ms   (" << S << " x 1 EvalSign)\n";
    std::cout << "vs bit-serial 32-bit stress : ~2130 ms/sensor  (~44 gate-bootstraps)\n";
    return (flag_ok == flag_tot) ? 0 : 4;
}
