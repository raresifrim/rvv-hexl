// openfhe_ntt_bench — single-thread forward-NTT latency using OpenFHE's
// ChineseRemainderTransformFTT (the negacyclic NTT its BFV/CKKS/binfhe use).
//
// Usage:  ./openfhe_ntt_bench [prime_bits] [N1 N2 N3 ...]
//   prime_bits : NTT prime bit-width (default 49). CycloOrder = 2N.
//   Ni         : ring sizes, powers of two (default: 8192 16384 32768).
//
// Reports median us/NTT over repeated in-place forward transforms, so it lines
// up directly with seal_ntt_bench for a same-machine SEAL-vs-OpenFHE compare.
#include "openfhe.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>

using namespace lbcrypto;

static int ilog2(size_t n) { int k = 0; while ((size_t(1) << k) < n) k++; return k; }

static void bench_n(size_t n, int prime_bits) {
    int logn = ilog2(n);
    usint cyclo = 2 * (usint)n;                       // cyclotomic order 2N
    NativeInteger modulus = FirstPrime<NativeInteger>(prime_bits, cyclo);
    NativeInteger root    = RootOfUnity<NativeInteger>(cyclo, modulus);

    ChineseRemainderTransformFTT<NativeVector> crt;
    crt.PreCompute(root, cyclo, modulus);             // twiddle tables, once

    std::mt19937_64 rng(12345);
    NativeVector a(n, modulus);
    for (size_t i = 0; i < n; i++) a[i] = NativeInteger(rng() % modulus.ConvertToInt());

    size_t iters = std::max<size_t>(500, (200000000ull) / (n * (size_t)logn));

    for (int i = 0; i < 50; i++)
        crt.ForwardTransformToBitReverseInPlace(root, cyclo, &a);   // warm

    std::vector<double> samples;
    for (int rep = 0; rep < 11; rep++) {
        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < iters; i++)
            crt.ForwardTransformToBitReverseInPlace(root, cyclo, &a);
        auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / iters);
    }
    std::sort(samples.begin(), samples.end());
    printf("  N=2^%-2d=%-6zu  prime=%2d-bit   %9.3f us/NTT   (%.2f ns/coeff)\n",
           logn, n, prime_bits, samples[samples.size()/2], samples[samples.size()/2]*1000.0/n);
}

int main(int argc, char **argv) {
    int prime_bits = (argc > 1) ? std::atoi(argv[1]) : 49;
    std::vector<size_t> sizes;
    for (int i = 2; i < argc; i++) sizes.push_back((size_t)std::strtoull(argv[i], nullptr, 10));
    if (sizes.empty()) sizes = { 8192, 16384, 32768 };

    printf("OpenFHE forward NTT (ChineseRemainderTransformFTT), single-thread\n");
    for (size_t n : sizes) bench_n(n, prime_bits);
    return 0;
}
