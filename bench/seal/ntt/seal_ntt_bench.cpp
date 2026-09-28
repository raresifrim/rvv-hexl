// seal_ntt_bench — single-thread forward-NTT latency using SEAL's
// ntt_negacyclic_harvey (the exact primitive HEXL/HEAX accelerate).
//
// Usage:  ./seal_ntt_bench [prime_bits] [N1 N2 N3 ...]
//   prime_bits : coefficient-modulus bit-width (default 49; NTT cost is nearly
//                width-independent since ops are 64-bit either way).
//   Ni         : ring sizes, powers of two (default: 8192 16384 32768).
//
// Reports median us/NTT over repeated in-place forward transforms.
#include "seal/seal.h"
#include "seal/util/ntt.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>

using namespace seal;
using namespace seal::util;

static int ilog2(size_t n) { int k = 0; while ((size_t(1) << k) < n) k++; return k; }

static void bench_n(size_t n, int prime_bits) {
    int logn = ilog2(n);
    auto mods = CoeffModulus::Create(n, { prime_bits });
    Modulus p = mods[0];
    auto pool = MemoryManager::GetPool();
    NTTTables tables(logn, p, pool);

    std::mt19937_64 rng(12345);
    std::vector<uint64_t> a(n);
    for (auto &x : a) x = rng() % p.value();

    // scale iteration count so total work is roughly constant across sizes
    size_t iters = std::max<size_t>(500, (200000000ull) / (n * (size_t)logn));

    for (int i = 0; i < 50; i++) ntt_negacyclic_harvey(CoeffIter(a.data()), tables); // warm

    std::vector<double> samples;
    for (int rep = 0; rep < 11; rep++) {
        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < iters; i++)
            ntt_negacyclic_harvey(CoeffIter(a.data()), tables);
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

    printf("SEAL forward NTT (ntt_negacyclic_harvey), single-thread\n");
    for (size_t n : sizes) bench_n(n, prime_bits);
    return 0;
}
