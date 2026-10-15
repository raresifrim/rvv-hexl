# Future work

Ideas that go beyond the drop-in Intel HEXL API, kept here so they are not lost.
Each entry: the idea, why it should pay off, what it needs, and how to measure it.

## Batch the external product's digit polynomials across vector lanes (OpenFHE binfhe)

*Noted 2026-10-13.*

**Idea.** In FHEW/TFHE bootstrapping, every blind-rotation step performs one or two external
products `ACC ⊡ RGSW(s_i)`. Each external product gadget-decomposes the two accumulator
polynomials into `d` digits each and transforms all `2d` digit polynomials to NTT form, one at a
time. Those `2d` polynomials share **the same modulus Q and the same twiddles**, so instead of
one NTT per polynomial (vectorizing across the coefficients of one polynomial), run them
together with **one polynomial per vector lane**: lane `k` computes the NTT of digit polynomial `k`.

**Why it should pay off.**
- Every stage becomes full width. The small-span problem of the per-polynomial RVV NTT (spans
  `t < VLMAX` run with `vl = t`) disappears: butterfly partners sit at the same lane in two
  registers at every stage, like the transposed layout measured while designing the RVV NTT
  (October 2026, 26-30% faster than per-stage slides on the X100), but without paying for a
  transpose.
- All lanes use the same twiddle at the same point, so twiddles stay scalar (no per-lane
  twiddle tables, unlike cross-limb batching in BFV/CKKS, where every limb has its own modulus).
- binfhe is the hot path of the port: about 5,000-10,000 NTTs at N = 1024/2048, Q ~ 2^27 per
  gate, all e32 (RVV32).
- Lane count fits: `2d` = 4-8 for typical `d` = 2-4. e32 at m1 gives VL = 8 on the X100
  (VLEN 256). The A100 (VLEN 1024, VL = 32 at m1) would need more polynomials per batch, for
  example several external products or several gates in flight.

**What it needs.**
1. *rvv-hexl:* a batched NTT extension (not part of the Intel HEXL API, like the 32-bit
   storage API), e.g. forward/inverse transforms of `B` polynomials with the same (N, q), on an
   interleaved layout (coefficient `j` of polynomial `k` at `j * B + k`), plus the matching
   element-wise multiply-accumulate with the bootstrapping key.
2. *OpenFHE binfhe:* the decomposition and the per-digit NTT loop live in
   `src/binfhe/lib/rgsw-acc-cggi.cpp` (`AddToAccCGGI`: `SignedDigitDecompose(params, ct, dct)`,
   then `dct[i].SetFormat(Format::EVALUATION)` per digit), with the same pattern in
   `rgsw-acc-dm.cpp` (`AddToAccDM`) and `rgsw-acc-lmkcdey.cpp`. The decomposition produces all
   digits of one coefficient together, so it can write the interleaved layout directly, with no
   separate transpose. The bootstrapping key would be stored in the same interleaved NTT form.
3. Shipping it means a patch to OpenFHE (as a patch file under `third_party/patches/`, like
   the word-size patch for openfhe-hexl), so it is no longer a pure drop-in.

**How to measure.**
1. Standalone first: batched NTT with `B` = 4, 8, 16, 32 lanes against `B` calls of the current
   per-polynomial RVV32 kernel (N = 1024/2048, 27-bit q), on both clusters.
2. If it pays, prototype the binfhe patch and run the IPCEI TFHE benches (`tfhe stress`,
   `tfhe flags`, `tfhe lut`) against stock OpenFHE and the drop-in rvv-hexl build.

**Related.** Cross-limb batching for BFV/CKKS is the same idea with one RNS limb per lane, but
it needs per-lane moduli and twiddles and competes with OpenFHE's OpenMP parallelism over limbs;
binfhe has a single modulus, which makes it the simpler and more valuable target. Also relevant
for the RVV ISA-extension stage of D01: a batched layout changes which instructions matter.
