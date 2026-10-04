# rvv-hexl porting guide

What every class and interface is for, what is already done, and what you have to write.
Read sections 1 and 2 first. Section 3 is the reference, section 4 the suggested order of
work, and sections 5 and 6 cover RVV design and methodology. Section 7 covers the second
consumer, Microsoft SEAL, which uses the same library unchanged.

---

## 1. How the port plugs into OpenFHE

OpenFHE does not call HEXL directly. The `openfhe-hexl` overlay replaces OpenFHE's native math
HAL (`math/hal/intnat-hexl/*`, `lattice/hal/hexl/*`) with versions that forward hot loops to
the `intel::hexl` API. That API is the only thing OpenFHE sees, so a library with **the same
headers, same namespace, same signatures and same semantics** is a drop-in replacement:

```
            OpenFHE v1.5.1 + openfhe-hexl v1.5.1.0   (make openfhe WITH_RVV_HEXL=ON)
                   │   #include "hexl/hexl.hpp"
                   │   find_package(HEXL 1.2.6)  ← cmake/HEXLConfig.cmake from `make install`
                   ▼
   include/hexl/*  public API ─────────────── identical to Intel HEXL 1.2.6         [DONE]
                   │
   src/*/<op>.cpp  argument checks + dispatch ─ HEXL_HAS_RVV && has_rvv ?          [DONE]
                   │                                    │
                   ▼                                    ▼
          <Op>Native (portable C++)          <Op>RVV / RVV32 / RVV64 (intrinsics)  [TODO]
          = "no RVV" path, = reference       = the port proper
```

### What OpenFHE actually calls

Measured by grepping openfhe-hexl v1.5.1.0, **including** its compile-time switches in
`math/math-hal.h`: `HEXL_MUL_ENABLE 1`, `HEXL_ADD_ENABLE 0`. So every HEXL call guarded by the
add switch is compiled out, and OpenFHE never reaches `EltwiseAddMod`. What is live:

| HEXL API | OpenFHE call site | Weight in our workloads |
|---|---|---|
| `NTT(N, q, root)`, `ComputeForward(d, d, 1, 1)`, `ComputeInverse(d, d, 1, 1)` | `transformnathexl-impl.h`: every forward/inverse transform | **~44% of a binfhe bootstrap, ~80% (with modmul) of the central node** (D01) |
| `EltwiseMultMod(r, a, b, n, q, 1)` | `NativeVector::ModMul`, `DCRTPoly::operator*` | #2 hotspot (NTT-domain products) |
| `EltwiseFMAMod(r, a, s, c\|nullptr, n, q, 1)` | scalar `ModMul`, RNS basis switching (`hexldcrtpoly-impl.h`) | BFV rescale / key switching |
| `EltwiseCmpAdd`, `EltwiseCmpSubMod` | `NativeVector::SwitchModulus` | modulus switching (binfhe, BFV) |
| `EltwiseReduceMod(r, a, n, q, 1, 1)` | `NativeVector::Mod` | occasional |
| `AlignedVector64<T>` | storage of **every** `NativeVector` | type only |
| `CMPINT` | argument of the two Cmp* calls | type only |

`EltwiseAddMod` (disabled by `HEXL_ADD_ENABLE 0`) and `EltwiseSubMod` (never called) are not
reached by OpenFHE's backend, but they are part of the API and **SEAL calls both, in both the
vector and the scalar form** (section 7). They are cheap to write and make the port a complete
HEXL 1.2.6.

### Three invariants OpenFHE imposes

1. **In place.** Almost every call has `result == operand1` (and FMA may have `result == arg3`).
   Every kernel must be correct under that aliasing. The tests check both ways.
2. **Concurrency.** OpenFHE caches one `NTT` per (N, q) in a static map and calls
   `ComputeForward` on the *same object* from many OpenMP threads. Never keep mutable scratch
   in the object. Use `thread_local` or the stack. `Ntt_ConcurrentUse` checks this.
3. **Two storage widths.** Upstream's HEXL HAL `reinterpret_cast`s every coefficient vector
   to `uint64_t*`, which only works at `NATIVE_SIZE=64`. rvv-hexl adds a `uint32_t` overload
   of every function it needs (`HEXL_RVV_HAS_32BIT_API`), and
   `third_party/patches/openfhe-hexl-wordsize.py` rewrites those 60 casts to OpenFHE's own
   `BasicInteger` (identical code at 64, the `uint32_t` API at 32). So `make openfhe
   WITH_RVV_HEXL=ON` works at `NATIVE_SIZE=64` **and** 32. At 32 OpenFHE caps moduli at 28
   bits, so everything takes the e32 RVV path with no 64↔32 conversion. At 64 the e32 kernels
   narrow on load and widen on store when `q < 2^30` (see section 5).

---

## 2. Layer model: what is done and what is yours

| Layer | Files | State |
|---|---|---|
| Public API declarations | `include/hexl/**` | **done**: exact upstream signatures, checked at compile time by `test/test-api-compat.cpp` |
| Header-inline helpers | `number-theory.hpp` (MultiplyFactor, ReduceMod, BarrettReduce64, MultiplyModLazy, …), `util/*.hpp` | **done**: part of the upstream header API |
| Entry points + dispatch | `src/eltwise/<op>.cpp` (top half), `src/ntt/ntt.cpp` (except one method) | **done** |
| CPU detection, port info, allocator | `src/util/cpu-features.*`, `aligned-allocator.cpp` | **done** |
| Scalar number theory | `src/number-theory/number-theory.cpp` | **TODO**: 11 functions |
| Native kernels | `src/eltwise/<op>.cpp` (bottom half), `src/ntt/ntt-radix-2.cpp` | **TODO** |
| Twiddle precomputation | `NTT::ComputeRootOfUnityPowers` in `src/ntt/ntt.cpp` | **TODO** |
| RVV kernels | `src/eltwise/<op>-rvv.cpp`, `src/ntt/ntt-rvv.cpp` | **TODO** |
| RVV helpers | `src/util/rvv-util.hpp` | **TODO** (signatures fixed, bodies empty) |

Every TODO body is `HEXL_NOT_IMPLEMENTED();`. It throws `std::logic_error("[rvv-hexl TODO] <fn>
(<file>:<line>)")`, so:
* `make test` reports each test as PASS / FAIL / **TODO ← name of the stub it hit**,
* benchmarks skip that entry and print the stub name,
* OpenFHE's unit tests print the stub name instead of crashing or silently computing garbage,
* `make todo` lists every remaining stub.

---

## 3. Reference: every class and interface

### 3.1 `hexl/util/*`: infrastructure (nothing to implement)

| Header | Purpose |
|---|---|
| `defines.hpp` | Compiler detection (`HEXL_USE_GNU`/`HEXL_USE_CLANG`), `HEXL_UNUSED`. Upstream generates it with CMake; here it is static. Adds `HEXL_RVV_PORT` so consumers can detect the port. |
| `types.hpp` | Global `int128_t` / `uint128_t` typedefs (RV64 lowers a 64×64→128 product to `mul`+`mulhu`). |
| `check.hpp` | `HEXL_CHECK(cond, msg)` / `HEXL_CHECK_BOUNDS`: argument validation. Compiled out in release. In `make BUILD=debug` it throws with the message. Keep the checks at the top of each entry point: they document the contract and catch misuse from OpenFHE. |
| `compiler.hpp`, `gcc.hpp`, `clang.hpp` | Scalar 128-bit helpers: `MultiplyUInt64` (full product), `MultiplyUInt64Hi<Shift>` (high part, one `mulhu` for Shift=64), `BarrettReduce128`, `DivideUInt128UInt64Lo` (precomputation only: a 128-bit division), `MSB` (uses `clz`, not upstream's `log2l`, which is a soft-float quad routine on RISC-V). Building blocks of your native kernels. |
| `util.hpp` | `enum class CMPINT` (8 predicates) and `Not()`. |
| `allocator.hpp` | `AllocatorBase` (virtual allocate/deallocate) and the CRTP `AllocatorInterface`. Lets callers (SEAL) inject memory pools into `NTT`. |
| `aligned-allocator.hpp` | `AlignedAllocator<T, Alignment>` and `AlignedVector64<T>`: 64-byte aligned `std::vector`. OpenFHE stores every coefficient vector in it, so every buffer you receive from OpenFHE is 64 B aligned. RVV unit-stride loads only need element alignment, but alignment keeps each vector load in the fewest cache lines. |
| `logging/logging.hpp` | `HEXL_VLOG(level, msg)`: debug-build tracing to stderr gated by `HEXL_VLOG=<n>`. Every dispatch logs `Calling <Kernel>` at level 3, so `HEXL_VLOG=3` shows which path ran. |

### 3.2 `hexl/number-theory/number-theory.hpp`: scalar modular arithmetic

**Purpose.** Everything scalar that the NTT constructor, the native kernels and the tests
build on. Nothing here is RISC-V specific and nothing is on a hot path (it runs once per
parameter set). Write for exactness and clarity.

**Done (inline, header API):** `MultiplyFactor` (Barrett/Shoup precomputation
`floor(operand·2^s / q)` for s ∈ {32, 52, 64}), `IsPowerOfTwo`, `Log2`, `MaximumValue`,
`MultiplyModLazy<S>` (Shoup product in [0, 2q)), `AddUInt64`, `BarrettReduce64<F>`,
`ReduceMod<F>` (fold [0, F·q) to [0, q)), `MontgomeryReduce`, `HenselLemma2adicRoot`.

**To implement** in `src/number-theory/number-theory.cpp`:

| Function | Contract | Hint |
|---|---|---|
| `ReverseBits(x, w)` | reverse the low `w` bits; `w=0 → 0` | loop is fine (table building only) |
| `InverseMod(x, q)` | x⁻¹ mod q, **q not necessarily prime** | extended Euclid in `int64_t` |
| `MultiplyMod(x, y, q)` | x·y mod q, x,y < q | `MultiplyUInt64` + `BarrettReduce128` |
| `MultiplyMod(x, y, y_precon, q)` | same; `y_precon = floor(y·2^64/q)` | Shoup: `Q = hi(x·y_precon)`, `r = x·y − Q·q ∈ [0,2q)`, one conditional subtract. (Upstream's doc comment calls y_precon `floor(2^64/q)`, which is wrong.) |
| `AddUIntMod`, `SubUIntMod` | x ± y mod q, x,y < q | one conditional correction |
| `PowMod(b, e, q)` | bᵉ mod q | square-and-multiply |
| `IsPrimitiveRoot(r, d, q)` | r has order exactly d (d = power of 2) | `r^(d/2) == q−1`; `r==0 → false` |
| `GeneratePrimitiveRoot(d, q)` | some primitive d-th root | random x, `x^((q−1)/d)`, test, retry ≤ 200 |
| `MinimalPrimitiveRoot(d, q)` | the **smallest** primitive d-th root | walk r, r³, r⁵, … (d/2 values), keep the min. The `NTT(N, q)` constructor depends on this being deterministic. |
| `IsPrime(n)` | exact for all 64-bit n | Miller–Rabin, witnesses {2..37} |
| `GeneratePrimes(k, b, small, N)` | k primes q ≡ 1 (mod 2N) in [2^b, 2^(b+1)) | walk up from 2^b+1 or down from the top in steps of 2N; the order is part of the contract |

Tests: `NumberTheory_*` (exhaustive/random vs a `__int128` oracle, known Carmichael numbers and
strong pseudoprimes, exact prime lists).

### 3.3 `hexl/eltwise/*`: seven elementwise operations

**Purpose.** Vectorised modular arithmetic on coefficient arrays: everything OpenFHE does on
polynomials apart from the NTT itself. Each op has three pieces:

```
src/eltwise/eltwise-<op>.cpp           public entry: HEXL_CHECKs, dispatch      [done]
                                       + <Op>Native kernel(s)                    [TODO]
src/eltwise/eltwise-<op>-internal.hpp  kernel declarations                      [done]
src/eltwise/eltwise-<op>-rvv.cpp       <Op>RVV (or RVV32/RVV64) kernel(s)       [TODO]
```

Every kernel is a **template on the storage word** (`Word = uint64_t` for the upstream API,
`uint32_t` for the 32-bit extension). You write one body; where the two widths want different
code, branch at compile time:

```cpp
template <typename Word>
void EltwiseAddModRVV(Word* result, const Word* a, const Word* b, uint64_t n, uint64_t q) {
  if constexpr (std::is_same_v<Word, uint64_t>) {
    // e64/m4 loop
  } else {
    // e32/m4 loop (q < 2^30 here)
  }
}
```

The branch not taken is never instantiated, so it can even use intrinsics that don't make
sense for the other type. (A full specialisation, `template <> void EltwiseAddModRVV<uint32_t>(...)`,
also works when the two versions share nothing.)

Dispatch rule (already written):
* **64-bit entry points:** RVV if compiled with V and `has_rvv`, else native. For `MultMod` and
  `FMAMod` the RVV side splits again: **`…RVV32<uint64_t>`** when `q < 2^30` (binfhe's ~27-bit
  moduli, computed in 32-bit lanes) and **`…RVV64`** otherwise (BFV's 60-bit primes).
* **32-bit entry points:** RVV (`…RVV<uint32_t>` / `…RVV32<uint32_t>`) when `q < 2^30`, else
  native `<uint32_t>`. `CmpAdd` has no modulus, so it always takes RVV. There is no RVV64 at
  32-bit storage.

`kMaxModulusRVV32 = 2^30` lives in `src/util/cpu-features.hpp`. For `Word = uint32_t` the
native kernels must still do their arithmetic in 64-bit: q can be up to 2^32 there, so sums
and lazy `[0, 4q)` values do not fit in 32 bits.

| Op | Semantics | Native: what to write | RVV: what to write |
|---|---|---|---|
| `EltwiseAddMod` (vv, vs) | r = (a + b) mod q, inputs < q < 2^63 | add + conditional subtract | e64/m4 strip loop; `vminu(s, s−q)` does the conditional subtract branch-free |
| `EltwiseSubMod` (vv, vs) | r = (a − b) mod q | subtract + conditional add | as above (`vmsltu` mask + masked add, or the minu trick on `a−b+q`) |
| `EltwiseMultMod` | r = a·b mod q, inputs < imf·q (imf ∈ {1,2,4}), output < q | reduce inputs (`ReduceMod<imf>`), Barrett with a precomputed factor (never `% q` on a 128-bit value, which is a libgcc call) | **RVV32**: no fixed multiplier, so Barrett on e32 lanes (`vmul`+`vmulhu` give both halves of the 64-bit product at e32 speed): the pre-shift Barrett at 32-bit words, `rvv::MulModBarrett32(a, b, q, mu, shift)`; for 30-bit q the final correction is needed twice. **RVV64**: `vmul`+`vmulhu` at e64, upstream's pre-shift Barrett (`rvv::MulModBarrett(a, b, q, mu, shift)`, μ and shift computed once per call). Requires q < 2^61: for 62-bit q the quotient estimate can be two short (measured) |
| `EltwiseFMAMod` | r = (a·s + c) mod q, `c` may be `nullptr`, imf ∈ {1,2,4,8} | reduce `s` once, Shoup factor once (`MultiplyFactor(s,64,q)`), per element `MultiplyMod(x,s,precon,q)` + `AddUIntMod` | Shoup with scalar broadcast (`.vx` forms); separate loops for `c == nullptr` |
| `EltwiseReduceMod` | r ≡ a (mod q), r < omf·q; imf ∈ {q ("any 64-bit"), 2, 4}, omf ∈ {1, 2} | three cases: Barrett (`BarrettReduce64<omf>`), `ReduceMod<2>`, `ReduceMod<4>` or one −2q | `vmulhu` Barrett + `vminu` chain; hoist the case switch out of the loop |
| `EltwiseCmpAdd` | r = cmp(a, bound) ? a + diff : a (plain wrapping add) | switch on `cmp` **outside** the loop | one `vms{eq,ne,ltu,leu,gtu,…}.vx` → mask → masked `vadd.vx` |
| `EltwiseCmpSubMod` | x = a mod q; r = cmp(**a**, bound) ? (x − diff) mod q : x, for any 64-bit a | compare on the raw value, then reduce, then conditional `SubUIntMod` | mask from the raw values, Barrett reduction, masked `SubMod` |

Tests: `Eltwise_*` sweep 24 sizes (every strip-mining tail at VLEN 128…1024), 16+ moduli
around the e32/e64 boundary (largest prime < 2^30, smallest > 2^30), every mod factor, in
place and out of place, plus `Eltwise_OpenFHESwitchModulusPattern`, which replays OpenFHE's
exact `SwitchModulus` calls.

### 3.4 `hexl/ntt/ntt.hpp`: `class NTT`, the core of the port

**Purpose.** Negacyclic number-theoretic transform over Z_q[X]/(X^N + 1): the operation behind
every polynomial multiplication, ~44% of a binfhe gate bootstrap on our boards. An `NTT` object
is built once per (N, q) and holds all twiddle tables. `ComputeForward/Inverse` are stateless
transforms that read them.

**Semantics (pinned by `Ntt_ForwardMatchesDefinition` against the O(N²) definition):**
* forward: natural order in, **bit-reversed** out:
  `out[i] = Σ_j in[j] · w^((2·rev(i)+1)·j) mod q`, w = the object's 2N-th root;
* inverse: bit-reversed in, natural out, **scaled by N⁻¹**, so `Inverse(Forward(x)) == x`;
* mod factors: forward accepts input < {1,2,4}·q and returns < {1,4}·q; inverse {1,2}/{1,2}.
  OpenFHE always uses (1, 1).

**Members:**

| Member | State | What it is |
|---|---|---|
| `NTT()`, `~NTT()`, move/copy | done | OpenFHE needs default construction + move-assignment (`unordered_map::operator[]`) |
| `NTT(N, q, root, alloc)` | done | validates, computes `m_w_inv`, calls `ComputeRootOfUnityPowers()` (**OpenFHE's constructor**) |
| `NTT(N, q, alloc)` | done | same with `MinimalPrimitiveRoot(2N, q)` |
| allocator template ctors, `AllocatorAdapter` | done | custom memory pools |
| `CheckArguments` | done | N power of 2 ≤ 2^20, q < 2^62, q ≡ 1 mod 2N, q prime (debug only) |
| `ComputeForward`, `ComputeInverse` | done (dispatch) | checks, then RVV32 / RVV64 / native radix-2 |
| getters (`GetRootOfUnityPowers`, `GetPrecon64…`, …) | done | read-only access to the tables |
| `GetAVX512…`, `…Precon52…` | kept, **empty** | source compatibility with upstream code only |
| `GetRVV32RootOfUnityPowers` + 3 more | done (getters) | rvv-hexl addition: `uint32_t` twiddle tables for the e32 kernels |
| **`ComputeRootOfUnityPowers()`** | **TODO** | fill every table (below) |

**`ComputeRootOfUnityPowers()`: what to fill:**

| Table | Content | Consumer |
|---|---|---|
| `m_root_of_unity_powers` | `[rev(i)] = w^i`, i < N | native forward, RVV64 forward, `GetRootOfUnityPowers()` (tested) |
| `m_precon64_root_of_unity_powers` | `floor(W·2^64/q)` per entry | native / RVV64 Shoup multiply (tested) |
| `m_precon32_root_of_unity_powers` | `floor(W·2^32/q)` per entry | API compatibility |
| `m_inv_root_of_unity_powers` | inverse twiddles **in the order your inverse kernel reads them** | native / RVV64 inverse |
| `m_precon64_inv_…`, `m_precon32_inv_…` | Shoup factors of the above | same |
| `m_rvv32_*` (4 tables, `uint32_t`) | only if q < 2^30: e32 twiddles + `floor(W·2^32/q)`, **layout of your choice** | RVV32 forward/inverse |

Upstream stores inverse twiddles "stage by stage" (for m = N/2 … 1, push `inv[m+i]`) so the
Gentleman–Sande loop reads them sequentially. The `m_rvv32_*` layout is the main design lever
for the final stages (section 5).

**Free-function kernels** (`src/ntt/ntt-internal.hpp`):

| Kernel | File | Role |
|---|---|---|
| `ForwardTransformToBitReverseRadix2` | `ntt-radix-2.cpp` | native forward: Cooley–Tukey with Harvey lazy butterflies, values in [0, 4q) between stages |
| `InverseTransformFromBitReverseRadix2` | `ntt-radix-2.cpp` | native inverse: Gentleman–Sande, N⁻¹ folded into the last stage |
| `ReferenceForward…` / `ReferenceInverse…` | `ntt-radix-2.cpp` | slow, fully-reduced textbook versions for diffing stage by stage (debug aid, not dispatched) |
| `ForwardTransformToBitReverseRVV32` / `InverseTransformFromBitReverseRVV32` | `ntt-rvv.cpp` | RVV, q < 2^30, 32-bit lanes: **the binfhe hot path, main deliverable** |
| `ForwardTransformToBitReverseRVV64` / `InverseTransformFromBitReverseRVV64` | `ntt-rvv.cpp` | RVV, 64-bit lanes: BFV's 49/60-bit primes |

Tests: `Ntt_*`: definition check (N ≤ 512), round trip up to N = 32768 for 11 modulus sizes,
in place vs out of place, operand untouched, every mod-factor combination, negacyclic
convolution through `EltwiseMultMod`, the public twiddle-table contract, the OpenFHE map
pattern, and 8 threads sharing one object.

### 3.5 Internal support (`src/util/`)

| File | Purpose | State |
|---|---|---|
| `cpu-features.hpp/.cpp` | `HEXL_HAS_RVV` (compile time: TU built with V), `has_rvv` (runtime: `AT_HWCAP` has V and `HEXL_DISABLE_RVV` unset), `CurrentVLenBits()`, `kMaxModulusRVV32` | done |
| `rvv-util.hpp` | the RVV vocabulary: `AddMod`, `SubMod`, `ReduceFromTwice`, `BarrettReduce` (any 64-bit x mod q, factor `MultiplyFactor(1, 64, q)`), `MulModShoupLazy`, `MulModBarrett` (e64) and `Load32`, `Store32` (overloaded on `uint64_t*` = narrow/widen and `uint32_t*` = plain vle32/vse32, which is what lets one e32 kernel serve both storage widths), `AddMod32`, `SubMod32`, `BarrettReduce32` (factor `MultiplyFactor(1, 32, q)`), `MulModShoupLazy32`, `MulModBarrett32` (e32). All helpers are LMUL-generic templates: the arithmetic ones deduce the LMUL from their arguments; `Load32<V32>(p, vl)` takes the lane type as a template argument (default `vuint32m1_t`) and `Store32` deduces it | **TODO**: write these first, then compose kernels from them. Each helper has its own test (`hexl-tests RvvUtil`, `test/test-rvv-util.cpp`), so it can be checked before any kernel uses it. The two Barrett helpers' factor convention is defined at the top of that file (`Barrett64Factors`, `Barrett32Factor`): edit it if you choose another. Change the signatures if a better decomposition emerges (and the matching test) |
| `not-implemented.hpp/.cpp` | `HEXL_NOT_IMPLEMENTED()` | done |
| `util-internal.hpp` | scalar `Compare(CMPINT, a, b)` for the native Cmp kernels | done |
| `aligned-allocator.cpp` | the global `mallocStrategy` | done |

### 3.6 `hexl/rvv/port-info.hpp`: `GetPortInfo()` (extension)

Reports RVV compiled/available/enabled, VLEN of the current hart, build flags and compiler.
The test runner prints it and `bench-hexl` writes it into every JSON `context`, so a result
file always says which code path produced it. (Same idea as the IPCEI suite's "manipulation
check": a silent scalar fallback must show up in the output.)

### 3.7 Not ported (on purpose)

`hexl/experimental/*` (SEAL key-switch and dyadic multiply, FFT-like, LR mat-vec): used by
neither OpenFHE's backend nor SEAL's core path. The AVX-512 tables are kept only as empty
getters.

---

## 4. Suggested order of work

Each milestone ends with a command that must pass.

| # | Work | Done when |
|---|---|---|
| M0 | `number-theory.cpp` (11 functions) | `make test` → all `NumberTheory_*` PASS |
| M1 | the 7 native eltwise kernels | `HEXL_DISABLE_RVV=1 build/rvv-release/bin/hexl-tests Eltwise` all PASS |
| M2 | `ComputeRootOfUnityPowers` (64-bit tables) + native radix-2 forward/inverse | `make ISA=scalar test` fully green, **0 TODO** |
| M3 | integration with the native path only | `make ISA=scalar openfhe openfhe-check WITH_RVV_HEXL=ON`: OpenFHE's own tests pass on the port. Same for SEAL: `make ISA=scalar seal seal-check WITH_RVV_HEXL=ON` (section 7) |
| M4 | `rvv-util.hpp` + RVV eltwise (e64 first, then e32 for Mult/FMA) | `hexl-tests RvvUtil` green for each helper as you write it, then `make test` green (both passes) |
| M5 | RVV32 NTT (+ `m_rvv32_*` tables): the binfhe path | `Ntt_*` green; `bench-hexl --benchmark_filter=NTT.*qbits:27` vs `HEXL_DISABLE_RVV=1` |
| M6 | RVV64 NTT | green; honest comparison against native for 60-bit q |
| M7 | full matrix on both K3 clusters | `bench/run.sh` |

Stop after M2/M3 and run the benchmarks once: `ISA=scalar` + native kernels is the fair "base
RISC-V" point, and `ISA=rvv` + `HEXL_DISABLE_RVV=1` shows what the compiler's auto-vectoriser
already gets from your native code.

---

## 5. RVV design notes (from the D01 measurements)

Intrinsics reference: `docs/rvv_intrinsics/` (`make rvv-intrinsics-doc`; see the README section
"RVV intrinsics reference" for where each family is listed).

* **SEW=e32 whenever q < 2^30.** On both K3 clusters the multiplier retires 8× more bits per
  cycle at e32 than at e64 (the e64 multiplier does one element per cycle). binfhe's moduli are
  ≤ 28 bits, so the e32 path covers the whole TFHE side.
* **LMUL=m4 for every kernel.** Light kernels (Add/Sub/CmpAdd/CmpSubMod) are bound by per-strip
  overhead: on EltwiseAddMod at n = 1K-64K, m4 was 2.3-3.4× faster than m1 on the X100 and 2-3×
  on the A100. Multiply kernels gain too (n = 4096, cycles/element on the X100, m1 → m4):
  MulModBarrett32 3.51 → 1.59, MulModShoupLazy32 2.66 → 1.13, MulModBarrett 11.0 → 6.4, Shoup
  e64 7.6 → 5.2; m4 is the best LMUL there. m8 is slower than m4 on the X100; on the A100 it is
  5-20% faster, but it leaves only 4 register groups, so kernels that keep more values live (NTT
  butterflies, MultMod with input reduction) will spill. All arithmetic helpers in
  `rvv-util.hpp` are LMUL-generic templates (overloaded intrinsics), so only the kernel's
  `vsetvl`/`vle`/`vse` name the LMUL. Fractional LMUL only exposes the low part of a register
  (verified on silicon). **mf2 is the smallest portable
  one**: X100 returns vl=0 and traps on e32/mf4 and mf8.
* **32-bit compute, either storage.** With 64-bit storage (`n64-rvvhexl`), load `e64,m8` and
  narrow (`vncvt.x.x.w`) to `e32,m4` (`rvv::Load32<vuint32m4_t>`), and widen (`vzext.vf2`)
  before the store. That pair reaches about 2× the load bandwidth of `e64,m2` → `e32,m1` on
  both clusters. In the NTT do
  this once, in the first and last stage, not as separate passes. With 32-bit storage
  (`n32-rvvhexl`) the same kernel reads and writes directly. `rvv::Load32`/`Store32` hide the
  difference. `[0, 4q) < 2^32` holds for q < 2^30, so lazy butterflies fit in 32-bit lanes.
* **Native kernels in an RVV build are compiled with V enabled**, so GCC may auto-vectorize
  them at -O3. For loops around the scalar `BarrettReduce64` this is 3-4x slower than scalar
  on the K3 (X100 15 vs 3.9 cycles/elem, A100 49 vs 18.7): put `#pragma GCC novector`
  before such loops. It also means `HEXL_DISABLE_RVV=1` is not a vector-free baseline; use
  `ISA=scalar` for that. Keep per-element conditions branch-free by selecting a value
  (`sub = c ? diff : 0U`, which becomes `czero`) rather than one of two results.
* **VLEN-agnostic, always.** X100 VLEN=256, A100 VLEN=1024, same binary. `vsetvl` in every
  strip, and never cache VLEN in a table built at construction time. The benchmark runner
  starts A100 processes through `ailaunch` (VLEN changes across clusters, so migration after
  vector code has run is unsafe).
* **Final NTT stages (t < VL) decide performance on this silicon, not instruction selection.**
  D01's microbenchmark: vectorising them across blocks (strided `vlse32` with stride 2t, or
  `vlseg2e32`/`vlseg4e32` for t = 1, 2, or replicated twiddles in `m_rvv32_*`) almost doubles
  the X100 gain. On A100, avoid strided access (in-register transposes instead).
* **Then block in the register file** (radix-4/8: 2–3 stages per memory pass). The transform
  sits >5× below the roofline balance point, so once lane occupancy is fixed, bandwidth
  becomes the limit. A100's 4× larger register file gains the most.
* **Hand instruction selection comes last.** Measured as the least profitable step before the
  above (marginal on X100, counter-productive on A100).
* RVA23 guarantees **Zvbb** (`vbrev.v`, `vrev8`, `vwsll`) and Zba/Zbb, useful for
  bit-reversal and shifts. Detect multi-letter extensions with `riscv_hwprobe` if you need them
  at runtime.

---

## 6. Testing and benchmarking methodology

**Correctness ladder:** `make rvv-hexl-test` (both dispatch paths against `__int128` oracles, which were
validated against upstream Intel HEXL in release and `HEXL_DEBUG` builds) → `make BUILD=debug
rvv-hexl-test` (argument contracts) → `make SANITIZE=address test` → spike at VLEN 256 and 1024 (laptop)
→ `make openfhe-check WITH_RVV_HEXL=ON` (OpenFHE's own `core_tests`/`pke_tests`/`binfhe_tests` on the port)
→ `make seal-check WITH_RVV_HEXL=ON` (BFV + CKKS round trips through every HEXL call SEAL makes;
with `SEAL_TESTS=ON`, also SEAL's own `sealtest`).

**Benchmark matrix** (`bench/run.sh`), with every axis changing exactly one variable:

| Comparison | Isolates |
|---|---|
| `n64-rvvhexl` vs **`n64`** | the HEXL backend at 64-bit words (same flags, identical bench binaries) |
| `n32-rvvhexl` vs **`n32`** | the HEXL backend at 32-bit words: the port vs the best stock binfhe configuration (D01's recommended baseline) |
| `n32-rvvhexl` vs `n64-rvvhexl` | the storage width alone (same e32 kernels; also `BM_NTTForward32` vs `BM_NTTForward/qbits:27` in bench-hexl) |
| `ISA=rvv` vs `ISA=scalar` | the vector unit (scalar `-march` = the rvv one minus V; on the K3 `rva23u64` minus V/Zv*) |
| `n{64,32}-rvvhexl` vs the same + `HEXL_DISABLE_RVV=1` | your RVV kernels vs your native kernels under the same auto-vectoriser |
| SEAL `rvvhexl` vs **`stock`** | the HEXL backend under SEAL (same flags, identical bench binaries). **`stock` SEAL's NTT is already partly auto-vectorised by GCC**, so it is a harder baseline than OpenFHE's scalar NTT |
| x100 vs a100 | VLEN 256 vs 1024, in-order vs out-of-order |

Never compare an `n64-rvvhexl` result with an `n32` run and call the difference "HEXL": it mixes the
backend with the word size (same caveat as the x86 HEXL runs in D01). Compare at equal word size.

**Suites:**
* `bench/hexl`: HEXL's own kernel set at upstream's sizes (n = 1024/4096/16384; 45-bit NTT
  primes) plus our moduli (27/49/60 bits) and `BM_MemcpyReference`, which gives the
  bandwidth line each eltwise kernel should be read against.
* `bench/openfhe`: the IPCEI benches (NTT microbench, BFV OpenMP sweep, TFHE stress/flags/LUT),
  same arguments as `ZKP+FHE Research/benchmark.sh`, so results line up with
  `results/benchmarks-k3-*.json`.
* OpenFHE upstream (`lib-benchmark`, `poly-benchmark-{4k,16k}`, `binfhe-ginx`, `VectorMath`,
  and on rvv-hexl builds the `*-hexl` variants Intel used to evaluate HEXL in OpenFHE).
* `bench/seal`: the IPCEI SEAL benches (`seal_ntt_bench`, BFV deployable and 60-bit), same
  arguments as `ZKP+FHE Research/benchmark.sh`; part of the `ipcei` suite of `bench/run.sh`.

---

## 7. Using the port with Microsoft SEAL

Microsoft SEAL has its own Intel HEXL backend, and rvv-hexl is a drop-in for it too, with **no
changes to SEAL** and none to rvv-hexl. Verified on the K3 (2026-09-28, SEAL 4.1.2, GCC 16.2,
`-march=rva23u64`): SEAL configures against rvv-hexl, compiles and links cleanly (all 11
`intel::hexl` symbols it references resolve to this library), and at runtime its calls reach
rvv-hexl (the first one hits a stub until M0/M2 are done).

```
                  Microsoft SEAL v4.1.2   (make seal WITH_RVV_HEXL=ON)
                   │   -DSEAL_USE_INTEL_HEXL=ON
                   │   find_package(HEXL 1.2.4)  ← satisfied by our 1.2.6 HEXLConfig (same major)
                   │   #include "hexl/hexl.hpp", links HEXL::hexl
                   ▼
   seal/util/ntt.cpp               NTT (forward + inverse, per (N, q) cache)
   seal/util/polyarithsmallmod.cpp five Eltwise* functions
```

### How it is wired

SEAL's switch is a plain CMake option, `-DSEAL_USE_INTEL_HEXL=ON`. `make seal WITH_RVV_HEXL=ON`
(re)installs rvv-hexl for the current ISA/BUILD and builds SEAL against it into
`build/seal/<ISA>/rvvhexl/`; `make seal` builds the stock reference into `build/seal/<ISA>/stock/`
with identical options (`third_party/seal.sh`). One constraint matters:

* **`SEAL_BUILD_DEPS` must be `OFF`.** With `ON`, SEAL downloads upstream Intel HEXL 1.2.5 (x86
  only) instead of looking for an installed HEXL. As a consequence SEAL's optional dependencies
  are disabled in both builds: Microsoft GSL (API sugar), zlib and Zstandard (compressed
  serialisation, which the IPCEI suite also turns off). If you need them, install them
  (`libmsgsl-dev`, `zlib1g-dev`, `libzstd-dev`) and turn them back on in `third_party/seal.sh`.
* `libseal` is static and does not embed HEXL: whoever links SEAL also links `libhexl` (SEAL's
  installed `SEALConfig.cmake` does this for CMake consumers; `mk/seal-bench.mk` and
  `make seal-check` do it for ours).

### What SEAL calls, and with which arguments

Every HEXL call site in SEAL 4.1.2 (there are no others):

| HEXL API | SEAL call site | Arguments SEAL uses |
|---|---|---|
| `NTT(N, q, root, MemoryPoolHandle, SimpleThreadSafePolicy)` | `ntt.cpp` `get_ntt()`: one object per (N, q), cached in a static map | the **allocator template constructor** |
| `NTT::ComputeForward(d, d, imf, omf)` | `ntt_negacyclic_harvey{_lazy}` | in place; **(4, 4)** lazy and **(4, 1)** |
| `NTT::ComputeInverse(d, d, imf, omf)` | `inverse_ntt_negacyclic_harvey{_lazy}` | in place; **(2, 2)** lazy and **(2, 1)** |
| `EltwiseAddMod` | `add_poly_coeffmod`, `add_poly_scalar_coeffmod` | **vector+vector and vector+scalar** |
| `EltwiseSubMod` | `sub_poly_coeffmod`, `sub_poly_scalar_coeffmod` | **vector+vector and vector+scalar** |
| `EltwiseMultMod(r, a, b, n, q, 4)` | `dyadic_product_coeffmod` | input factor **4** |
| `EltwiseFMAMod(r, a, s, nullptr, n, q, 8)` | `multiply_poly_scalar_coeffmod` | **no addend**, input factor **8** |
| `EltwiseReduceMod(r, a, n, q, q, 1)` | `modulo_poly_coeffs` | input factor **= q** (arbitrary 64-bit input), output < q |

Compared with OpenFHE (section 1), what this adds to your work:

1. **No per-call fallback.** With `SEAL_USE_INTEL_HEXL`, every one of these always goes to
   rvv-hexl. All of them must be correct; a slow one just stays on its native path at first.
2. **The whole HEXL argument space is exercised.** OpenFHE uses factors (1, 1) everywhere. SEAL
   uses the lazy NTT factors, `MultMod` with inputs < 4q, `FMAMod` with inputs < 8q and
   `ReduceMod` from arbitrary 64-bit values. These are all upstream semantics already covered
   by the `Ntt_*` / `Eltwise_*` tests (every mod-factor combination), so passing `make test`
   means passing SEAL's usage.
3. **The allocator internals are part of the contract.** SEAL specialises
   `intel::hexl::NTT::AllocatorAdapter<seal::MemoryPoolHandle>` and
   `AllocatorAdapter<seal::MemoryPoolHandle, SimpleThreadSafePolicy>`, both derived from
   `AllocatorInterface<...>` (`hexl/util/allocator.hpp`). Keep `NTT::AllocatorAdapter` a nested
   `template <class Adaptee, class... Args> struct` with the upstream constructor shapes, and the
   allocator template constructors, or SEAL stops compiling.
4. **Concurrency:** same invariant as OpenFHE: SEAL shares one cached `NTT` per (N, q) across
   threads, so `ComputeForward/Inverse` must not keep mutable state in the object.

### Performance expectations

* **SEAL is mostly an e64 workload.** Its moduli are typically 36-60 bits
  (`CoeffModulus::BFVDefault`: 36-37 bits at N = 4096, 43-44 at 8192, larger N up to 60), so it
  exercises the `RVV64` kernels (M6) far more than the e32 path. On the K3 the e64 multiplier
  does one element per cycle (section 5), so expect a smaller gain than on binfhe.
* **The baseline is SEAL's own NTT, which is already partly auto-vectorised by GCC** (~25% of
  its instructions are RVV). Measured 2026-09-27 with GCC 16.2 `rva23u64`, N = 16384, 49-bit q:
  **53.8 ns/element on the X100, 148.5 on the A100** (`seal_ntt_bench`). For comparison, stock
  OpenFHE's scalar NTT is 34.7 / 125.3 ns/element on the same build. That is the number the
  `RVV64` NTT has to beat under SEAL.
* The stock SEAL that `make seal` builds is a faithful baseline: on the X100 it reproduces the
  IPCEI suite's own SEAL build (53.5 vs 53.8 ns/element for the NTT, 26.0 vs 25.9 s for BFV
  deployable, 17.2 vs 17.2 s for BFV 60-bit), so disabling GSL/zlib/zstd costs nothing.

### Workflow

```sh
make seal-all                          # stock + rvvhexl (installs rvv-hexl first)
make seal-check                        # stock: must pass (sanity of the build itself)
make seal-check WITH_RVV_HEXL=ON       # BFV + CKKS through every call above; TODO until M0/M2
make seal-check WITH_RVV_HEXL=ON SEAL_TESTS=ON   # + SEAL's sealtest (apt install libgtest-dev)
make ISA=scalar seal seal-check WITH_RVV_HEXL=ON # the M3 integration check on the native path
make bench && bench/run.sh --suite ipcei         # SEAL benches for every SEAL build
```

`seal-check` exits 0 when all checks pass, 1 on a wrong result, 2 when it reaches a stub (it
names it, like `make test`). The smoke test prints which backend it ran on (stock, or rvv-hexl
with RVV compiled/enabled and the VLEN), so a silent fallback shows up in the output.

After changing a kernel, `make rvv-hexl-install` is enough for SEAL: `libhexl` is linked into
the smoke test and the benches at their link step, so rerun `make seal-check` / `make bench`. Only
a change to a public header (`include/hexl/*`, in particular `ntt.hpp`) needs `make seal
WITH_RVV_HEXL=ON` again, because SEAL compiles those headers into `libseal`.
