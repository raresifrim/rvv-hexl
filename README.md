# rvv-hexl

RISC-V (RVV 1.0) port of [Intel HEXL](https://github.com/intel/hexl) v1.2.6, built to plug
into OpenFHE through the same extension point Intel HEXL uses
([openfhe-hexl](https://github.com/openfheorg/openfhe-hexl)), and into Microsoft SEAL through
SEAL's own `SEAL_USE_INTEL_HEXL` option (no source changes on either side). It implements "Etapa 1" of the
IPCEI A14 plan (SENTHIPoli D01, *Direcția de accelerare*): an RVV-aware HEXL backend,
measured on the SpaceMiT K3 (X100 and A100 clusters) against stock OpenFHE and the no-RVV
baseline.

**Status: skeleton.** The public API, dispatch, build system, tests and benchmarks are in place.
Every kernel body is a stub that throws `[rvv-hexl TODO] <function>`. `make todo` lists
them. Read [docs/PORTING_GUIDE.md](docs/PORTING_GUIDE.md) before you start.

```
OpenFHE (NATIVE_SIZE 64|32) ── openfhe-hexl HAL ──┐
                                                  ├─ intel::hexl API ── rvv-hexl
Microsoft SEAL 4.1 ── SEAL_USE_INTEL_HEXL ────────┘  (this repo)       ├─ dispatch (done)
                                                                       ├─ native C++ kernels (TODO)
                                                                       └─ RVV kernels (TODO)
```

## Quick start (K3 / RISC-V Ubuntu)

```bash
sudo apt install build-essential g++-14 cmake git python3 libssl-dev libgmp-dev libbenchmark-dev
```

```bash
make info                 # toolchain, -march, paths, selected OpenFHE configuration
make rvv-hexl-test        # correctness tests: RVV dispatch, then HEXL_DISABLE_RVV=1 (native)
make todo                 # remaining stubs
```

Then, once kernels are implemented:

```bash
make openfhe                          # stock OpenFHE, NATIVE_SIZE = CPU word size (64)
make openfhe NATIVE_SIZE=32           # stock, 32-bit words (best stock binfhe config)
make openfhe WITH_RVV_HEXL=ON         # OpenFHE + openfhe-hexl on top of rvv-hexl
make openfhe NATIVE_SIZE=32 WITH_RVV_HEXL=ON   # same with 32-bit words (binfhe)
make openfhe-check WITH_RVV_HEXL=ON   # OpenFHE's own unit tests on the port
make ISA=scalar openfhe-all           # the four above, built without V (no-RVV baseline)
make seal-all                         # Microsoft SEAL 4.1.2: stock + on top of rvv-hexl
make seal-check WITH_RVV_HEXL=ON      # BFV + CKKS round trips through every HEXL call SEAL makes
make bench && make ISA=scalar bench
bench/run.sh                          # full matrix, both K3 clusters -> results/<host>-<date>/
```

## Layout

```
include/hexl/        PUBLIC API, source-identical to Intel HEXL 1.2.6 (+ RVV tables, port-info)
src/number-theory/   scalar modular arithmetic (primes, roots, inverses)       TODO
src/eltwise/         7 elementwise ops: <op>.cpp = entry + dispatch + native,  TODO
                     <op>-rvv.cpp = RVV kernels
src/ntt/             NTT class (ntt.cpp), native radix-2, RVV kernels          TODO
src/util/            CPU detection, RVV helpers (rvv-util.hpp, TODO), stub marker
test/                self-contained tests with independent __int128 oracles
test/seal/           seal-smoke.cpp: end-to-end SEAL check (make seal-check)
bench/hexl/          HEXL-standard microbenchmarks (Google Benchmark)
bench/uarch/         vector-unit microbenchmarks: cycles/insn, bandwidth, thread scaling (K3 tuning)
bench/openfhe/       IPCEI OpenFHE benches (verbatim from ZKP+FHE Research @37a6844)
bench/seal/          IPCEI SEAL benches (verbatim from ZKP+FHE Research @9d51ffd)
bench/run.sh         benchmark matrix runner (ISA x OpenFHE/SEAL variant x RVV on/off x cluster)
third_party/         OpenFHE / openfhe-hexl / SEAL / Google Benchmark build scripts
cmake/               HEXLConfig.cmake: lets OpenFHE and SEAL find this library as "HEXL 1.2.6"
docs/                PORTING_GUIDE.md: every class and interface, what to implement
docs/rvv_intrinsics/ all RVA23 RVV intrinsics + searchable index, local only (make rvv-intrinsics-doc)
docs/tools/          scripts that fetch and index docs/rvv_intrinsics
```

## Make targets and knobs

Three components, each with its own targets. Every OpenFHE target acts on the configuration
selected by `NATIVE_SIZE` / `WITH_RVV_HEXL` (and `ISA`); every SEAL target on `WITH_RVV_HEXL`
(and `ISA`).

| rvv-hexl | What it does |
|---|---|
| `make rvv-hexl` (default goal) | `libhexl.a` + `libhexl.so` in `build/<ISA>-<BUILD>/lib` |
| `make rvv-hexl-test` (`test`) | build + run the tests; an RVV build runs them twice (RVV path, then native path) |
| `make rvv-hexl-bench` | `bench-hexl` (Google Benchmark; `make gbench` builds it locally if there's no system package) |
| `make uarch-bench` | `bench-uarch`: cycles per RVV instruction, load/store bandwidth and thread scaling on the current cluster (`ISA=rvv`; see [bench/uarch/README.md](bench/uarch/README.md)) |
| `make rvv-hexl-install` (`install`) | headers, libs, `lib/cmake/hexl-1.2.6/HEXLConfig.cmake`, `hexl.pc` into `PREFIX` |

| OpenFHE | What it does |
|---|---|
| `make openfhe` | configure + build + install OpenFHE v1.5.1 into `build/openfhe/<ISA>/n<NATIVE_SIZE>[-rvvhexl]/` |
| `make openfhe-check` | OpenFHE's `core_tests`, `pke_tests`, `binfhe_tests` for that build |
| `make openfhe-bench` | the IPCEI benches against that build |
| `make openfhe-all` | the comparison set: `n64`, `n32`, `n64-rvvhexl`, `n32-rvvhexl` |
| `make openfhe-list` | what exists for this ISA (read from each install's `config_core.h`) |

| SEAL | What it does |
|---|---|
| `make seal` | configure + build + install SEAL v4.1.2 (static) into `build/seal/<ISA>/{stock,rvvhexl}/`; `WITH_RVV_HEXL=ON` builds it with `SEAL_USE_INTEL_HEXL=ON` on top of rvv-hexl |
| `make seal-check` | `test/seal/seal-smoke.cpp` (BFV + CKKS, checked exactly / within 1e-3) against that build; with `SEAL_TESTS=ON` also SEAL's `sealtest` |
| `make seal-bench` | the IPCEI SEAL benches (`seal_ntt_bench`, BFV deployable and 60-bit) against that build |
| `make seal-all` | the comparison pair: `stock`, `rvvhexl` |
| `make seal-list` | what exists for this ISA (read from each install's `seal/util/config.h`) |

| Everything | |
|---|---|
| `make bench` | `rvv-hexl-bench` + `uarch-bench` (rvv) + IPCEI benches for every OpenFHE and SEAL build + `ailaunch` |
| `make run-bench` | `bench/run.sh` |
| `make rvv-intrinsics-doc` | all RVA23 RVV intrinsics + `INDEX.md` + grep-able `rva23-intrinsics.tsv` in `docs/rvv_intrinsics/` (git-ignored) |
| `make reconfigure` | forget the cached board/toolchain checks (after a compiler upgrade, or on another board) |
| `make todo` / `info` / `clean` / `distclean` | |

| Variable | Values (default first) | Meaning |
|---|---|---|
| `ISA` | auto-detected `rvv`, or `scalar` (riscv64 only) | `scalar` = the rvv `-march` minus V: the base RISC-V point. Applies to rvv-hexl **and** OpenFHE. See [Board and toolchain detection](#board-and-toolchain-detection) |
| `BUILD` (or `CMAKE_BUILD_TYPE`) | `release`, `debug` | rvv-hexl only; `debug` enables every `HEXL_CHECK`, and `HEXL_VLOG=3` prints the dispatched kernel |
| `HEXL_SHARED_LIB` | `ON`, `OFF` | also build `libhexl.so` |
| `HEXL_TESTING` / `HEXL_BENCHMARK` | `OFF`, `ON` | `make rvv-hexl` also builds the test / bench binary |
| `NATIVE_SIZE` | CPU word size (64), `32` | OpenFHE's native integer width |
| `WITH_RVV_HEXL` | `OFF`, `ON` | build OpenFHE (or SEAL) with the HEXL backend on top of rvv-hexl (installs rvv-hexl of the same `ISA`/`BUILD` first; for OpenFHE, `NATIVE_SIZE` 64 or 32) |
| `OPENFHE_BENCHMARKS` / `OPENFHE_UNITTESTS` | `ON` / follows `WITH_RVV_HEXL` | build upstream OpenFHE's benchmark suite / unit tests |
| `SEAL_TESTS` | `OFF`, `ON` | also build SEAL's own `sealtest` (needs `libgtest-dev`) |
| `RISCV_MARCH` / `RISCV_SCALAR_MARCH` | auto-detected | force the `-march` for `ISA=rvv` / `ISA=scalar` (the toolchain checks still run) |
| `CROSS`, `RUN` | e.g. `riscv64-unknown-elf-`, `spike --isa=rv64gcv_zvl256b pk` | cross-build and run under a simulator |
| `SANITIZE` | `address`, `undefined` | sanitizer build |

`ON`/`OFF`, `1`/`0`, `yes`/`no` are all accepted.

Developing against OpenFHE: `libOPENFHEcore.so` loads `libhexl.so` from the rvv-hexl install,
so after changing a kernel `make rvv-hexl-install && make openfhe-check WITH_RVV_HEXL=ON` is
enough; no OpenFHE rebuild. If you change a **public header** (for example, add members to
`NTT`), rerun `make openfhe WITH_RVV_HEXL=ON`: OpenFHE compiles those headers into its own
code, and its incremental rebuild picks up the change.

Developing against SEAL: `libseal` is static and does not embed HEXL, so the smoke test and the
SEAL benches link `libhexl` themselves. After changing a kernel, `make rvv-hexl-install &&
make seal-check WITH_RVV_HEXL=ON` is enough; after a public-header change, rerun `make seal
WITH_RVV_HEXL=ON`. What SEAL calls, and with which arguments, is in the porting guide, section 7.

Runtime switch: `HEXL_DISABLE_RVV=1` forces the native kernels in an RVV build (the counterpart
of upstream's `HEXL_DISABLE_AVX512DQ`).

## RVV intrinsics reference (the whole RVA23 vector ISA)

`make rvv-intrinsics-doc` assembles every RVV C intrinsic of the RVA23 profile in
`docs/rvv_intrinsics/` (git-ignored, ~160 MB) and indexes it
(`docs/tools/fetch-rvv-intrinsics.sh` + `gen-rvv-intrinsics-index.py`):

* [rvv-intrinsic-doc](https://github.com/riscv-non-isa/rvv-intrinsic-doc) `v1.0-ratified`: V, Zvfh, Zvfhmin;
* from its `main` branch (pinned commit): vector crypto (Zvbb ⊃ Zvkb, Zvbc, Zvkg, Zvkned,
  Zvknh[ab], Zvksed, Zvksh) and BF16 (Zvfbfmin, Zvfbfwma). Draft extensions are left out.

| File under `docs/rvv_intrinsics/` | Content |
|---|---|
| `INDEX.md` | per extension: RVA23 status, on the K3?, counts, chapter files, what the compiler lacks |
| `rva23-intrinsics.tsv` | one line per intrinsic: `name ext rva23 k3 gcc kind section proto file` |
| `doc/rvv-intrinsic-spec.adoc` | naming scheme, types, masks, tuples, `vsetvl`/`vget`/`vlenb` |
| `auto-generated/intrinsic_funcs/NN_*.adoc` | base V chapters (`00` loads/stores … `08` utilities) |
| `auto-generated/vector-crypto/intrinsic_funcs/` | Zvbb, Zvbc, Zvkg, Zvkned, Zvknh, Zvksed, Zvksh |
| `auto-generated/bfloat16/intrinsic_funcs/` | Zvfbfmin, Zvfbfwma (not on the K3) |
| `…/policy_funcs/…`, `…/overloaded_intrinsic_funcs/` | `_tu`/`_mu`/`_tum`/`_tumu` variants, short overloaded names |

```bash
grep -w __riscv_vbrev_v_u32m1 docs/rvv_intrinsics/rva23-intrinsics.tsv
```

The `gcc` column comes from compiling every name with your RISC-V compiler (native gcc on the
K3, `riscv64-*-gcc` elsewhere; override with `PROBE_CC=`). GCC 16.1 implements everything
except Zvbb's elementwise `vcpop.v`, the `_tu`/`_mu` policy variants of `vclz`/`vctz`, and a
few BF16 permutation/move intrinsics.

## Board and toolchain detection

On RISC-V, `ISA` and `-march` are not assumed: `mk/detect-riscv.sh` checks, once per toolchain,
that RVV can actually be used, and falls back to `ISA=scalar` (with a warning, even if you asked
for `ISA=rvv`) when it can't. `make info` shows the outcome.

1. **Board** (native builds): every hart in `/proc/cpuinfo` must report `v`. The `-march` only
   uses extensions that **all** harts report. This matters because GCC can use any `-march`
   extension anywhere in the library, so a `-march` wider than the CPU crashes with SIGILL even
   with RVV switched off at runtime.
   * RVA23 board (SpaceMiT K3): `-march=rva23u64`
   * RVV 1.0 but not RVA23 (e.g. SpacemiT K1): the board's own extensions, V included
   * no V, or only T-Head's pre-1.0 `xtheadvector` (e.g. JH7110, TH1520): scalar
2. **Toolchain**: the compiler must accept that `-march` and compile the RVV C intrinsics
   **v1.0** API the port uses (policy variants, tuple segment loads…): GCC >= 14 or LLVM >= 17.
   `__riscv_v_intrinsic` must be at least `12000` (v0.12, the API v1.0 froze): distribution GCC
   14/15 report `12000`, upstream GCC 16 reports `1000000`.

The scalar `-march` is always the rvv one minus V (on the K3: `rva23u64` without V/Zv*), so
`ISA=rvv` vs `ISA=scalar` isolates the vector unit and nothing else. Results are cached in
`build/config/`, and `make reconfigure` runs the checks again. Build directories rebuild
automatically when the compiler or flags change.

The library also checks V at runtime (`AT_HWCAP`, `HEXL_DISABLE_RVV`). Cross builds (`CROSS=`)
cannot see the target board, so pass `RISCV_MARCH=` for anything other than RVA23.

## Develop on a laptop with spike

The RVV code can be compiled and tested without the board. With `riscv-gnu-toolchain`,
`riscv-isa-sim` and `riscv-pk` installed (Homebrew tap `riscv-software-src/riscv`, already on the
IPCEI Mac):

```bash
make CROSS=riscv64-unknown-elf- RISCV_MARCH=rv64gcv \
     RUN="spike --isa=rv64gcv_zvl256b $(brew --prefix riscv-pk)/riscv64-unknown-elf/bin/pk" \
     TEST_ARGS=--quick test
```

`zvl256b` emulates the X100 VLEN, `zvl1024b` the A100's. Spike checks correctness only; its
timings mean nothing. The build is RISC-V only: without `CROSS=`, `make` stops on a non-RISC-V
host.

## Relation to the upstream Intel HEXL / OpenFHE builds

Upstream Intel HEXL is a standalone CMake project:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHEXL_SHARED_LIB=ON -DCMAKE_INSTALL_PREFIX=/opt/hexl
cmake --build build -j                 # + unit-test and bench_hexl (HEXL_TESTING/HEXL_BENCHMARK default ON)
cmake --build build --target unittest
cmake --build build --target bench
cmake --install build
```

| Upstream Intel HEXL (CMake) | rvv-hexl (make) |
|---|---|
| `cmake --build build` | `make rvv-hexl` |
| `--target unittest` (GoogleTest `unit-test`) | `make rvv-hexl-test` (self-contained `hexl-tests`) |
| `--target bench` (`bench_hexl`) | `make rvv-hexl-bench` (`bench-hexl`) |
| `cmake --install build` | `make rvv-hexl-install` |
| `-DCMAKE_BUILD_TYPE=Debug` (turns on `HEXL_DEBUG`) | `BUILD=debug` (or `CMAKE_BUILD_TYPE=Debug`) |
| `-DHEXL_SHARED_LIB=ON` (default OFF) | `HEXL_SHARED_LIB=ON` (default ON: OpenFHE is built shared) |
| `-DHEXL_TESTING` / `-DHEXL_BENCHMARK` (default ON) | same names, default OFF |
| `-DHEXL_EXPERIMENTAL`, `HEXL_DOCS`, `HEXL_COVERAGE` | not ported |
| CPU feature detection with google/cpu_features at runtime | `HEXL_HAS_RVV` at compile time + `AT_HWCAP` at runtime |
| `HEXL_DISABLE_AVX512DQ=1` | `HEXL_DISABLE_RVV=1` |
| installs `lib/cmake/hexl-1.2.6/HEXLConfig.cmake` (`HEXL::hexl`) | installs the same package and target name |

On the OpenFHE side, upstream uses
[openfhe-configurator](https://github.com/openfheorg/openfhe-configurator): an interactive
script that stages the `openfhe-hexl` overlay onto an `openfhe-development` checkout and builds
it with `-DWITH_INTEL_HEXL=ON`. OpenFHE then either downloads and builds Intel HEXL itself
(CMake ExternalProject, v1.2.6) or uses a prebuilt one
(`-DINTEL_HEXL_PREBUILT=ON -DINTEL_HEXL_HINT_DIR=/opt/hexl`).
`make openfhe WITH_RVV_HEXL=ON` does the same staging non-interactively and always takes the
prebuilt route, with `INTEL_HEXL_HINT_DIR` pointing at the rvv-hexl install.
`WITH_RVV_HEXL=OFF` builds the plain `openfhe-development` checkout, which contains no HEXL code.

## Notes

* The repository path must not contain spaces (GNU make limitation).
* Upstream's HEXL backend only works at `NATIVE_SIZE=64`: it casts coefficient vectors to
  `uint64_t*`. rvv-hexl adds a `uint32_t` API (`HEXL_RVV_HAS_32BIT_API`), and
  `third_party/patches/openfhe-hexl-wordsize.py` makes the overlay use OpenFHE's own word type,
  so `WITH_RVV_HEXL=ON` also works at `NATIVE_SIZE=32`. See the porting guide, section 1.
* License: Apache-2.0, as upstream Intel HEXL (the public headers derive from it).
