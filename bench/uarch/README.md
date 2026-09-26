# bench-uarch: vector-unit microbenchmarks

`bench-uarch` measures the vector unit itself: cycles per instruction (throughput
and latency), load/store bandwidth against working-set size, and how several
threads share the unit. It doesn't link the HEXL library: `bench-hexl` measures
the port, this one measures the machine the port is tuned for.

```bash
make uarch-bench                                         # ISA=rvv only
taskset -c 0-7 build/rvv-release/bin/bench-uarch --scaling               # X100, VLEN 256
build/tools/ailaunch $PWD/build/rvv-release/bin/bench-uarch --scaling    # A100, VLEN 1024
bench/run.sh --suite uarch                               # both clusters, CSV into results/
```

For real cycle counts, allow user-space perf counters, and use the performance governor:

```bash
sudo sysctl kernel.perf_event_paranoid=2
```

Without perf access, the wall clock is multiplied by the cpufreq frequency. The header line says which clock was used. Under perf, it also prints the effective GHz, as a sanity check against the governor.

## How the kernels are built

`gen-kernels.py` writes one assembly function per (op, SEW, LMUL, kind) into the build directory, 257 in all. They are assembly, not intrinsics, so the compiler can't drop, reorder or re-vtype anything. Each loop body is 16 instructions under test:

- **`tp` (throughput):** up to 8 independent destination groups in rotation.
- **`lat` (latency):** every instruction consumes the previous result.

VL is always VLMAX, so `elem/cyc` compares directly across VLENs.

```bash
bench-uarch --list                     # group/op/sew/lmul/kind ids
bench-uarch --filter vmul,vmulhu       # substring match on the id
bench-uarch --filter e32/m1/ --no-stream
bench-uarch --group mem --quick
```

Groups:

| Group | Contents |
|---|---|
| `alu` | add/sub/and/shift, `vminu`, `vmsltu`, `vmerge`, a masked add, `vnsrl.wx` |
| `mul` | `vmul`, `vmulhu` (`.vv` and `.vx`), `vmacc`, `vwmulu`, `vwmaccu` |
| `perm` | `vrgather.vv`, `vslidedown`, `vslideup`, `vcompress` |
| `mem` | L1-resident `vle`/`vse`, masked `vle`, `vlse` (stride 2 elements and 64 B), `vluxei` (reversed), `vlseg2`/`vlseg4`, `vsseg2` |
| `xfer` | scalar-vector round trips: load, `vmv.x.s`, next address; `vmv.s.x` then `vmv.x.s`; `vmsltu` then `vcpop.m` |
| `vset` | `vsetvli` + `vadd` pairs: same vtype, e32 and e64 alternating, m1 and m2 alternating |
| `stream` | `vle64` m1/m4 and `vse64` m4 over 16 KiB … 64 MiB |

## What each test answers

The K3 design paper ("SpacemiT K3: A RVA23 RISC-V AI CPU with 60 TOPS AI Compute", preview of 2026-01-29) describes the vector units but leaves the numbers that decide kernel design open.

| Paper says | Test | What it decides in rvv-hexl |
|---|---|---|
| X100: 2 arithmetic ports, 128-bit micro-ops, so LMUL=1 is 2 micro-ops | `alu`/`mul` `tp` across mf2 … m4 | whether larger LMUL buys throughput or only hides latency |
| X100: complex permutes may have a single 128-bit unit | `perm` vs `alu` | the cost of the last NTT stages (in-register shuffles) |
| A100: integer multiply at SEW 32/64 is 2×128, while add is 2×256 | `mul` vs `alu`, e32 vs e64 | e32 kernels for q < 2^30; how many multiplies a Barrett or Shoup step can afford |
| X100: segment access goes through a transpose buffer in the load/store unit | `vlseg2e*` vs `vle*` | loading butterfly pairs with `vlseg2` vs `vle` + shuffles |
| X100: vector addresses come from the scalar load/store unit, data from the VPU | `xfer` latencies, `vlse` vs `vle` | keep reductions and branches out of the scalar domain; strided vs unit-stride NTT layouts |
| A100: 512-bit load channel per core | `mem` `vle` m1…m4, `stream` | bytes per cycle an eltwise kernel can reach |
| A100: "4 scalar cores, 2 vector cores" per cluster (ambiguous) | `--scaling` | whether two harts share a vector unit, and so the thread count for OpenFHE on A100 |
| decoupled vector instruction buffer | `vset` | the cost of `vsetvl` per strip and of switching SEW inside a kernel |

## Thread scaling (`--scaling`)

Every thread runs the same kernel with the same iteration count and starts from a shared barrier:

- `vadd.vv` e64, `vmul.vv` e64 and `vmulhu.vv` e32 (all m1);
- L1-hot `vle64`;
- a 16 MiB-per-thread `vle64` m4 stream.

Columns:

- **`speedup`:** the aggregate instruction rate against the first thread count.
- **`cpus`:** where each thread ran. `a>b` means it migrated; `*` means a VLEN different from the main thread, i.e. it landed on the other cluster.

By default the threads aren't pinned. `--cpus 8,9` pins thread *i* to the *i*-th CPU, to test specific hart pairs. If two harts share one vector unit, a pair shows about 1× on `vmul` but about 2× on scalar-limited work.

Under `ailaunch`, whether threads created after the migration stay on the A100 is itself something to check. Look for `*` in the `cpus` column.

## Laptop check (spike)

Spike validates encodings and register-group rules; its cycle counts are instruction counts, not timing. It needs `_zicntr` for `rdcycle`:

```bash
make CROSS=riscv64-unknown-elf- RISCV_MARCH=rv64gcv uarch-bench
spike --isa=rv64gcv_zicntr_zvl1024b pk build/rvv-release/bin/bench-uarch --min-ms 0.2 --reps 1 --stream-kib 16
```
