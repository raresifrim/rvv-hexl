#!/usr/bin/env bash
# bench/run.sh — run the benchmark matrix and collect every result in one folder.
#
#   bench/run.sh [--isa rvv,scalar] [--cluster x100,a100|none] [--suite hexl,ipcei,upstream]
#                [--quick] [-o DIR]
#
# Matrix (each axis skips what has not been built):
#   ISA      rvv     library/OpenFHE built with V (make ISA=rvv ...)
#            scalar  same scalar extensions, no V (make ISA=scalar ...): the no-RVV baseline
#   path     for every rvv-hexl binary on an rvv build, a second run with
#            HEXL_DISABLE_RVV=1 isolates the hand-written RVV kernels from the
#            compiler's auto-vectorisation of everything else
#   OpenFHE  every build under build/openfhe/<ISA>/ (make openfhe NATIVE_SIZE=.. WITH_RVV_HEXL=..,
#            or make openfhe-all for n64, n32, n64-rvvhexl); word size and backend
#            are read from each install's config_core.h
#   cluster  SpaceMiT K3: x100 (cpus 0-7, VLEN=256) and a100 (cpus 8-15, VLEN=1024,
#            entered through build/tools/ailaunch). Elsewhere: none.
#
# Suites:
#   hexl      bench-hexl: HEXL-standard kernels (Google Benchmark JSON). Builds
#             unchanged against upstream Intel HEXL on x86 for the same table.
#   ipcei     the IPCEI OpenFHE benches, same arguments as ZKP+FHE Research/benchmark.sh
#   upstream  OpenFHE's own Google-Benchmark suite (lib-benchmark, poly-benchmark-*,
#             binfhe-ginx, VectorMath; + the *-hexl ones on rvv-hexl builds)
#
# Before a real run on the K3: performance governor on all cores, idle machine
#   for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | sudo tee $c; done
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAS="" CLUSTERS="" SUITES="hexl,ipcei,upstream" QUICK=0 OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --isa) ISAS="$2"; shift ;;
    --cluster) CLUSTERS="$2"; shift ;;
    --suite) SUITES="$2"; shift ;;
    --quick) QUICK=1 ;;
    -o) OUT="$2"; shift ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
  shift
done

ARCH="$(uname -m)"
if [ -z "$ISAS" ]; then
  if [ "$ARCH" = riscv64 ]; then ISAS="rvv,scalar"; else ISAS="native"; fi
fi
IS_K3=0; [ -e /proc/set_ai_thread ] && IS_K3=1
if [ -z "$CLUSTERS" ]; then
  if [ "$IS_K3" = 1 ]; then CLUSTERS="x100,a100"; else CLUSTERS="none"; fi
fi
HOST="$(hostname -s 2>/dev/null || hostname)"
OUT="${OUT:-$ROOT/results/$HOST-$(date +%Y-%m-%d-%H%M)}"
mkdir -p "$OUT"
AILAUNCH="$ROOT/build/tools/ailaunch"

has() { case ",$1," in *",$2,"*) return 0 ;; *) return 1 ;; esac; }
say() { printf '\033[1;36m==> %s\033[0m\n' "$*"; }

# launch <cluster> <cmd...>: pin to the cluster. env VAR=... must be part of cmd.
launch() {
  local cl="$1"; shift
  case "$cl" in
    x100) taskset -c 0-7 "$@" ;;
    a100)
      [ -x "$AILAUNCH" ] || { echo "missing $AILAUNCH (make tools)" >&2; return 1; }
      # ailaunch execve()s its argument: needs an absolute program path.
      "$AILAUNCH" /usr/bin/env "$@" ;;
    none) "$@" ;;
  esac
}

# run <cluster> <name> <logfile> <cmd...>
run() {
  local cl="$1" name="$2" log="$3"; shift 3
  say "[$cl] $name"
  if ! launch "$cl" "$@" >"$log" 2>&1; then
    echo "   FAILED (see $log)"; tail -3 "$log" | sed 's/^/   /'
  fi
}

# ---------------------------------------------------------------------------
# manifest: everything needed to interpret the numbers later
# ---------------------------------------------------------------------------
{
  echo "date: $(date -Iseconds 2>/dev/null || date)"
  echo "host: $HOST"
  echo "uname: $(uname -a)"
  echo "rvv-hexl git: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo none) $(git -C "$ROOT" status --porcelain 2>/dev/null | grep -q . && echo '(dirty)')"
  echo "isas: $ISAS   clusters: $CLUSTERS   suites: $SUITES   quick: $QUICK"
  for isa in ${ISAS//,/ }; do
    echo "make info ($isa):"; make -s -C "$ROOT" ISA="$isa" info 2>/dev/null | sed 's/^/  /'
  done
  if [ -r /proc/cpuinfo ]; then
    echo "cpuinfo:"; grep -m4 -E "^(isa|uarch|model name|mvendorid|marchid)" /proc/cpuinfo | sed 's/^/  /'
  fi
  echo "governors: $(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
} >"$OUT/manifest.txt"

# ---------------------------------------------------------------------------
for cl in ${CLUSTERS//,/ }; do
  for isa in ${ISAS//,/ }; do
    tag="$isa-$cl"

    # ---- hexl microbenchmarks -------------------------------------------
    if has "$SUITES" hexl; then
      bin="$ROOT/build/$isa-release/bin/bench-hexl"
      if [ -x "$bin" ]; then
        gb=(--benchmark_format=json --benchmark_repetitions=$([ $QUICK = 1 ] && echo 1 || echo 3)
            --benchmark_report_aggregates_only=true)
        [ $QUICK = 1 ] && gb+=(--benchmark_min_time=0.05s)
        run "$cl" "bench-hexl $isa" "$OUT/hexl-$tag.log" \
          env BENCH_CLUSTER="$cl" "$bin" "${gb[@]}" --benchmark_out="$OUT/hexl-$tag.json"
        if [ "$isa" = rvv ]; then
          run "$cl" "bench-hexl $isa (HEXL_DISABLE_RVV=1)" "$OUT/hexl-$tag-norvv.log" \
            env BENCH_CLUSTER="$cl" HEXL_DISABLE_RVV=1 "$bin" "${gb[@]}" \
            --benchmark_out="$OUT/hexl-$tag-norvv.json"
        fi
      else
        echo "skip hexl ($isa): $bin not built (make ISA=$isa bench-hexl)"
      fi
    fi

    # ---- IPCEI OpenFHE benches + upstream OpenFHE benchmarks ------------
    for ofhe in "$ROOT/build/openfhe/$isa"/*/; do
      ofhe="${ofhe%/}"; v="$(basename "$ofhe")"
      cfg="$ofhe/install/include/openfhe/core/config_core.h"
      [ -f "$cfg" ] || continue
      nint="$(sed -n 's/^#define NATIVEINT \([0-9]*\).*/\1/p' "$cfg")"
      hexl=0; grep -q '^#define WITH_INTEL_HEXL' "$cfg" && hexl=1
      case "$v" in *-debug) hbuild=debug ;; *) hbuild=release ;; esac
      ldp="$ofhe/install/lib:$ofhe/install/lib64:$ROOT/build/$isa-$hbuild/install/lib"
      modes=("")
      [ "$hexl" = 1 ] && [ "$isa" = rvv ] && modes+=("HEXL_DISABLE_RVV=1")

      for mode in "${modes[@]}"; do
        sfx="$v${mode:+-norvv}"
        envs=(LD_LIBRARY_PATH="$ldp" ${mode:+"$mode"})

        if has "$SUITES" ipcei; then
          b="$ROOT/build/bench-openfhe/$isa/$v"
          if [ -d "$b" ]; then
            if [ $QUICK = 1 ]; then R=1; S="--sensors 2 --width 8"; else R=3; S="--sensors 5 --width 32"; fi
            if [ "$nint" != 32 ]; then   # 49/60-bit moduli need 64-bit words
              run "$cl" "ntt 49-bit $sfx" "$OUT/ipcei-ntt49-$tag-$sfx.txt" \
                env "${envs[@]}" "$b/openfhe_ntt_bench" 49 4096 8192 16384 32768
              run "$cl" "ntt 27-bit $sfx" "$OUT/ipcei-ntt27-$tag-$sfx.txt" \
                env "${envs[@]}" "$b/openfhe_ntt_bench" 27 1024 2048 4096
              run "$cl" "bfv openmp sweep $sfx" "$OUT/ipcei-bfv-$tag-$sfx.txt" \
                env "${envs[@]}" "$b/direction_a_lwe_openfhe_bench" --rounds $R --depth 10
            fi
            run "$cl" "tfhe stress $sfx" "$OUT/ipcei-tfhe-stress-$tag-$sfx.txt" \
              env "${envs[@]}" OMP_NUM_THREADS=1 "$b/direction_a_tfhe_bench" $S --rounds $R --compress
            run "$cl" "tfhe flags $sfx" "$OUT/ipcei-tfhe-flags-$tag-$sfx.txt" \
              env "${envs[@]}" OMP_NUM_THREADS=1 "$b/direction_a_tfhe_bench" --sensors 5 --rounds $R --mode flags --compress
            run "$cl" "tfhe lut $sfx" "$OUT/ipcei-tfhe-lut-$tag-$sfx.txt" \
              env "${envs[@]}" OMP_NUM_THREADS=1 "$b/direction_a_tfhe_lut_bench"
          else
            echo "skip ipcei ($isa/$v): make ISA=$isa bench"
          fi
        fi

        if has "$SUITES" upstream; then
          ub="$ofhe/build/bin/benchmark"
          if [ -d "$ub" ]; then
            gb=(--benchmark_format=json)
            [ $QUICK = 1 ] && gb+=(--benchmark_min_time=0.05s)
            for u in lib-benchmark poly-benchmark-4k poly-benchmark-16k binfhe-ginx VectorMath \
                     lib-hexl-benchmark poly-hexl-benchmark-4k poly-hexl-benchmark-16k VectorMath-hexl; do
              [ -x "$ub/$u" ] || continue
              case "$nint:$u" in 32:lib-benchmark|32:poly-*) continue ;; esac  # need 64-bit words
              run "$cl" "upstream $u $sfx" "$OUT/upstream-$u-$tag-$sfx.log" \
                env "${envs[@]}" OMP_NUM_THREADS=1 "$ub/$u" "${gb[@]}" \
                --benchmark_out="$OUT/upstream-$u-$tag-$sfx.json"
            done
          fi
        fi
      done
    done
  done
done

say "results in $OUT"
