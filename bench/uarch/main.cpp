// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// bench-uarch: vector-unit microbenchmarks for tuning the RVV kernels.
//
// Measures what the K3 papers leave open (see bench/uarch/README.md): cycles
// per instruction, throughput and latency, of the operations the NTT and eltwise
// kernels are built from, on whichever cluster the process runs:
//
//   taskset -c 0-7 build/rvv-release/bin/bench-uarch            X100 (VLEN 256)
//   build/tools/ailaunch $PWD/build/rvv-release/bin/bench-uarch A100 (VLEN 1024)
//
// Sections
//   single-thread table  every generated kernel (bench/uarch/gen-kernels.py)
//   stream               load/store bandwidth vs working-set size (L1/L2/DRAM)
//   scaling (--scaling)  the same kernels on 1..N threads: do harts share a
//                        vector unit or a load channel?
//
// The kernels are assembly, independent of the HEXL library: this binary
// measures the machine, bench-hexl measures the port.

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#if defined(__linux__)
#include <linux/perf_event.h>
#include <pthread.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <thread>
#define RVVU_LINUX 1
#endif

namespace {

using KernelFn = void (*)(uint64_t, void*, uint64_t);

#define RVVU_KERNEL(sym, ...) extern "C" void sym(uint64_t, void*, uint64_t);
#include "kernels.inc"
#undef RVVU_KERNEL

struct Kernel {
  const char* group;
  const char* name;
  const char* kind;
  int sew;
  int lmul_x8;  // LMUL * 8: mf2 = 4, m1 = 8, m2 = 16, m4 = 32
  int nf;       // segment fields (1 otherwise)
  int per_iter; // instructions under test per loop iteration
  KernelFn fn;
};

#define RVVU_KERNEL(sym, group, name, kind, sew, lmul_x8, nf, per) \
  {group, name, kind, sew, lmul_x8, nf, per, sym},
const Kernel kKernels[] = {
#include "kernels.inc"
};
#undef RVVU_KERNEL

std::string LmulName(int x8) {
  switch (x8) {
    case 4: return "mf2";
    case 8: return "m1";
    case 16: return "m2";
    case 32: return "m4";
    default: return "m8";
  }
}

std::string Id(const Kernel& k) {
  return std::string(k.group) + "/" + k.name + "/e" + std::to_string(k.sew) + "/" +
         LmulName(k.lmul_x8) + "/" + k.kind;
}

uint64_t VlenB() {
  uint64_t v;
  asm volatile("csrr %0, vlenb" : "=r"(v));
  return v;
}

/// Elements per instruction at VL = VLMAX.
uint64_t Vlmax(const Kernel& k) { return VlenB() * 8 * k.lmul_x8 / 8 / k.sew; }

// ---------------------------------------------------------------------------
// clocks
// ---------------------------------------------------------------------------
enum class ClockKind { kPerf, kRdcycle, kTime };

struct Options {
  std::vector<std::string> filters;
  std::vector<std::string> groups;
  bool list = false, stream = true, scaling = false, pin = false;
  std::vector<int> threads = {1, 2, 4, 8};
  std::vector<int> cpus;
  std::vector<uint64_t> stream_kib = {16, 64, 256, 1024, 4096, 65536};
  double min_ms = 20;
  int reps = 5;
  std::string clock = "auto";
  double freq_ghz = 0;
  std::string csv;
};

ClockKind g_clock = ClockKind::kTime;
double g_freq_ghz = 0;  // 0 = unknown (then only ns are reported)

uint64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

#ifdef RVVU_LINUX
/// Per-thread user-mode cycle counter. Needs kernel.perf_event_paranoid <= 2.
int OpenPerfCycles() {
  perf_event_attr a;
  memset(&a, 0, sizeof(a));
  a.type = PERF_TYPE_HARDWARE;
  a.size = sizeof(a);
  a.config = PERF_COUNT_HW_CPU_CYCLES;
  a.exclude_kernel = 1;
  a.exclude_hv = 1;
  return static_cast<int>(syscall(SYS_perf_event_open, &a, 0, -1, -1, 0));
}
thread_local int t_perf_fd = -1;
#endif

uint64_t ReadCycles() {
  switch (g_clock) {
#ifdef RVVU_LINUX
    case ClockKind::kPerf: {
      if (t_perf_fd < 0) t_perf_fd = OpenPerfCycles();
      uint64_t v = 0;
      if (t_perf_fd < 0 || read(t_perf_fd, &v, sizeof(v)) != sizeof(v)) return 0;
      return v;
    }
#endif
    case ClockKind::kRdcycle: {
      uint64_t v;
      asm volatile("rdcycle %0" : "=r"(v));
      return v;
    }
    default:
      return 0;
  }
}

struct Sample {
  double cycles;  // NaN when no cycle source and no frequency
  double ns;
};

Sample Measure(KernelFn fn, uint64_t iters, void* buf, uint64_t arg) {
  const uint64_t c0 = ReadCycles();
  const uint64_t t0 = NowNs();
  fn(iters, buf, arg);
  const uint64_t t1 = NowNs();
  const uint64_t c1 = ReadCycles();
  Sample s{NAN, static_cast<double>(t1 - t0)};
  if (g_clock != ClockKind::kTime) {
    s.cycles = static_cast<double>(c1 - c0);
  } else if (g_freq_ghz > 0) {
    s.cycles = s.ns * g_freq_ghz;
  }
  return s;
}

int CurrentCpu() {
#ifdef RVVU_LINUX
  return sched_getcpu();
#else
  return -1;
#endif
}

double CpufreqGhz(int cpu) {
#ifdef RVVU_LINUX
  if (cpu < 0) return 0;
  const std::string p =
      "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpufreq/scaling_cur_freq";
  if (FILE* f = fopen(p.c_str(), "r")) {
    double khz = 0;
    const int ok = fscanf(f, "%lf", &khz);
    fclose(f);
    if (ok == 1 && khz > 0) return khz / 1e6;
  }
#endif
  return 0;
}

std::string ClockName() {
  switch (g_clock) {
    case ClockKind::kPerf: return "perf_event cycles (user)";
    case ClockKind::kRdcycle: return "rdcycle";
    default:
      return g_freq_ghz > 0 ? "wall clock x " + std::to_string(g_freq_ghz).substr(0, 5) + " GHz"
                            : "wall clock (ns only)";
  }
}

void SetupClock(const Options& o) {
  g_freq_ghz = o.freq_ghz;
  if (o.clock == "rdcycle") {
    g_clock = ClockKind::kRdcycle;
    return;
  }
  if (o.clock == "time") {
    g_clock = ClockKind::kTime;
  } else {
#ifdef RVVU_LINUX
    const int fd = OpenPerfCycles();
    if (fd >= 0) {
      t_perf_fd = fd;
      g_clock = ClockKind::kPerf;
      return;
    }
    fprintf(stderr,
            "bench-uarch: perf_event_open(cycles) failed (%s); using the wall clock.\n"
            "  For real cycle counts: sudo sysctl kernel.perf_event_paranoid=2\n",
            strerror(errno));
    g_clock = ClockKind::kTime;
#else
    g_clock = ClockKind::kRdcycle;  // bare metal (spike + pk): rdcycle is always readable
    return;
#endif
  }
  if (g_freq_ghz <= 0) g_freq_ghz = CpufreqGhz(CurrentCpu());
}

// ---------------------------------------------------------------------------
// measurement
// ---------------------------------------------------------------------------
constexpr size_t kBufBytes = 1 << 20;  // L1-resident kernels touch at most a few KiB

void* AllocBuf(size_t bytes) {
  bytes = (bytes + 4095) & ~static_cast<size_t>(4095);  // aligned_alloc: a multiple of 4096
  void* p = aligned_alloc(4096, bytes);
  if (!p) {
    fprintf(stderr, "bench-uarch: out of memory (%zu bytes)\n", bytes);
    exit(1);
  }
  memset(p, 0, bytes);
  return p;
}

/// Iterations so that one call takes at least min_ms.
uint64_t Calibrate(KernelFn fn, void* buf, uint64_t arg, double min_ms) {
  uint64_t iters = 16;
  fn(iters, buf, arg);  // warm-up: page faults, I-cache, predictors
  for (;;) {
    const Sample s = Measure(fn, iters, buf, arg);
    if (s.ns >= min_ms * 1e6 || iters >= (1ULL << 40)) return iters;
    const double scale = s.ns > 0 ? min_ms * 1e6 / s.ns * 1.2 : 16;
    iters = static_cast<uint64_t>(iters * std::min(16.0, std::max(2.0, scale)));
  }
}

/// Best (minimum) of `reps` runs: the least-disturbed measurement.
Sample Best(KernelFn fn, uint64_t iters, void* buf, uint64_t arg, int reps) {
  Sample best{NAN, INFINITY};
  for (int r = 0; r < reps; ++r) {
    const Sample s = Measure(fn, iters, buf, arg);
    if (s.ns < best.ns) best = s;
  }
  return best;
}

struct Csv {
  FILE* f = nullptr;
  std::string cluster;
  void Row(const std::string& section, const std::string& id, int threads, int cpu, double vl,
           double cpi, double ns, double bytes_per_cycle, double gbps) {
    if (!f) return;
    fprintf(f, "%s,%s,%s,%llu,%d,%d,%.0f,%.4f,%.4f,%.4f,%.4f\n", cluster.c_str(), section.c_str(),
            id.c_str(), static_cast<unsigned long long>(VlenB() * 8), threads, cpu, vl, cpi, ns,
            bytes_per_cycle, gbps);
  }
};

bool Selected(const Kernel& k, const Options& o) {
  if (!o.groups.empty() &&
      std::find(o.groups.begin(), o.groups.end(), std::string(k.group)) == o.groups.end())
    return false;
  if (o.filters.empty()) return true;
  const std::string id = Id(k);
  for (const auto& f : o.filters)
    if (id.find(f) != std::string::npos) return true;
  return false;
}

void RunTable(const Options& o, Csv& csv) {
  void* buf = AllocBuf(kBufBytes);
  printf("\n%-44s %5s %9s %9s %10s\n", "test (group/op/sew/lmul/kind)", "VL", "cyc/insn",
         "ns/insn", "elem/cyc");
  std::string last_group;
  for (const Kernel& k : kKernels) {
    if (std::string(k.group) == "stream" || !Selected(k, o)) continue;
    if (last_group != k.group) {
      printf("-- %s\n", k.group);
      last_group = k.group;
    }
    memset(buf, 0, 64 * 1024);  // the xfer load chase needs zeros; stores dirty the buffer
    const uint64_t iters = Calibrate(k.fn, buf, 0, o.min_ms);
    const Sample s = Best(k.fn, iters, buf, 0, o.reps);
    const double n = static_cast<double>(iters) * k.per_iter;
    const double cpi = s.cycles / n, nspi = s.ns / n;
    const double vl = static_cast<double>(Vlmax(k)) * k.nf;
    char epc[32] = "-";
    if (strcmp(k.kind, "tp") == 0) snprintf(epc, sizeof(epc), "%.2f", vl / cpi);
    printf("%-44s %5.0f %9.2f %9.3f %10s\n", Id(k).c_str(), vl, cpi, nspi, epc);
    fflush(stdout);
    csv.Row("table", Id(k), 1, CurrentCpu(), vl, cpi, nspi, NAN, NAN);
  }
  free(buf);
}

[[maybe_unused]] const Kernel* Find(const char* group, const char* name, int sew, int lmul_x8, const char* kind) {
  for (const Kernel& k : kKernels)
    if (!strcmp(k.group, group) && !strcmp(k.name, name) && k.sew == sew &&
        k.lmul_x8 == lmul_x8 && !strcmp(k.kind, kind))
      return &k;
  return nullptr;
}

constexpr int kUnroll = 16;  // gen-kernels.py UNROLL

/// iterations of a stream kernel that cover `bytes` once
uint64_t StreamIters(const Kernel& k, uint64_t bytes) {
  const uint64_t per_iter_bytes = VlenB() * k.lmul_x8 / 8 * kUnroll;
  return std::max<uint64_t>(1, bytes / per_iter_bytes);
}

/// One pass over `bytes` repeated until min_ms; returns best bytes/cycle and GB/s
void StreamPoint(const Kernel& k, void* buf, uint64_t bytes, const Options& o, double* bpc,
                 double* gbps) {
  const uint64_t it = StreamIters(k, bytes);
  const uint64_t moved = it * VlenB() * k.lmul_x8 / 8 * kUnroll;
  k.fn(it, buf, 0);  // warm-up (fills the caches for the working set)
  uint64_t passes = 1;
  for (;;) {
    const uint64_t t0 = NowNs();
    for (uint64_t p = 0; p < passes; ++p) k.fn(it, buf, 0);
    if ((NowNs() - t0) >= o.min_ms * 1e6 / 4 || passes >= (1u << 24)) break;
    passes *= 2;
  }
  Sample best{NAN, INFINITY};
  for (int r = 0; r < o.reps; ++r) {
    const uint64_t c0 = ReadCycles(), t0 = NowNs();
    for (uint64_t p = 0; p < passes; ++p) k.fn(it, buf, 0);
    const uint64_t t1 = NowNs(), c1 = ReadCycles();
    Sample s{g_clock != ClockKind::kTime ? static_cast<double>(c1 - c0)
                                         : (g_freq_ghz > 0 ? (t1 - t0) * g_freq_ghz : NAN),
             static_cast<double>(t1 - t0)};
    if (s.ns < best.ns) best = s;
  }
  const double total = static_cast<double>(moved) * passes;
  *bpc = total / best.cycles;
  *gbps = total / best.ns;
}

void RunStream(const Options& o, Csv& csv) {
  const uint64_t max_bytes = *std::max_element(o.stream_kib.begin(), o.stream_kib.end()) * 1024;
  void* buf = AllocBuf(max_bytes + 4096);
  printf("\n-- stream: bandwidth vs working set (one thread)\n%-28s", "working set");
  std::vector<const Kernel*> ks;
  for (const Kernel& k : kKernels)
    if (!strcmp(k.group, "stream") && Selected(k, o)) ks.push_back(&k);
  for (auto* k : ks) printf(" %20s", (std::string(k->name) + " " + LmulName(k->lmul_x8)).c_str());
  printf("\n%-28s", "");
  for (size_t i = 0; i < ks.size(); ++i) printf(" %20s", "B/cyc     GB/s");
  printf("\n");
  for (uint64_t kib : o.stream_kib) {
    printf("%-28s", (std::to_string(kib) + " KiB").c_str());
    for (auto* k : ks) {
      double bpc, gbps;
      StreamPoint(*k, buf, kib * 1024, o, &bpc, &gbps);
      printf(" %9.2f %10.2f", bpc, gbps);
      fflush(stdout);
      csv.Row("stream-" + std::to_string(kib) + "KiB", Id(*k), 1, CurrentCpu(),
              static_cast<double>(kib * 1024), NAN, NAN, bpc, gbps);
    }
    printf("\n");
  }
  free(buf);
}

#ifdef RVVU_LINUX
// ---------------------------------------------------------------------------
// thread scaling
// ---------------------------------------------------------------------------
struct ThreadResult {
  int cpu_before = -1, cpu_after = -1;
  uint64_t vlenb = 0;
  bool pinned = false;
  Sample s{NAN, NAN};
};

void RunScaling(const Options& o, Csv& csv) {
  struct Probe {
    const Kernel* k;
    uint64_t bytes;  // 0 = L1 kernel; else per-thread stream working set
  };
  std::vector<Probe> probes;
  auto add = [&](const char* g, const char* n, int sew, int l, uint64_t bytes) {
    if (const Kernel* k = Find(g, n, sew, l, "tp")) probes.push_back({k, bytes});
  };
  add("alu", "vadd.vv", 64, 8, 0);
  add("mul", "vmul.vv", 64, 8, 0);
  add("mul", "vmulhu.vv", 32, 8, 0);
  add("mem", "vle64.v", 64, 8, 0);
  add("stream", "vle64.v", 64, 32, 16u << 20);

  const uint64_t vlenb0 = VlenB();
  printf("\n-- scaling: same work on every thread, all started together%s\n",
         o.pin ? " (pinned, --cpus)" : " (not pinned: the scheduler places threads)");
  printf("%-34s %3s %-24s %10s %12s %9s\n", "test", "thr", "cpus (vlen!=main: *)", "cyc/insn",
         "insn/ns (sum)", "speedup");
  for (const Probe& p : probes) {
    void* cal = AllocBuf(p.bytes ? p.bytes + 4096 : kBufBytes);
    const uint64_t iters = p.bytes ? StreamIters(*p.k, p.bytes)
                                   : Calibrate(p.k->fn, cal, 0, o.min_ms);
    const uint64_t passes = p.bytes ? 32 : 1;
    free(cal);
    double base_rate = 0;
    for (int T : o.threads) {
      std::vector<ThreadResult> res(T);
      std::vector<void*> bufs(T);
      for (int t = 0; t < T; ++t) bufs[t] = AllocBuf(p.bytes ? p.bytes + 4096 : kBufBytes);
      std::atomic<int> ready{0};
      std::atomic<bool> go{false};
      uint64_t wall0 = 0, wall1 = 0;
      std::vector<std::thread> th;
      for (int t = 0; t < T; ++t) {
        th.emplace_back([&, t] {
          ThreadResult& r = res[t];
          if (o.pin && !o.cpus.empty()) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(o.cpus[t % o.cpus.size()], &set);
            r.pinned = pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
          }
          p.k->fn(p.bytes ? iters : std::max<uint64_t>(1, iters / 16), bufs[t], 0);  // warm-up
          r.cpu_before = sched_getcpu();
          r.vlenb = VlenB();
          ready.fetch_add(1);
          while (!go.load(std::memory_order_acquire)) {
          }
          const uint64_t c0 = ReadCycles(), t0 = NowNs();
          for (uint64_t q = 0; q < passes; ++q) p.k->fn(iters, bufs[t], 0);
          const uint64_t t1 = NowNs(), c1 = ReadCycles();
          r.s = {g_clock == ClockKind::kPerf ? static_cast<double>(c1 - c0)
                                             : (g_freq_ghz > 0 ? (t1 - t0) * g_freq_ghz : NAN),
                 static_cast<double>(t1 - t0)};
          r.cpu_after = sched_getcpu();
        });
      }
      while (ready.load() < T) {
      }
      wall0 = NowNs();
      go.store(true, std::memory_order_release);
      for (auto& x : th) x.join();
      wall1 = NowNs();
      for (void* b : bufs) free(b);

      const double insn_per_thread =
          static_cast<double>(iters) * passes * (p.bytes ? kUnroll : p.k->per_iter);
      double cpi_sum = 0;
      std::string cpus;
      bool pin_failed = false;
      for (const auto& r : res) {
        cpi_sum += r.s.cycles / insn_per_thread;
        if (!cpus.empty()) cpus += ",";
        cpus += std::to_string(r.cpu_before);
        if (r.cpu_after != r.cpu_before) cpus += ">" + std::to_string(r.cpu_after);
        if (r.vlenb != vlenb0) cpus += "*";
        if (o.pin && !r.pinned) pin_failed = true;
      }
      const double rate = insn_per_thread * T / static_cast<double>(wall1 - wall0);
      if (T == o.threads.front()) base_rate = rate / T;
      const std::string name = Id(*p.k) + (p.bytes ? " " + std::to_string(p.bytes >> 20) + "MiB" : "");
      printf("%-34s %3d %-24s %10.2f %12.3f %8.2fx%s\n", name.c_str(), T, cpus.c_str(),
             cpi_sum / T, rate, rate / base_rate, pin_failed ? "  (pinning failed)" : "");
      fflush(stdout);
      csv.Row("scaling", name, T, -1, NAN, cpi_sum / T, 1.0 / rate * T, NAN, NAN);
    }
  }
}
#endif

// ---------------------------------------------------------------------------
std::vector<std::string> Split(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i <= s.size()) {
    const size_t j = s.find(',', i);
    const std::string t = s.substr(i, j == std::string::npos ? std::string::npos : j - i);
    if (!t.empty()) out.push_back(t);
    if (j == std::string::npos) break;
    i = j + 1;
  }
  return out;
}

void Usage() {
  printf(
      "usage: bench-uarch [options]\n"
      "  --list                list the tests (group/op/sew/lmul/kind) and exit\n"
      "  --filter A,B          only tests whose id contains A or B (e.g. vmul, e32/m1, /lat)\n"
      "  --group G,..          alu,mul,perm,mem,xfer,vset,stream\n"
      "  --no-stream           skip the working-set sweep\n"
      "  --stream-kib K,..     working sets in KiB (default 16,64,256,1024,4096,65536)\n"
      "  --scaling             thread-scaling section (Linux)\n"
      "  --threads 1,2,4,8     thread counts for --scaling\n"
      "  --cpus 8,9,..         pin scaling thread i to the i-th cpu of the list\n"
      "  --min-ms X            minimum time per measurement (default 20)\n"
      "  --reps N              repetitions, the best is reported (default 5)\n"
      "  --quick               --min-ms 4 --reps 2\n"
      "  --clock perf|time|rdcycle   cycle source (default: perf, else wall clock)\n"
      "  --freq-ghz F          core clock for --clock time (default: cpufreq)\n"
      "  --csv FILE            also write every number as CSV\n");
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        fprintf(stderr, "bench-uarch: %s needs a value\n", a.c_str());
        exit(2);
      }
      return argv[++i];
    };
    if (a == "--list") o.list = true;
    else if (a == "--filter") o.filters = Split(next());
    else if (a == "--group") o.groups = Split(next());
    else if (a == "--no-stream") o.stream = false;
    else if (a == "--stream-kib") {
      o.stream_kib.clear();
      for (auto& s : Split(next())) o.stream_kib.push_back(std::stoull(s));
    } else if (a == "--scaling") o.scaling = true;
    else if (a == "--threads") {
      o.threads.clear();
      for (auto& s : Split(next())) o.threads.push_back(std::stoi(s));
    } else if (a == "--cpus") {
      for (auto& s : Split(next())) o.cpus.push_back(std::stoi(s));
      o.pin = true;
    } else if (a == "--min-ms") o.min_ms = std::stod(next());
    else if (a == "--reps") o.reps = std::stoi(next());
    else if (a == "--quick") {
      o.min_ms = 4;
      o.reps = 2;
    } else if (a == "--clock") o.clock = next();
    else if (a == "--freq-ghz") o.freq_ghz = std::stod(next());
    else if (a == "--csv") o.csv = next();
    else if (a == "-h" || a == "--help") {
      Usage();
      return 0;
    } else {
      fprintf(stderr, "bench-uarch: unknown option %s\n", a.c_str());
      Usage();
      return 2;
    }
  }
  if (o.threads.empty() || o.stream_kib.empty() || o.reps < 1) {
    fprintf(stderr, "bench-uarch: empty --threads/--stream-kib or --reps < 1\n");
    return 2;
  }

  if (o.list) {
    for (const Kernel& k : kKernels)
      if (Selected(k, o)) printf("%s\n", Id(k).c_str());
    return 0;
  }

  SetupClock(o);
  Csv csv;
  if (const char* c = getenv("BENCH_CLUSTER")) csv.cluster = c;
  if (!o.csv.empty()) {
    csv.f = fopen(o.csv.c_str(), "w");
    if (!csv.f) {
      fprintf(stderr, "bench-uarch: cannot write %s\n", o.csv.c_str());
      return 1;
    }
    fprintf(csv.f, "cluster,section,test,vlen,threads,cpu,vl_or_bytes,cyc_per_insn,ns_per_insn,"
                   "bytes_per_cyc,gb_per_s\n");
  }

  printf("bench-uarch  VLEN %llu  cpu %d%s%s  clock: %s\n",
         static_cast<unsigned long long>(VlenB() * 8), CurrentCpu(),
         csv.cluster.empty() ? "" : "  cluster ", csv.cluster.c_str(), ClockName().c_str());
  printf("tp = throughput (independent instructions), lat = dependent chain; VL = VLMAX (x NF for "
         "segments)\n");
#ifdef RVVU_LINUX
  if (g_clock == ClockKind::kPerf) {  // sanity check of the counter against the wall clock
    const Kernel* k = Find("alu", "vadd.vv", 64, 8, "tp");
    void* b = AllocBuf(kBufBytes);
    const Sample s = Measure(k->fn, Calibrate(k->fn, b, 0, 10), b, 0);
    printf("effective clock %.3f GHz (perf cycles / wall time; expect ~2.4 X100, ~2.0 A100 under "
           "the performance governor)\n",
           s.cycles / s.ns);
    free(b);
  }
#endif

  RunTable(o, csv);
  if (o.stream) RunStream(o, csv);
#ifdef RVVU_LINUX
  if (o.scaling) RunScaling(o, csv);
#else
  if (o.scaling) printf("\n(--scaling needs Linux threads; skipped)\n");
#endif
  if (csv.f) fclose(csv.f);
  return 0;
}
