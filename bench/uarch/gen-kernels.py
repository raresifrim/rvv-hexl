#!/usr/bin/env python3
# Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
# SPDX-License-Identifier: Apache-2.0
"""Generate the bench-uarch kernels: one assembly function per measurement.

    gen-kernels.py <outdir>      -> <outdir>/kernels.S, <outdir>/kernels.inc

Every kernel has the C signature
    void k(uint64_t iters, void* buf, uint64_t arg)
and runs `iters` iterations of a fully unrolled body of PER instructions under
test, followed by `addi a0,a0,-1; bnez a0,1b`. Assembly rather than intrinsics
so that the compiler can neither drop, reorder nor re-vtype anything: what is
listed below is exactly what the vector unit sees.

Kinds
  tp   throughput: the PER instructions rotate over up to 8 independent
       destination groups, so only issue/execution bandwidth limits them
  lat  latency: every instruction depends on the previous one's result

Register allocation per LMUL (v0 is kept free for masks): source groups first
(vid.v initialised, so vrgather indices are always in range), then as many
aligned, non-overlapping destination groups as fit (max 8).
"""
import os
import re
import sys

UNROLL = 16
LMULS = {"mf2": 0.5, "m1": 1, "m2": 2, "m4": 4}
LMUL_X8 = {"mf2": 4, "m1": 8, "m2": 16, "m4": 32, "m8": 64}


def regs(lmul):
    """Registers in a group of this LMUL (fractional LMUL still takes one)."""
    return max(1, int(LMULS.get(lmul, 0) or {"m8": 8}[lmul]))


def wider(lmul):
    return {"mf2": "m1", "m1": "m2", "m2": "m4", "m4": "m8"}[lmul]


class Alloc:
    def __init__(self):
        self.used = {0}

    def take(self, n):
        for start in range(n, 32, n):
            if not any(r in self.used for r in range(start, start + n)):
                self.used.update(range(start, start + n))
                return start
        return None

    def many(self, n, limit=8):
        out = []
        while len(out) < limit:
            r = self.take(n)
            if r is None:
                break
            out.append(r)
        return out


kernels = []   # (symbol, id fields..., body text)


def vset(sew, lmul, mask_undisturbed=False):
    return f"vsetvli t0, zero, e{sew}, {lmul}, ta, {'mu' if mask_undisturbed else 'ma'}"


def mask_init(sew, lmul, mu=False):
    # v0 = 0b0101... as a mask, then back to the kernel's vtype
    return ["vsetvli t0, zero, e8, m1, ta, ma", "li t2, 0x55", "vmv.v.x v0, t2",
            vset(sew, lmul, mu)]


def emit(group, name, kind, sew, lmul, prologue, body, nf=1, per=None):
    sym = "rvvu_" + re.sub(r"[^A-Za-z0-9]+", "_", "_".join((group, name, kind, f"e{sew}", lmul)))
    assert all(k["sym"] != sym for k in kernels), sym
    per = per if per is not None else len(body)
    kernels.append(dict(sym=sym, group=group, name=name, kind=kind, sew=sew,
                        lmul=lmul, nf=nf, per=per, prologue=prologue, body=body))


# ---------------------------------------------------------------------------
# arithmetic / permutation
# ---------------------------------------------------------------------------
# shape: dst 'v' (LMUL), 'w' (2*LMUL, widening), 'm' (mask register)
#        src2 'v' or 'w' (narrowing: vs2 is 2*LMUL)
# tmpl placeholders: {d} dest, {a} vs2-ish source, {b} second source
# lat: template for the dependent chain ({d} <- op({p}, ...)), None = no lat
# acc: the destination is also an input (vmacc & co): lat chains through vd
OPS = [
    # group, name, template, dst, src2, sews, lmuls, lat, flags
    ("alu", "vadd.vv", "vadd.vv {d}, {a}, {b}", "v", "v", (32, 64), "all", "vadd.vv {d}, {p}, {b}", ""),
    ("alu", "vsub.vx", "vsub.vx {d}, {a}, t2", "v", "v", (32, 64), "all", None, ""),
    ("alu", "vand.vv", "vand.vv {d}, {a}, {b}", "v", "v", (32, 64), "all", None, ""),
    ("alu", "vsrl.vx", "vsrl.vx {d}, {a}, t3", "v", "v", (32, 64), "all", "vsrl.vx {d}, {p}, t3", ""),
    ("alu", "vminu.vv", "vminu.vv {d}, {a}, {b}", "v", "v", (32, 64), "all", "vminu.vv {d}, {p}, {b}", ""),
    ("alu", "vmsltu.vv", "vmsltu.vv {d}, {a}, {b}", "m", "v", (32, 64), "all", None, ""),
    ("alu", "vmerge.vvm", "vmerge.vvm {d}, {a}, {b}, v0", "v", "v", (32, 64), "all", None, "mask"),
    ("alu", "vadd.vv.masked", "vadd.vv {d}, {a}, {b}, v0.t", "v", "v", (32, 64), "all", None, "mask,mu"),
    ("mul", "vmul.vv", "vmul.vv {d}, {a}, {b}", "v", "v", (32, 64), "all", "vmul.vv {d}, {p}, {b}", ""),
    ("mul", "vmul.vx", "vmul.vx {d}, {a}, t2", "v", "v", (32, 64), "all", None, ""),
    ("mul", "vmulhu.vv", "vmulhu.vv {d}, {a}, {b}", "v", "v", (32, 64), "all", "vmulhu.vv {d}, {p}, {b}", ""),
    ("mul", "vmulhu.vx", "vmulhu.vx {d}, {a}, t2", "v", "v", (32, 64), "all", None, ""),
    ("mul", "vmacc.vv", "vmacc.vv {d}, {a}, {b}", "v", "v", (32, 64), "all", "vmacc.vv {d}, {a}, {b}", "acc"),
    ("mul", "vwmulu.vv", "vwmulu.vv {d}, {a}, {b}", "w", "v", (32,), ("mf2", "m1", "m2"), None, ""),
    ("mul", "vwmaccu.vv", "vwmaccu.vv {d}, {a}, {b}", "w", "v", (32,), ("mf2", "m1", "m2"), None, "acc"),
    ("alu", "vnsrl.wx", "vnsrl.wx {d}, {a}, t3", "v", "w", (32,), ("mf2", "m1", "m2"), None, ""),
    ("perm", "vrgather.vv", "vrgather.vv {d}, {a}, {b}", "v", "v", (32, 64), "all", "vrgather.vv {d}, {p}, {b}", ""),
    ("perm", "vslidedown.vi", "vslidedown.vi {d}, {a}, 1", "v", "v", (32, 64), "all", "vslidedown.vi {d}, {p}, 1", ""),
    ("perm", "vslideup.vi", "vslideup.vi {d}, {a}, 1", "v", "v", (32, 64), "all", None, ""),
    ("perm", "vcompress.vm", "vcompress.vm {d}, {a}, v0", "v", "v", (32, 64), "all", None, "mask"),
]

for group, name, tmpl, dst, src2, sews, lmuls, lat, flags in OPS:
    flags = flags.split(",") if flags else []
    for sew in sews:
        for lmul in (LMULS if lmuls == "all" else lmuls):
            if sew == 64 and lmul == "mf2":
                continue            # SEW=64 needs LMUL >= SEW/ELEN = 1
            g = regs(lmul)
            al = Alloc()
            a = al.take(regs(wider(lmul)) if src2 == "w" else g)
            b = al.take(g)
            dsz = {"v": g, "w": regs(wider(lmul)), "m": 1}[dst]
            mu = "mu" in flags
            pro = (mask_init(sew, lmul, mu) if "mask" in flags else [vset(sew, lmul, mu)])
            pro += [f"vid.v v{b}", "li t2, 3", "li t3, 7"]
            if src2 == "v":
                pro.insert(len(pro) - 2, f"vid.v v{a}")
            dests = al.many(dsz)
            body = [tmpl.format(d=f"v{dests[i % len(dests)]}", a=f"v{a}", b=f"v{b}")
                    for i in range(UNROLL)]
            emit(group, name, "tp", sew, lmul, pro, body)
            if lat:
                if "acc" in flags:     # chain through the accumulator
                    body = [lat.format(d=f"v{dests[0]}", a=f"v{a}", b=f"v{b}")] * UNROLL
                else:                  # ping-pong: dest never overlaps its own source
                    x, y = dests[0], dests[1]
                    body = [lat.format(d=f"v{(x, y)[i % 2]}", p=f"v{(y, x)[i % 2]}", b=f"v{b}")
                            for i in range(UNROLL)]
                emit(group, name, "lat", sew, lmul, pro + [f"vid.v v{dests[0]}", f"vid.v v{dests[1]}"], body)

# ---------------------------------------------------------------------------
# memory: L1-resident (the buffer is re-read every iteration)
# ---------------------------------------------------------------------------
def group_bytes(lmul):
    """t1 = bytes in one register group of this LMUL."""
    shift = {"mf2": ("srli", 1), "m1": None, "m2": ("slli", 1), "m4": ("slli", 2)}[lmul]
    return ["csrr t1, vlenb"] + ([f"{shift[0]} t1, t1, {shift[1]}"] if shift else [])


def bases(n=4):
    """a3..a6 = buf + k*t1 (4 disjoint groups): unit-stride accesses rotate over them."""
    out = ["mv a3, a1"]
    for i in range(1, n):
        out.append(f"add a{3 + i}, a{2 + i}, t1")
    return out


for sew in (32, 64):
    for lmul in LMULS:
        if sew == 64 and lmul == "mf2":
            continue
        g = regs(lmul)
        # unit stride load / masked load / store
        al = Alloc()
        dests = al.many(g)
        base = [vset(sew, lmul)] + group_bytes(lmul) + bases()
        emit("mem", f"vle{sew}.v", "tp", sew, lmul, base,
             [f"vle{sew}.v v{dests[i % len(dests)]}, (a{3 + i % 4})" for i in range(UNROLL)])
        emit("mem", f"vle{sew}.v.masked", "tp", sew, lmul,
             mask_init(sew, lmul, True) + group_bytes(lmul) + bases(),
             [f"vle{sew}.v v{dests[i % len(dests)]}, (a{3 + i % 4}), v0.t" for i in range(UNROLL)])
        emit("mem", f"vse{sew}.v", "tp", sew, lmul, base + [f"vid.v v{d}" for d in dests],
             [f"vse{sew}.v v{dests[i % len(dests)]}, (a{3 + i % 4})" for i in range(UNROLL)])
        # strided: every other element (NTT-like), and one element per 64-byte line
        for tag, stride in (("s2", 2 * sew // 8), ("s64B", 64)):
            emit("mem", f"vlse{sew}.v.{tag}", "tp", sew, lmul, [vset(sew, lmul), f"li t4, {stride}"],
                 [f"vlse{sew}.v v{dests[i % len(dests)]}, (a1), t4" for i in range(UNROLL)])
        # indexed, reversed order (a permutation inside one group's footprint)
        al = Alloc()
        idx = al.take(g)
        dests_i = al.many(g)
        sh = {32: 2, 64: 3}[sew]
        emit("mem", f"vluxei{sew}.v.rev", "tp", sew, lmul,
             [vset(sew, lmul), f"vid.v v{idx}", f"vrsub.vx v{idx}, v{idx}, t0",   # t0 = VLMAX
              f"vadd.vi v{idx}, v{idx}, -1", f"vsll.vi v{idx}, v{idx}, {sh}"],
             [f"vluxei{sew}.v v{dests_i[i % len(dests_i)]}, (a1), v{idx}" for i in range(UNROLL)])
        # segment loads/stores (NTT butterfly pairs / quads)
        for nf in (2, 4):
            if nf * g > 8:
                continue
            al = Alloc()
            segd = al.many(nf * g)
            # bases nf groups apart: consecutive segment accesses never overlap
            seg_base = ([vset(sew, lmul)] + group_bytes(lmul) + [f"slli t1, t1, {nf.bit_length() - 1}"]
                        + bases())
            emit("mem", f"vlseg{nf}e{sew}.v", "tp", sew, lmul, seg_base,
                 [f"vlseg{nf}e{sew}.v v{segd[i % len(segd)]}, (a{3 + i % 4})" for i in range(UNROLL)], nf=nf)
            if nf == 2:
                emit("mem", f"vsseg{nf}e{sew}.v", "tp", sew, lmul, seg_base,
                     [f"vsseg{nf}e{sew}.v v{segd[i % len(segd)]}, (a{3 + i % 4})" for i in range(UNROLL)], nf=nf)

# ---------------------------------------------------------------------------
# latency across the scalar <-> vector boundary (decoupled vector unit)
# ---------------------------------------------------------------------------
for sew in (32, 64):
    lmul = "m1"
    # load-to-use: vle -> vmv.x.s -> next address (the buffer is zero: same address)
    emit("xfer", f"vle{sew}+vmv.x.s", "lat", sew, lmul, [vset(sew, lmul)],
         [x for _ in range(UNROLL // 2) for x in
          (f"vle{sew}.v v8, (a1)", "vmv.x.s t4, v8", "add a1, a1, t4")], per=UNROLL // 2)
    # scalar -> vector -> scalar round trip
    emit("xfer", "vmv.s.x+vmv.x.s", "lat", sew, lmul, [vset(sew, lmul), "li t4, 0"],
         [x for _ in range(UNROLL // 2) for x in ("vmv.s.x v8, t4", "vmv.x.s t4, v8")], per=UNROLL // 2)
    # vector compare feeding a scalar decision: vmsltu.vx -> vcpop.m -> next compare
    emit("xfer", "vmsltu.vx+vcpop.m", "lat", sew, lmul, [vset(sew, lmul), "vid.v v8", "li t4, 0"],
         [x for _ in range(UNROLL // 2) for x in ("vmsltu.vx v1, v8, t4", "vcpop.m t4, v1")], per=UNROLL // 2)

# ---------------------------------------------------------------------------
# vsetvli: cost of re-configuring (a strip-mined loop does it every iteration)
# ---------------------------------------------------------------------------
for tag, types in (("same", ("e32, m1", "e32, m1")),
                   ("e32<->e64", ("e32, m1", "e64, m1")),
                   ("m1<->m2", ("e32, m1", "e32, m2"))):
    body = []
    for i in range(UNROLL):
        body += [f"vsetvli t0, a2, {types[i % 2]}, ta, ma", f"vadd.vv v{8 + 2 * (i % 4)}, v2, v4"]
    emit("vset", f"vsetvli.{tag}+vadd", "tp", 32, "m1",
         [vset(32, "m1"), "vid.v v2", "vid.v v4", "li a2, -1"], body, per=UNROLL)

# ---------------------------------------------------------------------------
# streaming: the C++ side sweeps working-set sizes (L1 / L2 / DRAM)
# arg-less: iters = bytes / (UNROLL * group bytes); a1 advances through the buffer
# ---------------------------------------------------------------------------
for op, lmul in (("vle64.v", "m1"), ("vle64.v", "m4"), ("vse64.v", "m4")):
    g = regs(lmul)
    dests = Alloc().many(g)
    body = []
    for i in range(UNROLL):
        body += [f"{op} v{dests[i % len(dests)]}, (a1)", "add a1, a1, t1"]
    emit("stream", op, "tp", 64, lmul, [vset(64, lmul)] + group_bytes(lmul), body, per=UNROLL)

# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
out = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "kernels.S"), "w") as f:
    f.write("# generated by bench/uarch/gen-kernels.py, do not edit\n")
    f.write("# void k(uint64_t iters /*a0*/, void* buf /*a1*/, uint64_t arg /*a2*/)\n")
    f.write("\t.text\n")
    for k in kernels:
        f.write(f"\n\t.globl {k['sym']}\n\t.type {k['sym']}, @function\n\t.p2align 4\n{k['sym']}:\n")
        for ins in k["prologue"]:
            f.write(f"\t{ins}\n")
        f.write("\t.p2align 4\n1:\n")
        for ins in k["body"]:
            f.write(f"\t{ins}\n")
        f.write("\taddi a0, a0, -1\n\tbnez a0, 1b\n\tret\n")
        f.write(f"\t.size {k['sym']}, .-{k['sym']}\n")
    f.write('\t.section .note.GNU-stack,"",@progbits\n')
with open(os.path.join(out, "kernels.inc"), "w") as f:
    f.write("// generated by bench/uarch/gen-kernels.py, do not edit\n")
    f.write("// RVVU_KERNEL(symbol, group, name, kind, sew, lmul_x8, nf, per_iter)\n")
    for k in kernels:
        f.write(f'RVVU_KERNEL({k["sym"]}, "{k["group"]}", "{k["name"]}", "{k["kind"]}", '
                f'{k["sew"]}, {LMUL_X8[k["lmul"]]}, {k["nf"]}, {k["per"]})\n')
print(f"bench-uarch: {len(kernels)} kernels -> {out}")
