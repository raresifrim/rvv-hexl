#!/usr/bin/env python3
"""Build a searchable index of every RVA23 RVV C intrinsic in docs/rvv_intrinsics.

    gen-rvv-intrinsics-index.py <docs/rvv_intrinsics> [--cc <riscv gcc>]

Parses the chapter files of rvv-intrinsic-doc (base V, vector-crypto, bfloat16),
classifies each intrinsic by ISA extension, marks its RVA23 status and whether
the SpaceMiT K3 implements that extension, and (with --cc) whether the given
compiler actually declares it. Writes

    INDEX.md               per-extension summary + what the compiler is missing
    rva23-intrinsics.tsv   one line per intrinsic (explicit, policy, overloaded)

The compiler probe compiles `(void)__riscv_xxx;` for every name with
-fsyntax-only: GCC declares all intrinsics it implements (explicit, policy and
overloaded) once riscv_vector.h is included, and reports the others as
"undeclared". Nothing is linked or run.
"""
import argparse
import collections
import concurrent.futures
import os
import re
import subprocess
import sys
import tempfile

# (directory relative to root, kind, family)
SOURCES = [
    ("auto-generated/intrinsic_funcs", "explicit", "base"),
    ("auto-generated/policy_funcs/intrinsic_funcs", "policy", "base"),
    ("auto-generated/overloaded_intrinsic_funcs", "overloaded", "base"),
    ("auto-generated/policy_funcs/overloaded_intrinsic_funcs", "policy-overloaded", "base"),
    ("auto-generated/vector-crypto/intrinsic_funcs", "explicit", "crypto"),
    ("auto-generated/vector-crypto/policy_funcs/intrinsic_funcs", "policy", "crypto"),
    ("auto-generated/vector-crypto/overloaded_intrinsic_funcs", "overloaded", "crypto"),
    ("auto-generated/vector-crypto/policy_funcs/overloaded_intrinsic_funcs", "policy-overloaded", "crypto"),
    ("auto-generated/bfloat16/intrinsic_funcs", "explicit", "bf16"),
    ("auto-generated/bfloat16/policy_funcs/intrinsic_funcs", "policy", "bf16"),
    ("auto-generated/bfloat16/overloaded_intrinsic_funcs", "overloaded", "bf16"),
    ("auto-generated/bfloat16/policy_funcs/overloaded_intrinsic_funcs", "policy-overloaded", "bf16"),
]

# extension -> (RVA23U64 status, implemented on SpaceMiT K3 per /proc/cpuinfo)
EXTENSIONS = collections.OrderedDict([
    ("V",           ("mandatory", True)),
    ("Zvfhmin",     ("mandatory", True)),
    ("Zvfh",        ("optional (expansion)", True)),
    ("Zvbb",        ("mandatory", True)),
    ("Zvbb/Zvkb",   ("mandatory (Zvbb)", True)),
    ("Zvbc",        ("optional (development)", True)),
    ("Zvkg",        ("optional (in Zvkng / Zvksg)", True)),
    ("Zvkned",      ("optional (in Zvkng)", True)),
    ("Zvknha/Zvknhb", ("optional (in Zvkng)", True)),
    ("Zvknhb",      ("optional (in Zvkng)", True)),
    ("Zvksed",      ("optional (in Zvksg)", True)),
    ("Zvksh",       ("optional (in Zvksg)", True)),
    ("Zvfbfmin",    ("optional (expansion)", False)),
    ("Zvfbfwma",    ("optional (expansion)", False)),
])

CRYPTO_FILE_EXT = {"zvbb": "Zvbb", "zvbc": "Zvbc", "zvkg": "Zvkg", "zvkned": "Zvkned",
                   "zvknh": "Zvknha/Zvknhb", "zvksed": "Zvksed", "zvksh": "Zvksh"}
ZVKB_OPS = {"vandn", "vbrev8", "vrev8", "vrol", "vror"}

PROTO_RE = re.compile(r"^(?P<ret>.*?)\b(?P<name>__riscv_\w+)\s*\((?P<args>.*)\)\s*;$")


def op_of(name):
    """__riscv_vfwcvt_f_f_v_f32m1 -> vfwcvt"""
    return name[len("__riscv_"):].split("_")[0]


def classify(family, fname, name, proto):
    op = op_of(name)
    if family == "crypto":
        key = os.path.basename(fname).split("_-_")[0].split("_", 1)[1]
        ext = CRYPTO_FILE_EXT.get(key, key)
        if ext == "Zvbb" and op in ZVKB_OPS:
            return "Zvbb/Zvkb"
        if ext == "Zvknha/Zvknhb" and "uint64" in proto:
            return "Zvknhb"  # SHA-512 (SEW=64) only in Zvknhb
        return ext
    if family == "bf16":
        return "Zvfbfwma" if op == "vfwmaccbf16" else "Zvfbfmin"
    if "float16" not in proto:
        return "V"
    # float16: Zvfhmin = f16<->f32 converts + everything that only moves data
    if name.startswith("__riscv_vfwcvt_f_f_v_f32") or name.startswith("__riscv_vfncvt_f_f_w_f16"):
        return "Zvfhmin"
    if op.startswith("vf") or op.startswith("vmf"):
        return "Zvfh"
    return "Zvfhmin"


def parse(root):
    rows = []
    for rel, kind, family in SOURCES:
        d = os.path.join(root, rel)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".adoc"):
                continue
            path = os.path.join(d, fn)
            section, in_code, buf = "", False, []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    s = line.rstrip("\n")
                    if s.startswith("===="):
                        section = s.lstrip("=").strip()
                    if s == "----":
                        in_code = not in_code
                        buf = []
                        continue
                    if not in_code or not s.strip():
                        continue
                    buf.append(s.strip())
                    if s.rstrip().endswith(";"):
                        proto = re.sub(r"\s+", " ", " ".join(buf)).replace("( ", "(")
                        buf = []
                        m = PROTO_RE.match(proto)
                        if not m:
                            continue
                        name = m.group("name")
                        rows.append({
                            "name": name, "kind": kind, "family": family,
                            "ext": classify(family, fn, name, proto),
                            "section": section, "proto": proto,
                            "file": os.path.relpath(path, root),
                        })
    return rows


def probe(cc, names):
    """Returns (march used, compiler version, set of names the compiler does NOT declare)."""
    env = dict(os.environ, LC_ALL="C")
    marches = ["rva23u64_zvbc_zvkg_zvkned_zvknhb_zvksed_zvksh_zvfbfmin_zvfbfwma",
               "rva23u64_zvbc_zvkg_zvkned_zvknhb_zvksed_zvksh", "rva23u64",
               "rv64gcv_zvbb_zvbc_zvkg_zvkned_zvknhb_zvksed_zvksh_zvfh", "rv64gcv"]
    march = None
    for m in marches:
        r = subprocess.run([cc, "-march=" + m, "-x", "c", "-fsyntax-only", "-"],
                           input="#include <riscv_vector.h>\n", text=True,
                           capture_output=True, env=env)
        if r.returncode == 0:
            march = m
            break
    if march is None:
        return None, None, None
    version = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()[0]

    names = sorted(names)
    chunks = [names[i:i + 6000] for i in range(0, len(names), 6000)]
    undeclared = re.compile(r"'(__riscv_\w+)' undeclared")

    def run(chunk):
        with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as t:
            t.write("#include <riscv_vector.h>\nvoid probe(void) {\n")
            t.writelines("  (void)%s;\n" % n for n in chunk)
            t.write("}\n")
        try:
            r = subprocess.run([cc, "-march=" + march, "-fsyntax-only", "-fmax-errors=0",
                                "-w", t.name], capture_output=True, text=True, env=env)
            return set(undeclared.findall(r.stderr))
        finally:
            os.unlink(t.name)

    missing = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        for s in ex.map(run, chunks):
            missing |= s
    return march, version, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--cc")
    a = ap.parse_args()

    rows = parse(a.root)
    if not rows:
        sys.exit("no intrinsics parsed under %s" % a.root)

    march = version = missing = None
    if a.cc:
        march, version, missing = probe(a.cc, {r["name"] for r in rows})
        if march is None:
            print("[rvv-intrinsics] %s cannot compile riscv_vector.h: skipping the compiler column" % a.cc)
    for r in rows:
        r["gcc"] = "?" if missing is None else ("no" if r["name"] in missing else "yes")
        status, k3 = EXTENSIONS.get(r["ext"], ("?", False))
        r["rva23"], r["k3"] = status, "yes" if k3 else "no"

    cols = ["name", "ext", "rva23", "k3", "gcc", "kind", "section", "proto", "file"]
    with open(os.path.join(a.root, "rva23-intrinsics.tsv"), "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(r[c] for c in cols) + "\n")

    # ---- INDEX.md ----
    count = collections.Counter((r["ext"], r["kind"]) for r in rows)
    miss = collections.Counter((r["ext"], r["kind"]) for r in rows if r["gcc"] == "no")
    kinds = ["explicit", "policy", "overloaded", "policy-overloaded"]
    files = collections.defaultdict(set)
    for r in rows:
        if r["kind"] == "explicit":
            files[r["ext"]].add(r["file"])

    out = []
    w = out.append
    w("# RVA23 RVV C intrinsics: index\n")
    w("Generated by `docs/tools/gen-rvv-intrinsics-index.py` (`make rvv-intrinsics-doc`). "
      "Sources: rvv-intrinsic-doc `v1.0-ratified` (base V, Zvfh, Zvfhmin) plus `main` for the "
      "ratified extensions the tag predates (vector crypto, BF16). Draft extensions are excluded.\n")
    if missing is not None:
        w("Compiler column: `%s`, probed with `-march=%s`. For overloaded names it only says "
          "that the name exists (e.g. `__riscv_vcpop` exists for masks even if the Zvbb "
          "overload does not): check the explicit name.\n" % (version, march))
    else:
        w("Compiler column: not probed (no RISC-V compiler found; set PROBE_CC).\n")
    w("SpaceMiT K3 column: extensions in the K3's `/proc/cpuinfo` isa string (identical on "
      "X100 and A100); check yours with `grep -m1 ^isa /proc/cpuinfo`.\n")
    w("| Extension | RVA23U64 | SpaceMiT K3 | explicit | policy (`_tu/_mu/...`) | overloaded | "
      "compiler: explicit / policy missing | chapter file(s) |")
    w("|---|---|---|---:|---:|---:|---|---|")
    for ext, (status, k3) in EXTENSIONS.items():
        if not any(count[(ext, k)] for k in kinds):
            continue
        n = [count[(ext, k)] for k in kinds]
        m = "n/a" if missing is None else "%d / %d" % (miss[(ext, "explicit")], miss[(ext, "policy")])
        fl = "<br>".join("`%s`" % p for p in sorted(files[ext]))
        w("| %s | %s | %s | %d | %d | %d | %s | %s |" % (
            ext, status, "yes" if k3 else "**no**", n[0], n[1], n[2] + n[3], m, fl))
    w("\nZvkt (mandatory) only guarantees data-independent timing: it has no intrinsics. "
      "Zvkng = Zvkned + Zvknhb + Zvkb + Zvkg + Zvkt; Zvksg = Zvksed + Zvksh + Zvkb + Zvkg + Zvkt.\n")

    if missing:
        w("## Intrinsics the compiler does not implement\n")
        w("Grouped by operation (explicit + policy names). Everything else in the index compiles.\n")
        w("| Extension | Operation | missing / total | example |")
        w("|---|---|---|---|")
        tot = collections.Counter()
        mis = collections.defaultdict(list)
        for r in rows:
            if r["kind"] in ("explicit", "policy"):
                key = (r["ext"], op_of(r["name"]))
                tot[key] += 1
                if r["gcc"] == "no":
                    mis[key].append(r["name"])
        for key in sorted(mis):
            w("| %s | `%s` | %d / %d | `%s` |" % (key[0], key[1], len(mis[key]), tot[key], mis[key][0]))
        w("")

    w("## Searching\n")
    w("```bash")
    w("cd docs/rvv_intrinsics")
    w("grep -w __riscv_vbrev_v_u32m1 rva23-intrinsics.tsv          # one intrinsic, all columns")
    w("awk -F'\\t' '$1 ~ /^__riscv_vmulhu.*u32m1$/' rva23-intrinsics.tsv # a family at one type")
    w("awk -F'\\t' '$2==\"Zvbb\" && $6==\"explicit\" && $5==\"yes\" {print $8}' rva23-intrinsics.tsv")
    w("grep -rn --include='*.adoc' vwsll auto-generated/              # the documentation itself")
    w("```\n")
    w("Columns of `rva23-intrinsics.tsv`: " + ", ".join("`%s`" % c for c in cols) + ".\n")

    with open(os.path.join(a.root, "INDEX.md"), "w") as f:
        f.write("\n".join(out) + "\n")

    n_expl = sum(1 for r in rows if r["kind"] == "explicit")
    print("[rvv-intrinsics] %d intrinsics indexed (%d explicit); %s" % (
        len(rows), n_expl,
        "compiler lacks %d of them" % sum(1 for r in rows if r["gcc"] == "no")
        if missing is not None else "compiler not probed"))
    print("[rvv-intrinsics] wrote %s/INDEX.md and rva23-intrinsics.tsv" % a.root)


if __name__ == "__main__":
    main()
