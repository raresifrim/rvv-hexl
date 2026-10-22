#!/usr/bin/env python3
# Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
# All rights reserved.
#
"""Make OpenFHE's key switching accumulate with ModMulAddEq (OpenFHE v1.5.1).

    openfhe-pke-keyswitch-multadd.py <staged openfhe-development tree>

Applied by third_party/openfhe.sh to the rvv-hexl tree only (stock builds stay
stock), after openfhe-hexl-multadd.py, which adds ModMulAddEq. Idempotent; fails
loudly if upstream changes shape. Both key-switching techniques (every
relinearization and rotation in BFV, BGV and CKKS):

- KeySwitchHYBRID::EvalFastKeySwitchCoreExt accumulates the digit x key products
  limb by limb as
      elements[0].SetElementAtIndex(i, elements[0].GetElementAtIndex(i) + cji * bji);
  a multiplication into a temporary polynomial, an addition into a second one,
  then a copy back. Each line becomes elements[.].GetAllElements()[i].ModMulAddEq(cji, .).
- KeySwitchBV::EvalFastKeySwitchCore (the technique the blindloop BFV bench uses)
  copies the whole evaluation key (every digit, every limb), multiplies each
  digit into its copy, then sums the copies in a second pass. It becomes one
  ModMulAddEq per digit and limb into two zero-initialised results over Ql: no
  key copy, no temporary, one pass.

ModMulAddEq is one fused call (EltwiseMultAddMod) when HEXL_MULADD_ENABLE is 1, the
two HEXL calls in place when it is 0. Same result: all operands are reduced
polynomials in EVALUATION format, limb j over the same modulus (Ql's j-th prime,
the key's first sizeQl limbs).
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1])
REL = "src/pke/lib/keyswitch/keyswitch-hybrid.cpp"
MARK = "rvv-hexl: patches/openfhe-pke-keyswitch-multadd.py"
LINE = re.compile(r"^(\s*)elements\[([01])\]\.SetElementAtIndex\(i, elements\[\2\]\.GetElementAtIndex\(i\) \+ cji \* ([ab]ji)\);$", re.M)

BV = "src/pke/lib/keyswitch/keyswitch-bv.cpp"
BV_OLD = """    std::vector<DCRTPoly> bv(evalKey->GetBVector());
    std::vector<DCRTPoly> av(evalKey->GetAVector());
    const auto diffQl    = bv[0].GetParams()->GetParams().size() - paramsQl->GetParams().size();
    const uint32_t limit = (*digits).size();
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(limit))
    for (uint32_t i = 0; i < limit; ++i) {
        bv[i].DropLastElements(diffQl);
        bv[i] *= (*digits)[i];
        av[i].DropLastElements(diffQl);
        av[i] *= (*digits)[i];
    }

    std::vector<DCRTPoly> res{std::move(bv[0]), std::move(av[0])};
    for (uint32_t i = 1; i < limit; ++i) {
        res[0] += bv[i];
        res[1] += av[i];
    }
    return std::make_shared<std::vector<DCRTPoly>>(std::move(res));
"""
BV_NEW = f"""    // {MARK}: one ModMulAddEq per digit and limb, no key copy
    const auto& bv       = evalKey->GetBVector();
    const auto& av       = evalKey->GetAVector();
    const uint32_t sizeQl = paramsQl->GetParams().size();
    const uint32_t limit  = (*digits).size();
    std::vector<DCRTPoly> res{{DCRTPoly(paramsQl, Format::EVALUATION, true), DCRTPoly(paramsQl, Format::EVALUATION, true)}};
#pragma omp parallel for num_threads(OpenFHEParallelControls.GetThreadLimit(sizeQl))
    for (uint32_t j = 0; j < sizeQl; ++j) {{
        auto& r0 = res[0].GetAllElements()[j];
        auto& r1 = res[1].GetAllElements()[j];
        for (uint32_t i = 0; i < limit; ++i) {{
            const auto& dij = (*digits)[i].GetElementAtIndex(j);
            r0.ModMulAddEq(dij, bv[i].GetElementAtIndex(j));
            r1.ModMulAddEq(dij, av[i].GetElementAtIndex(j));
        }}
    }}
    return std::make_shared<std::vector<DCRTPoly>>(std::move(res));
"""

for rel in (REL, BV):
    p = ROOT / rel
    s = p.read_text()
    if MARK in s:
        print(f"[openfhe-pke] {rel}: multiply-add patch already applied")
        continue
    if rel == REL:
        found = len(LINE.findall(s))
        if found != 2:
            sys.exit(f"[openfhe-pke] ERROR: {rel}: expected 2 accumulation lines, found {found}: "
                     f"upstream changed, review {__file__}")
        s = LINE.sub(lambda m: f"{m.group(1)}elements[{m.group(2)}].GetAllElements()[i].ModMulAddEq(cji, {m.group(3)});", s)
        print(f"[openfhe-pke] {rel}: {found} accumulations -> ModMulAddEq")
    else:
        if s.count(BV_OLD) != 1:
            sys.exit(f"[openfhe-pke] ERROR: {rel}: EvalFastKeySwitchCore body not found once: "
                     f"upstream changed, review {__file__}")
        s = s.replace(BV_OLD, BV_NEW)
        print(f"[openfhe-pke] {rel}: EvalFastKeySwitchCore -> ModMulAddEq per digit and limb")
    p.write_text(f"// {MARK}: key-switch accumulation uses ModMulAddEq\n" + s)
