"""P1-P3 verdict tables from the frozen validation runs.

Reads results/raw/valid_*.jsonl and produces, per model:

  * one row per (draw, arm): NLL wt2 / NLL c4 / KL / bpw / flips / opt time / peak VRAM --
    every draw individually, never only the aggregate;
  * paired bootstrap WITHIN each draw over evaluation windows (module - layer and
    module - gptq), 10k resamples, reported as mean [2.5%, 97.5%];
  * across draws: mean, range and the sign pattern of the effect (n=3 draws is a source of
    variability that must stay visible, not be averaged away);
  * the pre-registered verdict: clear PASS if all draws improve wt2 and C4/KL agree in
    direction; conditional PASS if one draw is ~neutral and none clearly negative.
"""
from __future__ import annotations

import glob
import json
import math
import random
import sys
from collections import defaultdict

RAW = "results/raw"
import os as _os
if not glob.glob(_os.path.join(RAW, "valid_*.jsonl")) and glob.glob("evidence/validation/valid_*.jsonl"):
    RAW = "evidence/validation"    # public release: the committed evidence


def bootstrap_diff(a, b, n=10000, seed=0):
    """Paired bootstrap of mean(a) - mean(b) over windows."""
    rng = random.Random(seed)
    m = len(a)
    d = [x - y for x, y in zip(a, b)]
    base = sum(d) / m
    lo_hi = []
    for _ in range(n):
        s = sum(d[rng.randrange(m)] for _ in range(m)) / m
        lo_hi.append(s)
    lo_hi.sort()
    return base, lo_hi[int(0.025 * n)], lo_hi[int(0.975 * n)]


def main():
    files = sorted(glob.glob("%s/valid_*.jsonl" % RAW) + glob.glob("%s/p3_llama1b*.jsonl" % RAW))
    by_model = defaultdict(lambda: defaultdict(dict))   # model -> seed -> arm -> rec
    nlls = defaultdict(dict)                            # (model, seed, arm, eval) -> [..]
    for f in files:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            m, s = r.get("model", "?"), r.get("calib_seed", -1)
            if "nlls" in r:
                nlls[(m, s, r["arm"], r["eval"])] = r["nlls"]
            elif "bpw" in r or r.get("arm") == "fp16":
                by_model[m][s][r["arm"]] = r

    out = []
    for model, seeds in sorted(by_model.items()):
        out += ["# P-validation: %s" % model, ""]
        out += ["| draw | arm | NLL wt2 | NLL c4 | KL | bpw | flips | opt s | VRAM MiB |",
                "|" + "---|" * 9]
        for s in sorted(seeds):
            for arm in ("gptq", "layer", "module", "block"):
                r = seeds[s].get(arm)
                if not r:
                    continue
                fl = r.get("flips") or {}
                fl_all = sum(v for k, v in fl.items() if not k.startswith("depth"))
                out.append("| %d | %s | %.5f | %.5f | %s | %.4f | %.3f | %.0f | %.0f |" % (
                    s, arm, r.get("nll_wt2", float("nan")), r.get("nll_c4", float("nan")),
                    ("%.5f" % r["kl_fp16"]) if "kl_fp16" in r else "-",
                    r.get("bpw", float("nan")),
                    fl_all / max(sum(1 for k in fl if not k.startswith("depth")), 1),
                    r.get("opt_s", 0), r.get("peak_vram_mib", 0)))
        out.append("")

        out += ["## Paired bootstrap within each draw (NLL difference, 10k resamples)", "",
                "| draw | eval | module - layer | module - gptq (primary) | verdict |",
                "|" + "---|" * 5]
        signs = {"wt2": [], "c4": []}
        for s in sorted(seeds):
            for ev in ("wt2", "c4"):
                key_m = (model, s, "module", ev)
                key_l = (model, s, "layer", ev)
                key_g = (model, s, "gptq", ev)
                if key_m not in nlls or key_g not in nlls:
                    continue
                dg, lo2, hi2 = bootstrap_diff(nlls[key_m], nlls[key_g], seed=1)
                if key_l in nlls:
                    dm, lo, hi = bootstrap_diff(nlls[key_m], nlls[key_l])
                    cell = "%+.5f [%+.5f, %+.5f]" % (dm, lo, hi)
                else:
                    cell = "n/a"
                sig = "improves" if hi2 < 0 else ("neutral" if lo2 < 0 else "WORSE")
                signs[ev].append(dg)
                out.append("| %d | %s | %s | %+.5f [%+.5f, %+.5f] | %s |"
                           % (s, ev, cell, dg, lo2, hi2, sig))
        out.append("")

        for ev in ("wt2", "c4"):
            v = signs[ev]
            if v:
                out.append("**%s across draws:** mean %+.5f, range [%+.5f, %+.5f], "
                           "signs %s" % (ev, sum(v) / len(v), min(v), max(v),
                                         "".join("-" if x < 0 else "+" for x in v)))
        if signs["wt2"]:
            neg = sum(1 for x in signs["wt2"] if x < 0)
            n = len(signs["wt2"])
            if neg == n and all(x < 0 for x in signs["c4"] or [1]):
                verdict = "CLEAR PASS" if n >= 3 else "on track (%d/%d draws in)" % (neg, n)
            elif neg >= max(1, n - 1):
                verdict = "conditional PASS"
            else:
                verdict = "NEEDS MORE DRAWS"
            out.append("\n**Pre-registered verdict (n=%d draws): %s**" % (n, verdict))
        out.append("")

    txt = "\n".join(out)
    import os as _os2
    _os2.makedirs("results/tables", exist_ok=True)
    open("results/tables/validation.md", "w", encoding="utf-8").write(txt)
    print(txt if "--print" in sys.argv else "wrote results/tables/validation.md")


if __name__ == "__main__":
    main()
