"""Addendum D3 read-out (see docs/FROZEN_PROTOCOL.md, D3a-D3c).

D3a  four-arm interaction: (module - gptq) - [(module_attn - gptq) + (module_mlp - gptq)]
     per window, paired bootstrap; gptq/module rows come from the frozen draw-0 validation
     file, the halves from d3a_*.jsonl (same model, seed, protocol; fp16 rows must agree).
D3b  full-model horizon at three estimation budgets: NLL, KL and obj_ratio per budget.
D3c  C4-calibrated gptq/module vs the wikitext-2-calibrated rows of the same draw.

usage: python scripts/analyze_d3.py [raw_dir] [validation_dir]
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_valid import bootstrap_diff  # noqa: E402

RAW = sys.argv[1] if len(sys.argv) > 1 else ("results/raw" if glob.glob("results/raw/d3*_*.jsonl") else "evidence/validation")
VAL = sys.argv[2] if len(sys.argv) > 2 else "evidence/validation"
MODEL = "Qwen/Qwen2.5-0.5B"


def load(path):
    win, rec = {}, {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("model") != MODEL or r.get("calib_seed", 0) != 0:
            continue
        if "nlls" in r:
            win[(r["arm"], r["eval"])] = r["nlls"]
        elif "bpw" in r:
            rec[r["arm"]] = r
    return win, rec


def mean(x):
    return sum(x) / len(x)


def boot(a, b):
    base, lo, hi = bootstrap_diff(a, b)
    return "%+.5f [%+.5f, %+.5f]" % (base, lo, hi)


def main():
    base_win, base_rec = load(os.path.join(VAL, "valid_qwen05_s0.jsonl"))

    # ---------------- D3a ----------------
    f = os.path.join(RAW, "d3a_qwen05_halves_s0.jsonl")
    if os.path.exists(f):
        w, rec = load(f)
        print("## D3a  attention half / MLP half / whole (draw 0, 3.25 bpw)\n")
        for ev in ("wt2", "c4"):
            fp, gq, mo = base_win[("fp16", ev)], base_win[("gptq", ev)], base_win[("module", ev)]
            assert max(abs(x - y) for x, y in zip(fp, w[("fp16", ev)])) < 1e-6, "fp16 rows differ"
            at, ml = w[("module_attn", ev)], w[("module_mlp", ev)]
            gap = mean(gq) - mean(fp)
            print("### %s   gptq gap %.4f" % (ev, gap))
            print("| arm | NLL | gap closed | arm - gptq (paired bootstrap) |")
            print("|---|---|---|---|")
            for name, x in (("module_attn", at), ("module_mlp", ml), ("module", mo)):
                print("| %s | %.5f | %+.1f%% | %s |" % (name, mean(x), (mean(gq) - mean(x)) / gap * 100,
                                                       boot(x, gq)))
            inter = [(m - g) - ((a - g) + (l - g)) for m, g, a, l in zip(mo, gq, at, ml)]
            zero = [0.0] * len(inter)
            print("| interaction (whole - sum of halves) | | %+.1f%% of gap | %s |"
                  % (-mean(inter) / gap * 100, boot(inter, zero)))
            print()
        for a in ("module_attn", "module_mlp"):
            r = rec[a]
            print("%-12s flips %s  obj_ratio %.4f  opt %.0f s  vram %.0f MiB  KL %.5f"
                  % (a, sum(v for k, v in (r.get("flips") or {}).items() if not k.startswith("depth")),
                     r["obj_ratio"], r["opt_s"], r["peak_vram_mib"], r["kl_fp16"]))
        print()

    # ---------------- D3c ----------------
    f = os.path.join(RAW, "d3c_qwen05_c4calib_s0.jsonl")
    if os.path.exists(f):
        w, rec = load(f)
        print("## D3c  calibration source: wikitext-2 (frozen) vs C4 train (same draw seed)\n")
        print("| eval | calib | gptq NLL | module NLL | module - gptq | gap closed |")
        print("|---|---|---|---|---|---|")
        for ev in ("wt2", "c4"):
            fp = base_win[("fp16", ev)]
            for src, ww in (("wikitext2", base_win), ("c4", w)):
                gq, mo = ww[("gptq", ev)], ww[("module", ev)]
                print("| %s | %s | %.5f | %.5f | %s | %+.1f%% |"
                      % (ev, src, mean(gq), mean(mo), boot(mo, gq),
                         (mean(gq) - mean(mo)) / (mean(gq) - mean(fp)) * 100))
        print()
        print("cross-source contrasts (paired over eval windows):")
        for ev in ("wt2", "c4"):
            print("  %s: gptq(c4 calib) - gptq(wt2 calib)     %s" % (ev, boot(w[("gptq", ev)], base_win[("gptq", ev)])))
            print("  %s: module(c4 calib) - module(wt2 calib) %s" % (ev, boot(w[("module", ev)], base_win[("module", ev)])))
            d_c4 = [m - g for m, g in zip(w[("module", ev)], w[("gptq", ev)])]
            d_wt = [m - g for m, g in zip(base_win[("module", ev)], base_win[("gptq", ev)])]
            print("  %s: (module-gptq | c4 calib) - (module-gptq | wt2 calib) %s" % (ev, boot(d_c4, d_wt)))
        for a in ("gptq", "module"):
            r = rec[a]
            print("%-7s c4-calib: KL %.5f  obj_ratio %s  opt %.0f s" % (a, r["kl_fp16"], r.get("obj_ratio"), r["opt_s"]))
        print()

    # ---------------- D3b ----------------
    rows = []
    for f in sorted(glob.glob(os.path.join(RAW, "d3b_qwen05_model_*.jsonl"))):
        w, rec = load(f)
        if "model" not in rec:
            continue
        r = rec["model"]
        rows.append((r["n_probe"], r["g_tokens"], r, w))
    if rows:
        print("## D3b  full-model horizon vs estimation budget (draw 0)\n")
        gq_w, gq_c = base_win[("gptq", "wt2")], base_win[("gptq", "c4")]
        fp_w = base_win[("fp16", "wt2")]
        gap = mean(gq_w) - mean(fp_w)
        print("| probes | G tokens | NLL wt2 | gap closed | model - gptq wt2 | NLL c4 | KL fp16 | obj_ratio | scope s | opt s |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for npb, gt, r, w in rows:
            print("| %d | %d | %.5f | %+.1f%% | %s | %.5f | %.5f | %.4f | %.0f | %.0f |"
                  % (npb, gt, r["nll_wt2"], (mean(gq_w) - r["nll_wt2"]) / gap * 100,
                     boot(w[("model", "wt2")], gq_w), r["nll_c4"], r["kl_fp16"],
                     r["obj_ratio"], r["scope_s"], r["opt_s"]))
        print("reference: gptq KL %.5f, module KL %.5f, module gap closed %+.1f%%"
              % (base_rec["gptq"]["kl_fp16"], base_rec["module"]["kl_fp16"],
                 (mean(gq_w) - mean(base_win[("module", "wt2")])) / gap * 100))


if __name__ == "__main__":
    main()
