"""Rate-distortion points (P6): per-rate paired bootstrap of module - gptq over windows.

Reads rd_*.jsonl (one file per exact bpw; the frozen method run through valid_eval with
--bits/--group overrides) plus the 3.25 bpw draw-0 validation file, and prints per rate and
eval set: fp16 / gptq / module NLL, the fraction of the gap closed, the paired bootstrap CI
of module - gptq (10k resamples over windows), and the equivalent bits saved by
piecewise-linear interpolation of the GPTQ rate-distortion curve.

usage: python scripts/analyze_rd.py [dir]        (default evidence/validation)
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_valid import bootstrap_diff  # noqa: E402

RAW = sys.argv[1] if len(sys.argv) > 1 else "evidence/validation"


def main():
    pts = {}                                    # bpw -> eval -> arm -> nlls
    files = sorted(glob.glob(os.path.join(RAW, "rd_*.jsonl"))
                   + glob.glob(os.path.join(RAW, "valid_qwen05_s0.jsonl")))
    for f in files:
        recs = [json.loads(l) for l in open(f, encoding="utf-8")]
        if any(r.get("calib_seed", 0) != 0 for r in recs):
            continue
        bpw = next((r["bpw"] for r in recs if "bpw" in r), None)
        if bpw is None:
            continue
        d = pts.setdefault(round(bpw, 2), defaultdict(dict))
        for r in recs:
            if "nlls" in r:
                d[r["eval"]][r["arm"]] = r["nlls"]
    mean = lambda x: sum(x) / len(x)
    for ev in ("wt2", "c4"):
        print("## %s (Qwen2.5-0.5B, calibration draw 0)" % ev)
        print("| bpw | fp16 | gptq | module | gap closed | module - gptq [2.5%, 97.5%] |")
        print("|---|---|---|---|---|---|")
        curve = []
        for bpw in sorted(pts):
            a = pts[bpw][ev]
            if not all(k in a for k in ("fp16", "gptq", "module")):
                continue
            fp, gq, mo = mean(a["fp16"]), mean(a["gptq"]), mean(a["module"])
            base, lo, hi = bootstrap_diff(a["module"], a["gptq"])
            curve.append((bpw, gq - fp, mo - fp))
            print("| %.2f | %.5f | %.5f | %.5f | %+.1f%% | %+.5f [%+.5f, %+.5f] |"
                  % (bpw, fp, gq, mo, (gq - mo) / (gq - fp) * 100, base, lo, hi))
        print()
        print("equivalent bits saved: rate at which the gptq curve (piecewise-linear) reaches "
              "the module gap, minus the actual rate")
        for i, (bpw, g, mg) in enumerate(curve):
            if mg > g:
                print("  %.2f bpw: post-pass harmful (module gap %.4f > gptq gap %.4f)"
                      % (bpw, mg, g))
                continue
            if i + 1 < len(curve):
                b1, g1, _ = curve[i + 1]
                if mg >= g1:
                    eq = bpw + (b1 - bpw) * (g - mg) / (g - g1)
                    print("  %.2f bpw: module gap %.4f = gptq gap at %.4f bpw -> +%.3f bits "
                          "(interpolated on [%.2f, %.2f])" % (bpw, mg, eq, eq - bpw, bpw, b1))
                    continue
            b0, g0, _ = curve[i - 1]
            eq = bpw + (bpw - b0) * (g - mg) / (g0 - g)
            print("  %.2f bpw: module gap %.4f -> +%.3f bits (EXTRAPOLATED with the slope of "
                  "[%.2f, %.2f]; no higher rate measured)" % (bpw, mg, eq - bpw, b0, bpw))
        print()


if __name__ == "__main__":
    main()
