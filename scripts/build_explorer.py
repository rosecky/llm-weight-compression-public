"""Build the static evidence explorer's data bundle (space/static/data.json).

Reads evidence/validation/*.jsonl and emits, per run file: model, calibration seed, per-arm
per-eval window NLLs (rounded to 5 decimals), per-arm bpw, and for wikitext-2 the first
tokens and an 80-token snippet of every window (Qwen tokenizer, frozen protocol windows).
"""
from __future__ import annotations

import glob
import json
import os

VAL = "evidence/validation"
OUT = "space/static/data.json"


def wt2_snippets():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="np").input_ids[0]
    n = ids.size // 2048
    W = ids[: n * 2048].reshape(n, 2048)
    return [{"first": tok.decode(W[i, :3]), "text": tok.decode(W[i, :80]).replace("\n", " ")} for i in range(n)]


def main():
    files = {}
    for path in sorted(glob.glob(os.path.join(VAL, "*.jsonl"))):
        f = os.path.basename(path)[:-6]
        entry = {"arms": {}, "bpw": {}, "kl": {}}
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            entry["model"] = r.get("model", "?")
            entry["seed"] = r.get("calib_seed", 0)
            if "nlls" in r:
                entry["arms"].setdefault(r["arm"], {})[r["eval"]] = [round(x, 5) for x in r["nlls"]]
            elif "bpw" in r:
                entry["bpw"][r["arm"]] = round(r["bpw"], 4)
                if "kl_fp16" in r:
                    entry["kl"][r["arm"]] = round(r["kl_fp16"], 5)
        files[f] = entry
    bundle = {"files": files, "wt2": wt2_snippets(),
              "dataset": "thinletter/llm-weight-compression-evidence",
              "repo": "https://github.com/rosecky/llm-weight-compression-public"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(bundle, open(OUT, "w", encoding="utf-8"), separators=(",", ":"), ensure_ascii=False)
    print("wrote %s (%.0f KB, %d files)" % (OUT, os.path.getsize(OUT) / 1024, len(files)))


if __name__ == "__main__":
    main()
