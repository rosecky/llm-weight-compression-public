"""Evidence explorer for thinletter/llm-weight-compression-evidence (Gradio, CPU only).

Every number is recomputed from the per-window NLLs in the dataset at request time; nothing
is hard-coded. Paired bootstrap = resample evaluation windows with replacement, 2 000 times.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import gradio as gr
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

DATASET = "thinletter/llm-weight-compression-evidence"
REPO = "https://github.com/rosecky/llm-weight-compression-public"
ARTICLE = "https://huggingface.co/datasets/%s/blob/main/ARTICLE.md" % DATASET

root = snapshot_download(DATASET, repo_type="dataset", allow_patterns=["evidence/validation/*.jsonl"])
VAL = os.path.join(root, "evidence", "validation")

# ---------------------------------------------------------------- load everything once
WIN = {}     # (file, arm, eval) -> np.array of per-window NLL
REC = {}     # (file, arm) -> summary row
META = {}    # file -> (model, calib_seed)
for path in sorted(glob.glob(os.path.join(VAL, "*.jsonl"))):
    f = os.path.basename(path)[:-6]
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        META[f] = (r.get("model", "?"), r.get("calib_seed", 0))
        if "nlls" in r:
            WIN[(f, r["arm"], r["eval"])] = np.asarray(r["nlls"], dtype=np.float64)
        elif "bpw" in r or r.get("arm") == "fp16":
            REC[(f, r["arm"])] = r
FILES = sorted({k[0] for k in WIN})


def arms_in(f):
    return sorted({a for (ff, a, e) in WIN if ff == f})


def boot(a: np.ndarray, b: np.ndarray, n=2000, seed=0):
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def fmt_ci(t):
    return "%+.4f [%+.4f, %+.4f]" % t


# ---------------------------------------------------------------- wikitext-2 window text
_wt2_text = None


def wt2_windows():
    """Token windows of the frozen protocol (Qwen tokenizer), for showing window text."""
    global _wt2_text
    if _wt2_text is None:
        try:
            from datasets import load_dataset
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
            ids = tok("\n\n".join(ds["text"]), return_tensors="np").input_ids[0]
            n = ids.size // 2048
            W = ids[: n * 2048].reshape(n, 2048)
            _wt2_text = [(tok.decode(W[i, :3]), tok.decode(W[i, :80]).replace("\n", " ")) for i in range(n)]
        except Exception as e:                     # offline or no tokenizer: degrade gracefully
            _wt2_text = [("?", "(window text unavailable: %s)" % e)] * 146
    return _wt2_text


# ---------------------------------------------------------------- views
def headline():
    rows = []
    groups = defaultdict(list)
    for f in FILES:
        if f.startswith(("valid_", "p3_llama1b")) and (f, "module", "wt2") in WIN and (f, "gptq", "wt2") in WIN:
            groups[META[f][0]].append(f)
    for model, fs in sorted(groups.items()):
        for f in sorted(fs, key=lambda x: META[x][1]):
            fp, gq, mo = WIN[(f, "fp16", "wt2")], WIN[(f, "gptq", "wt2")], WIN[(f, "module", "wt2")]
            m, lo, hi = boot(mo, gq)
            gap = gq.mean() - fp.mean()
            rows.append(dict(model=model.split("/")[-1], draw=META[f][1],
                             **{"fp16 NLL": round(fp.mean(), 4), "GPTQ NLL": round(gq.mean(), 4),
                                "+post-pass NLL": round(mo.mean(), 4),
                                "gap closed": "%+.1f%%" % (-m / gap * 100),
                                "module − GPTQ [95% CI]": fmt_ci((m, lo, hi)),
                                "perplexity": "%+.1f%%" % ((np.exp(mo.mean() - gq.mean()) - 1) * 100)}))
    return pd.DataFrame(rows)


def compare(f, arm_a, arm_b, ev):
    if (f, arm_a, ev) not in WIN or (f, arm_b, ev) not in WIN:
        return "arm not present in this file", None, None
    a, b = WIN[(f, arm_a, ev)], WIN[(f, arm_b, ev)]
    m, lo, hi = boot(a, b)
    d = a - b
    fp = WIN.get((f, "fp16", ev))
    gap_txt = ""
    if fp is not None and arm_b != "fp16":
        gap = b.mean() - fp.mean()
        gap_txt = " · gap closed vs fp16: %+.1f%%" % (-m / gap * 100) if gap > 0 else ""
    txt = ("**%s − %s on %s** (%d windows): mean %s%s  \nbetter on %d, worse on %d, median %+.4f, "
           "max %+.3f, min %+.3f" % (arm_a, arm_b, ev, len(d), fmt_ci((m, lo, hi)), gap_txt,
                                     int((d < 0).sum()), int((d > 0).sum()), float(np.median(d)),
                                     float(d.max()), float(d.min())))
    hist = pd.DataFrame({"Δ NLL per window": d})
    worst = np.argsort(-d)[:10]
    rows = [dict(window=int(i), **{arm_b: round(float(b[i]), 3), arm_a: round(float(a[i]), 3), "Δ": round(float(d[i]), 3)}) for i in worst]
    if ev == "wt2":
        wt = wt2_windows()
        for r in rows:
            r["first tokens"], r["text"] = wt[r["window"]]
    return txt, hist, pd.DataFrame(rows)


def rd_table():
    rows = []
    for f in FILES:
        if f.startswith("rd_") or f == "valid_qwen05_s0":
            if (f, "module", "wt2") not in WIN:
                continue
            fp, gq, mo = WIN[(f, "fp16", "wt2")], WIN[(f, "gptq", "wt2")], WIN[(f, "module", "wt2")]
            bpw = REC[(f, "module")]["bpw"]
            m, lo, hi = boot(mo, gq)
            rows.append({"bpw": round(bpw, 2), "GPTQ gap": round(gq.mean() - fp.mean(), 4),
                         "post-pass gap": round(mo.mean() - fp.mean(), 4),
                         "gap closed": "%+.1f%%" % (-m / (gq.mean() - fp.mean()) * 100),
                         "module − GPTQ [95% CI]": fmt_ci((m, lo, hi))})
    return pd.DataFrame(sorted(rows, key=lambda r: r["bpw"]))


def two_bit(prefixed, ev):
    if prefixed:
        f = "d4_qwen05_vq20_prefix"
    else:
        f = None
    if f:
        a, b = WIN[(f, "vq20_refine", ev)], WIN[(f, "vq20_plain", ev)]
    else:
        a, b = WIN[("p4_qwen05_vq20_refine", "vq20_refine", ev)], WIN[("p4_qwen05_vq20_plain", "vq20_plain", ev)]
    m, lo, hi = boot(a, b)
    d = a - b
    txt = ("**post-pass − plain, 2.12 bpw VQ, %s, %s** (%d windows): %s  \nbetter on %d, worse on %d, "
           "windows worse by >1 nat: %d, max %+.3f" % (ev, "with \"\\n\\n\" prefix" if prefixed else "frozen protocol (no prefix)",
                                                        len(d), fmt_ci((m, lo, hi)), int((d < 0).sum()), int((d > 0).sum()),
                                                        int((d > 1).sum()), float(d.max())))
    worst = np.argsort(-d)[:10]
    rows = [dict(window=int(i), plain=round(float(b[i]), 3), refine=round(float(a[i]), 3), Δ=round(float(d[i]), 3)) for i in worst]
    if ev == "wt2":
        wt = wt2_windows()
        for r in rows:
            r["first tokens"], r["text"] = wt[r["window"]]
    return txt, pd.DataFrame({"Δ NLL per window": d}), pd.DataFrame(rows)


# ---------------------------------------------------------------- UI
with gr.Blocks(title="llm-weight-compression: evidence explorer") as demo:
    gr.Markdown(
        "# Which objective should a post-training quantizer optimize? — evidence explorer\n"
        "Every number here is recomputed live from the per-window NLLs in "
        "[the dataset](https://huggingface.co/datasets/%s). Read the [article](%s); code and log in "
        "[the repository](%s). Paired bootstrap over evaluation windows, 2 000 resamples; "
        "negative Δ = the first arm is better." % (DATASET, ARTICLE, REPO))
    with gr.Tab("Headline"):
        gr.Markdown("Frozen protocol, 3.25 bits per weight, full wikitext-2 test, one row per calibration draw. "
                    "\"Gap closed\" = fraction of the GPTQ-vs-fp16 loss the post-pass recovers.")
        gr.DataFrame(headline(), wrap=True)
    with gr.Tab("Compare any two arms"):
        f_dd = gr.Dropdown(FILES, value="valid_qwen05_s0", label="run file (model / draw / point)")
        with gr.Row():
            a_dd = gr.Dropdown(arms_in("valid_qwen05_s0"), value="module", label="arm A")
            b_dd = gr.Dropdown(arms_in("valid_qwen05_s0"), value="gptq", label="arm B")
            ev_dd = gr.Radio(["wt2", "c4"], value="wt2", label="evaluation set")
        out_md = gr.Markdown()
        out_plot = gr.Plot(label="per-window Δ (A − B)")
        out_tbl = gr.DataFrame(label="ten windows where A is worst relative to B", wrap=True)

        def _upd_arms(f):
            arms = arms_in(f)
            return gr.update(choices=arms, value=arms[0] if "module" not in arms else "module"), gr.update(choices=arms, value="gptq" if "gptq" in arms else arms[-1])

        def _run(f, a, b, ev):
            txt, hist, tbl = compare(f, a, b, ev)
            fig = None
            if hist is not None:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(7, 2.6))
                ax.hist(hist["Δ NLL per window"], bins=40, color="#4477aa")
                ax.axvline(0, color="k", lw=0.8)
                ax.set_xlabel("Δ NLL per window (A − B)"); ax.set_ylabel("windows")
                fig.tight_layout()
            return txt, fig, tbl
        f_dd.change(_upd_arms, f_dd, [a_dd, b_dd])
        for c in (f_dd, a_dd, b_dd, ev_dd):
            c.change(_run, [f_dd, a_dd, b_dd, ev_dd], [out_md, out_plot, out_tbl])
        demo.load(_run, [f_dd, a_dd, b_dd, ev_dd], [out_md, out_plot, out_tbl])
    with gr.Tab("Rate–distortion"):
        gr.Markdown("Qwen2.5-0.5B, calibration draw 0, the same frozen method at four exact bit widths.")
        gr.DataFrame(rd_table())
    with gr.Tab("2 bits: the first-token tail"):
        gr.Markdown("A 2.12-bpw per-block vector-quantized Qwen2.5-0.5B, plain vs with the post-pass. "
                    "Without a prefix the ten worst windows all start with a digit-like token; with a fixed "
                    "\"\\n\\n\" prefix the tail disappears and the post-pass wins on both sets.")
        with gr.Row():
            pre_cb = gr.Checkbox(False, label="prefix every window with \"\\n\\n\"")
            ev2 = gr.Radio(["wt2", "c4"], value="wt2", label="evaluation set")
        md2 = gr.Markdown(); plot2 = gr.Plot(label="per-window Δ (post-pass − plain)"); tbl2 = gr.DataFrame(wrap=True)

        def _run2(p, ev):
            txt, hist, tbl = two_bit(p, ev)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 2.6))
            ax.hist(hist["Δ NLL per window"], bins=40, color="#cc6677"); ax.axvline(0, color="k", lw=0.8)
            ax.set_xlabel("Δ NLL per window (post-pass − plain)"); ax.set_ylabel("windows"); fig.tight_layout()
            return txt, fig, tbl
        pre_cb.change(_run2, [pre_cb, ev2], [md2, plot2, tbl2]); ev2.change(_run2, [pre_cb, ev2], [md2, plot2, tbl2])
        demo.load(_run2, [pre_cb, ev2], [md2, plot2, tbl2])
    gr.Markdown("Jan Rosecký, 2026. MIT. Experiments run with Claude (Anthropic) as an autonomous research agent under the author's direction.")

if __name__ == "__main__":
    demo.launch()
