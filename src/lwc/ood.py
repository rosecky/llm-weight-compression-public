"""Held-out corpora for asking whether compensation generalises off its calibration text.

Calibration is drawn from wikitext-2 train. Compensation that only works where the activation
statistics look like wikitext is compensation that has fitted the sample, so the evaluation set
has to include text whose statistics are genuinely different: other prose, code, mathematics,
another language, and longer context.

Availability is not assumed. Each source is attempted independently and a source that cannot be
loaded is *reported as unavailable* rather than silently skipped or quietly substituted, because
"we evaluated on OOD data" and "we tried to and three of five sources failed to download" are
very different claims.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

# (name, loader spec). Each spec is (hf_path, config, split, text_column).
SOURCES: List[Tuple[str, Tuple[str, Optional[str], str, str]]] = [
    ("wikitext2", ("Salesforce/wikitext", "wikitext-2-raw-v1", "test", "text")),
    ("wikitext103", ("Salesforce/wikitext", "wikitext-103-raw-v1", "test", "text")),
    ("ptb", ("ptb_text_only", "penn_treebank", "test", "sentence")),
    ("c4", ("allenai/c4", "en", "validation", "text")),
    ("code", ("bigcode/the-stack-smol", "data/python", "train", "content")),
    ("math", ("EleutherAI/proof-pile-2", "arxiv", "train", "text")),
    ("multilingual", ("wikimedia/wikipedia", "20231101.de", "train", "text")),
]


def _load_text(spec, max_docs: int = 2000, streaming: bool = False) -> Optional[str]:
    from datasets import load_dataset
    path, config, split, col = spec
    kw = dict(split=split)
    if config:
        kw["name"] = config
    if streaming:
        kw["streaming"] = True
    ds = load_dataset(path, **kw)
    if streaming:
        out, n = [], 0
        for row in ds:
            out.append(row[col])
            n += 1
            if n >= max_docs:
                break
        return "\n\n".join(out)
    if len(ds) > max_docs:
        ds = ds.select(range(max_docs))
    return "\n\n".join(ds[col])


def load_corpora(tokenizer, names: List[str], seqlen: int = 2048, n_seq: int = 16,
                 seed: int = 0, max_docs: int = 2000) -> Tuple[Dict[str, torch.Tensor],
                                                               Dict[str, str]]:
    """Returns {name: token ids (n_seq, seqlen)} plus {name: reason} for what failed."""
    by_name = dict(SOURCES)
    out: Dict[str, torch.Tensor] = {}
    failed: Dict[str, str] = {}
    g = torch.Generator().manual_seed(seed)
    for name in names:
        if name not in by_name:
            failed[name] = "unknown source"
            continue
        text = None
        for streaming in (False, True):
            try:
                text = _load_text(by_name[name], max_docs=max_docs, streaming=streaming)
                break
            except Exception as e:                       # noqa: BLE001 - report, do not hide
                failed[name] = "%s: %s" % (type(e).__name__, str(e)[:160])
        if text is None:
            continue
        enc = tokenizer(text, return_tensors="pt").input_ids[0]
        need = seqlen + 1
        if enc.numel() < need * 2:
            failed[name] = "only %d tokens, need >= %d" % (enc.numel(), need * 2)
            continue
        starts = torch.randint(0, enc.numel() - need, (n_seq,), generator=g)
        out[name] = torch.stack([enc[s:s + seqlen] for s in starts])
        failed.pop(name, None)
    return out, failed


@torch.no_grad()
def nll_on(model, ids: torch.Tensor, device: str) -> Tuple[float, float]:
    """Mean next-token NLL and its perplexity, over pre-tokenised windows."""
    tot, ntok = 0.0, 0
    for i in range(ids.shape[0]):
        x = ids[i:i + 1].to(device)
        out = model(x, labels=x)
        tot += float(out.loss) * (x.shape[1] - 1)
        ntok += x.shape[1] - 1
    nll = tot / max(ntok, 1)
    return nll, float(torch.exp(torch.tensor(nll)))
