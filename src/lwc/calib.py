"""Calibration activations.

We cache, for a small chosen subset of layers, a subsampled matrix X of shape
(in_features, n_samples) in fp16. That is enough to compute the exact activation-space error
||(W - What) X||_F / ||W X||_F, and also to derive the diagonal activation scale used by all
codecs (AWQ-style), which makes weight-MSE a first-order proxy for activation-MSE.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch

from .modelio import LayerRef, get_module


def get_wikitext2(tokenizer, seqlen: int = 2048, n_seq: int = 64, split="train", seed: int = 0):
    from datasets import load_dataset

    # the dataset id needs a namespace on recent `datasets` versions
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(seed)
    max_start = enc.numel() - seqlen - 1
    starts = torch.randint(0, max_start, (n_seq,), generator=g)
    return torch.stack([enc[s : s + seqlen] for s in starts])


def get_c4_train(tokenizer, seqlen: int = 2048, n_seq: int = 64, seed: int = 0,
                 pool_tokens: int = 2_000_000):
    """Calibration windows from the C4 *train* stream (the evaluation windows come from
    the *validation* split, so the two are disjoint by construction).

    Deterministic: the first `pool_tokens` tokens of the stream (fixed document order) form
    the pool; windows are drawn from it with the same seeded sampler as `get_wikitext2`, so
    a calibration draw is identified by (source, seed) exactly as before.
    """
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    buf, total = [], 0
    for doc in ds:
        t = tokenizer(doc["text"], return_tensors="pt").input_ids[0]
        buf.append(t)
        total += t.numel()
        if total >= pool_tokens:
            break
    enc = torch.cat(buf)[:pool_tokens]
    g = torch.Generator().manual_seed(seed)
    max_start = enc.numel() - seqlen - 1
    starts = torch.randint(0, max_start, (n_seq,), generator=g)
    return torch.stack([enc[s : s + seqlen] for s in starts])


def get_calib(source: str, tokenizer, seqlen: int, n_seq: int, seed: int):
    if source == "wikitext2":
        return get_wikitext2(tokenizer, seqlen=seqlen, n_seq=n_seq, split="train", seed=seed)
    if source == "c4":
        return get_c4_train(tokenizer, seqlen=seqlen, n_seq=n_seq, seed=seed)
    raise ValueError("unknown calibration source %r" % source)


@torch.no_grad()
def collect_layer_inputs(
    model,
    refs: List[LayerRef],
    input_ids: torch.Tensor,
    device: str = "cuda",
    max_samples: int = 4096,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Run the model and reservoir-subsample each target module's input tokens.

    Returns {module_name: X (in_features, n_samples) fp16 on CPU}.
    Memory-bounded: we keep at most `max_samples` columns per module.
    """
    store: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    g = torch.Generator(device="cpu").manual_seed(seed)
    names = {r.name for r in refs}

    def hook(name):
        def fn(mod, inp, out):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1])              # (tokens, in)
            n = x.shape[0]
            keep = min(max_samples, n)
            idx = torch.randperm(n, generator=g)[:keep].to(x.device)
            xs = x[idx].to(torch.float16).cpu()          # (keep, in)
            if name not in store:
                store[name] = xs
                counts[name] = keep
            else:
                cur = store[name]
                cat = torch.cat([cur, xs], 0)
                if cat.shape[0] > max_samples:
                    sel = torch.randperm(cat.shape[0], generator=g)[:max_samples]
                    cat = cat[sel]
                store[name] = cat
        return fn

    handles = []
    for r in refs:
        handles.append(get_module(model, r.name).register_forward_hook(hook(r.name)))
    try:
        for i in range(input_ids.shape[0]):
            model(input_ids[i : i + 1].to(device))
    finally:
        for h in handles:
            h.remove()
    return {k: v.T.contiguous() for k, v in store.items()}   # -> (in, n_samples)


def act_scale_from_X(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """s_j = RMS of input channel j. Column-scaling W by s makes weight-MSE a diagonal
    approximation of activation-MSE."""
    Xf = X.float()
    return Xf.pow(2).mean(dim=1).sqrt().clamp_min(eps)
