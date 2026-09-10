"""Calibration draws that are actually independent, at budgets up to half a million tokens.

The calibration-size question needs several *independent* draws at each budget. Sampling random
windows from wikitext-2 does not provide that: its training split is roughly 2.4M tokens, so
three draws of 524k tokens would cover two thirds of the corpus and overlap heavily. Any
"variation across draws" measured that way would be an underestimate of the real thing, in the
direction that flatters the method.

So the pool is drawn from wikitext-103 (about 100M tokens), the token stream is cut into
`n_parts` **contiguous disjoint** blocks, and each draw samples only inside its own block. Two
draws then share no token at all, by construction rather than by probability.

The tokenised stream is cached, because tokenising a few million tokens takes longer than every
quantization the sweep performs.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

CORPORA = {
    "wikitext103": ("Salesforce/wikitext", "wikitext-103-raw-v1", "train", "text"),
    "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1", "train", "text"),
}


def token_stream(tokenizer, corpus: str = "wikitext103", target_tokens: int = 4_000_000,
                 cache_dir: str = "cache", verbose: bool = True) -> torch.Tensor:
    """One long 1-D token stream, at least `target_tokens` long where the corpus allows."""
    os.makedirs(cache_dir, exist_ok=True)
    path = "%s/pool_%s_%dM.pt" % (cache_dir, corpus, max(1, target_tokens // 1_000_000))
    if os.path.exists(path):
        enc = torch.load(path, map_location="cpu")
        if enc.numel() >= target_tokens:
            return enc

    from datasets import load_dataset
    hf, cfg, split, col = CORPORA[corpus]
    ds = load_dataset(hf, name=cfg, split=split)
    # take documents until the character budget plausibly covers the token target; ~4 chars
    # per token is a safe overestimate of density for this tokenizer family
    need_chars = int(target_tokens * 5)
    chunks, total = [], 0
    for row in ds:
        t = row[col]
        if not t:
            continue
        chunks.append(t)
        total += len(t)
        if total >= need_chars:
            break
    enc = tokenizer("\n\n".join(chunks), return_tensors="pt").input_ids[0]
    if verbose:
        print("  token pool: %s -> %d tokens from %d documents"
              % (corpus, enc.numel(), len(chunks)))
    torch.save(enc, path)
    return enc


def draw_windows(stream: torch.Tensor, n_parts: int, part: int, seqlen: int, n_seq: int,
                 seed: int = 0) -> torch.Tensor:
    """`n_seq` windows of `seqlen`, sampled only from partition `part` of `n_parts`.

    Distinct `part` values therefore share no token, which is what makes a spread across draws
    a measurement of sampling variability rather than of window jitter inside one text.
    """
    total = stream.numel()
    size = total // n_parts
    lo = part * size
    hi = lo + size
    need = seqlen + 1
    if size < need * 2:
        raise ValueError("partition %d/%d holds %d tokens, too few for %d-token windows"
                         % (part, n_parts, size, seqlen))
    g = torch.Generator().manual_seed(seed + 1000 * part)
    starts = torch.randint(lo, hi - need, (n_seq,), generator=g)
    return torch.stack([stream[s:s + seqlen] for s in starts])


def budget_to_nseq(budget_tokens: int, seqlen: int) -> int:
    return max(1, budget_tokens // seqlen)
