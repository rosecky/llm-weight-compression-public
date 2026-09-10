"""Model loading + layer-wise streaming access to weights.

Everything here is deliberately layer-wise / streaming: the point is that the encoder must be
applicable to a 70B model on a desktop, so we never require the full model in GPU memory.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch

# Canonical projection names. Adapted at load time if the model uses different naming.
PROJ_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Which module's *input* each projection consumes (used for calibration hooks).
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


@dataclass
class LayerRef:
    name: str          # full module name
    layer_idx: int
    proj: str          # canonical proj type
    shape: Tuple[int, int]


def detect_proj(name: str) -> Optional[str]:
    for p in PROJ_TYPES:
        if name.endswith("." + p):
            return p
    # common alternates
    alt = {
        "wq": "q_proj", "wk": "k_proj", "wv": "v_proj", "wo": "o_proj",
        "w1": "gate_proj", "w3": "up_proj", "w2": "down_proj",
        "query": "q_proj", "key": "k_proj", "value": "v_proj", "dense": "o_proj",
        "fc1": "up_proj", "fc2": "down_proj",
    }
    tail = name.split(".")[-1]
    return alt.get(tail)


def load_model(model_id: str, device: str = "cpu", dtype=torch.float16, local_files_only=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, local_files_only=local_files_only,
    )
    model.eval()
    model.to(device)
    return model, tok


def list_linear_layers(model) -> List[LayerRef]:
    refs = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        proj = detect_proj(name)
        if proj is None:
            continue
        m = LAYER_RE.search(name)
        if m is None:
            continue
        refs.append(LayerRef(name=name, layer_idx=int(m.group(1)), proj=proj,
                             shape=tuple(mod.weight.shape)))
    refs.sort(key=lambda r: (r.layer_idx, PROJ_TYPES.index(r.proj) if r.proj in PROJ_TYPES else 99))
    return refs


def get_module(model, name: str) -> torch.nn.Module:
    mod = model
    for part in name.split("."):
        mod = getattr(mod, part) if not part.isdigit() else mod[int(part)]
    return mod


def iter_weights(model, refs: List[LayerRef], device="cpu") -> Iterator[Tuple[LayerRef, torch.Tensor]]:
    """Stream weights one matrix at a time (bounded memory)."""
    for r in refs:
        W = get_module(model, r.name).weight.detach()
        yield r, W.to(device=device, dtype=torch.float32)


def total_weights(refs: List[LayerRef]) -> int:
    return sum(r.shape[0] * r.shape[1] for r in refs)
