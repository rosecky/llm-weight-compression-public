"""Tile <-> matrix conversion. Tiles are the unit of local decoding."""
from __future__ import annotations

import torch


def tile_grid(out_f: int, in_f: int, th: int, tw: int):
    return (out_f + th - 1) // th, (in_f + tw - 1) // tw


def to_tiles(W: torch.Tensor, th: int, tw: int) -> torch.Tensor:
    """(out,in) -> (n_tiles, th*tw), row-major over the tile grid. Zero-pads if needed."""
    out_f, in_f = W.shape
    gh, gw = tile_grid(out_f, in_f, th, tw)
    ph, pw = gh * th - out_f, gw * tw - in_f
    if ph or pw:
        W = torch.nn.functional.pad(W, (0, pw, 0, ph))
    W = W.reshape(gh, th, gw, tw).permute(0, 2, 1, 3).reshape(gh * gw, th * tw)
    return W.contiguous()


def from_tiles(T: torch.Tensor, out_f: int, in_f: int, th: int, tw: int) -> torch.Tensor:
    gh, gw = tile_grid(out_f, in_f, th, tw)
    W = T.reshape(gh, gw, th, tw).permute(0, 2, 1, 3).reshape(gh * th, gw * tw)
    return W[:out_f, :in_f].contiguous()


def iter_tile_batches(W: torch.Tensor, th: int, tw: int, batch: int = 65536):
    """Streaming tile access with bounded memory (used by encoders that must scale)."""
    T = to_tiles(W, th, tw)
    for i in range(0, T.shape[0], batch):
        yield T[i : i + batch]
