---
title: llm-weight-compression evidence explorer
emoji: 📊
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
license: mit
datasets:
  - honza-rosecky/llm-weight-compression-evidence
---

Interactive explorer for the per-window evidence behind
[rosecky/llm-weight-compression-public](https://github.com/rosecky/llm-weight-compression-public):
paired bootstraps between any two arms, the rate–distortion points, and the 2-bit
first-token tail with and without a fixed prefix. Everything is recomputed in the browser
from `data.json`, which `scripts/build_explorer.py` in the repository generates from the
dataset; nothing is hard-coded. A Gradio version of the same explorer is in the repository
under `space/gradio/` for anyone with a PRO account.
