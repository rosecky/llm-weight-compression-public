---
title: llm-weight-compression evidence explorer
emoji: 📊
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: "6.26.0"
app_file: app.py
pinned: false
license: mit
datasets:
  - honza-rosecky/llm-weight-compression-evidence
---

Interactive explorer for the per-window evidence behind
[rosecky/llm-weight-compression-public](https://github.com/rosecky/llm-weight-compression-public):
paired bootstraps between any two arms, the rate–distortion points, and the 2-bit
first-token tail with and without a fixed prefix. Everything is recomputed live from the
dataset; nothing is hard-coded.
