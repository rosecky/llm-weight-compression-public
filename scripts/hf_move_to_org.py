"""Move the released HF repos from the personal namespace to an organization and rewrite links.

Steps (each idempotent):
  1. move  honza-rosecky/llm-weight-compression-evidence  -> ORG/llm-weight-compression-evidence  (dataset)
  2. move  honza-rosecky/llm-weight-compression-explorer  -> ORG/llm-weight-compression-explorer  (space)
  3. rewrite the old ids in the release texts, the dataset card, the Space card and the
     explorer bundle; rebuild data.json; re-upload card + ARTICLE.md + Space files.
Old URLs keep redirecting (Hub behaviour), so nothing published breaks.

usage: python scripts/hf_move_to_org.py --org thinletter [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

OLD = "honza-rosecky"
REPOS = [("llm-weight-compression-evidence", "dataset"), ("llm-weight-compression-explorer", "space")]
FILES_WITH_LINKS = [
    "docs/release/hf_blog.md", "docs/release/report.md", "docs/release/post.md",
    "docs/release/hf_dataset_card.md", "docs/release/ARTICLE.md",
    "space/static/README.md", "space/static/index.html", "space/gradio/app.py", "space/gradio/README.md",
    "scripts/build_explorer.py",
    "../llm-weight-compression-public/README.md",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    api = HfApi()
    # 1-2. moves
    for name, rtype in REPOS:
        src, dst = "%s/%s" % (OLD, name), "%s/%s" % (a.org, name)
        try:
            api.repo_info(dst, repo_type=rtype)
            print("already at", dst)
            continue
        except HfHubHTTPError:
            pass
        print("move", src, "->", dst)
        if not a.dry_run:
            api.move_repo(from_id=src, to_id=dst, repo_type=rtype)
    # 3. rewrite links
    for rel in FILES_WITH_LINKS:
        if not os.path.exists(rel):
            continue
        s = open(rel, encoding="utf-8").read()
        t = s
        for name, _ in REPOS:
            t = t.replace("%s/%s" % (OLD, name), "%s/%s" % (a.org, name))
            t = t.replace("%s-%s" % (OLD, name), "%s-%s" % (a.org, name))      # static.hf.space host
        if t != s:
            print("rewrite", rel)
            if not a.dry_run:
                open(rel, "w", encoding="utf-8").write(t)
    if a.dry_run:
        return
    subprocess.run([sys.executable, "scripts/build_explorer.py"], check=True)
    ds, sp = "%s/llm-weight-compression-evidence" % a.org, "%s/llm-weight-compression-explorer" % a.org
    api.upload_file(path_or_fileobj="docs/release/hf_dataset_card.md", path_in_repo="README.md", repo_id=ds, repo_type="dataset",
                    commit_message="links: moved to %s" % a.org)
    api.upload_file(path_or_fileobj="docs/release/ARTICLE.md", path_in_repo="ARTICLE.md", repo_id=ds, repo_type="dataset",
                    commit_message="links: moved to %s" % a.org)
    api.upload_folder(folder_path="space/static", repo_id=sp, repo_type="space", commit_message="links: moved to %s" % a.org)
    print("done: https://huggingface.co/datasets/%s  https://huggingface.co/spaces/%s" % (ds, sp))


if __name__ == "__main__":
    main()
