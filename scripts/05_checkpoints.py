"""Stage 05 (H2): score the same questions at log-spaced stage-1 checkpoints.

Picks the stage-1 branch nearest each target token count, runs stage 04 for it, then deletes that
checkpoint's weights from the Hugging Face cache (each is ~6 GB; disk is limited). The final stage-1
checkpoint (the primary model) is never deleted. Resumable: stage 04 skips finished shards.

Usage: python scripts/05_checkpoints.py [templates=[qa]] [dry_run=true]
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from freqsub.utils.config import load_config  # noqa: E402

TARGET_TOKENS_B = [25, 50, 100, 200, 400, 800, 1600, 3200]  # + the final checkpoint (~4001B), already scored


def stage1_branches(repo: str) -> list[tuple[int, str]]:
    from huggingface_hub import list_repo_refs

    out = []
    for b in list_repo_refs(repo).branches:
        m = re.match(r"stage1-step(\d+)-tokens(\d+)B$", b.name)
        if m:
            out.append((int(m.group(2)), b.name))
    return sorted(out)


def pick(branches: list[tuple[int, str]], targets: list[int]) -> list[tuple[int, str]]:
    chosen = []
    for t in targets:
        tok, name = min(branches, key=lambda x: abs(x[0] - t))
        if name not in [c[1] for c in chosen]:
            chosen.append((tok, name))
    return chosen


def delete_cached_revision(repo: str, revision: str) -> None:
    from huggingface_hub import scan_cache_dir

    cache = scan_cache_dir()
    for r in cache.repos:
        if r.repo_id != repo:
            continue
        hashes = [rev.commit_hash for rev in r.revisions if revision in rev.refs]
        if hashes:
            strategy = cache.delete_revisions(*hashes)
            print(f"deleting {revision}: frees {strategy.expected_freed_size_str}")
            strategy.execute()


def main() -> None:
    cfg = load_config()
    repo = cfg["model"]["name"]
    final = cfg["model"]["revision"]
    templates = cfg.get("templates") or ["qa"]
    chosen = pick(stage1_branches(repo), TARGET_TOKENS_B)
    print("checkpoints:", chosen)
    if cfg.get("dry_run"):
        return
    for tok, rev in chosen:
        if rev == final:
            continue
        for t in templates:
            cmd = [sys.executable, "-u", "scripts/04_score.py", f"templates=[{t}]", f"model.revision={rev}",
                   f"threads={cfg.get('threads', 2)}", f"batch_size={cfg.get('batch_size', 32)}"]
            if cfg.get("question_subset") and tok > 210:  # 21B-210B were already scored on all questions
                cmd.append(f"question_subset={cfg['question_subset']}")
            print("running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
        delete_cached_revision(repo, rev)


if __name__ == "__main__":
    main()
