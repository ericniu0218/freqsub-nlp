"""Sampled-context co-occurrence estimator.

Why this exists: infini-gram's CNF (AND) counts are *approximate* whenever a clause is frequent, and
in Week-0 testing they were severely and inconsistently biased downward (e.g. "Barack Obama AND
Honolulu" = 152 at default settings, 4,804 at the max clause frequency, 274,853 with a wider window).

Estimator: for subject s with exact occurrence count N_s, draw K occurrences uniformly at random
without replacement (all of them if N_s <= K), fetch a fixed-size token window around each, and
record which candidate entities appear in each window. Then

    p_hat(c | s)  = (# sampled windows mentioning c) / K
    cooc_hat(s,c) = p_hat(c | s) * N_s

p_hat is the unbiased sample proportion of subject mentions whose window mentions c, with standard
error sqrt(p(1-p)/K * (N_s-K)/(N_s-1)) (finite-population correction). When N_s <= K it is exact.
"""
from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..data.aliases import text_mentions
from .client import InfiniGramClient


@dataclass
class SubjectContexts:
    subject: str
    n_occurrences: int
    windows: list[str] = field(default_factory=list)
    n_blocked: int = 0
    exhaustive: bool = False

    @property
    def k(self) -> int:
        return len(self.windows)


def _window_text(doc: dict) -> str:
    return "".join(span[0] for span in doc.get("spans", []))


def sample_occurrences(segments: list[list[int]], k: int, rng: random.Random) -> list[tuple[int, int]]:
    """Uniform sample (without replacement) of k occurrences across all shards' suffix-array ranges."""
    sizes = [max(0, b - a) for a, b in segments]
    total = sum(sizes)
    if total == 0:
        return []
    picks = range(total) if total <= k else rng.sample(range(total), k)
    out = []
    for flat in sorted(picks):
        for shard, ((a, _b), size) in enumerate(zip(segments, sizes)):
            if flat < size:
                out.append((shard, a + flat))
                break
            flat -= size
    return out


def fetch_subject_contexts(client: InfiniGramClient, subject: str, k: int, max_disp_len: int,
                           seed: int, workers: int = 4) -> SubjectContexts:
    found = client.find(subject)
    n = int(found["cnt"])
    rng = random.Random(f"{seed}:{subject}")  # deterministic per subject => cache hits on re-runs
    occ = sample_occurrences(found["segment_by_shard"], k, rng)
    ctx = SubjectContexts(subject=subject, n_occurrences=n, exhaustive=(n <= k))
    if not occ:
        return ctx
    ids = found["token_ids"]

    def get(o):
        return client.get_doc_by_rank(o[0], o[1], ids, max_disp_len)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        docs = list(ex.map(get, occ))
    for d in docs:
        if d.get("blocked"):
            ctx.n_blocked += 1
            continue
        ctx.windows.append(_window_text(d))
    return ctx


@dataclass
class CoocEstimate:
    hits: int
    k: int
    n_subject: int
    exhaustive: bool

    @property
    def p_hat(self) -> float:
        return self.hits / self.k if self.k else float("nan")

    @property
    def cooc_hat(self) -> float:
        return self.p_hat * self.n_subject if self.k else float("nan")

    @property
    def se_p(self) -> float:
        if not self.k or self.exhaustive:
            return 0.0
        p, k, n = self.p_hat, self.k, self.n_subject
        fpc = (n - k) / (n - 1) if n > 1 else 0.0
        return math.sqrt(max(p * (1 - p), 0.0) / k * fpc)


def estimate_cooc(ctx: SubjectContexts, candidate_aliases: list[str]) -> CoocEstimate:
    hits = sum(text_mentions(w, candidate_aliases) for w in ctx.windows)
    return CoocEstimate(hits=hits, k=ctx.k, n_subject=ctx.n_occurrences, exhaustive=ctx.exhaustive)
