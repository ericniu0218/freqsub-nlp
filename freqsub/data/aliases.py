"""Alias normalization and matching.

Two different needs:
  * ``normalize`` -- loose canonical form for comparing a model's answer to gold aliases.
  * ``compile_alias_pattern`` -- strict, case-sensitive, word-bounded regex for finding an entity
    inside raw corpus text (case sensitivity avoids e.g. "US" matching the pronoun "us").
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache

_ARTICLES = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")

MIN_ALIAS_CHARS = 2  # aliases shorter than this are too ambiguous to search for


def normalize(text: str) -> str:
    """Canonical form: NFKC, lowercase, strip punctuation and leading articles, collapse spaces."""
    t = unicodedata.normalize("NFKC", text).lower().strip()
    t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    t = _ARTICLES.sub("", t)
    return t


def parse_alias_field(raw: str | list | None) -> list[str]:
    """PopQA stores alias lists as JSON strings. Be tolerant of empty or malformed values."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    raw = raw.strip()
    if not raw:
        return []
    try:
        val = json.loads(raw)
        return [str(x) for x in val] if isinstance(val, list) else [str(val)]
    except json.JSONDecodeError:
        return [raw]


def dedupe_aliases(primary: str, aliases: list[str], max_aliases: int | None = None) -> list[str]:
    """Primary name first, then aliases, deduplicated by normalized form, short ones dropped."""
    seen: set[str] = set()
    out: list[str] = []
    for a in [primary, *aliases]:
        a = a.strip()
        key = normalize(a)
        if len(a) < MIN_ALIAS_CHARS or not key or key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out[:max_aliases] if max_aliases else out


@lru_cache(maxsize=100_000)
def compile_alias_pattern(alias: str) -> re.Pattern:
    """Case-sensitive match with word boundaries that also work for aliases ending in punctuation."""
    escaped = re.escape(alias)
    return re.compile(rf"(?<!\w){escaped}(?!\w)")


def text_mentions(text: str, aliases: list[str]) -> bool:
    return any(compile_alias_pattern(a).search(text) for a in aliases)


def answer_matches(answer: str, gold_aliases: list[str]) -> bool:
    """Loose match for free-generation answers: normalized equality or whole-token containment."""
    a = normalize(answer)
    if not a:
        return False
    for g in gold_aliases:
        n = normalize(g)
        if not n:
            continue
        if a == n or re.search(rf"(?<!\w){re.escape(n)}(?!\w)", a):
            return True
    return False


# ---- corpus-search alias filtering ---------------------------------------------------------------
# Wikidata alias lists contain language/ISO codes that collide with common English words
# ("in" for India, "be" for Belgium, "by" for Belarus, "my" for Malaysia). Searching the corpus for
# those would make co-occurrence counts meaningless, so aliases used for *corpus search* are filtered.
# (Answer *scoring* keeps the full PopQA possible_answers list.)
SHORT_ALIAS_WHITELIST = {"USA", "U.S.", "U.S.A.", "UK", "U.K.", "UAE", "USSR", "DRC", "PRC", "ROC"}
MAX_SHORT_LEN = 3
RATIO_FLAG = 20.0  # alias more than 20x as frequent as the primary name -> needs manual review


def rule_filter_alias(primary: str, alias: str) -> str | None:
    """Return a drop reason, or None to keep. Rules are deterministic and model-independent."""
    if alias == primary:
        return None
    if alias in SHORT_ALIAS_WHITELIST:
        return None
    if primary[:1].isupper() and alias == alias.lower():
        return "lowercase_alias_of_proper_noun"
    if len(alias.replace(".", "")) <= MAX_SHORT_LEN:
        return "short_alias"
    return None


def filter_search_aliases(primary: str, alias_counts: dict[str, int],
                          overrides: dict[str, list[str]] | None = None) -> tuple[list[str], list[dict]]:
    """Apply rules + manual overrides. Returns (kept aliases, decision log rows).

    overrides: {"drop": [...], "keep": [...]} for this entity (manual review decisions, versioned in
    configs/alias_overrides.yaml). Ratio-flagged aliases are dropped unless explicitly kept.
    """
    overrides = overrides or {}
    drop_manual, keep_manual = set(overrides.get("drop", [])), set(overrides.get("keep", []))
    p_count = max(alias_counts.get(primary, 0), 1)
    kept, log = [], []
    for alias, cnt in alias_counts.items():
        reason = rule_filter_alias(primary, alias)
        ratio = cnt / p_count
        if reason is None and alias != primary and ratio > RATIO_FLAG:
            reason = "ratio_flag"
        if alias in keep_manual:
            reason = None
        if alias in drop_manual:
            reason = "manual_drop"
        log.append({"primary": primary, "alias": alias, "count": cnt, "ratio_to_primary": ratio,
                    "decision": "keep" if reason is None else "drop", "reason": reason})
        if reason is None:
            kept.append(alias)
    return kept, log
