"""Load PopQA and turn it into a clean, typed question table."""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .aliases import dedupe_aliases, parse_alias_field

POPQA_URL = "https://huggingface.co/datasets/akariasai/PopQA/resolve/main/test.tsv"


def download_popqa(dest: Path) -> Path:
    import requests

    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(POPQA_URL, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def load_popqa(path: Path, relations: list[str] | None = None, max_aliases: int = 5) -> pd.DataFrame:
    csv.field_size_limit(10**9)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    df = pd.DataFrame(rows)
    if relations:
        df = df[df["prop"].isin(relations)].copy()

    for col in ["id", "subj_id", "obj_id", "s_pop", "o_pop"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df["s_alias_list"] = [dedupe_aliases(s, parse_alias_field(a), max_aliases)
                          for s, a in zip(df["subj"], df["s_aliases"])]
    df["o_alias_list"] = [dedupe_aliases(o, parse_alias_field(a), max_aliases)
                          for o, a in zip(df["obj"], df["o_aliases"])]
    # possible_answers = every acceptable surface form, used for scoring correctness
    df["gold_answers"] = [dedupe_aliases(o, parse_alias_field(p))
                          for o, p in zip(df["obj"], df["possible_answers"])]

    keep = ["id", "prop", "subj", "subj_id", "obj", "obj_id", "s_pop", "o_pop", "question",
            "s_alias_list", "o_alias_list", "gold_answers"]
    df = df[keep].rename(columns={"prop": "relation"}).reset_index(drop=True)
    df = df.dropna(subset=["subj", "obj"])
    return df


def object_pool(df: pd.DataFrame, relation: str) -> pd.DataFrame:
    """All distinct gold objects for a relation: the typed candidate pool, one row per entity."""
    sub = df[df["relation"] == relation]
    pool = (sub.groupby("obj_id")
            .agg(obj=("obj", "first"), o_pop=("o_pop", "first"),
                 o_alias_list=("o_alias_list", "first"), n_gold=("id", "size"))
            .reset_index())
    return pool.sort_values("obj").reset_index(drop=True)
