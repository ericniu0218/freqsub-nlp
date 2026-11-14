"""Subject-level dev/test split, stratified by relation and popularity decile.

Splitting by *subject* (not question) guarantees no entity leaks between the set used to make design
choices (dev) and the set results are reported on (test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def assign_splits(df: pd.DataFrame, dev_frac: float = 0.2, seed: int = 0) -> pd.DataFrame:
    df = df.copy()
    subj = (df.groupby("subj_id")
            .agg(relation=("relation", "first"), s_pop=("s_pop", "max"))
            .reset_index())
    # popularity deciles within relation (rank first so ties don't break qcut)
    subj["pop_decile"] = (subj.groupby("relation")["s_pop"]
                          .transform(lambda s: pd.qcut(s.rank(method="first"), q=min(10, len(s)),
                                                       labels=False, duplicates="drop")))
    rng = np.random.default_rng(seed)
    subj["split"] = "test"
    for _, idx in subj.groupby(["relation", "pop_decile"]).groups.items():
        idx = list(idx)
        n_dev = int(round(len(idx) * dev_frac))
        dev_idx = rng.choice(idx, size=n_dev, replace=False) if n_dev else []
        subj.loc[dev_idx, "split"] = "dev"
    return df.merge(subj[["subj_id", "split"]], on="subj_id", how="left")
