"""Subject ambiguity: corpus mentions per Wikipedia pageview.

A subject string like "Cave" or "Tina" occurs millions of times in the corpus but almost never refers
to the PopQA entity, so its sampled contexts say nothing about that entity. We measure

    amb = log10( (corpus_count + 1) / (s_pop + 1) )

and flag subjects whose amb exceeds a threshold fixed on the DEV split only (pre-registered rule:
dev-split 90th percentile, or single-word subject above the dev 70th percentile). Primary analyses exclude flagged subjects; robustness checks include them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ambiguity_score(corpus_count: pd.Series, s_pop: pd.Series) -> pd.Series:
    return np.log10((corpus_count.astype(float) + 1) / (s_pop.astype(float) + 1))


def flag_ambiguous(df: pd.DataFrame, quantile: float = 0.9, single_word_quantile: float = 0.7
                   ) -> tuple[pd.Series, float]:
    """Flag = amb > dev P90, OR (single-word subject AND amb > dev P70).

    Rationale (decided from subject strings/counts only, before any model results): single-word
    names ("Dublin", "Logan", "Lima") are far more often ambiguous than multi-word names.
    df needs columns amb, split, subj. Returns (flags, P90 threshold).
    """
    dev = df.loc[df["split"] == "dev", "amb"]
    thr, thr_single = float(dev.quantile(quantile)), float(dev.quantile(single_word_quantile))
    single = ~df["subj"].str.contains(r"\s", regex=True)
    return (df["amb"] > thr) | (single & (df["amb"] > thr_single)), thr
