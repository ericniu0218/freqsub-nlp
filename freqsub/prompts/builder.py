"""Few-shot prompt construction.

Design rules (from "Calibrate Before Use", Zhao et al. 2021, which showed majority-label, recency and
common-token biases in few-shot prompts):
  * demonstrations come from the DEV split only and from subjects different from the target;
  * the 4 demo answers are spread across corpus-frequency quartiles (one each), so the prompt itself
    does not push the model toward frequent answers;
  * demo order is fixed by seed; the same demo set is used for every target in a template, except
    targets whose subject is a demo subject, which get a disjoint backup set.
Three paraphrased templates per relation; templates are frozen after dev-split selection.
"""
from __future__ import annotations

import random

import pandas as pd

TEMPLATES: dict[str, str] = {
    "qa": "Q: {question}\nA:",
    "question_answer": "Question: {question}\nAnswer:",
    "instructed": "Answer with the name of a country.\nQ: {question}\nA:",
}
DEMO_SEP = "\n\n"


def pick_demos(dev: pd.DataFrame, entity_freq: dict, n: int = 4, seed: int = 0,
               exclude_subj_ids: set | None = None) -> pd.DataFrame:
    """One demo per frequency quantile of the gold answer; distinct subjects and distinct answers."""
    exclude_subj_ids = exclude_subj_ids or set()
    d = dev[~dev["subj_id"].isin(exclude_subj_ids)].copy()
    d["gold_freq"] = d["obj_id"].map(entity_freq).fillna(0)
    d["fq"] = pd.qcut(d["gold_freq"].rank(method="first"), q=n, labels=False)
    rng = random.Random(seed)
    chosen, used_objs = [], set()
    for q in range(n):
        pool = d[(d["fq"] == q) & (~d["obj_id"].isin(used_objs))]
        row = pool.iloc[rng.randrange(len(pool))]
        chosen.append(row)
        used_objs.add(row["obj_id"])
    demos = pd.DataFrame(chosen)
    return demos.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle order


def render(template: str, question: str, answer: str | None = None) -> str:
    s = TEMPLATES[template].format(question=question)
    return s + (" " + answer if answer is not None else "")


def build_prompt(template: str, demos: pd.DataFrame, question: str) -> str:
    parts = [render(template, r["question"], r["obj"]) for _, r in demos.iterrows()]
    parts.append(render(template, question))
    return DEMO_SEP.join(parts)


class PromptBank:
    """Holds a primary and a backup demo set; returns the right prompt for each target."""

    def __init__(self, dev: pd.DataFrame, entity_freq: dict, seed: int = 0, n_demos: int = 4):
        self.primary = pick_demos(dev, entity_freq, n_demos, seed)
        self.backup = pick_demos(dev, entity_freq, n_demos, seed + 1,
                                 exclude_subj_ids=set(self.primary["subj_id"]))
        overlap = set(self.primary["subj_id"]) & set(self.backup["subj_id"])
        assert not overlap, "backup demos must be disjoint from primary demos"

    @classmethod
    def from_ids(cls, questions: pd.DataFrame, relation: str, primary: list, backup: list) -> "PromptBank":
        """Rebuild a bank from frozen demo subject ids, preserving their prompt order exactly."""
        q = questions[questions["relation"] == relation].drop_duplicates("subj_id").set_index("subj_id")
        bank = cls.__new__(cls)
        bank.primary = q.loc[list(primary)].reset_index()
        bank.backup = q.loc[list(backup)].reset_index()
        assert not set(primary) & set(backup), "backup demos must be disjoint from primary demos"
        assert (bank.primary["split"] == "dev").all() and (bank.backup["split"] == "dev").all(), "demos must be dev"
        return bank

    def demos_for(self, subj_id) -> tuple[pd.DataFrame, str]:
        if subj_id in set(self.primary["subj_id"]):
            return self.backup, "backup"
        return self.primary, "primary"

    def prompt(self, template: str, subj_id, question: str) -> tuple[str, str, list]:
        demos, which = self.demos_for(subj_id)
        return build_prompt(template, demos, question), which, list(demos["obj_id"])
