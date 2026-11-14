"""Data layer tests: alias handling, answer matching, split disjointness and stratification."""
import pandas as pd
import pytest

from freqsub.data.aliases import answer_matches, dedupe_aliases, normalize, parse_alias_field
from freqsub.data.splits import assign_splits
from freqsub.utils.io import ROOT


def test_normalize():
    assert normalize("  The United States! ") == "united states"
    assert normalize("Côte d’Ivoire") == normalize("côte d ivoire")


def test_parse_alias_field_tolerant():
    assert parse_alias_field('["a","b"]') == ["a", "b"]
    assert parse_alias_field("") == []
    assert parse_alias_field("not json") == ["not json"]
    assert parse_alias_field(None) == []


def test_dedupe_keeps_primary_first_and_drops_dupes():
    out = dedupe_aliases("United States", ["united states", "USA", "U", "the United States"])
    assert out == ["United States", "USA"]


def test_answer_matching():
    gold = ["United States of America", "USA", "United States"]
    assert answer_matches("the United States", gold)
    assert answer_matches("USA.", gold)
    assert not answer_matches("Canada", gold)
    assert not answer_matches("", gold)


def _toy(n=400):
    return pd.DataFrame({"id": range(n), "subj_id": [i // 2 for i in range(n)],  # 2 questions per subject
                         "relation": ["country"] * n, "s_pop": [i * 7 % 1000 for i in range(n)]})


def test_splits_disjoint_by_subject():
    df = assign_splits(_toy(), dev_frac=0.2, seed=0)
    dev, test = set(df[df.split == "dev"].subj_id), set(df[df.split == "test"].subj_id)
    assert dev and test and not dev & test


def test_splits_fraction_and_determinism():
    a, b = assign_splits(_toy(), seed=3), assign_splits(_toy(), seed=3)
    assert (a.split == b.split).all()
    frac = a.drop_duplicates("subj_id").split.eq("dev").mean()
    assert 0.15 < frac < 0.25


@pytest.mark.skipif(not (ROOT / "data_raw/popqa_test.tsv").exists(), reason="PopQA not downloaded")
def test_real_popqa_country_loads():
    from freqsub.data.popqa import load_popqa, object_pool
    df = load_popqa(ROOT / "data_raw/popqa_test.tsv", ["country"])
    assert len(df) == 838
    assert all(len(g) >= 1 for g in df.gold_answers)
    assert len(object_pool(df, "country")) == 107


@pytest.mark.skipif(not (ROOT / "artifacts/04_scores/2071aaf7f97d/demos.parquet").exists(), reason="no scoring run")
def test_frozen_demos_reproduce_scored_prompts():
    """configs/demos.yaml must rebuild exactly the demos (and order) used by the primary scoring run."""
    from freqsub.prompts.builder import PromptBank
    from freqsub.utils.config import load_config
    from freqsub.utils.stages import dir_01, frozen_demos
    cfg = load_config([])
    q = pd.read_parquet(dir_01(cfg) / "questions.parquet")
    bank = PromptBank.from_ids(q, "country", **frozen_demos("country"))
    stored = pd.read_parquet(ROOT / "artifacts/04_scores/2071aaf7f97d/demos.parquet")
    for name, demos in [("primary", bank.primary), ("backup", bank.backup)]:
        s = stored[stored["set"] == name]
        assert list(s["subj_id"]) == list(demos["subj_id"])
        assert list(s["question"]) == list(demos["question"]) and list(s["obj"]) == list(demos["obj"])


def test_stage_resolution_is_deterministic_and_never_by_mtime():
    from freqsub.utils.config import load_config
    from freqsub.utils.stages import cfg_02a, cfg_02b
    from freqsub.utils.io import config_hash
    cfg = load_config([])
    assert config_hash(cfg_02a(cfg)) == config_hash(cfg_02a(cfg))
    assert cfg_02b(cfg)["source"] == config_hash(cfg_02a(cfg))
