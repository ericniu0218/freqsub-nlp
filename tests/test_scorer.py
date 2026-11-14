"""The fast prefix-cached scorer must match the naive one-pass-per-candidate scorer."""
import os

import pytest

pytest.importorskip("transformers")

TINY = os.environ.get("FREQSUB_TEST_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")


@pytest.fixture(scope="module")
def lm():
    from freqsub.scoring.hf_scorer import load_model
    try:
        return load_model(TINY, dtype="float32", device="cpu")
    except Exception as e:  # offline CI without the model cached
        pytest.skip(f"test model unavailable: {e}")


CANDS = ["France", "United States of America", "Iran", "Côte d'Ivoire", "a", "New Zealand"]


def test_cached_matches_naive(lm):
    from freqsub.scoring.hf_scorer import score_cached, score_naive
    prompt = "Q: In what country is Paris?\nA:"
    naive = score_naive(lm, prompt, CANDS)
    fast, lengths = score_cached(lm, prompt, CANDS, batch_size=4)  # forces multiple uneven batches
    assert len(fast) == len(CANDS) and all(n >= 1 for n in lengths)
    for a, b in zip(naive, fast):
        assert abs(a - b) < 1e-3, (a, b)


def test_order_invariance(lm):
    from freqsub.scoring.hf_scorer import score_cached
    prompt = "Q: In what country is Tehran?\nA:"
    s1, _ = score_cached(lm, prompt, CANDS, batch_size=3)
    s2, _ = score_cached(lm, prompt, CANDS[::-1], batch_size=5)
    for a, b in zip(s1, s2[::-1]):
        assert abs(a - b) < 1e-3


def test_trailing_space_prompt_rejected(lm):
    from freqsub.scoring.hf_scorer import score_cached
    with pytest.raises(ValueError):
        score_cached(lm, "A: ", ["x"])
