"""Counts layer tests: caching, throttle handling, error policy, sampler correctness, estimator bias."""
import json
import random

import pytest

from freqsub.counts import client as client_mod
from freqsub.counts.client import InfiniGramClient, InfiniGramError
from freqsub.counts.contexts import (SubjectContexts, estimate_cooc, sample_occurrences)


class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class FakeSession:
    """Replays a scripted list of (status, body) responses and records calls."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.headers = {}

    def post(self, url, data, timeout):
        self.calls += 1
        status, body = self.script.pop(0) if self.script else (200, {"count": 1, "approx": False})
        return FakeResponse(status, body)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)


def make_client(tmp_path, script, **kw):
    kw.setdefault("cooldown_start_s", 0.0)
    return InfiniGramClient("idx", tmp_path / "c.sqlite", rate_per_s=1e6, session=FakeSession(script), **kw)


def test_cache_prevents_second_api_call(tmp_path):
    c = make_client(tmp_path, [(200, {"count": 5, "approx": False})])
    assert c.count("Paris")["count"] == 5
    assert c.count("Paris")["count"] == 5
    assert c.session.calls == 1 and c.stats["cache_hits"] == 1


def test_cache_persists_across_clients(tmp_path):
    make_client(tmp_path, [(200, {"count": 7, "approx": False})]).count("Rome")
    c2 = make_client(tmp_path, [])
    assert c2.count("Rome")["count"] == 7 and c2.session.calls == 0


def test_throttle_403_is_retried_then_succeeds(tmp_path):
    c = make_client(tmp_path, [(403, {"message": "Forbidden"}), (403, {"message": "Forbidden"}),
                               (200, {"count": 3, "approx": False})])
    assert c.count("Iran")["count"] == 3
    assert c.stats["throttle_pauses"] >= 1


def test_gives_up_after_max_retries(tmp_path):
    c = make_client(tmp_path, [(403, {})] * 50, max_retries=3)
    with pytest.raises(InfiniGramError):
        c.count("Iran")


def test_api_error_field_is_not_cached_and_raises(tmp_path):
    c = make_client(tmp_path, [(200, {"error": "bad params"})])
    with pytest.raises(InfiniGramError):
        c.count("x")
    assert len(c.cache) == 0


def test_non_retryable_http_error_raises_immediately(tmp_path):
    c = make_client(tmp_path, [(400, {"error": "nope"})])
    with pytest.raises(InfiniGramError):
        c.count("x")
    assert c.session.calls == 1


def test_cnf_counts_refused(tmp_path):
    c = make_client(tmp_path, [])
    with pytest.raises(ValueError):
        c.count("A AND B")


def test_approx_plain_count_raises(tmp_path):
    c = make_client(tmp_path, [(200, {"count": 1, "approx": True})])
    with pytest.raises(InfiniGramError):
        c.count("x")


# ---- sampler -------------------------------------------------------------------------------
def test_sampler_exhaustive_when_small():
    occ = sample_occurrences([[10, 13], [0, 0], [5, 7]], k=50, rng=random.Random(0))
    assert sorted(occ) == [(0, 10), (0, 11), (0, 12), (2, 5), (2, 6)]


def test_sampler_no_duplicates_and_in_range():
    segs = [[100, 1100], [0, 0], [50, 550]]
    occ = sample_occurrences(segs, k=200, rng=random.Random(1))
    assert len(occ) == len(set(occ)) == 200
    for s, r in occ:
        a, b = segs[s]
        assert a <= r < b


def test_sampler_is_proportional_to_shard_size():
    segs = [[0, 9000], [0, 1000]]
    hits0 = sum(sum(1 for s, _ in sample_occurrences(segs, 100, random.Random(i)) if s == 0) for i in range(200))
    assert abs(hits0 / (200 * 100) - 0.9) < 0.02


# ---- estimator -----------------------------------------------------------------------------
def test_estimator_unbiased_on_synthetic_population():
    """Population of 5000 mentions, 12% mention 'Chicago'. Mean estimate over many samples ~ truth."""
    rng = random.Random(0)
    pop = ["... lived in Chicago ..." if i < 600 else "... lived elsewhere ..." for i in range(5000)]
    ests = []
    for _ in range(400):
        ctx = SubjectContexts("s", n_occurrences=5000, windows=rng.sample(pop, 50))
        ests.append(estimate_cooc(ctx, ["Chicago"]).cooc_hat)
    mean = sum(ests) / len(ests)
    assert abs(mean - 600) / 600 < 0.05


def test_estimator_exact_when_exhaustive():
    ctx = SubjectContexts("s", n_occurrences=3, windows=["in Paris", "in Rome", "Paris again"], exhaustive=True)
    e = estimate_cooc(ctx, ["Paris"])
    assert e.hits == 2 and e.cooc_hat == 2 and e.se_p == 0.0


def test_estimator_case_sensitive_word_boundary():
    ctx = SubjectContexts("s", n_occurrences=2, windows=["give us a call", "the US Army"], exhaustive=True)
    assert estimate_cooc(ctx, ["US"]).hits == 1
    ctx2 = SubjectContexts("s", n_occurrences=1, windows=["a Parisian cafe"], exhaustive=True)
    assert estimate_cooc(ctx2, ["Paris"]).hits == 0
