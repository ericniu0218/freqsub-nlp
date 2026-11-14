"""infini-gram API client: persistent cache, global rate limit, retries with backoff.

Every response is cached in SQLite keyed by the exact request, so runs are resumable and re-running
an analysis never re-queries the API. All query types we use are deterministic for a fixed index.
"""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import requests

API_URL = "https://api.infini-gram.io/"
USER_AGENT = "freqsub-research/0.1"


class InfiniGramError(RuntimeError):
    pass


class TokenBucket:
    """Thread-safe rate limiter shared by all worker threads."""

    def __init__(self, rate_per_s: float, burst: int = 1):
        self.rate = rate_per_s
        self.capacity = max(1, burst)
        self.tokens = float(self.capacity)
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
                self.last = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait = (1 - self.tokens) / self.rate
            time.sleep(wait)


class ResponseCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, request TEXT, "
                          "response TEXT, fetched_at REAL)")
        self.lock = threading.Lock()

    @staticmethod
    def key(payload: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def get(self, payload: dict[str, Any]) -> dict | None:
        with self.lock:
            row = self.conn.execute("SELECT response FROM responses WHERE key=?",
                                    (self.key(payload),)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, payload: dict[str, Any], response: dict) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO responses VALUES (?,?,?,?)",
                              (self.key(payload), json.dumps(payload, sort_keys=True),
                               json.dumps(response), time.time()))
            self.conn.commit()

    def __len__(self) -> int:
        with self.lock:
            return self.conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]


class InfiniGramClient:
    def __init__(self, index: str, cache_path: Path, rate_per_s: float = 4.0, max_retries: int = 12,
                 timeout_s: float = 120.0, session: requests.Session | None = None,
                 cooldown_start_s: float = 60.0, cooldown_max_s: float = 600.0):
        self.index = index
        self.cache = ResponseCache(cache_path)
        self.bucket = TokenBucket(rate_per_s, burst=2)
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Content-Type": "application/json"})
        self.stats = {"cache_hits": 0, "api_calls": 0, "retries": 0, "throttle_pauses": 0}
        self._stats_lock = threading.Lock()
        # Shared throttle state: the API answers bursts with HTTP 403 for a few minutes. One pause,
        # shared by every worker thread, doubling on repeated blocks and reset by any success.
        self._throttle_lock = threading.Lock()
        self._blocked_until = 0.0
        self._cooldown_start = cooldown_start_s
        self._cooldown_max = cooldown_max_s
        self._cooldown = cooldown_start_s

    def _wait_if_throttled(self) -> None:
        while True:
            with self._throttle_lock:
                remaining = self._blocked_until - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 5.0))

    def _register_throttle(self) -> None:
        with self._throttle_lock:
            now = time.monotonic()
            if now >= self._blocked_until:  # only the first thread to see the block extends it
                self._blocked_until = now + self._cooldown * (0.8 + 0.4 * random.random())
                self._cooldown = min(self._cooldown * 2, self._cooldown_max)
                self._bump("throttle_pauses")

    def _register_success(self) -> None:
        with self._throttle_lock:
            self._cooldown = self._cooldown_start

    # ---- low level -------------------------------------------------------------------------
    def _bump(self, k: str) -> None:
        with self._stats_lock:
            self.stats[k] += 1

    def request(self, payload: dict[str, Any]) -> dict:
        payload = {"index": self.index, **payload}
        cached = self.cache.get(payload)
        if cached is not None:
            self._bump("cache_hits")
            return cached
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            self._wait_if_throttled()
            self.bucket.acquire()
            try:
                r = self.session.post(API_URL, data=json.dumps(payload), timeout=self.timeout_s)
                self._bump("api_calls")
                if r.status_code == 200:
                    data = r.json()
                    if "error" in data:
                        # API-level errors (bad params) are not transient: fail loudly, never cache
                        raise InfiniGramError(f"{data['error']} | payload={payload}")
                    self._register_success()
                    self.cache.put(payload, data)
                    return data
                if r.status_code in (403, 429):
                    # throttling (this API uses 403): pause every thread, not just this one
                    last_err = InfiniGramError(f"HTTP {r.status_code} (throttled)")
                    self._register_throttle()
                    self._bump("retries")
                    continue
                if r.status_code in (500, 502, 503, 504):
                    last_err = InfiniGramError(f"HTTP {r.status_code}")
                else:
                    raise InfiniGramError(f"HTTP {r.status_code}: {r.text[:300]} | payload={payload}")
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
            self._bump("retries")
            time.sleep(min(60, (2 ** attempt)) * (0.5 + random.random()))  # backoff with jitter
        raise InfiniGramError(f"gave up after {self.max_retries} attempts: {last_err} | payload={payload}")

    # ---- query types -----------------------------------------------------------------------
    def count(self, query: str) -> dict:
        """Exact n-gram count of a string (no AND/OR: those are approximate and biased, see plan)."""
        if " AND " in query or " OR " in query:
            raise ValueError("CNF counts are deliberately unsupported; use the context sampler")
        d = self.request({"query_type": "count", "query": query})
        if d.get("approx"):
            raise InfiniGramError(f"plain count unexpectedly approximate: {query!r}")
        return d

    def find(self, query: str) -> dict:
        return self.request({"query_type": "find", "query": query})

    def get_doc_by_rank(self, shard: int, rank: int, query_ids: list[int], max_disp_len: int) -> dict:
        return self.request({"query_type": "get_doc_by_rank", "s": shard, "rank": rank,
                             "query_ids": query_ids, "max_disp_len": max_disp_len})
