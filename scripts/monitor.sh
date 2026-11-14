#!/usr/bin/env bash
# Foreground monitor: keeps the session active while background stages run (the cloud container is
# reclaimed when idle). Relaunches dead jobs every minute; prints a progress line at the end.
# Usage: scripts/monitor.sh [minutes=9]
cd "$(dirname "$0")/.."
end=$(( $(date +%s) + ${1:-9} * 60 ))
while [ "$(date +%s)" -lt "$end" ]; do scripts/run_all.sh >/dev/null 2>&1; sleep 60; done
echo "$(date +%T) up $(cut -d. -f1 /proc/uptime)s | chunks=$(ls artifacts/02b_contexts/966ae51388dd/chunk_*.parquet 2>/dev/null | wc -l)/34 (primary=610 subjects=first 25) | $(tail -1 artifacts/02b_run.log | cut -c1-40)"
echo "  ckpt: $(grep -E 'running:' artifacts/05_run.log | tail -1 | grep -o 'tokens[0-9]*B') $(grep -E 'shard' artifacts/05_run.log | tail -1 | cut -c1-60)"
echo "  tmpl: $(grep -v -E 'Loading|Fetching|Warning' artifacts/04_run.log | grep shard | tail -1 | cut -c1-60) | mem $(tail -1 artifacts/mem.log)"
for f in artifacts/07_interp/*/pair_cache.sqlite; do [ -f "$f" ] && echo "  h3: $(.venv/bin/python -c "import sqlite3;print(sqlite3.connect('$f').execute('select count(*) from s').fetchone()[0])") pair scores cached ($f)"; done
