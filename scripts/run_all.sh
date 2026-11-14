#!/usr/bin/env bash
# Idempotent job supervisor: (re)launch every long-running stage that isn't already running.
# Safe to call any number of times, e.g. after a container restart. Every stage is resumable
# (API cache + finished shards/chunks are skipped), so a relaunch only redoes in-flight work.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p artifacts
rm -f artifacts/04_scores/*/*.tmp artifacts/02b_contexts/*/*.tmp

launch() {  # launch <name-pattern> <log> <command...>
  local pat="$1" log="$2"; shift 2
  if pgrep -f "$pat" >/dev/null; then echo "running: $pat"; else
    setsid nohup "$@" >> "artifacts/$log" 2>&1 < /dev/null &
    echo "launched: $pat"; fi
}

# memory/load logger (diagnostics for container restarts)
if ! pgrep -f "freqsub-memlog" >/dev/null; then
  setsid nohup bash -c 'exec -a freqsub-memlog bash -c "while true; do echo \"\$(date +%T) \$(free -m | awk \"/Mem/{print \\\$3\\\"MB\\\"}\") load=\$(cut -d\" \" -f1 /proc/loadavg)\" >> artifacts/mem.log; sleep 30; done"' >/dev/null 2>&1 < /dev/null &
fi

# 1) corpus contexts (network-bound, ~0.7 req/s)
# (complete: all 837 subjects sampled) launch "^\.venv/bin/python -u scripts/02b_contexts\.py" 02b_run.log $PY -u scripts/02b_contexts.py
# 2) H2 checkpoints, template qa (each finished checkpoint is skipped without loading the model)
# (complete) launch "^\.venv/bin/python -u scripts/05_checkpoints\.py" 05_run.log $PY -u scripts/05_checkpoints.py "templates=[qa]" threads=4 batch_size=32 question_subset=primary_test
# 3) H3 on the TEST split (pre-registered; resumable via a per-pair score cache)
# (complete) launch "^\.venv/bin/python -u scripts/07_interp\.py eval_split=test" 07_test.log $PY -u scripts/07_interp.py eval_split=test k=3 max_cases=1000 threads=2
