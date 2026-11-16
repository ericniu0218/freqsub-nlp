"""One-figure summary: co-occurrence coefficient on the test split, primary analysis and robustness checks.

Reads the tracked aggregate `artifacts/06_h1/*/results.json` (the one with split == "test"; selected by
content, never by mtime). Writes figures/summary_main.png. Same palette and surface as make_figures.py.
Usage: python scripts/make_summary_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from freqsub.utils.io import ARTIFACTS, ROOT  # noqa: E402

BLUE = "#2a78d6"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

hits = [p for p in (ARTIFACTS / "06_h1").glob("*/results.json") if json.loads(p.read_text()).get("split") == "test"]
if len(hits) != 1:
    raise RuntimeError(f"expected exactly one test-split H1 result, found {len(hits)}")
res = json.loads(hits[0].read_text())

rows = [("Primary analysis", res["primary"])]
for key, label in [
    ("template_question_answer", "Other prompt: question-answer"),
    ("template_instructed", "Other prompt: instructed"),
    ("length_normalized_choice", "Length-normalized scoring"),
    ("excl_gold_in_demos", "Drop questions whose answer is in the prompt examples"),
    ("incl_label_ambiguous", "Keep label-ambiguous answers"),
    ("incl_ambiguous", "Keep ambiguous-name subjects"),
]:
    rows.append((label, res["robustness"][key]))

fig, ax = plt.subplots(figsize=(9, 4.6))
fig.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
for i, (label, r) in enumerate(rows):
    c = r["clogit"]["log_cooc"]
    y = len(rows) - 1 - i
    ax.plot([c["ci_low"], c["ci_high"]], [y, y], color=BLUE, linewidth=2, solid_capstyle="round")
    ax.plot(c["coef"], y, "o", color=BLUE, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.text(0.745, y, f'{c["coef"]:.2f} [{c["ci_low"]:.2f}, {c["ci_high"]:.2f}]   n={r["n_questions"]}',
            va="center", ha="left", color=INK2, fontsize=9)
ax.axvline(0, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1], color=INK, fontsize=10)
ax.set_xlim(-0.05, 1.15)
ax.set_xlabel("Log-odds of being chosen per SD of co-occurrence (95% CI). 0 = no effect.", color=INK2, fontsize=9)
fig.text(0.01, 0.955, "OLMo-2 1B picks countries that co-occur with the subject in its training text",
         color=INK, fontsize=12, ha="left")
ax.tick_params(colors=INK2, labelsize=9)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.grid(True, axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
fig.text(0.01, 0.015, "PopQA country questions, test split. Archived aggregate: row-level data is not in this repo.",
         color=INK2, fontsize=8, ha="left")
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
out = ROOT / "figures" / "summary_main.png"
fig.savefig(out, dpi=160, facecolor=SURFACE)
print("wrote", out)
