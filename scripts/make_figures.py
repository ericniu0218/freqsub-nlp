"""Regenerate every report figure from artifacts alone. Each figure is skipped if its stage hasn't run.

Style (validated palette, light surface #fcfcfb): series blue #2a78d6, orange #eb6834; text in
neutral ink; 2px lines, >=8px markers, recessive grid, one y-axis per chart, legend for >=2 series
plus direct labels.
Usage: python scripts/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from freqsub.utils.io import ARTIFACTS, ROOT  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"  # validated palette slots 1-4
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FIG = ROOT / "figures"


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def find_stage(stage: str, **match) -> Path | None:
    """Select a stage directory by its manifest config (never by mtime). Raises if ambiguous."""
    hits = []
    for d in (ARTIFACTS / stage).glob("*/"):
        m = d / "manifest.json"
        if not m.exists():
            continue
        conf = json.loads(m.read_text()).get("config", {})
        if all(conf.get(k) == v for k, v in match.items()):
            hits.append(d)
    if len(hits) > 1:
        raise RuntimeError(f"{stage}: {len(hits)} directories match {match}; refine the selection")
    return hits[0] if hits else None


SPLIT = "test"
REV = "stage1-step1907359-tokens4001B"  # primary checkpoint (configs/default.yaml)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h


def fig_accuracy_vs_exposure() -> None:
    d = find_stage("06_h1", split=SPLIT, revision=REV)
    if not d:
        return
    t = pd.read_parquet(d / "analysis_table.parquet")
    t = t[(t["template"] == "qa") & (t["k"] > 0) & (~t["ambiguous"]) & (~t["label_ambiguous"]) & (t["split"] == SPLIT)]
    g = t[t["is_gold"] == 1][["qid", "log_cooc"]].merge(
        t[t["chosen"] == 1][["qid", "is_gold"]].rename(columns={"is_gold": "correct"}), on="qid")
    g["bin"] = pd.qcut(g["log_cooc"].rank(method="first"), 6, labels=False)
    rows = []
    for _, b in g.groupby("bin"):
        lo, hi = wilson(int(b["correct"].sum()), len(b))
        rows.append((np.expm1(b["log_cooc"].median()), b["correct"].mean(), lo, hi, len(b)))
    x, y, lo, hi, n = map(np.array, zip(*rows))
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.fill_between(x + 1, lo, hi, color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(x + 1, y, color=BLUE, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    style(ax, "Accuracy vs. how often the true fact appears in training data",
          "Estimated subject–answer co-occurrences in pretraining corpus (+1, log scale)", "Accuracy")
    ax.text(0.99, 0.02, f"OLMo-2 1B stage-1, PopQA country ({SPLIT} split), n={int(n.sum())}; sextile bins; 95% Wilson CIs",
            transform=ax.transAxes, ha="right", va="bottom", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "a_accuracy_vs_exposure.png", dpi=200)
    plt.close(fig)


def fig_dynamics() -> None:
    d = find_stage("09_dynamics", split=SPLIT)
    if not d:
        return
    t = pd.read_parquet(d / "dynamics.parquet").sort_values("tokens_B")
    if len(t) < 2:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for col, ci, color, label in [("beta_log_cooc", "ci_log_cooc", BLUE, "co-occurrence (substitution)"),
                                  ("beta_log_freq", "ci_log_freq", ORANGE, "overall frequency")]:
        lo = [c[0] for c in t[ci]]
        hi = [c[1] for c in t[ci]]
        ax.fill_between(t["tokens_B"], lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(t["tokens_B"], t[col], color=color, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        ax.annotate(label, (t["tokens_B"].iloc[-1], t[col].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", color=INK2, fontsize=8, va="center")
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xscale("log")
    style(ax, "Pull toward co-occurring vs. frequent answers over training",
          "Training tokens seen (billions, log scale)", "Conditional-logit coefficient (per SD)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "b_dynamics.png", dpi=200)
    plt.close(fig)


def fig_logit_lens() -> None:
    d = find_stage("07_interp", eval_split=SPLIT, rev=REV)
    if not d or not (d / "logit_lens.parquet").exists():
        return
    t = pd.read_parquet(d / "logit_lens.parquet")
    t = t[t["split"] == SPLIT]
    s = t.groupby("layer")["logit_diff"].agg(["mean", "sem", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.fill_between(s["layer"], s["mean"] - 1.96 * s["sem"], s["mean"] + 1.96 * s["sem"], color=BLUE,
                    alpha=0.15, linewidth=0)
    ax.plot(s["layer"], s["mean"], color=BLUE, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.axhline(0, color=INK2, linewidth=0.8)
    style(ax, "Wrong answers pull ahead in layers 10–13 (logit lens)",
          "Layer", "Logit(gold) − logit(chosen wrong), first token")
    ax.text(0.99, 0.02, f"{SPLIT} split, n={int(s['count'].max())} wrong predictions; 95% CIs", transform=ax.transAxes,
            ha="right", va="bottom", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "c_logit_lens.png", dpi=200)
    plt.close(fig)


def fig_calibration() -> None:
    """Dev tuning curves (why alpha was set where it was) + the single test-split outcome in the note."""
    d = find_stage("08_calibrate", eval=SPLIT, rev=REV)
    if not d:
        return
    r = json.loads((d / "results.json").read_text())
    styles = {"corpus_marginal": (BLUE, "o", "corpus frequency"), "corpus_cooc": (ORANGE, "s", "corpus co-occurrence"),
              "contextual_cal": (AQUA, "^", "contextual calibration"), "pmi_dc": (YELLOW, "D", "PMI (domain premise)")}
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for m, (color, marker, label) in styles.items():
        curve = r["methods"][m]["dev_curve"]
        a = [c[0] for c in curve]
        acc = [c[1] for c in curve]
        ax.plot(a, acc, color=color, linewidth=2, label=label)
        ax.plot(a[::8], acc[::8], marker, color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                linestyle="none")
        ax.annotate(label, (a[-1], acc[-1]), xytext=(6, 0), textcoords="offset points", color=INK2, fontsize=8,
                    va="center")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 2.9)
    style(ax, "Subtracting any prior lowers accuracy (dev split, used to tune α)",
          "α (strength of prior subtraction)", "Accuracy")
    test = r["methods"]["corpus_marginal"][f"{SPLIT}_all"]
    tuned = " / ".join(f"{r['methods'][m]['alpha_dev']:g}" for m in styles)
    ax.text(0.01, 0.97, f"Tuned α (frequency / co-occurrence / contextual / PMI): {tuned}\n"
            f"{SPLIT.capitalize()}-split accuracy unchanged at {test['acc_a']:.3f}",
            transform=ax.transAxes, color=INK2, fontsize=8, va="top")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="center right", bbox_to_anchor=(1.0, 0.72))
    fig.tight_layout()
    fig.savefig(FIG / "d_calibration.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    for f in [fig_accuracy_vs_exposure, fig_dynamics, fig_logit_lens, fig_calibration]:
        try:
            f()
            print("ok:", f.__name__)
        except Exception as e:  # one missing/partial stage must not block the others
            print("skipped:", f.__name__, "-", e)
    print(sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
