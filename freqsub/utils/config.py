"""Tiny config loader: YAML defaults + command-line dotlist overrides (OmegaConf)."""
from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

from .io import ROOT


def load_config(argv: list[str] | None = None, default: str = "configs/default.yaml") -> dict:
    argv = sys.argv[1:] if argv is None else argv
    base = OmegaConf.load(ROOT / default)
    cfg = OmegaConf.merge(base, OmegaConf.from_dotlist(argv))
    return OmegaConf.to_container(cfg, resolve=True)


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
