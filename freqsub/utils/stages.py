"""Deterministic artifact resolution.

Every stage directory is artifacts/<stage>/<config_hash(stage_config)>. Downstream stages must locate
their inputs by recomputing that hash from the current config, NEVER by "most recently modified
directory": directory mtimes are not reliable (copies/moves reset them), and picking by mtime once
silently selected an obsolete stage-02a run (see preregistration change log, 2026-11-10).
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from .config import resolve
from .io import ARTIFACTS, config_hash


def cfg_01(cfg: dict) -> dict:
    return {"data": cfg["data"], "seed": cfg["seed"]}


def cfg_02a(cfg: dict) -> dict:
    overrides = OmegaConf.to_container(OmegaConf.load(resolve("configs/alias_overrides.yaml")))
    return {"data": cfg["data"], "seed": cfg["seed"], "index": cfg["counts"]["index"], "overrides": overrides}


def cfg_02b(cfg: dict) -> dict:
    c = cfg["counts"]
    return {"source": config_hash(cfg_02a(cfg)), "seed": cfg["seed"], "index": c["index"],
            "k": c["contexts_per_subject"], "max_disp_len": c["max_disp_len"]}


def _must_exist(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"expected stage output {p} (run the upstream stage with the current config)")
    return p


def dir_01(cfg: dict) -> Path:
    return _must_exist(ARTIFACTS / "01_data" / config_hash(cfg_01(cfg)))


def dir_02a(cfg: dict) -> Path:
    return _must_exist(ARTIFACTS / "02a_counts" / config_hash(cfg_02a(cfg)))


def dir_02b(cfg: dict) -> Path:
    return _must_exist(ARTIFACTS / "02b_contexts" / config_hash(cfg_02b(cfg)))


def frozen_demos(relation: str) -> dict:
    """Demonstration subject ids frozen in configs/demos.yaml (identical prompts across all runs)."""
    d = OmegaConf.to_container(OmegaConf.load(resolve("configs/demos.yaml")))
    return d[relation]
