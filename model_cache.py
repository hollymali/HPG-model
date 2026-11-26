"""Reuse simulation results only when they were made with the current MODEL."""

import hashlib
from pathlib import Path

import numpy as np

import MODEL


MODEL_HASH = hashlib.sha256(Path(MODEL.__file__).read_bytes()).hexdigest()
CSV_MODEL_TAG = f"# model_sha256={MODEL_HASH}"


def matches_model(path):
    """Old, untagged results must be simulated again."""
    if not path.exists():
        return False
    if path.suffix == ".npz":
        with np.load(path) as saved:
            return "model_sha256" in saved and str(saved["model_sha256"]) == MODEL_HASH
    with path.open(encoding="utf-8") as saved:
        next(saved, "")  # CSV column names.
        return next(saved, "").strip() == CSV_MODEL_TAG


def csv_header(columns):
    """Keep CSV column names first and add model provenance as a comment."""
    return f"{columns}\n{CSV_MODEL_TAG}"
