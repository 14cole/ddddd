"""IDB polynomial evaluation helpers.

Evaluates polynomial coefficients from IDB/LSF-style data:
    param(alpha) = SUM(A_i * (alpha - alpha0)**i)

This module does not yet parse IDB files; it evaluates given coefficients.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import re
from pathlib import Path

import numpy as np


def eval_poly(coeffs: Iterable[float], alpha: float, alpha0: float = 0.0) -> float:
    """Evaluate polynomial sum A_i * (alpha - alpha0)**i."""
    a = np.array(list(coeffs), dtype=float)
    powers = np.power(alpha - alpha0, np.arange(len(a)))
    return float(np.dot(a, powers))


def eval_params(coeff_matrix: List[List[float]], alpha: float, alpha0: float = 0.0) -> List[float]:
    """Evaluate a list of parameter polynomials."""
    return [eval_poly(row, alpha, alpha0) for row in coeff_matrix]


# ---------------------- IDB file parsing (basic) --------------------------- #

_NUM_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?")


def parse_idb_file(path: str) -> Optional[Tuple[float, int, List[List[float]], List[List[float]]]]:
    """Parse a minimal subset of an .IDB file: alpha0, idbkode, CEPS and CMU coefficients.

    Returns (alpha0, idbkode, ceps_coeffs, cmu_coeffs) or None on failure.
    Each coeff list is a list of lists: one entry per parameter, coefficients 0..M.
    """
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None

    alpha0 = None
    idbkode = None
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("! START OF THE IDB DATA FILE"):
            start_idx = i + 1
            break
    if start_idx is None or start_idx >= len(lines):
        return None

    nums = _NUM_RE.findall(lines[start_idx])
    if len(nums) >= 2:
        alpha0 = float(nums[0])
        idbkode = int(float(nums[1]))
    else:
        return None

    def read_segments(idx: int, count: int) -> Tuple[List[List[float]], int]:
        coeffs: List[List[float]] = []
        pos = idx
        for _ in range(count):
            if pos >= len(lines):
                break
            deg_tokens = _NUM_RE.findall(lines[pos])
            if not deg_tokens:
                break
            deg = int(float(deg_tokens[0]))
            pos += 1
            if pos >= len(lines):
                break
            coeff_tokens = _NUM_RE.findall(lines[pos])
            coeff_list = [float(x) for x in coeff_tokens[: deg + 1]]
            coeffs.append(coeff_list)
            pos += 1
        return coeffs, pos

    # Determine CEPS/CMU segment counts from idbkode
    ce_count = 5
    cm_count = 0
    if idbkode is not None:
        idbke = idbkode // 10
        idbkm = idbkode % 10
        ce_count = 5
        if idbke in (3, 4, 5, 6):
            ce_count = 8
        cm_count = 0 if idbkm == 0 else (5 if idbkm in (1, 2) else 8)

    ceps_coeffs, pos = read_segments(start_idx + 1, ce_count)
    cmu_coeffs: List[List[float]] = []
    if cm_count > 0:
        cmu_coeffs, pos = read_segments(pos, cm_count)

    return alpha0, idbkode, ceps_coeffs, cmu_coeffs
