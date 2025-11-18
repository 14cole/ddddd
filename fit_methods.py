"""Curve-fit scaffolding mirroring VBIDB IDBKODE handling.

This module is a Python-side placeholder to experiment with reading .LST and
.ROP files and producing parameter fits per IDBKODE. The real math for Debye /
Lorentzian fits still lives in the FORTRAN sources (e.g., IDB2LST.FOR,
IDB2LSF.FOR, IDB2ROP.FOR). Here we:

- Parse a .LST file.
- Load referenced .ROP files (expected to sit next to the .LST).
- Run very simple placeholder fits using numpy (means and polynomial fits over
  ALPHA) to provide a working end-to-end pipeline. Replace these routines with
  the actual Debye/Lorentz logic from the FORTRAN code for production use.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import file_loader

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None


@dataclass
class ParameterFit:
    alpha: float
    params: Dict[str, float]
    rms: float
    source: str


@dataclass
class FitResult:
    idbkode: int
    ceps_params: List[ParameterFit]
    cmu_params: List[ParameterFit]
    poly_coeffs: Dict[str, List[float]]


# IDBKODE decoding as in IDB2LST.FOR (ceps digit, cmu digit).
CEPS_EXT = {1: ".dbe", 2: ".lne", 3: ".2de", 4: ".2le", 5: ".3de", 6: ".dle"}
CMU_EXT = {1: ".dbm", 2: ".lnm", 3: ".2dm", 4: ".2lm", 5: ".3dm", 6: ".dlm"}


def _simple_stat_params(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Placeholder: compute simple statistics from ROP rows.

    We use average EPSR/EPSI/MUR/MUI as stand-ins for Debye/Lorentz parameters.
    """
    if not rows or np is None:
        return {}
    epsr = np.array([r["epsr"] for r in rows], dtype=float)
    epsi = np.array([r["epsi"] for r in rows], dtype=float)
    mur = np.array([r["mur"] for r in rows], dtype=float)
    mui = np.array([r["mui"] for r in rows], dtype=float)
    return {
        "EPSR_mean": float(epsr.mean()),
        "EPSI_mean": float(epsi.mean()),
        "MUR_mean": float(mur.mean()),
        "MUI_mean": float(mui.mean()),
    }


def _fit_polynomials(samples: List[ParameterFit], degree: int = 2) -> Dict[str, List[float]]:
    """Fit simple polynomials param(alpha) as placeholder."""
    if np is None or not samples:
        return {}
    alphas = np.array([s.alpha for s in samples], dtype=float)
    poly: Dict[str, List[float]] = {}
    keys = set(itertools.chain.from_iterable(s.params.keys() for s in samples))
    for key in keys:
        ys = np.array([s.params.get(key, 0.0) for s in samples], dtype=float)
        deg = min(degree, max(len(samples) - 1, 1))
        coeffs = np.polyfit(alphas, ys, deg)
        poly[key] = coeffs.tolist()
    return poly


def _fit_from_rop_files(entries: List[Tuple[float, str]], base_dir: Path) -> List[ParameterFit]:
    """Load .ROP files and produce placeholder ParameterFit instances."""
    fits: List[ParameterFit] = []
    for alpha, stem in entries:
        rop_path = base_dir / f"{stem}.rop"
        if not rop_path.exists():
            continue
        rop_data = file_loader.load_rop(rop_path)
        params = _simple_stat_params(rop_data.get("rows", []))
        rms = 0.0  # placeholder; compute real RMS against fit if available
        fits.append(ParameterFit(alpha=alpha, params=params, rms=rms, source=rop_path.name))
    return fits


def fit_from_lst(lst_path: Path, poly_degree: int = 2) -> FitResult:
    """Load a .LST, read its .ROP companions, and return placeholder fits."""
    lst_data = file_loader.load_lst(lst_path)
    idbkode_raw = lst_data.get("idbkode")
    legacy_map = {0: 10, 1: 20, 2: 11, 3: 21, 4: 12, 5: 22}
    idbkode = legacy_map.get(idbkode_raw, idbkode_raw)
    entries = lst_data.get("entries", [])

    ceps_fits = _fit_from_rop_files(entries, lst_path.parent)
    cmu_fits: List[ParameterFit] = []

    poly = _fit_polynomials(ceps_fits, degree=poly_degree)
    return FitResult(
        idbkode=idbkode or 0,
        ceps_params=ceps_fits,
        cmu_params=cmu_fits,
        poly_coeffs=poly,
    )


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Fit curves from .LST/.ROP files (placeholder implementation).")
    parser.add_argument("lst", type=Path, help="Path to .LST file")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree for placeholder fit")
    args = parser.parse_args()

    result = fit_from_lst(args.lst, poly_degree=args.degree)
    print(json.dumps(result, default=lambda o: o.__dict__, indent=2))


if __name__ == "__main__":
    main()
