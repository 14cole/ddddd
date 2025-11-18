"""Lorentz dispersion helpers (closer to VBIDB FORTRAN parity).

Matches CEPSLNG.FOR and CEPSLN2.FOR:
  single: deps / (1 - x*(x - j*gamma)), x = f/fres
  double: sum of two such terms, plus epsv - j*sige/(f*A0)
Frequencies are in GHz.
"""

from __future__ import annotations

import numpy as np

A0 = 5.563249e-2  # 2*pi*eps0*1e9 (GHz scaling)


def _safe_freq(f_ghz: np.ndarray) -> np.ndarray:
    return np.maximum(f_ghz, 1e-6)


def single_lorentz(
    f_ghz: np.ndarray, fres: float, deps: float, epsv: float, gamma: float, sige: float = 0.0
) -> np.ndarray:
    f = _safe_freq(f_ghz)
    x = f / max(fres, 1e-12)
    g2 = gamma * gamma
    u = 1.0 / ((1.0 - x * x) ** 2 + x * x * g2)
    er = epsv + deps * (1.0 - x * x) * u
    ei = -deps * gamma * x * u - sige / (f * A0)
    return er + 1j * ei


def double_lorentz(
    f_ghz: np.ndarray,
    fres1: float,
    deps1: float,
    gamma1: float,
    fres2: float,
    deps2: float,
    gamma2: float,
    epsv: float,
    sige: float = 0.0,
) -> np.ndarray:
    f = _safe_freq(f_ghz)
    cond_term = -1j * sige / (f * A0)

    def _comp(fres: float, deps: float, gamma: float) -> np.ndarray:
        x = f / max(fres, 1e-12)
        return deps / (1.0 - x * (x - 1j * gamma))

    c1 = _comp(fres1, deps1, gamma1)
    c2 = _comp(fres2, deps2, gamma2)
    return epsv + cond_term + c1 + c2


def debye_lorentz_combo(
    f_ghz: np.ndarray,
    fresd: float,
    depsd: float,
    gammd: float,
    fresl: float,
    depsl: float,
    gamml: float,
    epsv: float,
    sige: float = 0.0,
) -> np.ndarray:
    """Debye+Lorentz combination matching CEPSDL2.FOR."""
    f = _safe_freq(f_ghz)
    gammd = abs(gammd)
    gammd = min(gammd, 10.0)
    gamml = max(min(gamml, 100.0), 0.001)
    # Base term with conductivity
    cond_term = -1j * sige / (f * A0)
    c0 = epsv + cond_term

    # Debye contribution with spreading (three poles)
    x1 = f / max(fresd, 1e-12)
    if gammd == 0.0:
        c1 = depsd / (1.0 + 1j * x1)
    else:
        dv = 10.0**gammd
        xm = x1 / dv
        xp = x1 * dv
        c1 = depsd * (1.0 / (1.0 + 1j * xm) + 1.0 / (1.0 + 1j * x1) + 1.0 / (1.0 + 1j * xp)) / 3.0

    # Lorentzian contribution
    x1 = f / max(fresl, 1e-12)
    c2 = depsl / (1.0 - x1 * (x1 - 1j * gamml))

    return c0 + c1 + c2


__all__ = [
    "single_lorentz",
    "double_lorentz",
    "debye_lorentz_combo",
]
