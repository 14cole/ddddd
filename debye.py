"""Debye dispersion helpers (closer to VBIDB FORTRAN parity).

Implements single/double/triple Debye evaluators using the same forms as
CEPSDBG.FOR and CEPSDB2/CEPSDB3:

    component = deps * avg(1 / (1 + j*x)) over x = f/fres, f/fres*dv, f/fres/dv
    dv = 10**gamma   (gamma clamped)
    eps  = epsv - j*sige/(f*A0) + sum(components)

where A0 = 2*pi*eps0*1e9. Frequencies are in GHz.
"""

from __future__ import annotations

import numpy as np


A0 = 5.563249e-2  # 2*pi*eps0*1e9 (GHz scaling)


def _safe_freq(f_ghz: np.ndarray) -> np.ndarray:
    return np.maximum(f_ghz, 1e-6)


def _debye_component(f_ghz: np.ndarray, fres: float, deps: float, gamma: float) -> np.ndarray:
    """Return complex Debye contribution (no epsv or conductivity)."""
    f = _safe_freq(f_ghz)
    gamma = abs(gamma)
    gamma = min(gamma, 10.0)
    x = f / max(fres, 1e-12)
    j = 1j
    if gamma == 0.0:
        return deps / (1.0 + j * x)
    dv = 10.0 ** gamma
    xm = x / dv
    xp = x * dv
    return deps * (1.0 / (1.0 + j * xm) + 1.0 / (1.0 + j * x) + 1.0 / (1.0 + j * xp)) / 3.0


def single_debye(
    f_ghz: np.ndarray,
    fres: float,
    deps: float,
    epsv: float,
    gamma: float = 0.0,
    sige: float = 0.0,
) -> np.ndarray:
    f = _safe_freq(f_ghz)
    cond_term = -1j * sige / (f * A0)
    comp = _debye_component(f, fres, deps, gamma)
    return epsv + cond_term + comp


def double_debye(
    f_ghz: np.ndarray,
    fres1: float,
    deps1: float,
    gamma1: float,
    fres2: float,
    deps2: float,
    gamma2: float,
    epsv: float,
    sige: float,
) -> np.ndarray:
    f = _safe_freq(f_ghz)
    cond_term = -1j * sige / (f * A0)
    c1 = _debye_component(f, fres1, deps1, gamma1)
    c2 = _debye_component(f, fres2, deps2, gamma2)
    return epsv + cond_term + c1 + c2


def triple_debye(
    f_ghz: np.ndarray,
    fres1: float,
    deps1: float,
    fres2: float,
    deps2: float,
    fres3: float,
    deps3: float,
    epsv: float,
    sige: float,
    gamma1: float = 0.0,
    gamma2: float = 0.0,
    gamma3: float = 0.0,
) -> np.ndarray:
    f = _safe_freq(f_ghz)
    cond_term = -1j * sige / (f * A0)
    c1 = _debye_component(f, fres1, deps1, gamma1)
    c2 = _debye_component(f, fres2, deps2, gamma2)
    c3 = _debye_component(f, fres3, deps3, gamma3)
    return epsv + cond_term + c1 + c2 + c3


__all__ = [
    "single_debye",
    "double_debye",
    "triple_debye",
]
