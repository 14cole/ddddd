"""IDB evaluation helpers (Debye/Lorentz models).

Dispatcher to evaluate complex permittivity/permeability for IDBKODE digits
using Debye/Lorentz approximations:
 1 = single Debye
 2 = single Lorentz
 3 = double Debye
 4 = double Lorentz
 5 = triple Debye
 6 = Debye + Lorentz (approximate: single Debye + single Lorentz)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import debye
import lorentz

def _pad(params: list[float], length: int, fill: float = 0.0) -> list[float]:
    if len(params) >= length:
        return params[:length]
    return params + [fill] * (length - len(params))


def eval_dispersion(idb_digit: int, params: list[float], f_ghz: np.ndarray) -> Optional[np.ndarray]:
    """Return complex dispersion for a given IDBKODE digit and parameter list."""
    try:
        if idb_digit == 1:  # single Debye: FRES, DEPS, EPSV, GAMMA, SIGE
            p = _pad(params, 5, 0.0)
            return debye.single_debye(f_ghz, p[0], p[1], p[2], p[3], p[4])
        if idb_digit == 2:  # single Lorentz: FRES, DEPS, EPSV, GAMMA, SIGE
            p = _pad(params, 5, 0.0)
            return lorentz.single_lorentz(f_ghz, p[0], p[1], p[2], p[3], p[4])
        if idb_digit == 3:  # double Debye: F1,D1,G1,F2,D2,G2,EPSV,SIGE
            p = _pad(params, 8, 0.0)
            return debye.double_debye(
                f_ghz,
                p[0],
                p[1],
                p[2],
                p[3],
                p[4],
                p[5],
                p[6],
                p[7],
            )
        if idb_digit == 4:  # double Lorentz: F1,D1,G1,F2,D2,G2,EPSV,SIGE
            p = _pad(params, 8, 0.0)
            return lorentz.double_lorentz(
                f_ghz,
                p[0],
                p[1],
                p[2],
                p[3],
                p[4],
                p[5],
                p[6],
                p[7],
            )
        if idb_digit == 5:  # triple Debye: F1,D1,F2,D2,F3,D3,EPSV,SIGE
            p = _pad(params, 8, 0.0)
            return debye.triple_debye(
                f_ghz,
                p[0],
                p[1],
                p[2],
                p[3],
                p[4],
                p[5],
                p[6],
                p[7],
            )
        if idb_digit == 6:  # Debye + Lorentz: Fd, Dd, Gd, Fl, Dl, Gl, EPSV, SIGE
            p = _pad(params, 8, 0.0)
            f = np.maximum(f_ghz, 1e-6)
            cond = -1j * p[7] / (f * debye.A0)
            deb = debye.single_debye(f, p[0], p[1], 0.0, p[2], 0.0)  # epsv=0, sige=0
            lor = lorentz.single_lorentz(f, p[3], p[4], 0.0, p[5], 0.0)
            return p[6] + cond + deb + lor
        return None
    except Exception:
        return None


__all__ = ["eval_dispersion"]
