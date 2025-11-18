"""Lightweight Debye/Lorentz fitters (work-in-progress).

Implements a simple coordinate-descent optimizer to approximate the legacy
Fortran CE2DB1/2/3 style least-square fits. Supports:
  - Single Debye (IDBKODE digit 1)
  - Double Debye (digit 3)
  - Triple Debye (digit 5)

Controls follow the same conventions:
  control = 0   -> estimate freely (initial guess from data)
  control > 0   -> fixed at that value (1e-8 is treated as zero)
  control < 0   -> use abs(control) as initial guess, free to move

NOTE: This is an initial port and may diverge from the precise Fortran search
heuristics. It is intended as a starting point for parity refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np

from . import debye
from . import lorentz


@dataclass
class FitResult:
    params: List[float]
    rms: float
    iterations: int
    converged: bool


@dataclass
class FitTuning:
    pf_init: float = 1e-2
    max_iter: int = 5000
    kx_div: int = 50  # stride divisor -> max(len(f)//kx_div,1)
    fres_min: float = 1e-4
    fres_max: float = 999.9999
    deps_min: float = 1e-4
    deps_max: float = 999.9999
    epsv_min: float = 0.0
    epsv_max: float = 999.9999
    gamma_min: float = 0.0
    gamma_max: float = 10.0
    sige_min: float = 0.0
    sige_max: float = 100.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "FitTuning":
        if not d:
            return cls()
        kwargs = {}
        int_fields = {"max_iter", "kx_div"}
        for field in (
            "pf_init",
            "max_iter",
            "kx_div",
            "fres_min",
            "fres_max",
            "deps_min",
            "deps_max",
            "epsv_min",
            "epsv_max",
            "gamma_min",
            "gamma_max",
            "sige_min",
            "sige_max",
        ):
            key = field.upper()
            if key in d:
                try:
                    if field in int_fields:
                        kwargs[field] = int(float(d[key]))
                    else:
                        kwargs[field] = float(d[key])
                except Exception:
                    continue
        return cls(**kwargs)


def _model_single(p: Sequence[float], f_ghz: np.ndarray) -> np.ndarray:
    return debye.single_debye(f_ghz, p[0], p[1], p[2], p[3], p[4])


def _model_double(p: Sequence[float], f_ghz: np.ndarray) -> np.ndarray:
    return debye.double_debye(f_ghz, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])


def _model_triple(p: Sequence[float], f_ghz: np.ndarray) -> np.ndarray:
    return debye.triple_debye(f_ghz, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])

def _model_lorentz_single(p: Sequence[float], f_ghz: np.ndarray) -> np.ndarray:
    return lorentz.single_lorentz(f_ghz, p[0], p[1], p[2], p[3], p[4])


def _model_lorentz_double(p: Sequence[float], f_ghz: np.ndarray) -> np.ndarray:
    return lorentz.double_lorentz(f_ghz, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])


def _model_debye_lorentz(p: Sequence[float], f_ghz: np.ndarray) -> np.ndarray:
    # Debye+Lorentz: Fd, Dd, Gd, Fl, Dl, Gl, EPSV, SIGE
    debye_part = debye.single_debye(f_ghz, p[0], p[1], 0.0, p[2], 0.0)
    lorentz_part = lorentz.single_lorentz(f_ghz, p[3], p[4], 0.0, p[5], 0.0)
    return p[6] + (debye_part - 0.0) + (lorentz_part - 0.0) - 1j * p[7] / (np.maximum(f_ghz, 1e-6) * debye.A0)

def _rmse(model: np.ndarray, meas: np.ndarray) -> float:
    diff = model - meas
    return float(np.sqrt(np.mean(diff.real**2 + diff.imag**2)))


def _initial_guess_single(f_ghz: np.ndarray, eps: np.ndarray) -> List[float]:
    fres = float(f_ghz[np.argmax(np.abs(eps.imag))]) if len(f_ghz) else 1.0
    epsr = eps.real
    deps = float(max(epsr) - min(epsr)) if len(epsr) else 1.0
    epsv = float(epsr[-1]) if len(epsr) else 1.0
    gamma = 0.3
    sige = 1e-4
    return [fres, deps, epsv, gamma, sige]


def _prepare_params(controls: Sequence[float], initial: List[float]) -> Tuple[List[float], List[bool]]:
    params = initial[:]
    fixed = [False] * len(initial)
    for i, c in enumerate(controls[: len(initial)]):
        if c > 0:
            params[i] = 0.0 if abs(c - 1e-8) < 1e-12 else float(c)
            fixed[i] = True
        elif c < 0:
            params[i] = abs(float(c))
        # c == 0 -> keep heuristic initial
    return params, fixed


def _coord_descent(
    params: List[float],
    fixed: List[bool],
    eval_model: Callable[[Sequence[float]], float],
    max_iter: int = 250,
    bounds: List[Tuple[float, float]] | None = None,
) -> Tuple[List[float], float, int, bool]:
    step = [max(abs(v) * 0.1, 1e-6) for v in params]
    best = eval_model(params)
    converged = False
    for it in range(max_iter):
        improved = False
        for i in range(len(params)):
            if fixed[i]:
                continue
            for delta in (step[i], -step[i]):
                candidate = params[:]
                candidate[i] += delta
                if bounds and i < len(bounds):
                    lo, hi = bounds[i]
                    candidate[i] = max(min(candidate[i], hi), lo)
                val = eval_model(candidate)
                if val < best:
                    best = val
                    params = candidate
                    improved = True
                    break
        # reduce step sizes if no improvement
        if not improved:
            step = [s * 0.5 for s in step]
            if max(step) < 1e-6:
                converged = True
                break
    return params, best, it + 1, converged


def _initial_guess_bounds(controls: Sequence[float], defaults: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    bounds = []
    for ctrl, (lo, hi) in zip(controls, defaults):
        if ctrl > 0:
            bounds.append((ctrl, ctrl))
        else:
            bounds.append((lo, hi))
    return bounds


def _ce2_style_loop(
    params: List[float],
    fixed: List[bool],
    step_funcs: List[Callable[[float, float], float]],
    clamp_funcs: List[Callable[[float], float]],
    objective_sum: Callable[[Sequence[float]], float],
    pf_init: float = 1e-2,
    iter_max: int = 5000,
) -> tuple[List[float], float, int, bool]:
    """Run CE2*-style coordinate scans with PF shrink/grow and SMIN/SREF logic."""
    pf = pf_init
    sf = 1.0
    best_sum = objective_sum(params)
    iter_count = 0
    converged = False

    while iter_count < iter_max and sf < 1024.1:
        for _ in range(10):
            sref = best_sum
            iter_count += 1
            if iter_count > iter_max:
                break

            for idx in range(len(params)):
                if fixed[idx]:
                    continue
                step_val = step_funcs[idx](params[idx], pf)
                base_params = params[:]
                base_sum = best_sum
                best_local = base_sum
                best_params = base_params
                for sign in (1.0, -1.0):
                    cand = base_params[:]
                    cand[idx] = clamp_funcs[idx](cand[idx] + sign * step_val)
                    s = objective_sum(cand)
                    if s < best_local:
                        best_local = s
                        best_params = cand
                params = best_params
                best_sum = best_local

            if best_sum >= sref:
                sf *= 2.0
                pf *= 0.5
                break
        else:
            sf *= 0.5
            pf *= 2.0
            continue

        if sf >= 1024.1 or iter_count >= iter_max:
            break

    if sf >= 1024.1:
        converged = True

    return params, best_sum, iter_count, converged


def fit_debye_single(
    f_ghz: Sequence[float],
    eps_real: Sequence[float],
    eps_imag: Sequence[float],
    controls: Sequence[float] | None = None,
    tuning: FitTuning | None = None,
) -> FitResult:
    f = np.asarray(f_ghz, dtype=float)
    eps = np.asarray(eps_real, dtype=float) + 1j * np.asarray(eps_imag, dtype=float)
    tuning = tuning or FitTuning()
    ctrl = list(controls or [])
    while len(ctrl) < 5:
        ctrl.append(0.0)

    epsr = eps.real if len(eps) else np.array([1.0])
    idx_i = int(np.argmax(np.abs(eps.imag))) if len(eps) else 0
    idx_j = int(np.argmax(epsr)) if len(epsr) else 0
    idx_k = int(np.argmin(epsr)) if len(epsr) else 0
    fres_guess = float(f[idx_i]) if len(f) else 1.0
    deps_guess = float(epsr[idx_j] - epsr[idx_k]) if len(epsr) else 1.0
    epsv_guess = float(epsr[-1]) if len(epsr) else 1.0
    gamma_guess = 0.3
    sige_guess = 1e-4

    def apply_ctrl(idx: int, guess: float) -> tuple[float, bool]:
        c = ctrl[idx]
        if abs(c - 1e-8) < 1e-12:
            return 0.0, True
        if c > 0:
            return float(c), True
        if c < 0:
            return abs(float(c)), False
        return guess, False

    fres, fix_f = apply_ctrl(0, fres_guess)
    deps, fix_d = apply_ctrl(1, deps_guess)
    epsv, fix_v = apply_ctrl(2, epsv_guess)
    gamma, fix_g = apply_ctrl(3, gamma_guess)
    sige, fix_s = apply_ctrl(4, sige_guess)

    params = [fres, deps, epsv, gamma, sige]
    fixed = [fix_f, fix_d, fix_v, fix_g, fix_s]

    kx = max(len(f) // max(tuning.kx_div, 1), 1)
    f_s = f[::kx] if len(f) else np.array([1.0])
    eps_s = eps[::kx] if len(eps) else np.array([1.0])
    sample_count = max(len(f_s), 1)

    def clamp_fres(v: float) -> float:
        return min(max(v, tuning.fres_min), tuning.fres_max)

    def clamp_deps(v: float) -> float:
        return min(max(v, tuning.deps_min), tuning.deps_max)

    def clamp_epsv(v: float) -> float:
        return min(max(v, tuning.epsv_min), tuning.epsv_max)

    def clamp_gamma(v: float) -> float:
        g = abs(v)
        if g < 1e-4:
            g = 0.0
        return min(max(g, tuning.gamma_min), tuning.gamma_max)

    def clamp_sige(v: float) -> float:
        return min(max(v, tuning.sige_min), tuning.sige_max)

    step_funcs = [
        lambda val, pf: max(pf, pf * val),
        lambda val, pf: max(pf, pf * val),
        lambda val, pf: max(pf, pf * max(abs(val), 1.0)),
        lambda val, pf: max(pf, pf * val),
        lambda val, pf: max(pf / 10.0, pf * val),
    ]
    clamp_funcs = [clamp_fres, clamp_deps, clamp_epsv, clamp_gamma, clamp_sige]

    def obj_sum(pvec: Sequence[float]) -> float:
        m = _model_single(pvec, f_s)
        diff = m - eps_s
        return float(np.sum(diff.real**2 + diff.imag**2))

    params, best_sum, iterations, converged = _ce2_style_loop(
        params, fixed, step_funcs, clamp_funcs, obj_sum, pf_init=tuning.pf_init, iter_max=tuning.max_iter
    )

    rms_full = _rmse(_model_single(params, f if len(f) else np.array([1.0])), eps if len(eps) else np.array([1.0]))
    rms_sample = np.sqrt(best_sum / sample_count) if np.isfinite(best_sum) else float("inf")
    return FitResult(params=params, rms=rms_full if np.isfinite(rms_full) else rms_sample, iterations=iterations, converged=converged)


def fit_debye_double(
    f_ghz: Sequence[float],
    eps_real: Sequence[float],
    eps_imag: Sequence[float],
    controls: Sequence[float] | None = None,
    tuning: FitTuning | None = None,
) -> FitResult:
    f = np.asarray(f_ghz, dtype=float)
    eps = np.asarray(eps_real, dtype=float) + 1j * np.asarray(eps_imag, dtype=float)
    tuning = tuning or FitTuning()
    ctrl = list(controls or [])
    while len(ctrl) < 8:
        ctrl.append(0.0)

    epsr = eps.real if len(eps) else np.array([1.0])
    idx_i = int(np.argmax(np.abs(eps.imag))) if len(eps) else 0
    idx_j = int(np.argmax(epsr)) if len(epsr) else 0
    idx_k = int(np.argmin(epsr)) if len(epsr) else 0
    fres_peak = float(f[idx_i]) if len(f) else 1.0
    deps_span = float(epsr[idx_j] - epsr[idx_k]) if len(epsr) else 1.0
    gamma_default = 0.3
    epsv_guess = float(epsr[-1]) if len(epsr) else 1.0
    sige_guess = 1e-8

    def apply_ctrl(idx: int, guess: float) -> tuple[float, bool]:
        c = ctrl[idx]
        if abs(c - 1e-8) < 1e-12:
            return 0.0, True
        if c > 0:
            return float(c), True
        if c < 0:
            return abs(float(c)), False
        return guess, False

    if abs(ctrl[4] - 1e-8) < 1e-12:
        fre1, fix_f1 = apply_ctrl(0, fres_peak or (np.max(f) / 3.0 if len(f) else 1.0))
        dep1, fix_d1 = apply_ctrl(1, deps_span)
        gam1, fix_g1 = apply_ctrl(2, gamma_default)
        fre2 = fres_peak or (2 * fre1 if fre1 else 1.0)
        dep2 = 0.0
        gam2 = gamma_default
        fix_f2 = fix_d2 = fix_g2 = True
    elif abs(ctrl[1] - 1e-8) < 1e-12:
        fre1 = fres_peak or (np.max(f) / 3.0 if len(f) else 1.0)
        dep1 = 0.0
        gam1 = gamma_default
        fix_f1 = fix_d1 = fix_g1 = True
        fre2, fix_f2 = apply_ctrl(3, fre1 * 2.0 if fre1 else 1.0)
        dep2, fix_d2 = apply_ctrl(4, deps_span)
        gam2, fix_g2 = apply_ctrl(5, gamma_default)
    else:
        fres_span = (float(f.max()) - float(f.min())) / 3.0 if len(f) else 1.0
        fre1, fix_f1 = apply_ctrl(0, fres_span if fres_span > 0 else 1.0)
        dep1, fix_d1 = apply_ctrl(1, deps_span / 2.0)
        gam1, fix_g1 = apply_ctrl(2, gamma_default)
        fre2, fix_f2 = apply_ctrl(3, 2 * (fres_span if fres_span > 0 else 1.0))
        dep2, fix_d2 = apply_ctrl(4, deps_span / 2.0)
        gam2, fix_g2 = apply_ctrl(5, gamma_default)

    epsv, fix_v = apply_ctrl(6, epsv_guess)
    sige, fix_s = apply_ctrl(7, sige_guess)

    params = [fre1, dep1, gam1, fre2, dep2, gam2, epsv, sige]
    fixed = [fix_f1, fix_d1, fix_g1, fix_f2, fix_d2, fix_g2, fix_v, fix_s]

    kx = max(len(f) // max(tuning.kx_div, 1), 1)
    f_s = f[::kx] if len(f) else np.array([1.0])
    eps_s = eps[::kx] if len(eps) else np.array([1.0])
    sample_count = max(len(f_s), 1)

    def clamp_fres(v: float) -> float:
        return min(max(v, tuning.fres_min), tuning.fres_max)

    def clamp_deps(v: float) -> float:
        return min(max(v, tuning.deps_min), tuning.deps_max)

    def clamp_gamma(v: float) -> float:
        g = abs(v)
        if g < 1e-4:
            g = 0.0
        return min(max(g, tuning.gamma_min), tuning.gamma_max)

    def clamp_epsv(v: float) -> float:
        return min(max(v, tuning.epsv_min), tuning.epsv_max)

    def clamp_sige(v: float) -> float:
        return min(max(v, tuning.sige_min), tuning.sige_max)

    step_funcs = [
        lambda val, pf: pf * val,
        lambda val, pf: max(pf, pf * val),
        lambda val, pf: max(pf, pf * val),
        lambda val, pf: pf * val,
        lambda val, pf: max(pf, pf * val),
        lambda val, pf: max(pf, pf * val),
        lambda val, pf: max(pf, pf * max(abs(val), 1.0)),
        lambda val, pf: max(pf / 10.0, pf * val),
    ]
    clamp_funcs = [
        clamp_fres,
        clamp_deps,
        clamp_gamma,
        clamp_fres,
        clamp_deps,
        clamp_gamma,
        clamp_epsv,
        clamp_sige,
    ]

    def obj_sum(pvec: Sequence[float]) -> float:
        m = _model_double(pvec, f_s)
        diff = m - eps_s
        return float(np.sum(diff.real**2 + diff.imag**2))

    params, best_sum, iterations, converged = _ce2_style_loop(
        params, fixed, step_funcs, clamp_funcs, obj_sum, pf_init=tuning.pf_init, iter_max=tuning.max_iter
    )

    rms_full = _rmse(_model_double(params, f if len(f) else np.array([1.0])), eps if len(eps) else np.array([1.0]))
    rms_sample = np.sqrt(best_sum / sample_count) if np.isfinite(best_sum) else float("inf")
    return FitResult(params=params, rms=rms_full if np.isfinite(rms_full) else rms_sample, iterations=iterations, converged=converged)


def fit_debye_triple(
    f_ghz: Sequence[float],
    eps_real: Sequence[float],
    eps_imag: Sequence[float],
    controls: Sequence[float] | None = None,
    tuning: FitTuning | None = None,
) -> FitResult:
    f = np.asarray(f_ghz, dtype=float)
    eps = np.asarray(eps_real, dtype=float) + 1j * np.asarray(eps_imag, dtype=float)
    tuning = tuning or FitTuning()
    ctrl = list(controls or [])
    while len(ctrl) < 8:
        ctrl.append(0.0)

    epsr = eps.real if len(eps) else np.array([1.0])
    idx_i = int(np.argmax(np.abs(eps.imag))) if len(eps) else 0
    idx_j = int(np.argmax(epsr)) if len(epsr) else 0
    idx_k = int(np.argmin(epsr)) if len(epsr) else 0
    fres_peak = float(f[idx_i]) if len(f) else 1.0
    deps_span = float(epsr[idx_j] - epsr[idx_k]) if len(epsr) else 1.0
    epsv_guess = float(epsr[-1]) if len(epsr) else 1.0
    sige_guess = 1e-8

    # Initialize defaults so seeds/fix flags always exist even with unusual controls.
    fres_span = (float(f.max()) - float(f.min())) / 4.0 if len(f) else 1.0
    if fres_span <= 0:
        fres_span = 1.0
    fres1 = fres_peak or fres_span
    deps1 = deps_span / 3.0
    fre2 = 2 * fres_span
    deps2 = deps_span / 3.0
    fre3 = 3 * fres_span
    deps3 = deps_span / 3.0
    fix_f1 = fix_d1 = fix_f2 = fix_d2 = fix_f3 = fix_d3 = False

    def apply_ctrl(idx: int, guess: float) -> tuple[float, bool]:
        c = ctrl[idx]
        if abs(c - 1e-8) < 1e-12:
            return 0.0, True
        if c > 0:
            return float(c), True
        if c < 0:
            return abs(float(c)), False
        return guess, False

    if abs(ctrl[4] - 1e-8) < 1e-12:
        fres1, fix_f1 = apply_ctrl(0, fres_peak or fres_span)
        deps1, fix_d1 = apply_ctrl(1, deps_span)
        fre2 = fres_peak or (2 * fres_span)
        dep2 = 0.0
        fre3 = 3 * fres_span
        dep3 = 0.0
        fix_f2 = fix_d2 = fix_f3 = fix_d3 = True
    elif abs(ctrl[1] - 1e-8) < 1e-12:
        fre1 = fres_peak or fres_span
        deps1 = 0.0
        fix_f1 = fix_d1 = True
        fre2, fix_f2 = apply_ctrl(2, 2 * fres_span)
        dep2, fix_d2 = apply_ctrl(3, deps_span / 2.0)
        fre3, fix_f3 = apply_ctrl(4, 3 * fres_span)
        dep3, fix_d3 = apply_ctrl(5, deps_span / 2.0)
    else:
        fre1, fix_f1 = apply_ctrl(0, fres_span)
        deps1, fix_d1 = apply_ctrl(1, deps_span / 3.0)
        fre2, fix_f2 = apply_ctrl(2, 2 * fres_span)
        deps2, fix_d2 = apply_ctrl(3, deps_span / 3.0)
        fre3, fix_f3 = apply_ctrl(4, 3 * fres_span)
        deps3, fix_d3 = apply_ctrl(5, deps_span / 3.0)

    epsv, fix_v = apply_ctrl(6, epsv_guess)
    sige, fix_s = apply_ctrl(7, sige_guess)

    def clamp_fres(v: float) -> float:
        return min(max(v, tuning.fres_min), tuning.fres_max)

    def clamp_deps(v: float) -> float:
        return min(max(v, tuning.deps_min), tuning.deps_max)

    def clamp_epsv(v: float) -> float:
        return min(max(v, tuning.epsv_min), tuning.epsv_max)

    def clamp_sige(v: float) -> float:
        return min(max(v, tuning.sige_min), tuning.sige_max)

    # Clamp initial seeds to respect tuning bounds before iteration/freeze takes effect.
    fres1, fre2, fre3 = clamp_fres(fres1), clamp_fres(fre2), clamp_fres(fre3)
    deps1, deps2, deps3 = clamp_deps(deps1), clamp_deps(deps2), clamp_deps(deps3)
    epsv = clamp_epsv(epsv)
    sige = clamp_sige(sige)

    params = [fres1, deps1, fre2, deps2, fre3, deps3, epsv, sige]
    fixed = [fix_f1, fix_d1, fix_f2, fix_d2, fix_f3, fix_d3, fix_v, fix_s]

    kx = max(len(f) // max(tuning.kx_div, 1), 1)
    f_s = f[::kx] if len(f) else np.array([1.0])
    eps_s = eps[::kx] if len(eps) else np.array([1.0])
    sample_count = max(len(f_s), 1)

    step_funcs = [
        lambda val, pf: pf * val,
        lambda val, pf: max(pf, pf * val),
        lambda val, pf: pf * val,
        lambda val, pf: max(pf, pf * val),
        lambda val, pf: pf * val,
        lambda val, pf: max(pf, pf * val),
        lambda val, pf: max(pf, pf * max(abs(val), 1.0)),
        lambda val, pf: max(pf / 10.0, pf * val),
    ]
    clamp_funcs = [
        clamp_fres,
        clamp_deps,
        clamp_fres,
        clamp_deps,
        clamp_fres,
        clamp_deps,
        clamp_epsv,
        clamp_sige,
    ]

    def obj_sum(pvec: Sequence[float]) -> float:
        m = debye.triple_debye(f_s, pvec[0], pvec[1], pvec[2], pvec[3], pvec[4], pvec[5], pvec[6], pvec[7])
        diff = m - eps_s
        return float(np.sum(diff.real**2 + diff.imag**2))

    params, best_sum, iterations, converged = _ce2_style_loop(
        params, fixed, step_funcs, clamp_funcs, obj_sum, pf_init=tuning.pf_init, iter_max=tuning.max_iter
    )

    rms_full = _rmse(
        debye.triple_debye(
            f if len(f) else np.array([1.0]),
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            params[5],
            params[6],
            params[7],
        ),
        eps if len(eps) else np.array([1.0]),
    )
    rms_sample = np.sqrt(best_sum / sample_count) if np.isfinite(best_sum) else float("inf")
    return FitResult(params=params, rms=rms_full if np.isfinite(rms_full) else rms_sample, iterations=iterations, converged=converged)


def fit_lorentz_single(
    f_ghz: Sequence[float],
    eps_real: Sequence[float],
    eps_imag: Sequence[float],
    controls: Sequence[float] | None = None,
    tuning: FitTuning | None = None,
) -> FitResult:
    f = np.asarray(f_ghz, dtype=float)
    eps = np.asarray(eps_real, dtype=float) + 1j * np.asarray(eps_imag, dtype=float)
    ctrl = list(controls or [])
    tuning = tuning or FitTuning()
    while len(ctrl) < 5:
        ctrl.append(0.0)

    epsr = eps.real if len(eps) else np.array([1.0])
    idx_i = int(np.argmax(np.abs(eps.imag))) if len(eps) else 0
    idx_j = int(np.argmax(epsr)) if len(epsr) else 0
    idx_k = int(np.argmin(epsr)) if len(epsr) else 0
    fres_guess = float(f[idx_i]) if len(f) else 1.0
    deps_guess = float(epsr[idx_j] - epsr[idx_k]) if len(epsr) else 1.0
    epsv_guess = float(epsr[-1]) if len(epsr) else 1.0
    gamma_guess = 0.3
    sige_guess = 1e-4

    def apply_ctrl(idx: int, guess: float) -> tuple[float, bool]:
        c = ctrl[idx]
        if abs(c - 1e-8) < 1e-12:
            return 0.0, True
        if c > 0:
            return float(c), True
        if c < 0:
            return abs(float(c)), False
        return guess, False

    fres, fix_f = apply_ctrl(0, fres_guess)
    deps, fix_d = apply_ctrl(1, deps_guess)
    epsv, fix_v = apply_ctrl(2, epsv_guess)
    gamma, fix_g = apply_ctrl(3, gamma_guess)
    sige, fix_s = apply_ctrl(4, sige_guess)

    params = [fres, deps, epsv, gamma, sige]
    fixed = [fix_f, fix_d, fix_v, fix_g, fix_s]

    kx = max(len(f) // max(tuning.kx_div, 1), 1)
    f_s = f[::kx] if len(f) else np.array([1.0])
    eps_s = eps[::kx] if len(eps) else np.array([1.0])
    sample_count = max(len(f_s), 1)

    def clamp_fres(v: float) -> float:
        return min(max(v, tuning.fres_min), tuning.fres_max)

    def clamp_deps(v: float) -> float:
        return min(max(v, tuning.deps_min), tuning.deps_max)

    def clamp_epsv(v: float) -> float:
        return min(max(v, tuning.epsv_min), tuning.epsv_max)

    def clamp_gamma(v: float) -> float:
        g = abs(v)
        if g < 1e-4:
            g = 0.0
        return min(max(g, tuning.gamma_min), tuning.gamma_max)

    def clamp_sige(v: float) -> float:
        return min(max(v, tuning.sige_min), tuning.sige_max)

    step_funcs = [
        lambda val, pf: pf * val,
        lambda val, pf: max(pf, pf * val),
        lambda val, pf: max(pf, pf * max(abs(val), 1.0)),
        lambda val, pf: max(pf, pf * val),
        lambda val, pf: max(pf / 10.0, pf * val),
    ]
    clamp_funcs = [clamp_fres, clamp_deps, clamp_epsv, clamp_gamma, clamp_sige]

    def obj_sum(pvec: Sequence[float]) -> float:
        m = _model_lorentz_single(pvec, f_s)
        diff = m - eps_s
        return float(np.sum(diff.real**2 + diff.imag**2))

    params, best_sum, iterations, converged = _ce2_style_loop(
        params, fixed, step_funcs, clamp_funcs, obj_sum, pf_init=tuning.pf_init, iter_max=tuning.max_iter
    )

    rms_full = _rmse(_model_lorentz_single(params, f if len(f) else np.array([1.0])), eps if len(eps) else np.array([1.0]))
    rms_sample = np.sqrt(best_sum / sample_count) if np.isfinite(best_sum) else float("inf")
    return FitResult(params=params, rms=rms_full if np.isfinite(rms_full) else rms_sample, iterations=iterations, converged=converged)


def fit_lorentz_double(
    f_ghz: Sequence[float],
    eps_real: Sequence[float],
    eps_imag: Sequence[float],
    controls: Sequence[float] | None = None,
    tuning: FitTuning | None = None,
) -> FitResult:
    f = np.asarray(f_ghz, dtype=float)
    eps = np.asarray(eps_real, dtype=float) + 1j * np.asarray(eps_imag, dtype=float)
    ctrl = list(controls or [])
    tuning = tuning or FitTuning()
    while len(ctrl) < 8:
        ctrl.append(0.0)

    epsr = eps.real if len(eps) else np.array([1.0])
    idx_i = int(np.argmax(np.abs(eps.imag))) if len(eps) else 0
    idx_j = int(np.argmax(epsr)) if len(epsr) else 0
    idx_k = int(np.argmin(epsr)) if len(epsr) else 0
    fres_peak = float(f[idx_i]) if len(f) else 1.0
    deps_span = float(epsr[idx_j] - epsr[idx_k]) if len(epsr) else 1.0
    gamma_default = 0.3
    epsv_guess = float(epsr[-1]) if len(epsr) else 1.0
    sige_guess = 1e-6

    def apply_ctrl(idx: int, guess: float) -> tuple[float, bool]:
        c = ctrl[idx]
        if abs(c - 1e-8) < 1e-12:
            return 0.0, True
        if c > 0:
            return float(c), True
        if c < 0:
            return abs(float(c)), False
        return guess, False

    dv = (float(f.max()) - float(f.min())) / 3.0 if len(f) else 1.0
    if dv <= 0:
        dv = 1.0

    if abs(ctrl[4] - 1e-8) < 1e-12:
        fre1, fix_f1 = apply_ctrl(0, fres_peak or dv)
        dep1, fix_d1 = apply_ctrl(1, deps_span)
        gam1, fix_g1 = apply_ctrl(2, gamma_default)
        fre2 = fres_peak or (2 * dv)
        dep2 = 0.0
        gam2 = gamma_default
        fix_f2 = fix_d2 = fix_g2 = True
    elif abs(ctrl[1] - 1e-8) < 1e-12:
        fre1 = fres_peak or dv
        dep1 = 0.0
        gam1 = gamma_default
        fix_f1 = fix_d1 = fix_g1 = True
        fre2, fix_f2 = apply_ctrl(3, 2 * dv)
        dep2, fix_d2 = apply_ctrl(4, deps_span)
        gam2, fix_g2 = apply_ctrl(5, gamma_default)
    else:
        fre1, fix_f1 = apply_ctrl(0, dv)
        dep1, fix_d1 = apply_ctrl(1, deps_span / 2.0)
        gam1, fix_g1 = apply_ctrl(2, gamma_default)
        fre2, fix_f2 = apply_ctrl(3, 2 * dv)
        dep2, fix_d2 = apply_ctrl(4, deps_span / 2.0)
        gam2, fix_g2 = apply_ctrl(5, gamma_default)

    epsv, fix_v = apply_ctrl(6, epsv_guess)
    sige, fix_s = apply_ctrl(7, sige_guess)

    params = [fre1, dep1, gam1, fre2, dep2, gam2, epsv, sige]
    fixed = [fix_f1, fix_d1, fix_g1, fix_f2, fix_d2, fix_g2, fix_v, fix_s]

    kx = max(len(f) // max(tuning.kx_div, 1), 1)
    f_s = f[::kx] if len(f) else np.array([1.0])
    eps_s = eps[::kx] if len(eps) else np.array([1.0])
    sample_count = max(len(f_s), 1)

    def clamp_fres(v: float) -> float:
        return min(max(v, tuning.fres_min), tuning.fres_max)

    def clamp_deps(v: float) -> float:
        return min(max(v, tuning.deps_min), tuning.deps_max)

    def clamp_gamma(v: float) -> float:
        g = abs(v)
        if g < 0.01:
            g = 0.01
        return min(max(g, tuning.gamma_min), tuning.gamma_max)

    def clamp_epsv(v: float) -> float:
        return min(max(v, tuning.epsv_min), tuning.epsv_max)

    def clamp_sige(v: float) -> float:
        return min(max(v, tuning.sige_min), tuning.sige_max)

    step_funcs = [
        lambda val, pf: pf * val,
        lambda val, pf: max(pf, pf * val),
        lambda val, pf: max(pf, pf * val),
        lambda val, pf: pf * val,
        lambda val, pf: max(pf, pf * val),
        lambda val, pf: max(pf, pf * val),
        lambda val, pf: max(pf, pf * max(abs(val), 1.0)),
        lambda val, pf: max(pf / 10.0, pf * val),
    ]
    clamp_funcs = [
        clamp_fres,
        clamp_deps,
        clamp_gamma,
        clamp_fres,
        clamp_deps,
        clamp_gamma,
        clamp_epsv,
        clamp_sige,
    ]

    def obj_sum(pvec: Sequence[float]) -> float:
        m = _model_lorentz_double(pvec, f_s)
        diff = m - eps_s
        return float(np.sum(diff.real**2 + diff.imag**2))

    params, best_sum, iterations, converged = _ce2_style_loop(
        params, fixed, step_funcs, clamp_funcs, obj_sum, pf_init=tuning.pf_init, iter_max=tuning.max_iter
    )

    rms_full = _rmse(_model_lorentz_double(params, f if len(f) else np.array([1.0])), eps if len(eps) else np.array([1.0]))
    rms_sample = np.sqrt(best_sum / sample_count) if np.isfinite(best_sum) else float("inf")
    return FitResult(params=params, rms=rms_full if np.isfinite(rms_full) else rms_sample, iterations=iterations, converged=converged)


def fit_debye_lorentz(
    f_ghz: Sequence[float],
    eps_real: Sequence[float],
    eps_imag: Sequence[float],
    controls: Sequence[float] | None = None,
    tuning: FitTuning | None = None,
) -> FitResult:
    f = np.asarray(f_ghz, dtype=float)
    eps = np.asarray(eps_real, dtype=float) + 1j * np.asarray(eps_imag, dtype=float)
    tuning = tuning or FitTuning()

    ctrl = list(controls) if controls else []
    while len(ctrl) < 8:
        ctrl.append(0.0)

    epsr = eps.real if len(eps) else np.array([1.0])
    idx_i = int(np.argmax(np.abs(eps.imag))) if len(eps) else 0
    idx_j = int(np.argmax(epsr)) if len(epsr) else 0
    idx_k = int(np.argmin(epsr)) if len(epsr) else 0
    fres_peak = float(f[idx_i]) if len(f) else 1.0
    eps_span_full = float(epsr[idx_j] - epsr[idx_k]) if len(epsr) else 1.0
    fres_span = (float(f.max()) - float(f.min())) / 3.0 if len(f) else 1.0
    fres_default = fres_span if fres_span > 0 else 1.0
    gamma_default = 0.3
    epsv_default = float(epsr[-1]) if len(epsr) else 1.0
    sige_default = 1e-8

    # Start with sane defaults so variables are always initialized even if controls are odd.
    fres1 = fres_peak or fres_default
    deps1 = eps_span_full / 2.0
    gamm1 = gamma_default
    fres2 = fres_peak or (2 * fres_default)
    deps2 = eps_span_full / 2.0
    gamm2 = gamma_default
    fixf1 = fixd1 = fixg1 = False
    fixf2 = fixd2 = fixg2 = False

    def _apply_ctrl(idx: int, guess: float) -> tuple[float, bool]:
        c = ctrl[idx]
        if abs(c - 1e-8) < 1e-12:
            return 0.0, True
        if c > 0:
            return float(c), True
        if c < 0:
            return abs(float(c)), False
        return guess, False

    freeze_debye = abs(ctrl[1] - 1e-8) < 1e-12
    freeze_lorentz = abs(ctrl[4] - 1e-8) < 1e-12

    if freeze_lorentz:
        fres1, fixf1 = _apply_ctrl(0, fres1)
        deps1, fixd1 = _apply_ctrl(1, eps_span_full)
        gamm1, fixg1 = _apply_ctrl(2, gamm1)
        # Lorentz leg forced to a negligible oscillator; Fortran also freezes its controls.
        fres2 = fres_peak or fres_default
        deps2 = 0.0
        gamm2 = gamma_default
        fixf2 = fixd2 = fixg2 = True
    elif freeze_debye:
        fres1 = fres_peak or fres_default
        deps1 = 0.0
        gamm1 = gamma_default
        fixf1 = fixd1 = fixg1 = True
        fres2, fixf2 = _apply_ctrl(3, fres2)
        deps2, fixd2 = _apply_ctrl(4, eps_span_full)
        gamm2, fixg2 = _apply_ctrl(5, gamm2)
    else:
        fres1, fixf1 = _apply_ctrl(0, fres1)
        deps1, fixd1 = _apply_ctrl(1, eps_span_full / 2.0)
        gamm1, fixg1 = _apply_ctrl(2, gamm1)
        fres2, fixf2 = _apply_ctrl(3, fres2)
        deps2, fixd2 = _apply_ctrl(4, eps_span_full / 2.0)
        gamm2, fixg2 = _apply_ctrl(5, gamm2)

    epsv, fixv = _apply_ctrl(6, epsv_default)
    sige, fixs = _apply_ctrl(7, sige_default)

    def _clamp_fres(val: float) -> float:
        return min(max(val, tuning.fres_min), tuning.fres_max)

    def _clamp_deps(val: float) -> float:
        return min(max(val, tuning.deps_min), tuning.deps_max)

    def _clamp_gamma(val: float) -> float:
        g = abs(val)
        if g < 1e-4:
            g = 0.0
        return min(max(g, tuning.gamma_min), tuning.gamma_max)

    def _clamp_epsv(val: float) -> float:
        return min(max(val, tuning.epsv_min), tuning.epsv_max)

    def _clamp_sige(val: float) -> float:
        return min(max(val, tuning.sige_min), tuning.sige_max)

    # Clamp seeds to tuning ranges before iteration.
    fres1, fres2 = _clamp_fres(fres1), _clamp_fres(fres2)
    deps1, deps2 = _clamp_deps(deps1), _clamp_deps(deps2)
    gamm1, gamm2 = _clamp_gamma(gamm1), _clamp_gamma(gamm2)
    epsv = _clamp_epsv(epsv)
    sige = _clamp_sige(sige)

    params = [fres1, deps1, gamm1, fres2, deps2, gamm2, epsv, sige]
    fixed = [fixf1, fixd1, fixg1, fixf2, fixd2, fixg2, fixv, fixs]

    def _obj_sum(pvec: Sequence[float], fvals: np.ndarray, epsvals: np.ndarray) -> float:
        model = lorentz.debye_lorentz_combo(
            fvals, pvec[0], pvec[1], pvec[2], pvec[3], pvec[4], pvec[5], pvec[6], pvec[7]
        )
        diff = model - epsvals
        return float(np.sum(diff.real**2 + diff.imag**2))

    kx = max(len(f) // max(tuning.kx_div, 1), 1)
    f_s = f[::kx] if len(f) else np.array([1.0])
    eps_s = eps[::kx] if len(eps) else np.array([1.0 + 0j])
    sample_count = max(len(f_s), 1)

    def _scan(idx: int, step: float, clamp_func):
        nonlocal params, best_sum
        base_params = params[:]
        base_sum = _obj_sum(base_params, f_s, eps_s)
        best_local = base_sum
        best_params = base_params
        for sign in (1.0, -1.0):
            cand = base_params[:]
            cand[idx] = clamp_func(cand[idx] + sign * step)
            s = _obj_sum(cand, f_s, eps_s)
            if s < best_local:
                best_local = s
                best_params = cand
        params = best_params
        best_sum = best_local

    pf = tuning.pf_init
    sf = 1.0
    iter_max = tuning.max_iter
    iter_count = 0
    best_sum = _obj_sum(params, f_s, eps_s)
    converged = False

    while iter_count < iter_max and sf < 1024.1:
        for _ in range(10):
            sref = best_sum
            iter_count += 1
            if iter_count > iter_max:
                break

            if not fixed[0]:
                _scan(0, pf * params[0], _clamp_fres)
            if not fixed[1]:
                _scan(1, max(pf, pf * params[1]), _clamp_deps)
            if not fixed[2]:
                _scan(2, max(pf, pf * params[2]), _clamp_gamma)
            if not fixed[3]:
                _scan(3, pf * params[3], _clamp_fres)
            if not fixed[4]:
                _scan(4, max(pf, pf * params[4]), _clamp_deps)
            if not fixed[5]:
                _scan(5, max(pf, pf * params[5]), _clamp_gamma)
            if not fixed[6]:
                _scan(6, max(pf, pf * params[6]), _clamp_epsv)
            if not fixed[7]:
                _scan(7, max(pf / 10.0, pf * params[7]), _clamp_sige)

            if best_sum >= sref:
                sf *= 2.0
                pf *= 0.5
                break
            else:
                sref = best_sum
        else:
            sf *= 0.5
            pf *= 2.0
            continue
        if sf >= 1024.1 or iter_count >= iter_max:
            break

    if sf >= 1024.1:
        converged = True

    rms_sample = np.sqrt(best_sum / sample_count) if np.isfinite(best_sum) else float("inf")
    rms_full = _rmse(
        lorentz.debye_lorentz_combo(f if len(f) else np.array([1.0]), params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]),
        eps if len(eps) else np.array([1.0 + 0j]),
    )

    return FitResult(params=params, rms=rms_full if np.isfinite(rms_full) else rms_sample, iterations=iter_count, converged=converged)


__all__ = [
    "FitResult",
    "fit_debye_single",
    "fit_debye_double",
    "fit_debye_triple",
    # Lorentz / Debye-Lorentz
    "fit_lorentz_single",
    "fit_lorentz_double",
    "fit_debye_lorentz",
]
