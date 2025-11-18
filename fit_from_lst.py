"""Generate LSF-style fit files from an .LST/.ROP set.

Usage (library):
    paths = generate_fits_from_lst(Path("WaveX.LST"), out_dir=Path("out"))

Notes:
- Fits use lightweight Python ports in `idb_fit` (Debye/Lorentz).
- Output .lsf files are simple text summaries (not full legacy format).
- IDBKODE drives which model is used per CEPS (tens digit) and CMU (ones digit).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from . import file_loader
from . import idb_fit
from . import idb_poly
from . import debye
from . import lorentz


CEPS_EXT = {1: ".dbe", 2: ".lne", 3: ".2de", 4: ".2le", 5: ".3de", 6: ".dle"}
CMU_EXT = {1: ".dbm", 2: ".lnm", 3: ".2dm", 4: ".2lm", 5: ".3dm", 6: ".dlm"}


def _param_names_ce(digit: int) -> List[str]:
    if digit <= 2:
        return ["FRES", "DEPS", "EPSV", "GAMMA", "SIGE"]
    if digit == 5:
        return ["FRES1", "DEPS1", "FRES2", "DEPS2", "FRES3", "DEPS3", "EPSV", "SIGE"]
    if digit == 6:
        return ["FRES_D", "DEPS_D", "GAMM_D", "FRES_L", "DEPS_L", "GAMM_L", "EPSV", "SIGE"]
    return ["FRES1", "DEPS1", "GAMM1", "FRES2", "DEPS2", "GAMM2", "EPSV", "SIGE"]


def _param_names_cmu(digit: int) -> List[str]:
    if digit <= 2:
        return ["FR_M", "DMUR", "MURV", "GAMMU", "SIGM"]
    if digit == 5:
        return ["FR_M1", "DMUR1", "FR_M2", "DMUR2", "FR_M3", "DMUR3", "MURV", "SIGM"]
    if digit == 6:
        return ["FR_MD", "DMUR_D", "GAMM_D", "FR_ML", "DMUR_L", "GAMM_L", "MURV", "SIGM"]
    return ["FR_M1", "DMUR1", "GAMM1", "FR_M2", "DMUR2", "GAMM2", "MURV", "SIGM"]


def _fmt(num: float, width: int = 9, prec: int = 4) -> str:
    return f"{num:{width}.{prec}g}"


def _write_combined_lsf(
    path: Path,
    lst_path: Path,
    idbkode: int,
    idbke: int,
    idbkm: int,
    ce_rows: List[Tuple[float, List[float], float, str]],
    cm_rows: List[Tuple[float, List[float], float, str]],
) -> None:
    lines: List[str] = []
    lines.append(f"! AUTO-GENERATED {path.name}")
    lines.append("")
    lines.append(f"LSF CEPS DATA FOR LST FILE {lst_path}")
    lines.append("")
    lines.append("!     NALPHA     IDBKODE        IDBKE        EXTE       NPARS")
    lines.append(
        f"{len(ce_rows):12d}{idbkode:12d}{idbke:12d}{CEPS_EXT.get(idbke,''):>12s}{len(_param_names_ce(idbke)):12d}"
    )
    lines.append("")
    header_ce = ["ALPHA"] + _param_names_ce(idbke) + ["RMSE", "LSF FILE"]
    lines.append("! " + " ".join(f"{h:>10s}" for h in header_ce))
    for alpha, params, rms, fname in ce_rows:
        vals = [alpha] + params + [rms]
        lines.append(" ".join(f"{v:10.4g}" for v in vals) + f"    {fname}")
    lines.append("")
    lines.append(f"LSF CMU DATA FOR LST FILE {lst_path}")
    lines.append("")
    lines.append("!     NALPHA     IDBKODE        IDBKM        EXTM       NPARS")
    lines.append(
        f"{len(cm_rows):12d}{idbkode:12d}{idbkm:12d}{CMU_EXT.get(idbkm,''):>12s}{len(_param_names_cmu(idbkm)):12d}"
    )
    lines.append("")
    header_cm = ["ALPHA"] + _param_names_cmu(idbkm) + ["RMSM", "LSF FILE"]
    lines.append("! " + " ".join(f"{h:>10s}" for h in header_cm))
    for alpha, params, rms, fname in cm_rows:
        vals = [alpha] + params + [rms]
        lines.append(" ".join(f"{v:10.4g}" for v in vals) + f"    {fname}")
    lines.append("")

    # Parameter tables vs alpha
    if ce_rows:
        lines.append("DLPARE VS ALPHA:")
        lines.append(" " * 5 + "ALPHA " + " ".join(f"{i:9d}" for i in range(1, len(_param_names_ce(idbke)) + 1)))
        for alpha, params, _, _ in ce_rows:
            lines.append(f"{alpha:10.4g} " + " ".join(f"{p:9.4g}" for p in params))
        lines.append("")
    if cm_rows:
        lines.append("DLPARME VS ALPHA:")
        lines.append(" " * 5 + "ALPHA " + " ".join(f"{i:9d}" for i in range(1, len(_param_names_cmu(idbkm)) + 1)))
        for alpha, params, _, _ in cm_rows:
            lines.append(f"{alpha:10.4g} " + " ".join(f"{p:9.4g}" for p in params))
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _select_fitter(digit: int, magnetic: bool) -> Callable:
    """Return the fitting function for an IDBKODE digit."""
    if digit == 1:
        return idb_fit.fit_debye_single if magnetic else idb_fit.fit_debye_single
    if digit == 2:
        return idb_fit.fit_lorentz_single if magnetic else idb_fit.fit_lorentz_single
    if digit == 3:
        return idb_fit.fit_debye_double if magnetic else idb_fit.fit_debye_double
    if digit == 4:
        return idb_fit.fit_lorentz_double if magnetic else idb_fit.fit_lorentz_double
    if digit == 5:
        return idb_fit.fit_debye_triple
    if digit == 6:
        return idb_fit.fit_debye_lorentz
    raise ValueError(f"Unsupported IDBKODE digit: {digit}")


def _load_tuning(ini_data: Dict[str, Dict[str, str]] | None, digit: int) -> idb_fit.FitTuning:
    """Extract tuning overrides from INI; look for section TUNING or TUNING<digit>."""
    if not ini_data:
        return idb_fit.FitTuning()
    # Digit-specific section takes priority (e.g., TUNING3)
    digit_key = f"TUNING{digit}"
    section = ini_data.get(digit_key) or ini_data.get("TUNING")
    return idb_fit.FitTuning.from_dict(section)


def _write_lsf(path: Path, alpha: float, rms: float, param_names: Sequence[str], params: Sequence[float]) -> None:
    """Write a per-alpha LSF file in a legacy-friendly format."""
    lines = []
    lines.append(f"! AUTO-GENERATED {path.name}")
    lines.append(f"! ALPHA {alpha}")
    lines.append("")
    lines.append("LEAST SQUARE FIT TO MODEL PARAMETERS")
    lines.append("!    " + "  ".join(f"{name:>10s}" for name in param_names + ["RMS"]))
    lines.append("     " + "  ".join(f"{v:10.6g}" for v in list(params) + [rms]))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _evaluate_model(
    digit: int, params: Sequence[float], f_ghz: Sequence[float], *, magnetic: bool
) -> np.ndarray:
    f = np.array(f_ghz, dtype=float)
    p = list(params)
    if magnetic:
        if digit == 1:
            return debye.single_debye(f, p[0], p[1], p[2], p[3], p[4])
        if digit == 2:
            return lorentz.single_lorentz(f, p[0], p[1], p[2], p[3], p[4])
        if digit == 3:
            return debye.double_debye(f, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
        if digit == 4:
            return lorentz.double_lorentz(f, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
        if digit == 5:
            return debye.triple_debye(f, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], gamma1=0.0, gamma2=0.0, gamma3=0.0)
        if digit == 6:
            # Debye+Lorentz hybrid: Fres_d, Deps_d, Gamm_d, Fres_l, Deps_l, Gamm_l, EPSV, SIGM
            debye_part = debye.single_debye(f, p[0], p[1], 0.0, p[2], 0.0)
            lorentz_part = lorentz.single_lorentz(f, p[3], p[4], 0.0, p[5], 0.0)
            return p[6] + (debye_part - 0.0) + (lorentz_part - 0.0) - 1j * p[7] / (np.maximum(f, 1e-6) * lorentz.A0)
    else:
        if digit == 1:
            return debye.single_debye(f, p[0], p[1], p[2], p[3], p[4])
        if digit == 2:
            return lorentz.single_lorentz(f, p[0], p[1], p[2], p[3], p[4])
        if digit == 3:
            return debye.double_debye(f, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
        if digit == 4:
            return lorentz.double_lorentz(f, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
        if digit == 5:
            return debye.triple_debye(f, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], gamma1=0.0, gamma2=0.0, gamma3=0.0)
        if digit == 6:
            debye_part = debye.single_debye(f, p[0], p[1], 0.0, p[2], 0.0)
            lorentz_part = lorentz.single_lorentz(f, p[3], p[4], 0.0, p[5], 0.0)
            return p[6] + (debye_part - 0.0) + (lorentz_part - 0.0) - 1j * p[7] / (np.maximum(f, 1e-6) * lorentz.A0)
    return np.zeros_like(f, dtype=complex)


def _write_full_output(
    path: Path,
    alpha: float,
    model_label: str,
    param_names: Sequence[str],
    params: Sequence[float],
    rms: float,
    controls: Sequence[float] | None,
    rop_rows: List[Dict],
    magnetic: bool,
    digit: int,
    sample_id: str | None,
) -> None:
    """Emit a legacy-like file with control seeds, fitted params, and measured vs fit tables."""
    ctrl_vals = list(controls or [])
    if len(ctrl_vals) < len(param_names):
        ctrl_vals += [0.0] * (len(param_names) - len(ctrl_vals))
    ctrl_names = [f"{name}0" for name in param_names]
    freqs = [row["fg_hz"] for row in rop_rows]
    f_start = freqs[0] if freqs else 0.0
    f_stop = freqs[-1] if freqs else 0.0
    nf = len(freqs)

    model = _evaluate_model(digit, params, freqs, magnetic=magnetic)

    lines: List[str] = []
    now = datetime.now()
    lines.append(f"DATE: {now:%d-%b-%Y};  TIME: {now:%H:%M:%S.%f}"[:-3])
    lines.append("")
    lines.append(f"{model_label} LSF FILE:  {path.name}")
    input_suffix = path.with_suffix(".rop").name
    lines.append(f"INPUT DATA FILE: {input_suffix}")
    lines.append(f"SAMPLE ID:       {sample_id or ''}")
    lines.append("")
    lines.append(f"FGHZ START, STOP, AND NUMBER: {f_start:8.3g}; {f_stop:8.3g}; {nf:6d}")
    lines.append("")
    lines.append("PARAMETER CONTROL;  IF PARAMETER (WITH \"0\" SUFFIX) IS:")
    lines.append("= 0 - LET THE COMPUTER ESTIMATE THE LEAST SQUARE FIT.")
    lines.append("> 0 - KEEP THE PARAMETER FIXED AT THAT VALUE.")
    lines.append("      TO FIX PARAMETER TO ZERO, SET IT EQUAL TO 1.E-8.")
    lines.append("< 0 - USE THE NEGATIVE OF THAT VALUE FOR INITIAL GUESS.")
    lines.append("")
    lines.append("  " + " ".join(f"{h:>10s}" for h in ctrl_names))
    lines.append("  " + " ".join(f"{c:10.4g}" for c in ctrl_vals[: len(ctrl_names)]))
    lines.append("")
    lines.append("LEAST SQUARE FIT TO MODEL PARAMETERS:")
    lines.append("     " + " ".join(f"{h:>10s}" for h in param_names + ["RMS"]))
    lines.append("     " + " ".join(f"{v:10.4f}" for v in list(params) + [rms]))
    lines.append("")

    if magnetic:
        lines.append("COMPARISON OF MEASURED (_M) AND LSF MODEL (_L) CMU VS FREQUENCY:")
        lines.append("   #      FGHZ       MUR_M     MUR_D   DEL_MUR       MUI_M     MUI_L   DEL_MUI")
    else:
        lines.append("COMPARISON OF MEASURED (_M) AND LSF DEBYE (_D) CEPS VS FREQUENCY:")
        lines.append("   #      FGHZ      EPSR_M    EPSR_D    DEL_ER      EPSI_M    EPSI_D    DEL_EI")

    for idx, row in enumerate(rop_rows, start=1):
        f = row["fg_hz"]
        meas_real = row["mur"] if magnetic else row["epsr"]
        meas_imag = row["mui"] if magnetic else row["epsi"]
        fit_val = model[idx - 1] if idx - 1 < len(model) else 0.0
        fit_real = float(np.real(fit_val))
        fit_imag = float(np.imag(fit_val))
        del_real = meas_real - fit_real
        del_imag = meas_imag - fit_imag
        if magnetic:
            lines.append(
                f"{idx:4d}{f:10.3f}  {meas_real:9.3f} {fit_real:9.3f} {del_real:9.3f}"
                f"  {meas_imag:9.3f} {fit_imag:9.3f} {del_imag:9.3f}"
            )
        else:
            lines.append(
                f"{idx:4d}{f:10.3f}  {meas_real:9.3f} {fit_real:9.3f} {del_real:9.3f}"
                f"  {meas_imag:9.3f} {fit_imag:9.3f} {del_imag:9.3f}"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _model_label(digit: int, magnetic: bool) -> str:
    if magnetic:
        mapping = {1: "DEBYE", 2: "LORENTZ", 3: "DOUBLE DEBYE", 4: "DOUBLE LORENTZ", 5: "TRIPLE DEBYE", 6: "DEBYE+LORENTZ"}
    else:
        mapping = {1: "DEBYE", 2: "LORENTZ", 3: "DOUBLE DEBYE", 4: "DOUBLE LORENTZ", 5: "TRIPLE DEBYE", 6: "DEBYE+LORENTZ"}
    return mapping.get(digit, "MODEL")


def generate_fits_from_lst(
    lst_path: Path,
    out_dir: Path | None = None,
    controls_ceps: Dict[float, List[float]] | None = None,
    controls_cmu: Dict[float, List[float]] | None = None,
    tuning_ini: Dict[str, Dict[str, str]] | None = None,
    pwr_ceps: List[float] | None = None,
    pwr_cmu: List[float] | None = None,
) -> Dict[str, List[Path]]:
    """Read an .LST and produce fitted .LSF summaries from corresponding .ROP files."""
    out_dir = out_dir or lst_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    legacy_idbk_map = {0: 10, 1: 20, 2: 11, 3: 21, 4: 12, 5: 22}

    lst_data = file_loader.load_lst(lst_path)
    raw_idbk = lst_data.get("idbkode")
    idbkode = legacy_idbk_map.get(raw_idbk, raw_idbk)
    entries = lst_data.get("entries", [])
    if idbkode is None or not entries:
        raise ValueError(f"{lst_path} missing IDBKODE or entries")

    idbke = idbkode // 10
    idbkm = idbkode % 10
    for digit, label in ((idbke, "CEPS"), (idbkm, "CMU")):
        if digit not in CEPS_EXT and digit not in CMU_EXT and not (label == "CMU" and digit == 0):
            raise ValueError(f"{lst_path} has unsupported {label} IDB digit {digit} (IDBKODE {idbkode})")

    created: Dict[str, List[Path]] = {"ceps": [], "cmu": []}
    combined_rows_ce: List[Tuple[float, List[float], float, str]] = []
    combined_rows_cmu: List[Tuple[float, List[float], float, str]] = []
    alpha_list: List[float] = []
    pwr_ceps = pwr_ceps or []
    pwr_cmu = pwr_cmu or []

    for alpha, stem in entries:
        rop_path = lst_path.parent / f"{stem}.rop"
        if not rop_path.exists():
            continue
        rop = file_loader.load_rop(rop_path)
        rows = rop.get("rows", [])
        if not rows:
            continue
        f_ghz = [r["fg_hz"] for r in rows]

        # CEPS fit
        if idbke in CEPS_EXT:
            epsr = [r["epsr"] for r in rows]
            epsi = [r["epsi"] for r in rows]
            fitter = _select_fitter(idbke, magnetic=False)
            ctrl_vals = None
            if controls_ceps and alpha in controls_ceps:
                ctrl_vals = controls_ceps[alpha]
            tuning = _load_tuning(tuning_ini, idbke)
            result = fitter(f_ghz, epsr, epsi, controls=ctrl_vals, tuning=tuning)
            ext = CEPS_EXT[idbke]
            lsf_path = out_dir / f"{stem}{ext}"
            _write_full_output(
                lsf_path,
                alpha,
                _model_label(idbke, magnetic=False),
                _param_names_ce(idbke),
                result.params,
                result.rms,
                ctrl_vals,
                rows,
                False,
                idbke,
                rop.get("sample_id"),
            )
            created["ceps"].append(lsf_path)
            combined_rows_ce.append((alpha, list(result.params), result.rms, lsf_path.name))

        # CMU fit
        if idbkm and idbkm in CMU_EXT:
            mur = [r["mur"] for r in rows]
            mui = [r["mui"] for r in rows]
            fitter = _select_fitter(idbkm, magnetic=True)
            ctrl_vals = None
            if controls_cmu and alpha in controls_cmu:
                ctrl_vals = controls_cmu[alpha]
            tuning = _load_tuning(tuning_ini, idbkm)
            result = fitter(f_ghz, mur, mui, controls=ctrl_vals, tuning=tuning)
            ext = CMU_EXT[idbkm]
            lsf_path = out_dir / f"{stem}{ext}"
            _write_full_output(
                lsf_path,
                alpha,
                _model_label(idbkm, magnetic=True),
                _param_names_cmu(idbkm),
                result.params,
                result.rms,
                ctrl_vals,
                rows,
                True,
                idbkm,
                rop.get("sample_id"),
            )
            created["cmu"].append(lsf_path)
            combined_rows_cmu.append((alpha, list(result.params), result.rms, lsf_path.name))
        alpha_list.append(alpha)

    # Write combined .LSF summary similar to legacy
    combined_path = out_dir / (lst_path.stem + ".LSF")
    _write_combined_lsf(
        combined_path,
        lst_path,
        idbkode,
        idbke,
        idbkm,
        combined_rows_ce,
        combined_rows_cmu,
    )

    created.setdefault("combined", []).append(combined_path)

    # Write a lightweight IDB file using polynomial fits of the current parameters.
    try:
        alpha_sorted = sorted(alpha_list)
        alpha0 = alpha_sorted[0] if alpha_sorted else 0.0
        ce_coeffs: List[List[float]] = []
        if combined_rows_ce:
            param_count_ce = len(combined_rows_ce[0][1])
            for idx in range(param_count_ce):
                vals = [row[1][idx] for row in sorted(combined_rows_ce, key=lambda r: r[0])]
                degree = int(pwr_ceps[idx]) if idx < len(pwr_ceps) else 3
                ce_coeffs.append(_fit_poly_coeffs(alpha_sorted, vals, alpha0, max_degree=degree))
        cm_coeffs: List[List[float]] = []
        if combined_rows_cmu:
            param_count_cm = len(combined_rows_cmu[0][1])
            for idx in range(param_count_cm):
                vals = [row[1][idx] for row in sorted(combined_rows_cmu, key=lambda r: r[0])]
                degree = int(pwr_cmu[idx]) if idx < len(pwr_cmu) else 3
                cm_coeffs.append(_fit_poly_coeffs(alpha_sorted, vals, alpha0, max_degree=degree))
        idb_path = out_dir / (lst_path.stem + ".IDB")
        _write_idb(
            idb_path,
            alpha0,
            idbkode,
            ce_coeffs,
            cm_coeffs,
            alpha_sorted,
            sorted(combined_rows_ce, key=lambda r: r[0]),
            sorted(combined_rows_cmu, key=lambda r: r[0]),
            _param_names_ce(idbke),
            _param_names_cmu(idbkm),
        )
        created.setdefault("idb", []).append(idb_path)
    except Exception:
        pass

    return created


def _fit_poly_coeffs(
    alphas: List[float],
    values: List[float],
    alpha0: float,
    max_degree: int = 3,
) -> List[float]:
    """Fit a polynomial in (alpha - alpha0) to the provided values."""
    pts = [(a, v) for a, v in zip(alphas, values) if np.isfinite(v)]
    if not pts:
        return [0.0]
    xs, ys = zip(*pts)
    xs_shift = np.array(xs, dtype=float) - alpha0
    ys_arr = np.array(ys, dtype=float)
    deg = min(max_degree, len(xs_shift) - 1)
    if deg < 0:
        deg = 0
    try:
        coeffs_desc = np.polyfit(xs_shift, ys_arr, deg)
        # polyfit returns highest degree first; reverse to ascending power
        coeffs = list(coeffs_desc[::-1])
    except Exception:
        coeffs = [float(np.nanmean(ys_arr))]
    return coeffs


def _write_idb(
    path: Path,
    alpha0: float,
    idbkode: int,
    ce_coeffs: List[List[float]],
    cm_coeffs: List[List[float]],
    alpha_sorted: List[float],
    ce_rows: List[Tuple[float, List[float], float, str]],
    cm_rows: List[Tuple[float, List[float], float, str]],
    ce_params: Sequence[str],
    cm_params: Sequence[str],
) -> None:
    lines: List[str] = []
    now = datetime.now()
    lines.append(f"! DATE: {now:%d-%b-%Y};  TIME: {now:%H:%M:%S.%f}"[:-3])
    lines.append(f"! IDB DATA FILE {path}")
    lines.append("!")
    lines.append("! IDB DATA FILE SYNTAX:")
    lines.append("! FIRST DATA LINE IS ALPHA0 AND IDBKODE.")
    lines.append("! IDBKODE IDENTIFIES THE TYPE OF LSF MODEL USED FOR CEPS/CMU.")
    lines.append("!")
    lines.append("")
    # CEPS LSF summary
    if ce_rows:
        lines.append("!--------------------------------------------------------------------------")
        lines.append("! LSF OF CEPS PARAMETERS VS ALPHA")
        header = ["ALPHA"] + list(ce_params) + ["RMSE"]
        lines.append("! " + " ".join(f"{h:>10s}" for h in header))
        for alpha, params, rms, _fname in ce_rows:
            row_vals = [alpha] + params + [rms]
            lines.append(" ".join(f"{v:10.4g}" for v in row_vals))
        lines.append("")

    # CMU LSF summary
    if cm_rows:
        lines.append("!--------------------------------------------------------------------------")
        lines.append("! LSF OF CMU PARAMETERS VS ALPHA")
        header = ["ALPHA"] + list(cm_params) + ["RMSM"]
        lines.append("! " + " ".join(f"{h:>10s}" for h in header))
        for alpha, params, rms, _fname in cm_rows:
            row_vals = [alpha] + params + [rms]
            lines.append(" ".join(f"{v:10.4g}" for v in row_vals))
        lines.append("")

    # IDB estimates from polynomial coefficients
    if ce_coeffs:
        lines.append("!--------------------------------------------------------------------------")
        lines.append("! IDB ESTIMATE OF CEPS PARAMETERS VS ALPHA")
        lines.append("! " + " ".join(f"{h:>10s}" for h in ["ALPHA"] + list(ce_params)))
        for a in alpha_sorted:
            vals = idb_poly.eval_params(ce_coeffs, a, alpha0)
            lines.append(" ".join([f"{a:10.4g}"] + [f"{v:10.4g}" for v in vals]))
        lines.append("")
    if cm_coeffs:
        lines.append("!--------------------------------------------------------------------------")
        lines.append("! IDB ESTIMATE OF CMU PARAMETERS VS ALPHA")
        lines.append("! " + " ".join(f"{h:>10s}" for h in ["ALPHA"] + list(cm_params)))
        for a in alpha_sorted:
            vals = idb_poly.eval_params(cm_coeffs, a, alpha0)
            lines.append(" ".join([f"{a:10.4g}"] + [f"{v:10.4g}" for v in vals]))
        lines.append("")

    lines.append("! START OF THE IDB DATA FILE.")
    lines.append(f"{alpha0:12.4E}{idbkode:6d}")
    lines.append("")
    for coeffs in ce_coeffs:
        deg = max(len(coeffs) - 1, 0)
        lines.append(f"{deg:5d}")
        lines.append(" ".join(f"{c:12.6E}" for c in coeffs))
        lines.append("")
    for coeffs in cm_coeffs:
        deg = max(len(coeffs) - 1, 0)
        lines.append(f"{deg:5d}")
        lines.append(" ".join(f"{c:12.6E}" for c in coeffs))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["generate_fits_from_lst"]
