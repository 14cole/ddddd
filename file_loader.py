"""Utility loaders for VBIDB-related file formats.

Supported inputs (case-insensitive):
- .lst: pairs of ALPHA and file stem plus NALPHA/IDBKODE header.
- .lsf: least-square-fit summary tables for CEPS/CMU.
- .idb: interpolated data base files (tables + polynomial segments).
- .dbe/.dbm/.lne/.lnm/.2de/.2dm/.2le/.2lm/.3de/.3dm/.dle/.dlm: single
  Debye/Lorentz/Lorentz-Debye LSF result headers.
- .rop: measured material data files.
- .ini: simple key/value config via configparser.

Use load_file(path) to dispatch based on extension.
"""

from __future__ import annotations

import argparse
import configparser
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


NUM_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?")


def _read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\r\n") for line in f]


def _parse_numbers(text: str) -> List[float]:
    return [float(m.group(0)) for m in NUM_RE.finditer(text)]


def load_lst(path: Path) -> Dict:
    lines = _read_lines(path)
    data_lines = [
        ln for ln in lines if ln.strip() and not ln.strip().startswith("!")
    ]
    if not data_lines:
        return {"nalphas": 0, "idbkode": None, "entries": []}

    header_nums = _parse_numbers(data_lines[0])
    if len(header_nums) < 2:
        raise ValueError(f"Missing NALPHA/IDBKODE in {path}")
    nalpha = int(header_nums[0])
    idbkode = int(header_nums[1])

    entries: List[Tuple[float, str]] = []
    for i in range(1, len(data_lines), 2):
        try:
            alpha_vals = _parse_numbers(data_lines[i])
            alpha = float(alpha_vals[0])
        except (IndexError, ValueError):
            continue
        if i + 1 >= len(data_lines):
            break
        stem = data_lines[i + 1].strip()
        entries.append((alpha, stem))

    if len(entries) != nalpha:
        raise ValueError(f"{path} expected {nalpha} entries but found {len(entries)}")

    return {"nalphas": nalpha, "idbkode": idbkode, "entries": entries}


def load_lsfit_header(path: Path) -> Dict:
    lines = _read_lines(path)
    header_line: Optional[str] = None
    for line in lines:
        if "LEAST SQUARE FIT" in line.upper():
            header_line = line
            break

    numeric_line: Optional[str] = None
    for line in lines:
        if not line.strip() or line.strip().startswith("!"):
            continue
        if NUM_RE.search(line):
            numeric_line = line
            break

    values = _parse_numbers(numeric_line or "")
    return {"header": header_line, "values": values}


def _consume_table(
    lines: List[str], start_idx: int, expected_cols: Optional[int] = None
) -> Tuple[Dict, int, List[str]]:
    headers = re.sub(r"^[! ]+", "", lines[start_idx]).split()
    rows = []
    idx = start_idx + 1
    warnings: List[str] = []
    # Use expected column count if provided, else infer from headers (minus any trailing label column)
    exp = expected_cols if expected_cols is not None else len(headers)
    while idx < len(lines):
        raw = lines[idx]
        if not raw.strip():
            break
        if raw.lstrip().startswith("!"):
            if rows:
                break
            idx += 1
            continue
        nums = _parse_numbers(raw)
        if not nums:
            break
        if exp and len(nums) < exp:
            warnings.append(f"Row {idx+1} truncated: expected {exp} numbers, found {len(nums)}")
        suffix = raw[raw.rfind(str(nums[-1])) + len(str(nums[-1])) :].strip()
        row = {"values": nums}
        if suffix:
            row["label"] = suffix
        rows.append(row)
        idx += 1
    return {"headers": headers, "rows": rows}, idx, warnings


def load_lsf(path: Path) -> Dict:
    lines = _read_lines(path)
    tables = []
    warnings: List[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if "ALPHA" in line.upper() and line.lstrip().startswith("!"):
            # Expect at least one numeric header to estimate column count
            headers = re.sub(r"^[! ]+", "", lines[idx]).split()
            expected_cols = max(len(headers) - 1, 0)  # minus potential label column
            table, idx, w = _consume_table(lines, idx, expected_cols if expected_cols else None)
            warnings.extend([f"{path}: {msg}" for msg in w])
            tables.append(table)
        else:
            idx += 1
    return {"tables": tables, "warnings": warnings}


def load_idb(path: Path) -> Dict:
    lines = _read_lines(path)
    idx = 0
    ceps_table = cmu_table = None
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("!     ALPHA      FRES"):
            ceps_table, idx = _consume_table(lines, idx)
            continue
        if line.startswith("!     ALPHA     FR_M1"):
            cmu_table, idx = _consume_table(lines, idx)
            continue
        idx += 1

    # Polynomial segment parsing.
    poly_ceps: List[Dict] = []
    poly_cmu: List[Dict] = []
    al0: Optional[float] = None
    idbkode: Optional[int] = None
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("! START OF THE IDB DATA FILE"):
            start_idx = i + 1
            break
    if start_idx is not None:
        nums = _parse_numbers(lines[start_idx]) if start_idx < len(lines) else []
        if len(nums) >= 2:
            al0, idbkode = nums[0], int(nums[1])
        idx = start_idx + 1
        current = poly_ceps
        while idx < len(lines):
            power_line = lines[idx].strip()
            if not power_line:
                idx += 1
                continue
            pwr_nums = _parse_numbers(power_line)
            if len(pwr_nums) != 1:
                idx += 1
                continue
            pwr = int(pwr_nums[0])
            idx += 1
            if idx >= len(lines):
                break
            coeffs = _parse_numbers(lines[idx])
            current.append({"power": pwr, "coeffs": coeffs})
            idx += 1
            # Heuristic: switch to CMU after five CEPS segments if CMU exists.
            if idbkode and len(current) == 5 and current is poly_ceps:
                current = poly_cmu

    return {
        "alpha_fits": {"ceps": ceps_table, "cmu": cmu_table},
        "polynomials": {"ceps": poly_ceps, "cmu": poly_cmu},
        "alpha0": al0,
        "idbkode": idbkode,
    }


def load_rop(path: Path) -> Dict:
    lines = _read_lines(path)
    sample_id = None
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line.upper().startswith("! SAMPLE ID:"):
            sample_id = line.split(":", 1)[1].strip()
        if line and not line.startswith("!"):
            break
        idx += 1
    if idx >= len(lines):
        return {"sample_id": sample_id, "nf": 0, "rows": []}

    nf_dens = _parse_numbers(lines[idx])
    if len(nf_dens) < 2:
        raise ValueError(f"Cannot parse NF/DENS in {path}")
    nf = int(nf_dens[0])
    dens = nf_dens[1]
    rows = []
    for raw in lines[idx + 1 : idx + 1 + nf]:
        nums = _parse_numbers(raw)
        if len(nums) < 5:
            continue
        rows.append(
            {
                "fg_hz": nums[0],
                "mur": nums[1],
                "mui": nums[2],
                "epsr": nums[3],
                "epsi": nums[4],
            }
        )
    return {"sample_id": sample_id, "nf": nf, "dens": dens, "rows": rows}


def load_ini(path: Path) -> Dict:
    config = configparser.ConfigParser()
    config.read(path)
    # Normalize keys to upper-case for easier lookups; values remain strings.
    data: Dict[str, Dict[str, str]] = {}
    for section in config.sections():
        section_data = {}
        for key, val in config.items(section):
            section_data[key.upper()] = val
        data[section.upper()] = section_data
    return data


EXT_TO_LOADER = {
    ".lst": load_lst,
    ".lsf": load_lsf,
    ".idb": load_idb,
    ".rop": load_rop,
    ".ini": load_ini,
}

LSF_EXTS = {
    ".dbe",
    ".dbm",
    ".lne",
    ".lnm",
    ".2de",
    ".2dm",
    ".2le",
    ".2lm",
    ".3de",
    ".3dm",
    ".dle",
    ".dlm",
}


def load_file(path_like) -> Dict:
    path = Path(path_like)
    ext = path.suffix.lower()
    if ext in EXT_TO_LOADER:
        return EXT_TO_LOADER[ext](path)
    if ext in LSF_EXTS:
        return load_lsfit_header(path)
    raise ValueError(f"Unsupported file type: {ext}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Load VBIDB-format files.")
    parser.add_argument("paths", nargs="+", help="Files to inspect.")
    args = parser.parse_args(argv)

    for name in args.paths:
        try:
            data = load_file(name)
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: error -> {exc}")
            continue
        print(f"{name}:")
        print(json.dumps(data, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
