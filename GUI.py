"""GUI with EPS and MUS tabs for viewing LST-derived data."""

from __future__ import annotations

import re
import math
from pathlib import Path
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from Methods import file_loader
from Methods import debye
from Methods import lorentz
from Methods import idb_eval
from Methods import idb_poly
from Methods import idb_fit
from Methods import fit_from_lst

try:
    import numpy as np
except ImportError:  # numpy is optional
    np = None


# IDBKODE mappings
LEGACY_IDBKODE_MAP = {0: 10, 1: 20, 2: 11, 3: 21, 4: 12, 5: 22}
CEPS_EXT_MAP = {1: ".dbe", 2: ".lne", 3: ".2de", 4: ".2le", 5: ".3de", 6: ".dle"}
CMU_EXT_MAP = {1: ".dbm", 2: ".lnm", 3: ".2dm", 4: ".2lm", 5: ".3dm", 6: ".dlm"}


def _param_names_ce(idbke: int) -> List[str]:
    if idbke <= 2:
        return ["FRES", "DEPS", "EPSV", "GAMMA", "SIGE"]
    if idbke == 5:
        return ["FRES1", "DEPS1", "FRES2", "DEPS2", "FRES3", "DEPS3", "EPSV", "SIGE"]
    if idbke == 6:
        return ["FRES_D", "DEPS_D", "GAMM_D", "FRES_L", "DEPS_L", "GAMM_L", "EPSV", "SIGE"]
    return ["FRES1", "DEPS1", "GAMM1", "FRES2", "DEPS2", "GAMM2", "EPSV", "SIGE"]


def _param_names_cmu(idbkm: int) -> List[str]:
    if idbkm <= 2:
        return ["FR_M", "DMUR", "MURV", "GAMMU", "SIGM"]
    if idbkm == 5:
        return ["FR_M1", "DMUR1", "FR_M2", "DMUR2", "FR_M3", "DMUR3", "MURV", "SIGM"]
    if idbkm == 6:
        return ["FR_MD", "DMUR_D", "GAMM_D", "FR_ML", "DMUR_L", "GAMM_L", "MURV", "SIGM"]
    return ["FR_M1", "DMUR1", "GAMM1", "FR_M2", "DMUR2", "GAMM2", "MURV", "SIGM"]


def _select_fitter_model(digit: int):
    if digit == 1:
        return idb_fit.fit_debye_single
    if digit == 2:
        return idb_fit.fit_lorentz_single
    if digit == 3:
        return idb_fit.fit_debye_double
    if digit == 4:
        return idb_fit.fit_lorentz_double
    if digit == 5:
        return idb_fit.fit_debye_triple
    if digit == 6:
        return idb_fit.fit_debye_lorentz
    raise ValueError(f"Unsupported IDBKODE digit: {digit}")


NUM_PATTERN = re.compile(r"(?<![A-Za-z_])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?(?![A-Za-z_])")


def _extract_params_from_lsf(path: Path, param_count: int) -> (List[float], float, Optional[List[float]]):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    controls: Optional[List[float]] = None
    fallback_nums: Optional[List[float]] = None
    for i, line in enumerate(lines):
        if "PARAMETER CONTROL" in line.upper():
            for j in range(i + 1, len(lines)):
                ln = lines[j].strip()
                if not ln or ln.startswith("!"):
                    continue
                nums = [float(x) for x in NUM_PATTERN.findall(ln)]
                if not nums:
                    continue
                # Prefer the first line that supplies all parameter controls;
                # keep the first numeric line as a fallback if counts are short.
                if len(nums) >= param_count:
                    controls = nums[:param_count]
                    break
                fallback_nums = fallback_nums or nums
            break
    if controls is None and fallback_nums:
        controls = fallback_nums[:param_count]
    if controls is not None and len(controls) < param_count:
        controls += [0.0] * (param_count - len(controls))

    idx = 0
    while idx < len(lines):
        if "LEAST SQUARE FIT TO" in lines[idx].upper():
            break
        idx += 1
    while idx < len(lines):
        line = lines[idx].strip()
        if line and not line.startswith("!"):
            nums = [float(x) for x in NUM_PATTERN.findall(line)]
            if nums:
                params = nums[:param_count]
                rms = nums[param_count] if len(nums) > param_count else float("nan")
                return params, rms, controls
        idx += 1
    return [], float("nan"), controls


class TabContext:
    def __init__(self, name: str, notebook: ttk.Notebook, external_hook=None) -> None:
        self.name = name
        self.frame = ttk.Frame(notebook)
        self.external_hook = external_hook

        self.frame.columnconfigure(0, weight=1, uniform=name)
        self.frame.columnconfigure(1, weight=1, uniform=name)
        self.frame.rowconfigure(0, weight=3, uniform=name)
        self.frame.rowconfigure(1, weight=1, uniform=name)

        plot_area = ttk.LabelFrame(self.frame, text="Plot Area")
        plot_area.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        plot_area.columnconfigure(0, weight=1)
        plot_area.rowconfigure(1, weight=1)
        self.plot_title = ttk.Label(plot_area, text="Parameter vs. Alpha (PWR = --, RMS = --)")
        self.plot_title.grid(row=0, column=0, sticky="w", padx=6, pady=(6, 2))
        self.plot_canvas = tk.Canvas(plot_area, background="white")
        self.plot_canvas.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)

        quadrant = ttk.Frame(self.frame)
        quadrant.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        quadrant.columnconfigure(0, weight=1)
        quadrant.rowconfigure(0, weight=1)
        quadrant.rowconfigure(1, weight=1)

        ini_frame = ttk.LabelFrame(quadrant, text="INI Parameter Controls")
        ini_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        ini_frame.columnconfigure(0, weight=1)
        ini_frame.rowconfigure(1, weight=1)
        ini_header = ttk.Frame(ini_frame)
        ini_header.grid(row=0, column=0, sticky="w", padx=6, pady=(6, 0))
        self.update_btn = ttk.Button(ini_header, text="Update Parameter")
        self.update_btn.pack(side="left")
        self.refit_btn = ttk.Button(ini_header, text="Refit Alpha")
        self.refit_btn.pack(side="left", padx=(6, 0))
        self.tuning_btn = ttk.Button(ini_header, text="Apply Tuning")
        self.tuning_btn.pack(side="left", padx=(6, 0))

        ini_table_frame = ttk.Frame(ini_frame)
        ini_table_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        for i, text in enumerate(["File", "Alpha", "Parameter", "Mode", "Magnitude"]):
            ttk.Label(ini_table_frame, text=text).grid(row=0, column=i, sticky="w", padx=4, pady=2)
        ini_table_frame.columnconfigure(0, weight=2)
        ini_table_frame.columnconfigure(1, weight=1)
        ini_table_frame.columnconfigure(2, weight=1)
        ini_table_frame.columnconfigure(3, weight=1)
        ini_table_frame.columnconfigure(4, weight=1)

        self.poly_frame = ttk.LabelFrame(quadrant, text="Polynomial Fit")
        self.poly_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self.poly_frame.columnconfigure(0, weight=1)
        self.poly_frame.rowconfigure(1, weight=1)
        self.poly_table = ttk.Treeview(self.poly_frame, show="headings", height=4)
        self.tuning_vars = {
            "PF_INIT": tk.StringVar(value="1e-2"),
            "MAX_ITER": tk.StringVar(value="5000"),
            "KX_DIV": tk.StringVar(value="50"),
            "FRES_MIN": tk.StringVar(value="1e-4"),
            "FRES_MAX": tk.StringVar(value="999.9999"),
            "DEPS_MIN": tk.StringVar(value="1e-4"),
            "DEPS_MAX": tk.StringVar(value="999.9999"),
            "EPSV_MIN": tk.StringVar(value="0"),
            "EPSV_MAX": tk.StringVar(value="999.9999"),
            "GAMMA_MIN": tk.StringVar(value="0"),
            "GAMMA_MAX": tk.StringVar(value="10"),
            "SIGE_MIN": tk.StringVar(value="0"),
            "SIGE_MAX": tk.StringVar(value="100"),
        }

        bottom = ttk.Frame(self.frame)
        bottom.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=8, pady=8)
        bottom.columnconfigure(0, weight=7, uniform=f"{name}-bottom")
        bottom.columnconfigure(1, weight=3, uniform=f"{name}-bottom")
        bottom.rowconfigure(0, weight=1)

        main_frame = ttk.LabelFrame(bottom, text="Main Data")
        main_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        main_header = ttk.Frame(main_frame)
        main_header.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 0))
        main_header.columnconfigure(0, weight=1)
        ttk.Label(main_header, text="Dynamic Columns").grid(row=0, column=0, sticky="w")
        self.main_table = ttk.Treeview(main_frame, show="headings", height=2)
        self.main_table.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        main_scroll = ttk.Scrollbar(main_frame, orient="vertical", command=self.main_table.yview)
        self.main_table.configure(yscrollcommand=main_scroll.set)
        main_scroll.grid(row=1, column=1, sticky="ns", pady=6)

        self.side_frame = ttk.LabelFrame(bottom, text="LSF / IDB / DELTA VALUES AT ALPHAS")
        self.side_frame.grid(row=0, column=1, sticky="nsew")
        self.side_frame.columnconfigure(0, weight=1)
        self.side_frame.rowconfigure(0, weight=1)
        side_cols = ["Alpha", "LSF", "IDB", "DELTA"]
        self.side_table = ttk.Treeview(self.side_frame, columns=side_cols, show="headings", height=2)
        for col in side_cols:
            self.side_table.heading(col, text=col)
            self.side_table.column(col, anchor="center", width=80)
        self.side_table.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        side_scroll = ttk.Scrollbar(self.side_frame, orient="vertical", command=self.side_table.yview)
        self.side_table.configure(yscrollcommand=side_scroll.set)
        side_scroll.grid(row=0, column=1, sticky="ns", pady=6)

        # Stateful fields
        self.current_params: List[str] = []
        self.selected_idx: int = 0
        self.poly_state: Dict[str, object] = {"headers": ["C1", "C2", "C3"], "values": [0, 0, 0], "selected": 0}
        self.ini_table_frame = ini_table_frame
        self.ini_rows: List[tk.Widget] = []
        self.ini_vars: List[tk.Variable] = []
        self.row_meta: List[Dict[str, object]] = []
        self.alpha_params: Dict[float, List[float]] = {}
        self.controls_by_file: Dict[str, List[float]] = {}
        self.idb_digit: Optional[int] = None
        self.idb_coeffs: Optional[tuple[float, List[List[float]]]] = None
        self.alpha_list: List[float] = []

        # Wire handlers
        self._bind_handlers()

    def _bind_handlers(self) -> None:
        self.poly_table.bind("<Button-1>", self._on_poly_click)
        self.plot_canvas.bind("<Configure>", lambda e: self.update_plot())
        self.main_table.bind("<Button-1>", self._on_main_click)
        poly_header = ttk.Frame(self.poly_frame)
        poly_header.grid(row=0, column=0, sticky="w", padx=6, pady=(6, 0))
        ttk.Button(poly_header, text="Increase PWR", command=self._increase_pwr).pack(side="left", padx=(0, 4))
        ttk.Button(poly_header, text="Decrease PWR", command=self._decrease_pwr).pack(side="left")
        self.poly_frame.rowconfigure(1, weight=1)
        self.poly_frame.columnconfigure(0, weight=1)
        self.poly_table.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        poly_scrollbar = ttk.Scrollbar(self.poly_frame, orient="vertical", command=self.poly_table.yview)
        self.poly_table.configure(yscrollcommand=poly_scrollbar.set)
        poly_scrollbar.grid(row=1, column=1, sticky="ns", pady=6)

        # Tuning controls (simple grid of key/value entries)
        tuning_frame = ttk.LabelFrame(self.poly_frame, text="Tuning Overrides")
        tuning_frame.grid(row=2, column=0, sticky="nsew", padx=6, pady=6, columnspan=2)
        tuning_frame.columnconfigure(1, weight=1)
        r = 0
        for key, var in self.tuning_vars.items():
            ttk.Label(tuning_frame, text=key).grid(row=r // 3, column=(r % 3) * 2, sticky="w", padx=2, pady=2)
            ttk.Entry(tuning_frame, textvariable=var, width=10).grid(row=r // 3, column=(r % 3) * 2 + 1, sticky="w", padx=2, pady=2)
            r += 1

    def set_headers(self, headers: List[str]) -> None:
        if not headers:
            headers = ["C1"]
        self.poly_state["headers"] = headers
        vals = self.poly_state["values"]
        if len(vals) < len(headers):
            vals.extend([0] * (len(headers) - len(vals)))
        elif len(vals) > len(headers):
            self.poly_state["values"] = vals[: len(headers)]
        self.set_selected(0)

    def set_selected(self, idx: int) -> None:
        if not self.current_params:
            return
        self.selected_idx = max(0, min(idx, len(self.current_params) - 1))
        self.poly_state["selected"] = self.selected_idx
        self.refresh_poly()
        self.refresh_headings()
        self.update_plot()

    def set_update_handler(self, handler) -> None:
        try:
            self.update_btn.config(command=handler)
        except Exception:
            pass

    def set_refit_handler(self, handler) -> None:
        try:
            self.refit_btn.config(command=handler)
        except Exception:
            pass

    def sync_ini_controls(self) -> None:
        """Push current INI UI values back into row_meta controls."""
        if not self.current_params or not self.ini_vars:
            return
        # ini_vars holds [mode_var, mag_var] per row in order
        vars_per_row = 2
        for idx, item in enumerate(self.main_table.get_children(), start=0):
            row_index = idx
            if row_index >= len(self.row_meta):
                continue
            offset = row_index * vars_per_row
            if offset + 1 >= len(self.ini_vars):
                break
            mode_var = self.ini_vars[offset]
            mag_var = self.ini_vars[offset + 1]
            mode = str(mode_var.get())
            try:
                mag = float(mag_var.get())
            except Exception:
                mag = 0.0
            ctrl_val = 0.0
            if mode.startswith("Estimate"):
                ctrl_val = 0.0
            elif mode.startswith("Fixed"):
                ctrl_val = mag
            elif mode.startswith("Init Guess"):
                ctrl_val = -abs(mag)
            elif mode.startswith("Fix to Zero"):
                ctrl_val = 1e-8
            ctrls = self.row_meta[row_index].get("controls", [])
            while len(ctrls) < len(self.current_params):
                ctrls.append(0.0)
            ctrls[self.selected_idx] = ctrl_val
            self.row_meta[row_index]["controls"] = ctrls

    def refresh_headings(self) -> None:
        columns = self.main_table["columns"]
        for idx, col in enumerate(columns):
            label = col
            param_idx = idx - 1
            if 0 <= param_idx < len(self.current_params) and param_idx == self.selected_idx:
                label = f"{col} *"
            self.main_table.heading(col, text=label)

    def refresh_poly(self) -> None:
        headers = self.poly_state["headers"]
        col_count = max(len(headers), 1)
        vals = self.poly_state["values"]
        if len(vals) < col_count:
            vals.extend([0] * (col_count - len(vals)))
        elif len(vals) > col_count:
            self.poly_state["values"] = vals[:col_count]
        self.poly_state["selected"] = min(self.selected_idx, col_count - 1)

        columns = ["Metric"] + headers
        self.poly_table["columns"] = columns
        for idx, col in enumerate(columns):
            label = col
            if col != "Metric" and idx - 1 == self.selected_idx:
                label = f"{col} *"
            self.poly_table.heading(col, text=label)
            self.poly_table.column(col, anchor="center", width=85)

        def compute_rms(param_idx: int) -> Optional[float]:
            if np is None:
                return None
            col_param = param_idx + 1
            pts = []
            for item in self.main_table.get_children():
                vals_row = list(self.main_table.item(item).get("values", []))
                if len(vals_row) <= col_param:
                    continue
                try:
                    alpha = float(vals_row[0])
                    yval = float(vals_row[col_param])
                except (TypeError, ValueError):
                    continue
                pts.append((alpha, yval))
            if len(pts) < 2:
                return None
            xs = np.array([p[0] for p in pts], dtype=float)
            ys = np.array([p[1] for p in pts], dtype=float)
            deg = int(max(self.poly_state["values"][param_idx], 0))
            deg = min(deg, len(pts) - 1)
            try:
                coeffs = np.polyfit(xs, ys, deg)
                poly = np.poly1d(coeffs)
                residuals = ys - poly(xs)
                return float(np.sqrt(np.mean(residuals**2)))
            except Exception:
                return None

        self.poly_table.delete(*self.poly_table.get_children())
        self.poly_table.insert("", "end", values=["PWR"] + self.poly_state["values"])
        rms_cells: List[object] = []
        for idx in range(col_count):
            rms = compute_rms(idx)
            rms_cells.append("--" if rms is None else f"{rms:.4g}")
        self.poly_table.insert("", "end", values=["RMS"] + rms_cells)
        self.update_ini_table()

    def _increase_pwr(self) -> None:
        self.poly_state["values"][self.selected_idx] += 1
        self.refresh_poly()
        self.update_plot()
        if callable(self.external_hook):
            self.external_hook()

    def _decrease_pwr(self) -> None:
        self.poly_state["values"][self.selected_idx] -= 1
        self.refresh_poly()
        self.update_plot()
        if callable(self.external_hook):
            self.external_hook()

    def _on_poly_click(self, event: tk.Event) -> None:
        region = self.poly_table.identify_region(event.x, event.y)
        if region not in ("cell", "heading"):
            return
        col_id = self.poly_table.identify_column(event.x)
        try:
            col_index = int(col_id.lstrip("#")) - 1
        except ValueError:
            return
        if col_index >= 1:
            self.set_selected(col_index - 1)
            if callable(self.external_hook):
                self.external_hook()

    def _on_main_click(self, event: tk.Event) -> None:
        region = self.main_table.identify_region(event.x, event.y)
        if region != "heading":
            return
        col_id = self.main_table.identify_column(event.x)
        try:
            col_index = int(col_id.lstrip("#")) - 1
        except ValueError:
            return
        param_idx = col_index - 1
        if 0 <= param_idx < len(self.current_params):
            self.set_selected(param_idx)
            if callable(self.external_hook):
                self.external_hook()

    def update_side_table(self) -> None:
        if not self.current_params:
            return
        self.side_table.delete(*self.side_table.get_children())
        self.side_frame.config(
            text=f"LSF / IDB / DELTA VALUES AT ALPHAS ({self.current_params[self.selected_idx]})"
        )
        col_idx = self.selected_idx + 1
        rows = []
        for item in self.main_table.get_children():
            vals = list(self.main_table.item(item).get("values", []))
            if len(vals) <= col_idx:
                continue
            try:
                alpha = float(vals[0])
                lsf_val = float(vals[col_idx])
            except (TypeError, ValueError):
                continue
            rows.append((alpha, lsf_val))
        rows.sort(key=lambda x: x[0])

        idb_vals = ["--"] * len(rows)
        if np is not None and len(rows) >= 2:
            deg = int(max(self.poly_state["values"][self.selected_idx], 0))
            deg = min(deg, len(rows) - 1)
            xs = np.array([r[0] for r in rows], dtype=float)
            ys = np.array([r[1] for r in rows], dtype=float)
            try:
                coeffs = np.polyfit(xs, ys, deg)
                poly = np.poly1d(coeffs)
                idb_vals = [float(poly(a)) for a, _ in rows]
            except Exception:
                pass

        for (alpha, lsf_val), idb_val in zip(rows, idb_vals):
            delta = lsf_val - idb_val if isinstance(idb_val, float) else None
            self.side_table.insert(
                "",
                "end",
                values=[
                    alpha,
                    f"{lsf_val:.4g}",
                    f"{idb_val:.4g}" if isinstance(idb_val, float) else "--",
                    f"{delta:.4g}" if isinstance(delta, float) else "--",
                ],
            )
        self.update_ini_table()

    def update_plot(self) -> None:
        if not self.current_params:
            return
        col_idx = self.selected_idx + 1
        rows = []
        for item in self.main_table.get_children():
            vals = list(self.main_table.item(item).get("values", []))
            if len(vals) <= col_idx:
                continue
            try:
                alpha = float(vals[0])
                yval = float(vals[col_idx])
            except (TypeError, ValueError):
                continue
            rows.append((alpha, yval))
        rows.sort(key=lambda x: x[0])

        w = max(self.plot_canvas.winfo_width(), 220)
        h = max(self.plot_canvas.winfo_height(), 220)
        margin = 40
        self.plot_canvas.delete("all")

        def fmt(num: float) -> str:
            return f"{num:.4g}"

        param_name = self.current_params[self.selected_idx]
        pwr_val = (
            self.poly_state["values"][self.selected_idx]
            if self.selected_idx < len(self.poly_state["values"])
            else 0
        )

        rms_txt = "--"
        if np is not None and len(rows) >= 2:
            deg = int(max(pwr_val, 0))
            deg = min(deg, max(len(rows) - 1, 1))
            xs_arr = np.array([r[0] for r in rows], dtype=float)
            ys_arr = np.array([r[1] for r in rows], dtype=float)
            try:
                coeffs = np.polyfit(xs_arr, ys_arr, deg)
                poly = np.poly1d(coeffs)
                residuals = ys_arr - poly(xs_arr)
                rms_val = np.sqrt(np.mean(residuals**2))
                rms_txt = fmt(rms_val)
            except Exception:
                pass
        self.plot_title.config(text=f"{param_name} vs. Alpha (PWR = {fmt(pwr_val)}, RMS = {rms_txt})")

        if len(rows) < 2:
            self.plot_canvas.create_text(w // 2, h // 2, text="Not enough data to plot", fill="gray")
            return

        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if xmax == xmin:
            xmax += 1
            xmin -= 1
        if ymax == ymin:
            ymax += 1
            ymin -= 1

        def to_screen(x: float, y: float) -> tuple[int, int]:
            sx = margin + (x - xmin) / (xmax - xmin) * (w - 2 * margin)
            sy = h - margin - (y - ymin) / (ymax - ymin) * (h - 2 * margin)
            return int(sx), int(sy)

        self.plot_canvas.create_line(margin, margin, margin, h - margin, fill="#444")
        self.plot_canvas.create_line(margin, h - margin, w - margin, h - margin, fill="#444")

        def ticks(vmin: float, vmax: float, count: int = 4) -> List[float]:
            if vmax == vmin:
                return [vmin] * count
            step = (vmax - vmin) / max(count - 1, 1)
            return [vmin + i * step for i in range(count)]

        for xv in ticks(xmin, xmax):
            sx, sy = to_screen(xv, ymin)
            self.plot_canvas.create_line(sx, h - margin, sx, h - margin + 5, fill="#444")
            self.plot_canvas.create_text(sx, h - margin + 12, text=fmt(xv), anchor="n", fill="#444")

        for yv in ticks(ymin, ymax):
            sx, sy = to_screen(xmin, yv)
            self.plot_canvas.create_line(margin - 5, sy, margin, sy, fill="#444")
            self.plot_canvas.create_text(margin - 8, sy, text=fmt(yv), anchor="e", fill="#444")

        for x, y in rows:
            sx, sy = to_screen(x, y)
            self.plot_canvas.create_line(sx - 5, sy, sx + 5, sy, fill="#d62728", width=2)
            self.plot_canvas.create_line(sx, sy - 5, sx, sy + 5, fill="#d62728", width=2)

        degree = int(max(pwr_val, 0))
        if np is not None and len(rows) >= 2 and degree >= 1:
            deg = min(degree, len(rows) - 1)
            xs_np = np.array(xs, dtype=float)
            ys_np = np.array(ys, dtype=float)
            try:
                coeffs = np.polyfit(xs_np, ys_np, deg)
                poly = np.poly1d(coeffs)
                fit_x = np.linspace(xs_np.min(), xs_np.max(), 200)
                fit_y = poly(fit_x)
                points = [to_screen(float(xv), float(yv)) for xv, yv in zip(fit_x, fit_y)]
                for (sx0, sy0), (sx1, sy1) in zip(points[:-1], points[1:]):
                    self.plot_canvas.create_line(sx0, sy0, sx1, sy1, fill="#1f77b4", width=2)
            except Exception:
                pass

        self.update_side_table()

    def update_ini_table(self) -> None:
        """Populate INI Parameter Controls table."""
        for widget in self.ini_rows:
            widget.destroy()
        self.ini_rows = []
        self.ini_vars = []
        if not self.current_params:
            return
        col_idx = self.selected_idx + 1
        mode_options = ["Estimate (0)", "Fixed (>0)", "Init Guess (<0)", "Fix to Zero (1e-8)"]

        def set_control(row_index: int, ctrl_val: float) -> None:
            if 0 <= row_index < len(self.row_meta):
                ctrls = self.row_meta[row_index].get("controls", [])
                while len(ctrls) < len(self.current_params):
                    ctrls.append(0.0)
                ctrls[self.selected_idx] = ctrl_val
                self.row_meta[row_index]["controls"] = ctrls

        def ensure_controls(row_index: int, file_name: str) -> List[float]:
            """Verify controls are present; re-parse the source file if missing."""
            if not (0 <= row_index < len(self.row_meta)):
                return []
            meta = self.row_meta[row_index]
            ctrls = meta.get("controls", []) or []
            if len(ctrls) >= len(self.current_params):
                return ctrls
            path_val = meta.get("path")
            if isinstance(path_val, Path) and path_val.exists():
                _, _, parsed_ctrls = _extract_params_from_lsf(path_val, len(self.current_params))
                if parsed_ctrls:
                    ctrls = list(parsed_ctrls[: len(self.current_params)])
                    while len(ctrls) < len(self.current_params):
                        ctrls.append(0.0)
                    meta["controls"] = ctrls
            return ctrls

        for row_idx, item in enumerate(self.main_table.get_children(), start=1):
            vals = list(self.main_table.item(item).get("values", []))
            if len(vals) <= col_idx:
                continue
            alpha = vals[0]
            try:
                mag_val = float(vals[col_idx])
            except Exception:
                mag_val = vals[col_idx]
            fname = vals[-1] if vals else ""
            ctrl_val = 0.0
            if row_idx - 1 < len(self.row_meta):
                ctrls = ensure_controls(row_idx - 1, fname)
                if len(ctrls) > self.selected_idx:
                    ctrl_val = ctrls[self.selected_idx]
            try:
                print(f"[INI-TABLE] file={fname}, alpha={alpha}, param={self.current_params[self.selected_idx]}, ctrl={ctrl_val}")
            except Exception:
                pass

            widgets = [
                ttk.Label(self.ini_table_frame, text=fname),
                ttk.Label(self.ini_table_frame, text=alpha),
                ttk.Label(self.ini_table_frame, text=self.current_params[self.selected_idx]),
            ]
            def infer_mode(val: float) -> str:
                if abs(val - 1e-8) < 1e-10:
                    return mode_options[3]
                if val == 0:
                    return mode_options[0]
                if val > 0:
                    return mode_options[1]
                return mode_options[2]

            mode_var = tk.StringVar(value=infer_mode(ctrl_val))
            mode_box = ttk.Combobox(
                self.ini_table_frame, values=mode_options, state="readonly", width=16, textvariable=mode_var
            )
            widgets.append(mode_box)
            if mode_var.get().startswith("Fix to Zero"):
                init_mag = "0"
            elif ctrl_val < 0:
                init_mag = str(abs(ctrl_val))
            else:
                init_mag = str(ctrl_val) if ctrl_val != 0 else ""

            mag_var = tk.StringVar(value=init_mag)
            mag_entry = ttk.Entry(self.ini_table_frame, width=12, textvariable=mag_var)
            widgets.append(mag_entry)

            def refresh_applied(*_):
                mode = mode_var.get()
                try:
                    mag = float(mag_var.get())
                except Exception:
                    mag = 0.0

                if mode.startswith("Estimate"):
                    set_control(row_idx - 1, 0.0)
                elif mode.startswith("Fixed"):
                    set_control(row_idx - 1, mag)
                elif mode.startswith("Init Guess"):
                    set_control(row_idx - 1, -abs(mag))
                elif mode.startswith("Fix to Zero"):
                    set_control(row_idx - 1, 1e-8)

            mode_var.trace_add("write", refresh_applied)
            mag_var.trace_add("write", refresh_applied)
            mode_box.set(mode_var.get())
            refresh_applied()
            for col, w in enumerate(widgets):
                w.grid(row=row_idx, column=col, sticky="nsew", padx=2, pady=1)
            self.ini_rows.extend(widgets)
            self.ini_vars.extend([mode_var, mag_var])


def build_gui() -> tk.Tk:
    root = tk.Tk()
    root.title("EPS / MUS / IDB vs ROP")
    root.geometry("900x650")

    style = ttk.Style()
    base_theme = style.theme_use()
    base_bg = root.cget("bg")
    base_fg = style.lookup("TLabel", "foreground", default="black")
    base_entry_bg = style.lookup("TEntry", "fieldbackground", default="white")
    base_btn_bg = style.lookup("TButton", "background", default=base_bg)
    base_tree_bg = style.lookup("Treeview", "background", default="white")
    base_tree_fg = style.lookup("Treeview", "foreground", default=base_fg)

    status_var = tk.StringVar(value="No .LST file loaded")
    lst_state: Dict[str, object] = {"path": None}
    rop_data: Dict[float, Dict[str, object]] = {}
    alpha_values: List[float] = []

    header = ttk.Frame(root)
    header.pack(fill="x", padx=10, pady=(10, 0))

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=(6, 6))

    status_bar = ttk.Frame(root)
    status_bar.pack(side="bottom", fill="x", padx=10, pady=(0, 10))
    ttk.Label(status_bar, textvariable=status_var, anchor="w").pack(side="left")

    # Define update hook placeholder; will be set after plots are ready.
    rop_update_hook = None

    eps_ctx = TabContext("eps", notebook, external_hook=lambda: update_rop_plots())
    mus_ctx = TabContext("mus", notebook, external_hook=lambda: update_rop_plots())
    notebook.add(eps_ctx.frame, text="EPS")
    notebook.add(mus_ctx.frame, text="MUS")
    rop_frame = ttk.Frame(notebook)
    notebook.add(rop_frame, text="IDB vs ROP")
    trends_frame = ttk.Frame(notebook)
    notebook.add(trends_frame, text="Trends")

    rop_frame.columnconfigure(0, weight=1)
    rop_frame.columnconfigure(1, weight=1)
    rop_frame.rowconfigure(1, weight=1)
    rop_frame.rowconfigure(2, weight=1)

    alpha_select_frame = ttk.Frame(rop_frame)
    alpha_select_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=6)
    ttk.Label(alpha_select_frame, text="Alpha:").pack(side="left")
    alpha_var = tk.StringVar()
    alpha_box = ttk.Combobox(alpha_select_frame, textvariable=alpha_var, state="normal", width=12)
    alpha_box.pack(side="left", padx=6)
    fit_mode_var = tk.StringVar(value="pwr")
    ttk.Label(alpha_select_frame, text="Mode: PWR Fit (dynamic)").pack(side="left", padx=(12, 0))

    ceps_plot = tk.Canvas(rop_frame, background="white", height=140)
    cmu_plot = tk.Canvas(rop_frame, background="white", height=140)
    ceps_plot.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=8, pady=(8, 4))
    cmu_plot.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=8, pady=(4, 8))

    # Trends tab: parameter vs alpha
    trends_frame.columnconfigure(0, weight=1)
    trends_frame.rowconfigure(1, weight=1)
    trend_top = ttk.Frame(trends_frame)
    trend_top.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
    trend_smooth_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(trend_top, text="Smooth (moving avg)", variable=trend_smooth_var).pack(side="left", padx=10)
    trend_plots = ttk.Frame(trends_frame)
    trend_plots.grid(row=1, column=0, sticky="nsew", padx=10, pady=8)
    trend_plots.columnconfigure(0, weight=1)
    trend_plots.columnconfigure(1, weight=1)
    trend_plots.rowconfigure(0, weight=1)
    # EPS column
    trend_eps_frame = ttk.Frame(trend_plots)
    trend_eps_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
    trend_eps_frame.rowconfigure(1, weight=1)
    trend_param_buttons_eps = ttk.Frame(trend_eps_frame)
    trend_param_buttons_eps.grid(row=0, column=0, sticky="ew", pady=(0, 4))
    trend_canvas_eps = tk.Canvas(trend_eps_frame, background="white", height=260)
    trend_canvas_eps.grid(row=1, column=0, sticky="nsew")
    # MU column
    trend_mu_frame = ttk.Frame(trend_plots)
    trend_mu_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
    trend_mu_frame.rowconfigure(1, weight=1)
    trend_param_buttons_mu = ttk.Frame(trend_mu_frame)
    trend_param_buttons_mu.grid(row=0, column=0, sticky="ew", pady=(0, 4))
    trend_canvas_mu = tk.Canvas(trend_mu_frame, background="white", height=260)
    trend_canvas_mu.grid(row=1, column=0, sticky="nsew")
    trend_param_vars_eps: Dict[str, tk.BooleanVar] = {}
    trend_param_vars_mu: Dict[str, tk.BooleanVar] = {}
    trend_smooth_var.trace_add("write", lambda *_: update_trends_tab())

    def draw_measured_vs_idb(canvas: tk.Canvas, rows: List[Dict[str, float]], real_key: str, imag_key: str, title: str) -> None:
        canvas.delete("all")
        if not rows:
            canvas.create_text(canvas.winfo_width() // 2, canvas.winfo_height() // 2, text="No data", fill="gray")
            return
        xs = [r["fg_hz"] for r in rows]
        real_meas = [r[real_key] for r in rows]
        imag_meas = [r[imag_key] for r in rows]
        real_idb = [r.get("idb_real") for r in rows]
        imag_idb = [r.get("idb_imag") for r in rows]
        # Fallback to measured if IDB not provided
        if not rows or any(v is None for v in real_idb) or len(real_idb) != len(real_meas):
            real_idb = real_meas
        if not rows or any(v is None for v in imag_idb) or len(imag_idb) != len(imag_meas):
            imag_idb = imag_meas

        def fmt(num: float) -> str:
            return str(int(round(num)))

        w = max(canvas.winfo_width(), 160)
        h = max(canvas.winfo_height(), 160)
        margin = 40

        xmin, xmax = min(xs), max(xs)
        if xmax == xmin:
            xmax += 1
            xmin -= 1
        yall_min = min(real_meas + real_idb + imag_meas + imag_idb)
        yall_max = max(real_meas + real_idb + imag_meas + imag_idb)
        if yall_max == yall_min:
            yall_max += 1
            yall_min -= 1

        def to_screen(x: float, y: float) -> tuple[int, int]:
            sx = margin + (x - xmin) / (xmax - xmin) * (w - 2 * margin)
            sy = h - margin - (y - yall_min) / (yall_max - yall_min) * (h - 2 * margin)
            return int(sx), int(sy)

        # Axes (single shared Y)
        is_dark = getattr(canvas, "dark_mode", False)
        axis_color = "#f0f0f0" if is_dark else "#444"
        text_color = "#fafafa" if is_dark else "#1f1f1f"
        canvas.create_line(margin, margin, margin, h - margin, fill=axis_color)
        canvas.create_line(margin, h - margin, w - margin, h - margin, fill=axis_color)
        canvas.create_text(w - margin, h - margin + 14, text="Freq (GHz)", anchor="e", fill=text_color)
        canvas.create_text(margin, margin - 10, text=title, anchor="w", fill=text_color)

        def freq_ticks(vmin: float, vmax: float) -> List[float]:
            a = math.floor(vmin)
            b = math.ceil(vmax)
            ticks = list(range(int(a), int(b) + 1))
            if len(ticks) < 2:
                ticks = [vmin, vmax]
            return ticks

        for xv in freq_ticks(xmin, xmax):
            sx, sy = to_screen(xv, yall_min)
            canvas.create_line(sx, h - margin, sx, h - margin + 5, fill=axis_color)
            canvas.create_text(sx, h - margin + 14, text=str(int(round(xv))), anchor="n", fill=text_color)

        for yv in np.linspace(yall_min, yall_max, num=5):
            sx, sy = to_screen(xmin, yv)
            canvas.create_line(margin - 5, sy, margin, sy, fill=axis_color)
            canvas.create_text(margin - 8, sy, text=fmt(yv), anchor="e", fill=text_color)

        def draw_series(vals_y, color, dash=None, width=2):
            pts = [to_screen(x, y) for x, y in zip(xs, vals_y)]
            for (sx0, sy0), (sx1, sy1) in zip(pts[:-1], pts[1:]):
                canvas.create_line(sx0, sy0, sx1, sy1, fill=color, width=width, dash=dash)

        # Draw measured first, then IDB dashed on top so both are visible even when identical.
        draw_series(real_meas, "#1f77b4")                # blue measured real
        draw_series(imag_meas, "#2ca02c")                # green measured imag
        draw_series(real_idb, "#d62728", dash=(4, 2))    # red idb real
        draw_series(imag_idb, "#ff7f0e", dash=(4, 2))    # orange idb imag

        # Legend
        legend_items = [
            ("Meas Real", "#1f77b4"),
            ("IDB Real", "#d62728"),
            ("Meas Imag", "#2ca02c"),
            ("IDB Imag", "#ff7f0e"),
        ]
        lx = margin + 10
        ly = margin + 10
        for i, (label, color) in enumerate(legend_items):
            canvas.create_line(lx, ly + i * 14, lx + 20, ly + i * 14, fill=color, width=2)
            canvas.create_text(lx + 26, ly + i * 14, text=label, anchor="w", fill=text_color)

    def update_rop_plots(*_) -> None:
        try:
            alpha = float(alpha_var.get())
        except Exception:
            ceps_plot.delete("all")
            cmu_plot.delete("all")
            status_var.set("Alpha is not a number")
            return

        data = rop_data.get(alpha)
        fallback_alpha = None
        if not data:
            if alpha_values:
                # Use nearest measured alpha for frequency grid/measurements
                nearest = min(alpha_values, key=lambda a: abs(a - alpha))
                data = rop_data.get(nearest, {})
                fallback_alpha = nearest
            else:
                ceps_plot.delete("all")
                cmu_plot.delete("all")
                status_var.set(f"No data available for alpha {alpha}")
                return
        rows = data.get("rows", [])

        if fallback_alpha is not None:
            status_var.set(f"No measured data at alpha {alpha}; using measured grid from {fallback_alpha}")

        def build_idb_rows(tab_ctx: TabContext, real_key: str, imag_key: str) -> List[Dict[str, float]]:
            if tab_ctx.idb_digit is None or not tab_ctx.alpha_params:
                return rows

            params_at_alpha: List[float] = []
            degs = tab_ctx.poly_state["values"]
            alphas = np.array(tab_ctx.alpha_list, dtype=float)
            for i in range(len(tab_ctx.current_params)):
                vals = np.array([tab_ctx.alpha_params[a][i] for a in tab_ctx.alpha_list], dtype=float)
                deg = int(max(degs[i] if i < len(degs) else 0, 0))
                deg = min(deg, max(len(alphas) - 1, 0))
                try:
                    if deg <= 0 or len(alphas) <= 1:
                        fit_val = vals.mean() if len(vals) else 0.0
                    else:
                        coeffs_np = np.polyfit(alphas, vals, deg)
                        fit_val = np.polyval(coeffs_np, alpha)
                    params_at_alpha.append(float(fit_val))
                except Exception:
                    params_at_alpha.append(float(vals.mean()) if len(vals) else 0.0)

            freq = [r["fg_hz"] for r in rows]
            f_arr = np.array(freq, dtype=float)
            vals = idb_eval.eval_dispersion(tab_ctx.idb_digit, params_at_alpha, f_arr)
            if vals is None:
                status_var.set(f"{tab_ctx.name.upper()} IDB eval failed in {mode_lbl} mode")
                return rows

            out = []
            for base, comp in zip(rows, vals):
                entry = dict(base)
                entry["idb_real"] = float(comp.real)
                entry["idb_imag"] = float(comp.imag)
                out.append(entry)
            return out

        mode_lbl = "PWR Fit"
        ceps_rows = build_idb_rows(eps_ctx, "epsr", "epsi")
        cmu_rows = build_idb_rows(mus_ctx, "mur", "mui")
        draw_measured_vs_idb(ceps_plot, ceps_rows, "epsr", "epsi", f"CEPS: Measured vs IDB ({mode_lbl})")
        draw_measured_vs_idb(cmu_plot, cmu_rows, "mur", "mui", f"CMU: Measured vs IDB ({mode_lbl})")

    alpha_box.bind("<<ComboboxSelected>>", update_rop_plots)

    def _on_tab_changed(event=None):
        try:
            current = notebook.tab(notebook.select(), "text")
        except Exception:
            current = ""
        if current == "IDB vs ROP":
            root.after_idle(update_rop_plots)
        elif current == "Trends":
            root.after_idle(update_trends_tab)

    notebook.bind("<<NotebookTabChanged>>", _on_tab_changed)

    dark_mode = {"enabled": False}

    def apply_dark_mode(enabled: bool) -> None:
        dark_mode["enabled"] = enabled
        if enabled:
            bg = "#1b1b1b"
            fg = "#f2f2f2"
            alt = "#252525"
            border = "#3a3a3a"
            accent = "#4a4a4a"
            combo_bg = "#2d2d2d"
            style.theme_use("clam")
            for elem in ("TFrame", "TLabelframe"):
                style.configure(elem, background=bg)
            style.configure("TLabelframe.Label", background=bg, foreground=fg)
            style.configure("TLabel", background=bg, foreground=fg)
            style.configure("TButton", background=accent, foreground=fg)
            style.configure("TNotebook", background=border)
            style.configure("TNotebook.Tab", background=alt, foreground=fg)
            style.map("TNotebook.Tab", background=[("selected", accent)])
            style.configure("Treeview", background=alt, fieldbackground=alt, foreground=fg, bordercolor=border)
            style.configure("Treeview.Heading", background=accent, foreground=fg, relief="flat")
            style.configure("TEntry", fieldbackground=combo_bg, background=combo_bg, foreground=fg, insertcolor=fg)
            style.configure("TCombobox", fieldbackground=combo_bg, background=combo_bg, foreground=fg, arrowcolor=fg, bordercolor=border)
            root.configure(background=bg)
            eps_ctx.plot_canvas.config(background=bg)
            mus_ctx.plot_canvas.config(background=bg)
            ceps_plot.config(background=bg)
            cmu_plot.config(background=bg)
            trend_canvas_eps.config(background=bg)
            trend_canvas_mu.config(background=bg)
        else:
            style.theme_use(base_theme)
            style.map("TNotebook.Tab", background=[], foreground=[])
            style.configure("TFrame", background=base_bg)
            style.configure("TLabelframe", background=base_bg)
            style.configure("TLabelframe.Label", background=base_bg, foreground=base_fg)
            style.configure("TLabel", background=base_bg, foreground=base_fg)
            style.configure("TButton", background=base_btn_bg, foreground=base_fg)
            style.configure("TNotebook", background=base_bg)
            style.configure("TNotebook.Tab", background=base_bg, foreground=base_fg)
            style.configure("Treeview", background=base_tree_bg, fieldbackground=base_tree_bg, foreground=base_tree_fg, bordercolor=base_bg)
            style.configure("Treeview.Heading", background=base_bg, foreground=base_fg, relief="raised")
            style.configure("TEntry", fieldbackground=base_entry_bg, background=base_entry_bg, foreground=base_fg, insertcolor=base_fg)
            style.configure("TCombobox", fieldbackground=base_entry_bg, background=base_entry_bg, foreground=base_fg, arrowcolor=base_fg, bordercolor=base_bg)
            root.configure(background=base_bg)
            eps_ctx.plot_canvas.config(background="white")
            mus_ctx.plot_canvas.config(background="white")
            ceps_plot.config(background="white")
            cmu_plot.config(background="white")
            trend_canvas_eps.config(background="white")
            trend_canvas_mu.config(background="white")
        # Propagate dark flag to ROP canvases for tick/legend colors
        ceps_plot.dark_mode = enabled
        cmu_plot.dark_mode = enabled
        update_rop_plots()
        update_trends_tab()

    def toggle_dark_mode() -> None:
        apply_dark_mode(not dark_mode["enabled"])

    def _get_trend_points(param_name: str, magnetic: bool) -> tuple[list[float], list[float]]:
        """Collect alpha values and chosen parameter for CEPS (magnetic=False) or CMU (magnetic=True)."""
        if not param_name:
            return [], []
        alphas = []
        values = []
        lookup = combined_rows_cmu_alphas if magnetic else combined_rows_ce_alphas
        names = combined_param_names_cmu if magnetic else combined_param_names_ce
        if not names or param_name not in names:
            return [], []
        idx = names.index(param_name)
        for alpha in sorted(lookup.keys()):
            params = lookup[alpha]
            if idx < len(params):
                alphas.append(alpha)
                values.append(params[idx])
        return alphas, values

    def _draw_trend(canvas: tk.Canvas, series: list[tuple[list[float], list[float], str]], title: str) -> None:
        canvas.delete("all")
        is_dark = getattr(canvas, "dark_mode", False)
        axis_color = "#e0e0e0" if is_dark else "#666"
        text_color = "#f8f8f8" if is_dark else "#1f1f1f"
        if not series:
            canvas.create_text(canvas.winfo_width() // 2, canvas.winfo_height() // 2, text="No data", fill="gray")
            return
        w = max(canvas.winfo_width(), 200)
        h = max(canvas.winfo_height(), 180)
        margin = 40
        all_alphas = [a for s in series for a in s[0]]
        all_vals = [v for s in series for v in s[1]]
        xmin, xmax = min(all_alphas), max(all_alphas)
        ymin, ymax = min(all_vals), max(all_vals)
        if xmax == xmin:
            xmax += 1
            xmin -= 1
        if ymax == ymin:
            ymax += 1
            ymin -= 1

        def to_screen(x: float, y: float) -> tuple[int, int]:
            sx = margin + (x - xmin) / (xmax - xmin) * (w - 2 * margin)
            sy = h - margin - (y - ymin) / (ymax - ymin) * (h - 2 * margin)
            return int(sx), int(sy)

        canvas.create_text(margin, margin - 16, text=title, anchor="w", fill=text_color)
        for xv in (xmin, xmax):
            sx, sy = to_screen(xv, ymin)
            canvas.create_line(sx, h - margin, sx, h - margin + 5, fill=axis_color)
            canvas.create_text(sx, h - margin + 12, text=f"{xv:.3g}", anchor="n", fill=text_color)
        for yv in (ymin, ymax):
            sx, sy = to_screen(xmin, yv)
            canvas.create_line(margin - 5, sy, margin, sy, fill=axis_color)
            canvas.create_text(margin - 8, sy, text=f"{yv:.3g}", anchor="e", fill=text_color)

        color_cycle = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
        for idx, (alphas, vals, label) in enumerate(series):
            vals_plot = vals
            if trend_smooth_var.get() and len(vals) > 3:
                win = max(2, max(len(vals) // 10, 2))
                smoothed = []
                for i in range(len(vals)):
                    lo = max(0, i - win // 2)
                    hi = min(len(vals), i + win // 2 + 1)
                    smoothed.append(sum(vals[lo:hi]) / (hi - lo))
                vals_plot = smoothed
            pts = [to_screen(a, v) for a, v in zip(alphas, vals_plot)]
            color = color_cycle[idx % len(color_cycle)]
            for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
                canvas.create_line(x0, y0, x1, y1, fill=color, width=2)
            for (x, y) in pts:
                canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline=color)
            canvas.create_text(margin + 10, margin + 12 + idx * 12, text=label, anchor="w", fill=color)

    def update_trends_tab(*_) -> None:
        series_eps: list[tuple[list[float], list[float], str]] = []
        for name, var in trend_param_vars_eps.items():
            if var.get():
                a_eps, v_eps = _get_trend_points(name, magnetic=False)
                if a_eps and v_eps:
                    series_eps.append((a_eps, v_eps, name))
        _draw_trend(trend_canvas_eps, series_eps, "EPS params vs Alpha")

        series_mu: list[tuple[list[float], list[float], str]] = []
        for name, var in trend_param_vars_mu.items():
            if var.get():
                a_mu, v_mu = _get_trend_points(name, magnetic=True)
                if a_mu and v_mu:
                    series_mu.append((a_mu, v_mu, name))
        _draw_trend(trend_canvas_mu, series_mu, "MU params vs Alpha")
    combined_rows_ce_alphas: Dict[float, List[float]] = {}
    combined_rows_cmu_alphas: Dict[float, List[float]] = {}
    combined_param_names_ce: List[str] = []
    combined_param_names_cmu: List[str] = []

    def _load_lst_from_path(lst_path: Path) -> None:
        try:
            lst_data = file_loader.load_lst(lst_path)
        except Exception as exc:  # noqa: BLE001
            status_var.set(f"Failed to load {lst_path.name}: {exc}")
            return
        # Try to load optional INI alongside the LST with same stem.
        ini_path = lst_path.with_suffix(".ini")
        ini_data = None
        if ini_path.exists():
            try:
                ini_data = file_loader.load_ini(ini_path)
                lst_state["ini"] = ini_data
                # Apply tuning defaults to UI if present
                tuning_section = ini_data.get("TUNING")
                if tuning_section:
                    for key, var in eps_ctx.tuning_vars.items():
                        if key in tuning_section:
                            var.set(tuning_section[key])
                    for key, var in mus_ctx.tuning_vars.items():
                        if key in tuning_section:
                            var.set(tuning_section[key])
            except Exception:
                lst_state["ini"] = None
        else:
            lst_state["ini"] = None

        idbkode_raw = lst_data.get("idbkode")
        if idbkode_raw is None:
            status_var.set(f"{lst_path.name}: missing IDBKODE")
            return
        idbkode = LEGACY_IDBKODE_MAP.get(idbkode_raw, idbkode_raw)
        idbke = idbkode // 10
        idbkm = idbkode % 10

        rop_data.clear()
        alpha_values.clear()
        base_dir = lst_path.parent
        for alpha, stem in lst_data.get("entries", []):
            rop_path = base_dir / f"{stem}.rop"
            if not rop_path.exists():
                continue
            try:
                rop_data[alpha] = file_loader.load_rop(rop_path)
                alpha_values.append(alpha)
            except Exception:
                continue
        alpha_values.sort()
        combined_rows_ce_alphas.clear()
        combined_rows_cmu_alphas.clear()
        combined_param_names_ce.clear()
        combined_param_names_cmu.clear()
        # Preload LSF-derived parameters for trends
        try:
            idbkode_raw = lst_data.get("idbkode")
            idbkode = LEGACY_IDBKODE_MAP.get(idbkode_raw, idbkode_raw)
            idbke = idbkode // 10
            idbkm = idbkode % 10
        except Exception:
            idbke = idbkm = 0
        for alpha, stem in lst_data.get("entries", []):
            if idbke:
                lsf_ce = (base_dir / f"{stem}{CEPS_EXT_MAP.get(idbke, '')}")
                if lsf_ce.exists():
                    params, _, _ = _extract_params_from_lsf(lsf_ce, len(_param_names_ce(idbke)))
                    if params:
                        combined_rows_ce_alphas[alpha] = params
                        if not combined_param_names_ce:
                            combined_param_names_ce.extend(_param_names_ce(idbke))
            if idbkm:
                lsf_cm = (base_dir / f"{stem}{CMU_EXT_MAP.get(idbkm, '')}")
                if lsf_cm.exists():
                    params, _, _ = _extract_params_from_lsf(lsf_cm, len(_param_names_cmu(idbkm)))
                    if params:
                        combined_rows_cmu_alphas[alpha] = params
                        if not combined_param_names_cmu:
                            combined_param_names_cmu.extend(_param_names_cmu(idbkm))
        # Build parameter toggle buttons
        def _rebuild_param_buttons(frame: ttk.Frame, names: List[str], vars_dict: Dict[str, tk.BooleanVar]):
            for child in frame.winfo_children():
                child.destroy()
            vars_dict.clear()
            for name in names:
                var = tk.BooleanVar(value=False)
                vars_dict[name] = var
                btn = ttk.Checkbutton(frame, text=name, variable=var, command=lambda: update_trends_tab())
                btn.pack(side="left", padx=2, pady=2)

        _rebuild_param_buttons(trend_param_buttons_eps, combined_param_names_ce, trend_param_vars_eps)
        _rebuild_param_buttons(trend_param_buttons_mu, combined_param_names_cmu, trend_param_vars_mu)
        if alpha_values:
            alpha_box["values"] = [str(a) for a in alpha_values]
            alpha_var.set(str(alpha_values[0]))
            update_rop_plots()
            update_trends_tab()

        idb_parsed = None
        idb_path = lst_path.with_suffix(".IDB")
        if idb_path.exists():
            idb_parsed = idb_poly.parse_idb_file(str(idb_path))

        def populate_tab(ctx: TabContext, use_cmu: bool) -> None:
            digit = idbkm if use_cmu else idbke
            ext_map = CMU_EXT_MAP if use_cmu else CEPS_EXT_MAP
            ext = ext_map.get(digit)
            entries = lst_data.get("entries", [])
            if not ext or not entries:
                return
            param_names = _param_names_cmu(digit) if use_cmu else _param_names_ce(digit)
            param_count = len(param_names)

            ctx.current_params = param_names
            ctx.selected_idx = 0
            ctx.row_meta = []
            ctx.alpha_params = {}
            ctx.alpha_list = []
            ctx.controls_by_file = {}
            # Attach IDB polynomial coefficients if available
            ctx.idb_coeffs = None
            if idb_parsed is not None:
                alpha0, idb_parsed_code, ce_coeffs, cm_coeffs = idb_parsed
                coeffs = cm_coeffs if use_cmu else ce_coeffs
                if coeffs:
                    ctx.idb_coeffs = (alpha0, coeffs)

            def _mode_and_mag(val: float) -> tuple[str, float]:
                if abs(val - 1e-8) < 1e-10:
                    return "Fix to Zero", 0.0
                if val == 0:
                    return "Estimate", 0.0
                if val > 0:
                    return "Fixed", val
                return "Init Guess", abs(val)

            columns = ["Alpha"] + param_names + ["RMS", "Filename"]
            ctx.main_table["columns"] = columns
            for col in columns:
                anchor = "w" if col in {"Alpha", "Filename"} else "center"
                width = 90
                if col == "Alpha":
                    width = 70
                if col == "Filename":
                    width = 140
                ctx.main_table.heading(col, text=col)
                ctx.main_table.column(col, anchor=anchor, width=width, stretch=True)
            ctx.main_table.delete(*ctx.main_table.get_children())

            base_dir_local = lst_path.parent
            for alpha_val, stem in sorted(entries, key=lambda t: t[0]):
                file_path = base_dir_local / f"{stem}{ext}"
                if not file_path.exists():
                    continue
                params, rms, controls = _extract_params_from_lsf(file_path, param_count)
                if len(params) < param_count:
                    params += [float('nan')] * (param_count - len(params))
                ctx.main_table.insert("", "end", values=[alpha_val] + params + [rms, file_path.name])
                control_vals = controls[:param_count] if controls else [0.0] * param_count
                ctx.row_meta.append({"alpha": alpha_val, "controls": control_vals, "params": params, "path": file_path})
                ctx.controls_by_file[file_path.name] = control_vals
                ctx.alpha_params[alpha_val] = params
                ctx.alpha_list.append(alpha_val)
                # Debug print for modes/magnitudes
                try:
                    mode_printout = []
                    for name, val in zip(param_names, control_vals):
                        mode, mag = _mode_and_mag(val)
                        mode_printout.append(f"{name}={mode} (mag {mag})")
                    print(f"[INI] {file_path.name}: " + "; ".join(mode_printout))
                except Exception:
                    print(f"[INI] {file_path.name}: unable to derive modes")
            ctx.idb_digit = digit

            ctx.set_headers(param_names)
            ctx.refresh_headings()
            ctx.update_plot()

        populate_tab(eps_ctx, use_cmu=False)
        populate_tab(mus_ctx, use_cmu=True)
        status_var.set(f"Loaded: {lst_path.name}")
        lst_state["path"] = lst_path
        update_rop_plots()

    def load_lst() -> None:
        filename = filedialog.askopenfilename(
            title="Select .LST file", filetypes=[("LST files", "*.lst"), ("All files", "*.*")]
        )
        if not filename:
            return
        _load_lst_from_path(Path(filename))

    def edit_lst() -> None:
        lst_path = lst_state.get("path")
        if not lst_path:
            status_var.set("Load an .LST first")
            return
        try:
            lst_data = file_loader.load_lst(Path(lst_path))
        except Exception as exc:  # noqa: BLE001
            status_var.set(f"Failed to read LST: {exc}")
            return

        entries = lst_data.get("entries", [])
        idbkode_val = lst_data.get("idbkode") or 0

        win = tk.Toplevel(root)
        win.title(f"Edit LST – {Path(lst_path).name}")

        ttk.Label(win, text="IDBKODE:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        idbk_var = tk.StringVar(value=str(idbkode_val))
        ttk.Entry(win, textvariable=idbk_var, width=10).grid(row=0, column=1, sticky="w", padx=6, pady=4)

        table = ttk.Frame(win)
        table.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=6, pady=6)
        table.columnconfigure(0, weight=1)
        table.columnconfigure(1, weight=1)
        ttk.Label(table, text="Alpha").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Label(table, text="Stem").grid(row=0, column=1, sticky="w", padx=4, pady=2)

        alpha_vars: list[tk.StringVar] = []
        stems: list[str] = []
        for idx, (alpha_val, stem) in enumerate(entries, start=1):
            a_var = tk.StringVar(value=str(alpha_val))
            alpha_vars.append(a_var)
            stems.append(stem)
            ttk.Entry(table, textvariable=a_var, width=12).grid(row=idx, column=0, sticky="w", padx=4, pady=2)
            ttk.Label(table, text=stem).grid(row=idx, column=1, sticky="w", padx=4, pady=2)

        def apply_changes() -> None:
            try:
                new_idbk = int(float(idbk_var.get()))
            except Exception:
                messagebox.showerror("Edit LST", "Enter a valid IDBKODE (integer).")
                return
            new_entries: list[tuple[float, str]] = []
            for a_var, stem in zip(alpha_vars, stems):
                try:
                    aval = float(a_var.get())
                except Exception:
                    messagebox.showerror("Edit LST", f"Invalid alpha value: {a_var.get()}")
                    return
                new_entries.append((aval, stem))

            # Write LST with updated IDBKODE/alphas
            lines = [f"{len(new_entries)} {new_idbk}"]
            for aval, stem in new_entries:
                lines.append(f"{aval}")
                lines.append(stem)
            try:
                Path(lst_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
            except OSError as exc:  # noqa: BLE001
                messagebox.showerror("Edit LST", f"Unable to save LST: {exc}")
                return
            _load_lst_from_path(Path(lst_path))
            win.destroy()

        btns = ttk.Frame(win)
        btns.grid(row=2, column=0, columnspan=2, pady=8)
        ttk.Button(btns, text="Apply", command=apply_changes).pack(side="left", padx=6)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="left", padx=6)

    ttk.Button(header, text="Load .LST", command=load_lst).pack(side="left")
    ttk.Button(header, text="Edit .LST", command=edit_lst).pack(side="left", padx=(4, 0))
    ttk.Button(header, text="Dark Mode", command=toggle_dark_mode).pack(side="left", padx=(4, 0))

    def _build_control_map(ctx: TabContext) -> Dict[float, List[float]]:
        mapping: Dict[float, List[float]] = {}
        for entry in ctx.row_meta:
            alpha_val = entry.get("alpha")
            ctrls = entry.get("controls", [])
            if alpha_val is None:
                continue
            mapping[float(alpha_val)] = list(ctrls)
        return mapping

    def _refit_selected(ctx: TabContext, magnetic: bool) -> None:
        lst_path = lst_state.get("path")
        if not lst_path:
            status_var.set("Load an .LST first")
            return
        selection = ctx.main_table.selection()
        if not selection:
            status_var.set("Select a row in the main table to refit.")
            return
        item_id = selection[0]
        vals = list(ctx.main_table.item(item_id).get("values", []))
        if len(vals) < 2:
            status_var.set("Selection is missing values.")
            return
        try:
            alpha_val = float(vals[0])
        except Exception:
            status_var.set("Unable to parse alpha for selection.")
            return
        rop_entry = rop_data.get(alpha_val)
        if not rop_entry:
            status_var.set(f"No ROP data for alpha {alpha_val}.")
            return
        rows = rop_entry.get("rows", [])
        if not rows:
            status_var.set(f"No ROP rows for alpha {alpha_val}.")
            return
        controls = next((m.get("controls", []) for m in ctx.row_meta if m.get("alpha") == alpha_val), [])
        try:
            fitter = _select_fitter_model(ctx.idb_digit or 0)
        except Exception as exc:  # noqa: BLE001
            status_var.set(f"Unsupported IDBKODE digit: {exc}")
            return
        tuning_ini = lst_state.get("ini")
        tuning = None
        if tuning_ini and "TUNING" in tuning_ini:
            tuning = idb_fit.FitTuning.from_dict(tuning_ini["TUNING"])
        else:
            tuning = idb_fit.FitTuning.from_dict({k: v.get() for k, v in ctx.tuning_vars.items()})
        f_ghz = [r["fg_hz"] for r in rows]
        if magnetic:
            real = [r["mur"] for r in rows]
            imag = [r["mui"] for r in rows]
        else:
            real = [r["epsr"] for r in rows]
            imag = [r["epsi"] for r in rows]
        result = fitter(f_ghz, real, imag, controls=controls, tuning=tuning)
        param_count = len(ctx.current_params)
        params = list(result.params)[:param_count]
        if len(params) < param_count:
            params += [float("nan")] * (param_count - len(params))
        new_values = [alpha_val] + params + [result.rms, vals[-1]]
        ctx.main_table.item(item_id, values=new_values)
        ctx.alpha_params[alpha_val] = params
        for meta in ctx.row_meta:
            if meta.get("alpha") == alpha_val:
                meta["params"] = params
                break
        ctx.update_plot()
        ctx.update_ini_table()
        status_var.set(f"Refit alpha {alpha_val} complete.")
        # Update trend caches and buttons
        if not magnetic:
            combined_rows_ce_alphas[alpha_val] = params
            if not combined_param_names_ce:
                combined_param_names_ce.extend(ctx.current_params)
            _rebuild_param_buttons(trend_param_buttons_eps, combined_param_names_ce, trend_param_vars_eps)
        else:
            combined_rows_cmu_alphas[alpha_val] = params
            if not combined_param_names_cmu:
                combined_param_names_cmu.extend(ctx.current_params)
            _rebuild_param_buttons(trend_param_buttons_mu, combined_param_names_cmu, trend_param_vars_mu)
        update_trends_tab()

    def generate_fits() -> None:
        lst_path = lst_state.get("path")
        if not lst_path:
            status_var.set("Load an .LST first")
            return
        try:
            # Ensure we capture the latest UI-edited control values.
            eps_ctx.sync_ini_controls()
            mus_ctx.sync_ini_controls()
            ce_ctrls = _build_control_map(eps_ctx)
            mu_ctrls = _build_control_map(mus_ctx)
            tuning_ini = lst_state.get("ini")
            # Merge live tuning overrides from UI into TUNING section so they apply to fits.
            if tuning_ini is None:
                tuning_ini = {}
            tuning_ini = dict(tuning_ini)
            tuning_ini.setdefault("TUNING", {})
            for key, var in eps_ctx.tuning_vars.items():
                tuning_ini["TUNING"][key] = var.get()
            created = fit_from_lst.generate_fits_from_lst(
                Path(lst_path),
                controls_ceps=ce_ctrls,
                controls_cmu=mu_ctrls,
                tuning_ini=tuning_ini,
                pwr_ceps=eps_ctx.poly_state.get("values"),
                pwr_cmu=mus_ctx.poly_state.get("values"),
            )
            total = sum(len(v) for v in created.values())
            status_var.set(f"Generated {total} LSF files from {Path(lst_path).name}")
            # Reload the freshly written outputs so the tables/plots stay in sync.
            try:
                _load_lst_from_path(Path(lst_path))
            except Exception:
                pass
        except Exception as exc:  # noqa: BLE001
            status_var.set(f"Fit generation failed: {exc}")
    ttk.Button(header, text="Generate Fits", command=generate_fits).pack(side="left", padx=(6, 0))

    # Wire the per-tab "Update Parameter" buttons to re-run fits using current controls.
    eps_ctx.set_update_handler(generate_fits)
    mus_ctx.set_update_handler(generate_fits)
    eps_ctx.set_refit_handler(lambda: _refit_selected(eps_ctx, magnetic=False))
    mus_ctx.set_refit_handler(lambda: _refit_selected(mus_ctx, magnetic=True))
    return root


def main() -> None:
    root = build_gui()
    root.mainloop()


if __name__ == "__main__":
    main()
