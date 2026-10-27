#!/usr/bin/env python3
"""Parity test: the q study must reproduce the upstream Python study.

Builds a date-partitioned HDB from the CSV with q/load.q, runs the full study
with q/run.q, runs the vendored upstream study (python/study_py.py) on the same
CSV, then compares six tables row by row on their keys:

  ic_series         market, signal, h, date        per-date rank IC
  ic_table          market, signal, h              IC/t/p/ICIR/hit, neutral, regime, BH q
  fama_macbeth_10d  market, signal                 joint coefficients
  costs             market, signal, h              spreads, turnover, net returns, DSR
  walk_forward_10d  market, fold                   out-of-sample combination IC
  health            market, signal                 trailing vs full IC, status

Numbers match when |q - py| <= atol + rtol*|py| (same null pattern required);
labels, counts and flags must be equal.

Run:
  python python/parity.py                                  # committed synthetic sample
  python python/parity.py --csv data/local/US.csv --work out/us
  Q=/path/to/q python python/parity.py                     # q not on PATH
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import study_py  # noqa: E402

KEYS = {"ic_series": ["market", "signal", "h", "date"], "ic_table": ["market", "signal", "h"],
        "fama_macbeth_10d": ["market", "signal"], "costs": ["market", "signal", "h"],
        "walk_forward_10d": ["market", "fold"], "health": ["market", "signal"]}

# Cells whose upstream value is floating-point noise, so no two correct
# implementations agree on it. lowvol is -vol63 and vol63 is one of the two
# neutralisation exposures, so rank(lowvol) is a linear function of rank(vol)
# and the neutralised residual is round-off (median |residual| ~2e-16). Only
# the null pattern of these cells is compared.
EXEMPT = {("ic_table", "lowvol"): {"neutral_ic", "neutral_t"}}


def find_q() -> str | None:
    for c in (os.environ.get("Q"), shutil.which("q"), str(Path.home() / ".kx/bin/q")):
        if c and Path(c).exists():
            return c
    return None


def run_q(q: str, csv: Path | None, n: int, start: str, work: Path,
          pq: dict | None = None) -> dict[str, pd.DataFrame]:
    """Load (CSV via q/load.q, or the Parquet warehouse via q/pqload.q), then run the study."""
    hdb, out = work / "db", work / "q"
    shutil.rmtree(hdb, ignore_errors=True)
    load = (["q/load.q", "-csv", str(csv), "-hdb", str(hdb)] if pq is None else
            ["q/pqload.q", "-root", pq["root"], "-mkts", *pq["mkts"], "-from", pq["from"].replace("-", "."),
             "-to", pq["to"].replace("-", "."), "-names", str(pq["names"]), "-hdb", str(hdb)])
    for args in (load,
                 ["q/run.q", "-hdb", str(hdb), "-univ", str(n),
                  "-start", start.replace("-", "."), "-out", str(out)]):
        p = subprocess.run([q, *args, "-q"], cwd=ROOT, capture_output=True, text=True)
        sys.stdout.write(p.stdout.split("\nmarket ")[0].rstrip() + "\n")
        if p.returncode:
            sys.exit(f"q {args[0]} failed ({p.returncode}):\n{p.stderr}{p.stdout}")
    return {k: pd.read_csv(out / f"{k}.csv") for k in KEYS
            if (out / f"{k}.csv").stat().st_size > 1}


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """Dates to datetime, True/False strings to bool, whatever the writer was."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        s = df[c].astype(str)
        if s.str.fullmatch(r"\d{4}[-.]\d{2}[-.]\d{2}").all():
            df[c] = pd.to_datetime(s.str.replace(".", "-", regex=False))
        elif set(s.unique()) <= {"True", "False"}:
            df[c] = s == "True"
    return df


def compare(name: str, q: pd.DataFrame | None, py: pd.DataFrame,
            atol: float, rtol: float) -> dict:
    if q is None or py is None or len(py) == 0:
        same = (q is None or len(q) == 0) and (py is None or len(py) == 0)
        return {"table": name, "rows": 0, "max_abs_diff": 0.0, "mismatches": 0 if same else 1,
                "pass": same, "worst": ""}
    q, py = _norm(q), _norm(py)
    keys = KEYS[name]
    for k in keys:
        if k == "date":
            q[k], py[k] = pd.to_datetime(q[k]), pd.to_datetime(py[k])
        elif k in ("h", "fold"):
            q[k], py[k] = q[k].astype(int), py[k].astype(int)
        else:
            q[k], py[k] = q[k].astype(str), py[k].astype(str)
    m = py.merge(q, on=keys, how="outer", suffixes=("_py", "_q"), indicator=True)
    bad, worst, wcol = int((m["_merge"] != "both").sum()), 0.0, ""
    for c in [c for c in py.columns if c not in keys]:
        a, b = m[f"{c}_q"], m[f"{c}_py"]
        if pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(b):
            a, b = pd.to_numeric(a, errors="coerce").astype(float), b.astype(float)
            nan_diff = a.isna() != b.isna()
            d = (a - b).abs()
            skip = pd.Series(False, index=m.index)
            for (tbl, sig), cols in EXEMPT.items():
                if tbl == name and c in cols:
                    skip |= m["signal"] == sig
            off = ((d > atol + rtol * b.abs()) & ~skip) | nan_diff
            d = d.where(~skip)
            bad += int(off.sum())
            if d.notna().any() and d.max() > worst:
                worst, wcol = float(d.max()), c
        else:
            if pd.api.types.is_bool_dtype(b):
                a = a.astype(str).isin(["1", "True", "true"])
            bad += int((a.astype(str) != b.astype(str)).sum())
    return {"table": name, "rows": len(m), "max_abs_diff": worst, "mismatches": bad,
            "pass": bad == 0, "worst": wcol}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=str(ROOT / "data/sample/ohlcv.csv"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--work", default=str(ROOT / "out"))
    ap.add_argument("--parquet", metavar="ROOT", help="read the warehouse directly (q: pq module; "
                    "Python: DuckDB) instead of --csv")
    ap.add_argument("--mkts", nargs="+", default=["US"])
    ap.add_argument("--from", dest="d0", default="2016-06-01")
    ap.add_argument("--to", dest="d1", default="2022-12-31")
    ap.add_argument("--names", type=int, default=200)
    a = ap.parse_args()

    q = find_q()
    if q is None:
        sys.exit("q not found: install KDB-X (see README) or set Q=/path/to/q")
    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if a.parquet:
        import extract_sample
        root = str(Path(a.parquet).expanduser())
        pq = {"root": root, "mkts": a.mkts, "from": a.d0, "to": a.d1, "names": a.names}
        res_q = run_q(q, None, a.n, a.start, work, pq)
        src = pd.concat([extract_sample.query(root, m, a.names, a.d0, a.d1) for m in a.mkts],
                        ignore_index=True)
        label = f"parquet {root} {' '.join(a.mkts)} {a.d0}..{a.d1}, top {a.names}"
    else:
        csv = Path(a.csv).resolve()
        res_q = run_q(q, csv, a.n, a.start, work)
        src, label = csv, csv.name
    res_py = study_py.run(src, a.n, a.start)
    rows = [compare(k, res_q.get(k), res_py.get(k), a.atol, a.rtol) for k in KEYS]
    table = pd.DataFrame(rows)
    print(f"\nparity on {label}: n={a.n}, start={a.start}, atol={a.atol:g}, rtol={a.rtol:g}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2e}"))
    ok = bool(table["pass"].all())
    for (tbl, sig), cols in EXEMPT.items():
        print(f"exempt (null pattern only): {tbl} {sig} {sorted(cols)} - degenerate upstream value, see README")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
