#!/usr/bin/env python3
"""Parity test: per-date rank IC from q must match python/reference.py.

Steps: build a date-partitioned HDB from the CSV with q/load.q, run q/run.q,
run the Python reference on the same CSV, then compare every (date, signal)
row: same valid-name count, same null pattern, |IC_q - IC_py| <= tol.

Run:
  python python/parity.py                                  # committed synthetic sample
  python python/parity.py --csv data/local/US.csv --n 100  # a real slice (extract_sample.py)
  Q=/path/to/q python python/parity.py                     # q not on PATH
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import reference as ref  # noqa: E402


def find_q() -> str | None:
    for c in (os.environ.get("Q"), shutil.which("q"), str(Path.home() / ".kx/bin/q")):
        if c and Path(c).exists():
            return c
    return None


def run_q(q: str, csv: Path, mkt: str, n: int, start: str, work: Path) -> pd.DataFrame:
    hdb, out = work / "db", work / "ic_q.csv"
    shutil.rmtree(hdb, ignore_errors=True)
    for args in (["q/load.q", "-csv", str(csv), "-hdb", str(hdb)],
                 ["q/run.q", "-hdb", str(hdb), "-mkt", mkt, "-n", str(n),
                  "-start", start.replace("-", "."), "-out", str(out)]):
        p = subprocess.run([q, *args, "-q"], cwd=ROOT, capture_output=True, text=True)
        sys.stdout.write(p.stdout)
        if p.returncode:
            sys.exit(f"q {args[0]} failed ({p.returncode}):\n{p.stderr}")
    df = pd.read_csv(out)
    df["date"] = pd.to_datetime(df["date"], format="%Y.%m.%d")
    return df


def compare(q: pd.DataFrame, py: pd.DataFrame, tol: float) -> tuple[bool, pd.DataFrame]:
    m = py.merge(q, on=["date", "signal"], how="outer", suffixes=("_py", "_q"), indicator=True)
    rows, ok = [], True
    for k, g in m.groupby("signal", sort=False):
        missing = int((g["_merge"] != "both").sum())
        n_diff = int((g["n_py"] != g["n_q"]).sum())
        nan_diff = int((g["ic_py"].isna() != g["ic_q"].isna()).sum())
        d = (g["ic_py"] - g["ic_q"]).abs()
        dmax = float(d.max()) if d.notna().any() else 0.0
        good = missing == 0 and n_diff == 0 and nan_diff == 0 and dmax <= tol
        ok &= good
        rows.append({"signal": k, "dates": len(g), "valid_ic": int(g["ic_py"].notna().sum()),
                     "max_abs_diff": dmax, "n_mismatch": n_diff, "null_mismatch": nan_diff,
                     "unmatched_rows": missing, "pass": good})
    return ok, pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=str(ROOT / "data/sample/ohlcv.csv"))
    ap.add_argument("--mkt", default=None, help="market code in the CSV (default: the only one)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--work", default=str(ROOT / "out"))
    a = ap.parse_args()

    q = find_q()
    if q is None:
        sys.exit("q not found: install KDB-X (see README) or set Q=/path/to/q")
    csv, work = Path(a.csv).resolve(), Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    mkt = a.mkt or pd.read_csv(csv, usecols=["mkt"], nrows=1)["mkt"].iloc[0]

    ic_q = run_q(q, csv, mkt, a.n, a.start, work)
    ic_py = ref.run(csv, a.n, a.start)
    ok, table = compare(ic_q, ic_py, a.tol)

    L = math.ceil(ref.H / 5)
    sq, sp = ref.summary(ic_q, L), ref.summary(ic_py, L)
    table = table.merge(sp[["signal", "mean_ic", "nw_t"]], on="signal")
    table["mean_ic_q"] = sq["mean_ic"].values
    print(f"\nparity on {csv.name}: n={a.n}, start={a.start}, tol={a.tol:g}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
