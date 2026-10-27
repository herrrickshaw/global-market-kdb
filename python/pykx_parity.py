#!/usr/bin/env python3
"""Parity through PyKX: run the q study in-process and compare to upstream.

Same check as python/parity.py, without an HDB or a q subprocess: the CSV is
read by pandas, handed to q as a table, and .study.run executes inside the
embedded q. Needs a licensed PyKX (pip install pykx plus a KDB-X licence;
PyKX's unlicensed mode cannot execute q code). Point it at the licence with
QLIC, e.g. QLIC=~/.kx.

Run:  QLIC=~/.kx python python/pykx_parity.py [--csv FILE --n 100 --start 2019-01-01]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import study_py  # noqa: E402
from parity import EXEMPT, KEYS, compare  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=str(ROOT / "data/sample/ohlcv.csv"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    a = ap.parse_args()

    import pykx as kx
    if not kx.licensed:
        sys.exit("PyKX is unlicensed and cannot run q code; install a KDB-X licence (see README)")
    for f in ("stats", "signals", "study"):
        kx.q(f'system "l {ROOT}/q/{f}.q"')

    df = pd.read_csv(a.csv, parse_dates=["date"])
    df = df.loc[df["close"] > 0, ["date", "mkt", "sym", "open", "high", "low", "close", "volume"]]
    df["volume"] = df["volume"].astype("float64")
    t = kx.q('{update `date$date from x}', kx.toq(df))      # mkt/sym arrive as symbols
    res = kx.q(".study.run", t, a.n, kx.q(f'"D"$"{a.start}"'))
    res_q = {k: res[k].pd() for k in KEYS if len(res[k])}

    res_py = study_py.run(a.csv, a.n, a.start)
    table = pd.DataFrame([compare(k, res_q.get(k), res_py.get(k), a.atol, a.rtol) for k in KEYS])
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2e}"))
    for (tbl, sig), cols in EXEMPT.items():
        print(f"exempt (null pattern only): {tbl} {sig} {sorted(cols)}")
    ok = bool(table["pass"].all())
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
