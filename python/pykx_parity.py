#!/usr/bin/env python3
"""Parity through PyKX: run q/signals.q in-process and compare to the reference.

Same check as python/parity.py, without an HDB or a q subprocess: the CSV is
read by pandas, handed to q as a table, and .sig.build / .sig.ic run inside
the embedded q. Needs a licensed PyKX (pip install pykx plus a KDB-X licence;
PyKX's unlicensed mode cannot execute q code).

Run:  python python/pykx_parity.py [--csv FILE --n 100 --start 2019-01-01]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import reference as ref  # noqa: E402
from parity import compare  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=str(ROOT / "data/sample/ohlcv.csv"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--tol", type=float, default=1e-10)
    a = ap.parse_args()

    import pykx as kx
    if not kx.licensed:
        sys.exit("PyKX is unlicensed and cannot run q code; install a KDB-X licence (see README)")
    kx.q(f'system "l {ROOT}/q/signals.q"')

    df = pd.read_csv(a.csv, parse_dates=["date"])
    df = df.loc[df["close"] > 0, ["date", "sym", "close", "high", "volume"]]
    df["volume"] = df["volume"].astype("float64")
    t = kx.q('{update `date$date from x}', kx.toq(df))   # sym arrives as symbol, date as timestamp
    s = kx.q(".sig.build", t, a.n, kx.q(f'"D"$"{a.start}"'))
    ic_q = kx.q(".sig.ic", s).pd()
    ic_q["date"] = pd.to_datetime(ic_q["date"])
    ic_q["signal"] = ic_q["signal"].astype(str)

    ok, table = compare(ic_q, ref.run(a.csv, a.n, a.start), a.tol)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
