#!/usr/bin/env python3
"""Check python/reference.py against the original upstream code.

Imports signal_research.build_signals and signal_validation.rank_ic_series from
a local checkout of herrrickshaw/global-market-scanners, feeds them the same
sample panels (cast to float32, as upstream's load_market does), and reports
how far the upstream per-date IC is from this repo's float64 reference.
The gap is float32 rounding only; formulas are shared.

Run:
  python python/check_upstream.py data/sample/ohlcv.csv \
      --scanners ~/repos/global-market-scanners
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference as ref  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv")
    ap.add_argument("--scanners", default=str(Path.home() / "repos/global-market-scanners"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--start", default="2019-01-01")
    a = ap.parse_args()

    sys.path.insert(0, a.scanners)
    import signal_research as sr
    import signal_validation as sv

    sr.UNIVERSE_N["SYN"] = a.n
    sr.START = a.start
    sr.HORIZONS = (ref.H,)
    p = {k: v.astype("float32") for k, v in ref.load_panels(a.csv).items()}
    S_up, F_up, _, _ = sr.build_signals(p, "SYN")

    ours = ref.run(a.csv, a.n, a.start)
    worst = 0.0
    print(f"{'signal':8s} {'dates':>5s} {'max|dIC|':>10s} {'mean IC up':>11s} {'mean IC ref':>11s}")
    for k in ref.SIGNALS:
        up = sv.rank_ic_series(S_up[k], F_up[ref.H])
        mine = ours[ours.signal == k].set_index("date")["ic"]
        up, mine = up.align(mine, join="outer")
        same_nan = bool((up.isna() == mine.isna()).all())
        d = float(np.nanmax(np.abs(up - mine))) if mine.notna().any() else 0.0
        worst = max(worst, d)
        print(f"{k:8s} {int(mine.notna().sum()):5d} {d:10.2e} {up.mean():11.6f} {mine.mean():11.6f}"
              + ("" if same_nan else "  NaN pattern differs"))
    print(f"worst per-date |IC(upstream float32) - IC(reference float64)| = {worst:.2e}")


if __name__ == "__main__":
    main()
