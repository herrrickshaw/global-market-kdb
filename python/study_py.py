#!/usr/bin/env python3
"""Run the upstream study (the parity oracle) on an interchange CSV.

Calls python/upstream/signal_research.py's own study_market() for every market in
the CSV, then repeats the cross-market post-processing from its main(): the
BH-FDR q-values, the Harvey-Liu-Zhu |t| >= 3 flag and the Deflated Sharpe over
all cost rows. Only the data loader is swapped. Upstream's load_market() reads
the Parquet warehouse and casts to float32; here panels come from the CSV in
float64, the same numbers q reads.

Writes to --out: ic_series.csv, ic_table.csv, fama_macbeth_10d.csv,
walk_forward_10d.csv, costs.csv, health.csv.

Run:  python python/study_py.py data/sample/ohlcv.csv --n 100 --start 2019-01-01 --out out/py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "upstream"))
import signal_research as sr  # noqa: E402
import signal_validation as sv  # noqa: E402


def load_panels(csv: str | Path) -> dict[str, dict[str, pd.DataFrame]]:
    df = pd.read_csv(csv, parse_dates=["date"])
    df = df[df["close"] > 0]
    out = {}
    for mkt, g in df.groupby("mkt"):
        out[mkt] = {name: g.pivot_table(index="date", columns="sym", values=col, aggfunc="last")
                         .sort_index().astype("float64")
                    for col, name in (("close", "Close"), ("high", "High"), ("volume", "Volume"))}
    return out


def run(csv, n: int, start: str) -> dict[str, pd.DataFrame]:
    panels = load_panels(csv)
    sr.START = start
    sr.load_market = lambda root, mkt: {k: v.copy() for k, v in panels[mkt].items()}
    series = []
    for m in panels:
        sr.UNIVERSE_N[m] = n
        sr.MARKET_NAME.setdefault(m, m)      # unknown markets fall back to the 0.3% cost
        S, F, _, _ = sr.build_signals(panels[m], m)
        for k in S:
            for h in sr.HORIZONS:
                ic = sv.rank_ic_series(S[k], F[h])
                series.append(pd.DataFrame({"market": m, "signal": k, "h": h,
                                            "date": ic.index, "ic": ic.values}))
    with contextlib.redirect_stdout(io.StringIO()):
        results = {m: sr.study_market(m, None) for m in panels}

    # --- upstream main(), cross-market post-processing --------------------------
    ic = pd.DataFrame([r for m in results.values() for r in m["ic"]])
    ic["q_bh"] = sv.bh_fdr(ic["p"])
    ic["hlz_t3"] = ic["nw_t"].abs() >= 3.0
    fm = pd.concat([m["fm"] for m in results.values()], ignore_index=True)
    wfs = [m["wf"] for m in results.values() if m["wf"] is not None and len(m["wf"])]
    wf = pd.concat(wfs, ignore_index=True) if wfs else pd.DataFrame()
    cost = pd.DataFrame([r for m in results.values() for r in m["cost"]])
    srs = [np.nanmean(s) / np.nanstd(s, ddof=1) for s in cost["_ls_series"] if len(s) > 10]
    sr_var = float(np.nanvar(srs, ddof=1))
    cost["dsr"] = [sv.deflated_sharpe(s, n_trials=len(cost), sr_trials_var=sr_var)[1]
                   if len(s) > 10 else np.nan for s in cost["_ls_series"]]
    cost = cost.drop(columns="_ls_series")
    health = pd.DataFrame([{"market": m, "signal": k, **v}
                           for m, r in results.items() for k, v in r["health"].items()])
    return {"ic_series": pd.concat(series, ignore_index=True), "ic_table": ic,
            "fama_macbeth_10d": fm, "walk_forward_10d": wf, "costs": cost, "health": health}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv")
    ap.add_argument("--n", type=int, default=100, help="universe size for the liquidity gate")
    ap.add_argument("--start", default="2019-01-01", help="first formation date")
    ap.add_argument("--out", default="out/py")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in run(a.csv, a.n, a.start).items():
        t.to_csv(out / f"{name}.csv", index=False, float_format="%.17g", date_format="%Y-%m-%d")
        print(f"{name}: {len(t)} rows")


if __name__ == "__main__":
    main()
